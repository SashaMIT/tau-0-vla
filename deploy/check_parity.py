"""eager vs optim numerical-parity check for the Tau0VLA deploy acceleration.

Feeds the SAME tokenized VLM input and the SAME fixed noise tensor through both
inference backends and reports how far their action outputs diverge:

  - eager : max_prefix_len=0  → DynamicKVCache, no padding, no CUDA graph, no compile
  - optim : max_prefix_len=N  → StaticKVCache + CUDA-graph prefix + torch.compile denoise

Only ``max_prefix_len`` (and the lazy graph/compile caches) differ between the two
runs — everything upstream (encode → tokenize → noise) is identical, so any output
difference is attributable to the acceleration path alone.

Because bf16 + CUDA-graph replay + torch.compile legitimately perturb the last few
mantissa bits, exact equality is not expected; the pass/fail verdict uses a loose
tolerance and the report breaks the diff down by active vs. inactive action dims.

Usage (inside the deploy container):
  python -m deploy.check_parity --model "$CKPT_DIR" --max-prefix-len 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from deploy._bootstrap import (
    discover_checkpoint_config_modules,
    ensure_configs_registered,
    ensure_policy_manifest,
)
from deploy.policy import Tau0VLAPolicy
from deploy.warmup import _configure_inference_mode, _dummy_payload
from tau0_vla.data import encode_payload


def _sample_with_noise(policy, model_input, noise):
    """Mirror ``Tau0VLAModel.sample_action`` but inject a fixed noise tensor so the
    only stochastic source is pinned identically across backends."""
    model = policy.model
    state = model_input["state"]
    if state.ndim == 3:
        state = state.squeeze(1)
    actions = model.flow_matching.sample_actions(
        state=state,
        input_ids=model_input["input_ids"],
        attention_mask=model_input["attention_mask"],
        pixel_values=model_input.get("pixel_values"),
        image_grid_thw=model_input.get("image_grid_thw"),
        pixel_values_videos=model_input.get("pixel_values_videos"),
        video_grid_thw=model_input.get("video_grid_thw"),
        action_mask=model_input.get("action_mask"),
        noise=noise.clone(),  # defensive: each run gets its own copy
    )
    return actions[:, :, : model.config.action_dim].detach().cpu().float().numpy()[0]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--route", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-prefix-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=2e-2, help="verdict abs tolerance")
    p.add_argument("--rtol", type=float, default=1e-2, help="verdict rel tolerance")
    args = p.parse_args()

    ensure_configs_registered()
    ensure_policy_manifest(args.model, verbose=False)
    discover_checkpoint_config_modules(args.model)
    policy = Tau0VLAPolicy.from_checkpoint(args.model, route=args.route, device=args.device)
    print(f"[parity] route={policy.data_spec.finch_config_name}")

    # Deterministic input: encode + tokenize ONCE → identical for both modes.
    payload = _dummy_payload(policy, seed=args.seed)
    encoded = encode_payload(payload, policy.data_spec)
    model_input = policy._tokenize_vlm(encoded)

    model = policy.model
    fm = model.flow_matching
    device = policy.device
    dtype = next(model.parameters()).dtype
    bsize = model_input["state"].shape[0]

    # Fixed noise (same tensor for both backends) — the only randomness in sample_actions.
    g = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        (bsize, model.config.n_action_steps, model.config.max_action_dim),
        generator=g, device=device, dtype=dtype,
    )
    has_mask = "action_mask" in model_input
    prefix_len = int(model_input["input_ids"].shape[1])
    print(f"[parity] prefix_len={prefix_len}  max_prefix_len={args.max_prefix_len}  "
          f"action_mask={has_mask}  noise={tuple(noise.shape)}  dtype={dtype}")

    # --- eager baseline ---
    _configure_inference_mode(policy, "eager", 0)
    assert fm.config.max_prefix_len == 0
    a_eager = _sample_with_noise(policy, model_input, noise)

    # --- optim (1st call captures graph + compiles; 2nd is steady-state) ---
    _configure_inference_mode(policy, "optim", args.max_prefix_len)
    a_optim1 = _sample_with_noise(policy, model_input, noise)
    a_optim2 = _sample_with_noise(policy, model_input, noise)
    used_static = fm._cuda_graph_prefix is not None
    print(f"[parity] optim engaged StaticKVCache/CUDA-graph path: {used_static} "
          f"(max_prefix_len={fm.config.max_prefix_len})")
    if not used_static:
        print("[parity] WARNING: optim fell back to eager (no CUDA?) — parity is trivially true "
              "but the acceleration was NOT exercised.")

    active = None
    if has_mask:
        am = model_input["action_mask"].detach().cpu().float().numpy().reshape(-1)[: a_eager.shape[-1]]
        active = am > 0.5

    def report(name: str, a: np.ndarray, b: np.ndarray) -> float:
        d = np.abs(a - b)
        rel = d / (np.abs(b) + 1e-8)
        print(f"\n=== {name} ===")
        print(f"  shape={a.shape}  max_abs_diff={d.max():.3e}  mean_abs_diff={d.mean():.3e}  "
              f"max_rel_diff={rel.max():.3e}")
        for atol in (1e-3, 5e-3, 1e-2, 2e-2, 5e-2):
            print(f"  allclose(atol={atol:.0e}, rtol={args.rtol:.0e}): {np.allclose(a, b, atol=atol, rtol=args.rtol)}")
        if active is not None:
            if active.any():
                da = d[:, active]
                print(f"  active   dims ({int(active.sum()):2d}): max_abs_diff={da.max():.3e}  mean={da.mean():.3e}")
            if (~active).any():
                di = d[:, ~active]
                print(f"  inactive dims ({int((~active).sum()):2d}): max_abs_diff={di.max():.3e}  (should be ~0)")
        return float(d.max())

    max_diff = report("optim (steady / 2nd call)  vs  eager", a_optim2, a_eager)
    report("optim (1st call, capture)  vs  eager", a_optim1, a_eager)
    report("optim 2nd  vs  optim 1st  (replay determinism)", a_optim2, a_optim1)

    for arr, nm in ((a_eager, "eager"), (a_optim2, "optim")):
        if not np.isfinite(arr).all():
            print(f"  [WARN] {nm} output contains NaN/Inf!")

    verdict = "PASS" if np.allclose(a_optim2, a_eager, atol=args.atol, rtol=args.rtol) else "FAIL"
    print(f"\n[parity] VERDICT (atol={args.atol}, rtol={args.rtol}): {verdict}   max_abs_diff={max_diff:.3e}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
