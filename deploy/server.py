"""HTTP server — ``python -m deploy.server --model <ckpt> [--route <leaf>]``."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
del _sys, _Path

import argparse
import logging
import pickle

import numpy as np
from fastapi import FastAPI, Request

from deploy._bootstrap import (
    discover_checkpoint_config_modules,
    ensure_configs_registered,
    ensure_policy_manifest,
    resolve_deploy_io,
)
from deploy.policy import Tau0VLAPolicy
from deploy.warmup import _configure_inference_mode, _run_dummy_input_warmup
from tau0_vla.data import action_slices as _action_slices

logger = logging.getLogger(__name__)


def _split_action_chunk(action: np.ndarray, slices) -> dict:
    """Flat ``[chunk, native_dim]`` → ``{name: [[...], ...]}`` per ``tau0_vla.data``
    ``action_slices``. Inline: this is the server's /act wire format — it
    has exactly one caller, doesn't warrant a shared helper."""
    arr = np.asarray(action, dtype=np.float32)
    return {name: arr[..., o: o + d].tolist() for name, o, d in slices}


def build_app(policy: Tau0VLAPolicy, *, adapter: str | None = None) -> FastAPI:
    # Embodiment-specific wire knowledge (SDK payload keys, camera aliases,
    # state channel map, SDK action column order) lives in layer 3 of an
    # adapter, ``adapters/<robot>/deploy_io.py``. Which adapter is a property of
    # the checkpoint, read off its Data Spec — see ``resolve_deploy_io``.
    deploy_io = resolve_deploy_io(policy.data_spec, override=adapter)
    logger.info("adapter deploy_io: %s", deploy_io.__name__)
    state_fd = deploy_io.load_state_field_descriptions(policy.data_spec.artifacts_dir)
    adapt = deploy_io.build_payload_adapter(
        cam_keys=policy.data_spec.cam_keys,
        state_fd=state_fd,
        state_dim=deploy_io.state_dim_from_field_descriptions(state_fd),
    )

    # Built once per process — drives the dict-shape /act response.
    action_slices = _action_slices(policy.data_spec)
    logger.info("action slices: %s", action_slices)

    # /act_lerobot_bytes returns a bare positional chunk the A2D SDK applies
    # index-for-index. Unified-40D routes emit grippers-first; the SDK expects
    # arms-first (registry action_groups) — remap. None for component routes
    # (already native order).
    sdk_action_perm = deploy_io.build_sdk_action_perm(policy.data_spec, action_slices)
    if sdk_action_perm is not None:
        logger.info("/act_lerobot_bytes SDK-native remap (unified→action_groups): %s",
                    sdk_action_perm)

    app = FastAPI()

    @app.post("/act_lerobot_bytes")
    async def act(request: Request):
        actions = policy.infer(adapt(pickle.loads(await request.body())))["actions"]
        # Unified routes: reorder columns into the SDK's native action layout
        # (arms-then-grippers) before serialising. Component routes: identity.
        arr = deploy_io.apply_sdk_action_perm(actions, sdk_action_perm)
        return arr.tolist()

    @app.post("/act")
    async def act_canonical(request: Request):
        """Direct canonical-payload inference: body is pickle of the exact
        ``{prompt, images, state, meta}`` ``policy.infer`` expects. Response
        is a dict keyed by canonical component name
        (``arm_joint`` / ``eef_pose`` / ``gripper`` / ``waist`` / ...), each
        value a nested list shape ``[chunk, native_dim]``."""
        payload = pickle.loads(await request.body())
        # A mismatch between the caller's prompt and the templated one is the
        # first thing to check when a policy behaves differently over HTTP than
        # in open loop, so log both. DEBUG, not INFO: this is on the hot path.
        logger.debug(
            "/act raw-prompt=%r → templated=%r",
            str(payload.get("prompt", ""))[:120],
            policy.data_spec.format_instruction(str(payload.get("prompt", "")))[:160],
        )
        actions = policy.infer(payload)["actions"]
        return deploy_io.canonicalize_action_dict(_split_action_chunk(actions, action_slices))

    @app.get("/health")
    async def health():
        return {"status": "ok", "route": policy.data_spec.finch_config_name}

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--route", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=10088)
    p.add_argument("--device", default=None)
    p.add_argument("--adapter", default=None,
                   help="Dotted adapter package whose deploy_io serves this checkpoint, e.g. "
                        "tau0_vla.adapters.my_robot. Normally unnecessary — it is read off the "
                        "checkpoint's Data Spec; pass it only for a config registered outside "
                        "the adapters package.")
    p.add_argument("--infer-mode", default="optim", choices=["optim", "eager"],
                   help="Tau0VLA inference mode: optim enables static prefix/compile when available.")
    p.add_argument("--max-prefix-len", type=int, default=0,
                   help="Static prefix length for optim mode; 0 keeps checkpoint value or uses 256.")
    p.add_argument("--warmup-steps", type=int, default=3,
                   help="Dummy deploy-path warmup calls before serving; set 0 to disable.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_configs_registered()
    ensure_policy_manifest(args.model, verbose=False)
    # A config registered through `config_modules` rather than a
    # `configs/*/data.py` is not on disk where the walk above looks; the
    # checkpoint records which modules to import.
    discover_checkpoint_config_modules(args.model)

    policy = Tau0VLAPolicy.from_checkpoint(args.model, route=args.route, device=args.device)
    logger.info("route=%s", policy.data_spec.finch_config_name)

    _configure_inference_mode(policy, args.infer_mode, args.max_prefix_len)
    _run_dummy_input_warmup(policy, warmup_steps=args.warmup_steps)

    import uvicorn
    uvicorn.run(build_app(policy, adapter=args.adapter), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
