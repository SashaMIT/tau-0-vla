#!/usr/bin/env python3
"""Reduce unified norm-stats partials into one per-embodiment 40D stats file.

Reads every ``.npz`` partial written by ``compute_unified_ft_stats.py``, groups
them by ``registry_key``, and combines the **mergeable sufficient statistics**
losslessly::

    mean = Σx / count
    std  = sqrt(Σx² / count − mean²)

That is exact and independent of how the work was sharded, so splitting by repo
costs nothing. q01/q99 are estimated from the concatenated reservoirs (NaN-masked
per dimension, subsampled to ``_RES_FINAL`` rows) and therefore DO move with
``--seed``.

Writes ``per_embodiment[<registry_key>][state|action]`` — what the unified route
looks its statistics up in — plus a pooled global ``norm_stats`` block as a
fallback for a body with no matching key.

Usage::

    PYTHONPATH=src:. python3 scripts/norm_stats/merge_stats.py \\
        --partials <dir-of-npz> --out <path>/my-task-unified-40d.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tau0_vla.data.stats import NormStats, serialize_json  # noqa: E402

_RES_FINAL = 200_000  # rows kept for the final quantile estimate per block


def _finalize(sum_, sumsq, count, res, rng, compute_quantiles=True) -> NormStats:
    count_safe = np.maximum(count, 1.0)
    mean = sum_ / count_safe
    var = np.maximum(sumsq / count_safe - mean * mean, 0.0)
    std = np.sqrt(var)
    if not compute_quantiles:
        return NormStats(mean=mean, std=std)
    if len(res) > _RES_FINAL:
        res = res[rng.choice(len(res), _RES_FINAL, replace=False)]
    if len(res):
        with np.errstate(all="ignore"):
            q01 = np.nan_to_num(np.nanpercentile(res, 1, axis=0), nan=0.0)
            q99 = np.nan_to_num(np.nanpercentile(res, 99, axis=0), nan=0.0)
    else:
        q01 = np.zeros_like(mean)
        q99 = np.zeros_like(mean)
    return NormStats(mean=mean, std=std, q01=q01, q99=q99)


def _accum(parts, role):
    total = sumsq = count = None
    res = []
    for part in parts:
        total = part[f"{role}_sum"] if total is None else total + part[f"{role}_sum"]
        sumsq = part[f"{role}_sumsq"] if sumsq is None else sumsq + part[f"{role}_sumsq"]
        count = part[f"{role}_count"] if count is None else count + part[f"{role}_count"]
        rows = part[f"{role}_res"]
        if rows.size:
            res.append(rows)
    pooled = np.concatenate(res, 0) if res else np.zeros((0, len(total)), np.float32)
    return total, sumsq, count, pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--partials", required=True, help="directory or glob of *.npz partials")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-quantiles", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="final quantile subsample; mean/std ignore it"
    )
    args = parser.parse_args()

    pattern = (
        args.partials
        if any(c in args.partials for c in "*?[")
        else str(Path(args.partials) / "*.npz")
    )
    # Sorted so the float64 accumulation order — and therefore the last bits of
    # mean/std — is reproducible for a given set of partials.
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no partials at {pattern}")
    rng = np.random.default_rng(args.seed)
    quantiles = not args.no_quantiles

    by_key: dict[str, list] = {}
    for path in files:
        data = np.load(path, allow_pickle=False)
        by_key.setdefault(str(data["registry_key"]), []).append(data)

    # Statistics fitted over different anchor sets are not mergeable: averaging
    # them yields a file that describes neither stream, and every consumer
    # downstream accepts it silently. Partials written before this field existed
    # report "unknown" and are treated as one group.
    filters = {str(d.get("anchor_filter", "unknown")) for parts in by_key.values() for d in parts}
    if len(filters) > 1:
        raise SystemExit(
            "refusing to merge partials fitted over different anchor sets:\n"
            + "".join(f"  {f}\n" for f in sorted(filters))
            + "Recompute them with matching --action-horizon / --positive-labels / "
            "--negative-labels, or merge each group into its own file."
        )

    print(f"merging {len(files)} partials -> {len(by_key)} embodiments")
    print(f"  anchor set: {sorted(filters)[0]}")

    per_embodiment: dict[str, dict[str, NormStats]] = {}
    processed: dict[str, int] = {}
    for key, parts in sorted(by_key.items()):
        per_embodiment[key] = {
            role: _finalize(*_accum(parts, role), rng, quantiles) for role in ("state", "action")
        }
        frames = int(sum(int(p["n_frames"]) for p in parts))
        processed[key] = frames
        print(f"  {key:36s} shards={len(parts):2d} frames={frames}")

    # Pooled global fallback: merge every partial regardless of key. A body whose
    # registry key is absent from per_embodiment normalizes against this instead.
    all_parts = [p for parts in by_key.values() for p in parts]
    global_stats = {
        role: _finalize(*_accum(all_parts, role), rng, quantiles) for role in ("state", "action")
    }

    summary = {
        "type": "UnifiedPerEmbodimentNormStats",
        "source_config": "all_unified",
        "layout_dim": int(len(global_stats["state"].mean)),
        "compute_quantiles": quantiles,
        "processed_by_key": processed,
        "merged_from_shards": len(files),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        serialize_json(global_stats, config_summary=summary, per_embodiment=per_embodiment)
    )
    print(f"wrote {out_path}  ({len(per_embodiment)} per_embodiment keys)")


if __name__ == "__main__":
    main()
