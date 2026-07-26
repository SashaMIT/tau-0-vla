#!/usr/bin/env python3
"""Full-data, training-faithful unified-40D norm stats for a finetune leaf.

The component path (``python -m tau0_vla.data compute_norm`` → ``save_norm_stats``)
fits statistics over *component* vectors — for example, 14 arm joints plus
2 grippers. A unified route does not normalize in that space: the
``UnifiedAssembler`` scatters those components into the 40D Unified Layout and
looks its statistics up under ``per_embodiment[<registry_key>]``. Handing it a
component-fitted file fails loudly (``operands could not be broadcast together
with shapes (40,) (14,)``). This script produces the file it actually needs.

How it stays faithful to training:

* anchors are ALL the frames the training stream uses — the same
  ``build_segment_index_ranges`` call the pipeline makes, with
  ``tail_buffer=action_horizon-1``. No per-episode sampling, no caps.
  The label filter defaults to the shipped example's
  (``--positive-labels l3 --negative-labels error_frame``) and **must match the
  ``FrameFilter`` your Robot Config declares** — statistics fitted over a
  different anchor set than training uses are wrong, and nothing downstream
  detects it.
* one process per repo. Each reads the needed parquet COLUMNS once via pyarrow
  (orders of magnitude faster than per-row access), verifies the row order
  against the meta episode bounds, then streams every anchor through the
  training ``UnifiedAssembler`` with ``disable_component_normalization=True`` —
  so the numbers are gathered in exactly the space training normalizes in.
* writes ``.npz`` partials that ``merge_stats.py`` reduces, so a repo that fails
  can be re-run alone.

mean/std are EXACT over all anchors (mergeable sufficient statistics, so the
shard split does not affect them). q01/q99 come from a per-repo reservoir of
``--res-cap`` rows (default 400k), pooled and subsampled again at merge — they
are estimates, and the estimate moves with ``--seed``.

Usage::

    PYTHONPATH=src:. python3 scripts/norm_stats/compute_unified_ft_stats.py \\
        --body g1_a2d_joint_unified --repos <repo1> <repo2> ... \\
        --action-horizon 30 --partials-dir <dir>
    PYTHONPATH=src:. python3 scripts/norm_stats/merge_stats.py \\
        --partials <dir> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from tau0_vla.data.robots import UNIFIED_ROBOT_CLASSES  # noqa: E402
from tau0_vla.data.robots.unified import UNIFIED_DIM  # noqa: E402
from tau0_vla.data.sampler import build_segment_index_ranges  # noqa: E402


def _repo_seed(repo: str, base: int) -> int:
    """Reservoir seed for one repo.

    Deliberately NOT ``hash(repo)``: CPython salts ``hash`` of a str per
    process, so that would make the quantiles unreproducible across runs of the
    same command. A digest of the basename keeps each repo's stream independent
    while staying stable.
    """
    digest = hashlib.sha256(Path(repo).name.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") + base) % (2**31)


def _anchor_filter_key(args_d: dict) -> str:
    """Canonical description of the anchor set, stored in every partial.

    Two partials are only mergeable if they were fitted over the same stream.
    The horizon sets the tail buffer and the two label lists set the filter, so
    all three belong in the key; sorting the labels makes flag order irrelevant.
    """
    positive = ",".join(sorted(args_d["positive_labels"]))
    negative = ",".join(sorted(args_d["negative_labels"]))
    return f"h={args_d['action_horizon']};+[{positive}];-[{negative}]"


class FastSuff:
    """Mergeable sufficient statistics + an O(batch) reservoir.

    The reservoir update is in-place. A naive version that concatenates the pool
    on every post-fill update is O(cap) per anchor, which stalls at full-data
    scale (5M+ anchors).
    """

    def __init__(self, dim, rng, cap):
        self.sum = np.zeros(dim, np.float64)
        self.sumsq = np.zeros(dim, np.float64)
        self.count = np.zeros(dim, np.float64)
        self.pool = np.zeros((cap, dim), np.float32)
        self.cap = cap
        self.fill = 0
        self.seen = 0
        self.rng = rng

    def update(self, values, mask):
        v = np.asarray(values, np.float64).reshape(-1, np.asarray(values).shape[-1])
        m = np.broadcast_to(np.asarray(mask, np.float64), v.shape)
        vm = np.where(m > 0, v, 0.0)
        self.sum += vm.sum(0)
        self.sumsq += (vm * vm).sum(0)
        self.count += (m > 0).sum(0)
        # Masked-out dims go in as NaN so the merge's nanpercentile skips them
        # per dimension rather than dropping the whole row.
        rows = np.where(m > 0, v, np.nan).astype(np.float32)
        k = len(rows)
        if self.fill < self.cap:
            take = min(k, self.cap - self.fill)
            self.pool[self.fill : self.fill + take] = rows[:take]
            self.fill += take
            rows = rows[take:]
            self.seen += take
            k = len(rows)
        if k:
            # Classic reservoir: replace with probability cap/seen, in place.
            idx = self.rng.integers(0, self.seen + np.arange(1, k + 1))
            repl = idx < self.cap
            if repl.any():
                self.pool[idx[repl]] = rows[repl]
            self.seen += k

    def arrays(self):
        pool = self.pool[: self.fill] if self.fill < self.cap else self.pool
        return self.sum, self.sumsq, self.count, pool, np.float64(self.seen)


def _episode_bounds_from_meta(repo: str) -> tuple[np.ndarray, np.ndarray]:
    """``(from, to)`` arrays indexed by ``episode_index``.

    Reads ``meta/episodes.jsonl`` (v2.x) or the ``meta/episodes/**/*.parquet``
    table (v3.0), whichever the repo has.
    """
    rows = []
    jsonl = Path(repo) / "meta" / "episodes.jsonl"
    if jsonl.exists():
        with open(jsonl) as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        import pyarrow.dataset as pads

        columns = ["episode_index", "dataset_from_index", "dataset_to_index"]
        table = pads.dataset(str(Path(repo) / "meta" / "episodes"), format="parquet").to_table(
            columns=columns
        )
        rows = [
            {"episode_index": int(a), "dataset_from_index": int(b), "dataset_to_index": int(c)}
            for a, b, c in zip(*(table.column(k).to_pylist() for k in columns))
        ]
    n = max(int(r["episode_index"]) for r in rows) + 1
    ep_from = np.zeros(n, np.int64)
    ep_to = np.zeros(n, np.int64)
    for row in rows:
        i = int(row["episode_index"])
        ep_from[i] = int(row.get("dataset_from_index", row.get("from_index", 0)))
        ep_to[i] = int(row.get("dataset_to_index", row.get("to_index", 0)))
    return ep_from, ep_to


class _DsShim:
    """The surface ``build_segment_index_ranges`` touches on a dataset."""

    def __init__(self, ep_from, ep_to, n):
        self.episode_data_index = {"from": ep_from, "to": ep_to}
        self.episodes = np.asarray(
            [i for i in range(len(ep_from)) if ep_to[i] > ep_from[i]], np.int64
        )
        self._n = int(n)

    def __len__(self):
        return self._n


class _MetaShim:
    """The surface ``build_segment_index_ranges`` touches on dataset metadata."""

    def __init__(self, repo: str):
        self.root = repo
        self.info = json.loads((Path(repo) / "meta" / "info.json").read_text())
        self.episodes = {}


def _read_columns_pyarrow(repo: str, cols: list[str]) -> dict[str, np.ndarray]:
    import pyarrow.dataset as pads

    dset = pads.dataset(str(Path(repo) / "data"), format="parquet")
    available = set(dset.schema.names)
    picked = {}
    for col in cols + ["episode_index"]:
        name = col if col in available else col.split(".")[-1]
        if name not in available:
            raise KeyError(f"column {col!r} not in {sorted(available)[:20]}")
        picked[col] = name
    table = dset.to_table(columns=sorted(set(picked.values())))
    out = {}
    for col, name in picked.items():
        arr = table.column(name).to_numpy(zero_copy_only=False)
        if arr.dtype == object:  # list column -> 2D
            arr = np.stack(arr)
        out[col] = arr
    return out


def _process_repo(job):
    repo, args_d = job
    started = time.time()
    body = args_d["body"]
    horizon = args_d["action_horizon"]
    cls = UNIFIED_ROBOT_CLASSES[body]
    # repo_id is irrelevant here: we read the parquet ourselves and only need
    # the class's layout declarations and its assembler.
    cfg = cls(repo_id="x")
    raw = cfg._raw_repack_structure()
    assembler = cfg._build_component_assembler(
        field_descriptions={}, disable_component_normalization=True
    )
    inputs = {k: col for k, col in raw.items() if k.endswith("_raw")}
    cols = sorted(set(inputs.values()))

    data = _read_columns_pyarrow(repo, cols)
    episode = np.asarray(data["episode_index"], np.int64)
    read_done = time.time()

    # pyarrow row order must match the meta episode bounds, or every anchor
    # index below addresses the wrong frame.
    ep_from, ep_to = _episode_bounds_from_meta(repo)
    for i in range(len(ep_from)):
        start, end = int(ep_from[i]), int(ep_to[i])
        if end > start:
            segment = episode[start:end]
            assert (segment == i).all(), f"{repo}: row order mismatch at episode {i} [{start}:{end})"
    print(
        f"[{Path(repo).name}] columns read {read_done - started:.0f}s, "
        f"row order verified against meta",
        flush=True,
    )

    ranges, _ = build_segment_index_ranges(
        dataset_metas=[_MetaShim(repo)],
        datasets=[_DsShim(ep_from, ep_to, len(episode))],
        tail_buffer=max(0, horizon - 1),
        positive_labels=list(args_d["positive_labels"]),
        negative_labels=list(args_d["negative_labels"]),
    )
    anchors = (
        np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in ranges])
        if len(ranges)
        else np.zeros(0, np.int64)
    )

    rng = np.random.default_rng(_repo_seed(repo, args_d["seed"]))
    state_stats = FastSuff(UNIFIED_DIM, rng, args_d["res_cap"])
    action_stats = FastSuff(UNIFIED_DIM, rng, args_d["res_cap"])

    offsets = np.arange(horizon, dtype=np.int64)
    stream_started = time.time()
    for k, anchor in enumerate(anchors):
        anchor = int(anchor)
        window = anchor + offsets  # tail_buffer keeps anchor+H-1 inside the segment
        sample = {"images": {}, "prompt": ""}
        for key, col in inputs.items():
            arr = data[col]
            sample[key] = arr[window] if "action" in key else arr[anchor]
        out = assembler(sample)
        state_stats.update(out["state"], out["state_mask"])
        action_stats.update(out["action"], out["action_mask"])
        if k and k % 500_000 == 0:
            rate = (time.time() - stream_started) / k * 1e3
            print(
                f"  [{Path(repo).name}] {k:,}/{len(anchors):,} ({rate:.2f} ms/anchor)",
                flush=True,
            )

    s_sum, s_sumsq, s_count, s_res, _ = state_stats.arrays()
    a_sum, a_sumsq, a_count, a_res, _ = action_stats.arrays()
    out_path = Path(args_d["partials_dir"]) / f"{body}.{Path(repo).name}.npz"
    np.savez(
        out_path,
        registry_key=cls._unified_registry_key,
        state_sum=s_sum,
        state_sumsq=s_sumsq,
        state_count=s_count,
        state_res=s_res,
        action_sum=a_sum,
        action_sumsq=a_sumsq,
        action_count=a_count,
        action_res=a_res,
        n_frames=len(anchors),
        # The anchor set these statistics were fitted over. merge_stats.py
        # refuses to reduce partials that disagree here: statistics from two
        # different frame filters are not mergeable, and averaging them would
        # produce a plausible-looking file fitted over neither stream.
        anchor_filter=_anchor_filter_key(args_d),
    )
    return (Path(repo).name, len(anchors), time.time() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--body",
        required=True,
        help=f"unified robot class key; one of {sorted(UNIFIED_ROBOT_CLASSES)}",
    )
    parser.add_argument("--repos", nargs="+", required=True, help="dataset roots, one per shard")
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=30,
        help="must match the training config's action_horizon (sets tail_buffer)",
    )
    parser.add_argument(
        "--positive-labels",
        nargs="*",
        default=["l3"],
        help="annotation tracks a frame must carry to be an anchor; must match "
        "your Robot Config's FrameFilter(positive=...)",
    )
    parser.add_argument(
        "--negative-labels",
        nargs="*",
        default=["error_frame"],
        help="annotation tracks that disqualify a frame; must match your Robot "
        "Config's FrameFilter(negative=...)",
    )
    parser.add_argument(
        "--res-cap", type=int, default=400_000, help="reservoir rows per repo, for q01/q99"
    )
    parser.add_argument("--seed", type=int, default=0, help="reservoir seed; mean/std ignore it")
    parser.add_argument("--partials-dir", required=True)
    args = parser.parse_args()
    if args.body not in UNIFIED_ROBOT_CLASSES:
        parser.error(f"unknown --body {args.body!r}; choose from {sorted(UNIFIED_ROBOT_CLASSES)}")
    Path(args.partials_dir).mkdir(parents=True, exist_ok=True)

    jobs = [(repo, vars(args)) for repo in args.repos]
    total = 0
    with Pool(processes=len(jobs)) as pool:
        for name, count, elapsed in pool.imap_unordered(_process_repo, jobs):
            total += count
            print(f"[done] {name}: {count:,} anchors in {elapsed / 60:.1f} min", flush=True)
    print(
        f"TOTAL {total:,} anchors. Merge with:\n"
        f"  PYTHONPATH=src:. python3 scripts/norm_stats/merge_stats.py "
        f"--partials {args.partials_dir} --out <json>"
    )


if __name__ == "__main__":
    main()
