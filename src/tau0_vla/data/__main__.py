from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _ensure_local_hf_cache() -> None:
    cache_root = Path.cwd() / ".cache" / "huggingface"
    datasets_cache = cache_root / "datasets"
    hub_cache = cache_root / "hub"
    transformers_cache = cache_root / "transformers"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tau0_vla.data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute_norm = subparsers.add_parser("compute_norm", help="compute and save norm stats for a registered config")
    compute_norm.add_argument("config_name", help="registered Finch config name")
    compute_norm.add_argument(
        "--config-module",
        help="python module to import explicitly so the config can register itself",
    )
    compute_norm.add_argument(
        "--config-file",
        type=Path,
        help="python file to import explicitly so the config can register itself",
    )
    compute_norm.add_argument(
        "--output-dir",
        type=Path,
        help="directory to write norm stats into; defaults to config.norm_stats_dir",
    )
    compute_norm.add_argument(
        "--max-samples",
        type=int,
        help="optional cap on sampled records when computing norm stats",
    )
    compute_norm.add_argument(
        "--max-ratio",
        type=float,
        help="sample this fraction of the corpus (in (0, 1]); resolved to "
        "max_samples = floor(total_frames * ratio). Mutually exclusive "
        "with --max-samples. Prefer this over --max-samples when the "
        "manifest size may change (e.g. switching 2k → 5k) so coverage "
        "scales automatically.",
    )
    compute_norm.add_argument(
        "--include-video",
        action="store_true",
        help="include video fields during stats computation",
    )
    compute_norm.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallelise the fast path across N worker processes (default 1 = single-process, "
        "identical to legacy behaviour). Safe: mean/std are bit-stable w.r.t. worker count; "
        "reservoir-based q01/q99 may differ within bounded sampling noise.",
    )

    return parser


def _import_config_file(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    module_name = f"_finch_config_{resolved.stem}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config file: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "compute_norm":
        _ensure_local_hf_cache()
        if args.config_file is not None:
            _import_config_file(args.config_file)
        if args.config_module is not None:
            importlib.import_module(args.config_module)
        from tau0_vla.data.stats import save_norm_stats

        path = save_norm_stats(
            args.config_name,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            max_ratio=args.max_ratio,
            include_video=args.include_video,
            num_workers=args.workers,
        )
        print(path)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
