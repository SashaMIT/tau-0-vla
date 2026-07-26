import logging
import os

from transformers.utils import logging as hf_logging


# =========================
# Rank-aware Formatter
# =========================
class RankFormatter(logging.Formatter):
    def format(self, record):
        node = os.environ.get("NODE_RANK", os.environ.get("SLURM_NODEID", "0"))
        local_rank = os.environ.get("LOCAL_RANK", "0")
        rank = os.environ.get("RANK", "0")

        try:
            rank_int = int(rank)
        except ValueError:
            rank_int = -1

        record.rank_info = f"N{node},R{rank_int:03d},GPU{local_rank}"
        return super().format(record)


# =========================
# Python logging setup
# =========================


def setup_py_logging(
    level="info",
    only_main_process=True,
):
    level_map = {
        "debug": hf_logging.DEBUG,
        "info": hf_logging.INFO,
        "warning": hf_logging.WARNING,
        "error": hf_logging.ERROR,
    }
    handler = logging.StreamHandler()
    formatter = RankFormatter(
        fmt="[%(levelname)s] %(asctime)s  [%(rank_info)s] [%(filename)s:%(lineno)d] (%(funcName)s) | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0

    handler.setFormatter(formatter)
    if only_main_process and not is_main:
        logging.basicConfig(handlers=[handler])
    else:
        logging.basicConfig(level=level_map[level], handlers=[handler])
