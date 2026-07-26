from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DatasetShardInfo:
    physically_split: bool
    global_rank: int
    world_size: int
    physical_rank: int
    physical_world_size: int
    logical_rank: int
    logical_world_size: int
    use_manual_distributed_sampler: bool


def get_dataset_shard_info(data_args) -> DatasetShardInfo:
    """Resolve physical dataset split and shard-internal sampler ranks."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = int(torch.distributed.get_world_size())
        global_rank = int(torch.distributed.get_rank())
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_rank = int(os.environ.get("RANK", "0"))

    physically_split = bool(getattr(data_args, "physically_split", False))
    requested_splits = int(getattr(data_args, "num_physical_shards", 0) or 0)

    if requested_splits < 0:
        raise ValueError(f"data_args.num_physical_shards must be >= 0, got {requested_splits}")

    if not physically_split:
        return DatasetShardInfo(
            physically_split=False,
            global_rank=global_rank,
            world_size=world_size,
            physical_rank=global_rank,
            physical_world_size=world_size,
            logical_rank=0,
            logical_world_size=1,
            use_manual_distributed_sampler=False,
        )

    if requested_splits > 0:
        if requested_splits > world_size:
            raise ValueError(
                "data_args.num_physical_shards must be <= distributed world_size "
                f"when physically_split=True, got num_physical_shards={requested_splits}, "
                f"world_size={world_size}."
            )
        physical_world_size = requested_splits
        physical_rank = global_rank % physical_world_size
        logical_rank = global_rank // physical_world_size
        logical_world_size = ((world_size - 1 - physical_rank) // physical_world_size) + 1
        use_manual_distributed_sampler = world_size > 1
    else:
        physical_world_size = world_size
        physical_rank = global_rank
        logical_rank = 0
        logical_world_size = 1
        use_manual_distributed_sampler = False

    return DatasetShardInfo(
        physically_split=True,
        global_rank=global_rank,
        world_size=world_size,
        physical_rank=physical_rank,
        physical_world_size=physical_world_size,
        logical_rank=logical_rank,
        logical_world_size=logical_world_size,
        use_manual_distributed_sampler=use_manual_distributed_sampler,
    )


def attach_shard_info(dataset, shard_info: DatasetShardInfo) -> None:
    dataset.physically_split = shard_info.physically_split
    dataset.global_rank = shard_info.global_rank
    dataset.world_size = shard_info.world_size
    dataset.physical_shard_rank = shard_info.physical_rank
    dataset.physical_shard_world_size = shard_info.physical_world_size
    dataset.logical_shard_rank = shard_info.logical_rank
    dataset.logical_shard_world_size = shard_info.logical_world_size
    dataset.use_manual_distributed_sampler = shard_info.use_manual_distributed_sampler
