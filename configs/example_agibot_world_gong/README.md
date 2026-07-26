# AgiBot World gong-striking example

This is the worked post-training example for the bundled
[`example_data/`](../../example_data/) subset of AgiBot World. It contains 25
episodes of the **Strike the gong** task and uses the head, left-wrist and
right-wrist cameras. The detailed instruction is read from each frame's
`instruction_segments` metadata.

Then launch from the released warm-start checkpoint:

```bash
bash scripts/train.sh configs/example_agibot_world_gong/train.yaml \
    --model_name_or_path /path/to/tau-0-vla-checkpoint
```

`norm_stats.json` is shipped with the example. It was fitted over all 5,484
instruction-annotated action anchors in the bundled subset with a 30-step
action horizon and the `l3`/`instruction_segments` filter. Its SHA-256 is
`5ee161d34b95613605c466751b9c79e2fdddaea2d0d93c9f93b03c78a10bb416`.

If the dataset, action horizon or frame filter changes, recompute it:

```bash
PYTHONPATH=src:. python3 scripts/norm_stats/compute_unified_ft_stats.py \
    --body g1_agibot_unified \
    --action-horizon 30 \
    --repos example_data \
    --positive-labels l3 \
    --negative-labels \
    --partials-dir /tmp/agibot_world_gong_stats

PYTHONPATH=src:. python3 scripts/norm_stats/merge_stats.py \
    --partials /tmp/agibot_world_gong_stats \
    --out configs/example_agibot_world_gong/norm_stats.json
```

The released checkpoint is a training starting point, not a deployable policy.
The post-training run writes its own `finch_data_spec/` and
`policy_manifest.json`, which are required for open-loop evaluation and serving.

## Verified data contract

The example adapter was checked against the bundled metadata, parquet tensors,
videos, and decoded training samples.

| public key | source feature | checked shape |
|---|---|---|
| `head` | `observation.images.top_head` | `480×640×3`, resized to `224×224×3` |
| `wrist_left` | `observation.images.hand_left` | `480×640×3`, resized to `224×224×3` |
| `wrist_right` | `observation.images.hand_right` | `480×640×3`, resized to `224×224×3` |
| state joints for FK | `observation.state[28:42]` | 14 |
| action joints for FK | `action[16:30]` | 14 per horizon step |
| state grippers | `observation.state[[0, 1]]` | 2 |
| action grippers | `action[[0, 1]]` | 2 per horizon step |
| instruction | `meta/info.json → instruction_segments` | one string per valid anchor |

The dataset's stored EEF channels are intentionally not consumed. This public
adapter uses the 14 joint values directly and therefore does not require or
ship a G01 URDF.

The resulting unified layout is:

| unified indices | value | mask |
|---|---|---|
| `[18]` | left gripper | 1 |
| `[19]` | right gripper | 1 |
| `[24:31]` | left arm joints (7D) | 1 |
| `[32:39]` | right arm joints (7D) | 1 |
| all other slots | inactive padding | 0 |

Joint actions are made relative to the current joint state before
normalization; gripper actions remain absolute. Every checked state/action
sample therefore has 16 active dimensions, with shapes `(40,)` and
`(30, 40)`.
