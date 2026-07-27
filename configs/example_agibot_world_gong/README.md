# AgiBot World example

This config post-trains τ₀-VLA on the bundled
[`example_data/`](../../example_data/) subset. It is a G1 joint-control route:
no URDF or joint-to-EEF conversion is used.

```bash
bash scripts/train.sh configs/example_agibot_world_gong/train.yaml \
    --model_name_or_path /path/to/tau-0-vla-checkpoint
```

## Input contract

| model-facing value | dataset source |
|---|---|
| `head` image | `observation.images.top_head` |
| `wrist_left` image | `observation.images.hand_left` |
| `wrist_right` image | `observation.images.hand_right` |
| instruction | `instruction_segments` in `meta/info.json` |
| native state | `observation.state` |
| native action | `action` |

`FrameFilter(positive=["l3"], negative=[])` keeps anchors covered by an
instruction segment, and the complete 30-step action chunk stays inside that
same segment.

## Native-to-unified mapping

Ranges use Python slice notation.

| semantic value | native state columns | native action columns | unified slots |
|---|---:|---:|---:|
| left gripper | `0:1` | `0:1` | `18` |
| right gripper | `1:2` | `1:2` | `19` |
| left arm, 7 joints | `28:35` | `16:23` | `24:31` |
| right arm, 7 joints | `35:42` | `23:30` | `32:39` |

The resulting sample has:

```text
state        (40,)       float32
action       (30, 40)    float32
state_mask   (40,)       float32
action_mask  (40,)       float32
```

Active slots are `18`, `19`, `24:31`, and `32:39`: 16 dimensions total.
Slots `31` and `39` are the unused eighth-joint padding positions. All other
slots are zero with mask `0`.

Arm targets are joint deltas relative to the current state. Gripper targets
remain absolute.

The native state is 163D and the native action is 36D. Only the 16 columns
listed above are used by this joint route; all other native fields, including
the inline EEF metadata, are ignored.

## Normalization statistics

The bundled `norm_stats.json` matches this dataset, `l3` filter, joint mapping,
and 30-step horizon. Recompute it after changing any of them:

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

The runtime selects `per_embodiment["g1_agibot_36"]`; verify that both state
and action statistics in that block are 40D.
