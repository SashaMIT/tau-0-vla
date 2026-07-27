# AgiBot World example

This config post-trains τ₀-VLA on the bundled
[`example_data/`](../../example_data/) subset. It reads the head and two wrist
cameras, uses `instruction_segments` as the language instruction, and controls
the G1 arms and grippers in joint space.

```bash
bash scripts/train.sh configs/example_agibot_world_gong/train.yaml \
    --model_name_or_path /path/to/tau-0-vla-checkpoint
```

The 40D unified layout activates 16 dimensions:

| source | unified slots |
|---|---|
| left/right gripper | `[18:20]` |
| left arm joints | `[24:31]` |
| right arm joints | `[32:39]` |

`norm_stats.json` matches the bundled data, 30-step action horizon, and `l3`
frame filter. Recompute it after changing any of these:

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
