# Config template

Copy this directory for a new post-training task:

```bash
cp -r configs/_template configs/my_task
grep -rn 'YOUR_' configs/my_task
```

| file | purpose |
|---|---|
| `data.py` | dataset paths, robot adapter, cameras, prompts, frame filters, and normalization stats |
| `train.yaml` | model checkpoint and training arguments |

Keep the two files together. The function decorated with `@register_config` in
`data.py` must match `data_args.config_name` in `train.yaml`; camera names and
image size must also agree.

Then:

1. Prepare a [supported LeRobot dataset](../../src/tau0_vla/data/DATASET_FORMAT.md).
2. Add or select a [robot adapter](../../src/tau0_vla/adapters/README.md).
3. Compute normalization statistics.
4. Launch `bash scripts/train.sh configs/my_task/train.yaml`.
