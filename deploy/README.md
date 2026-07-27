# Deployment

The `deploy` package loads post-trained τ₀-VLA checkpoints for local inference,
HTTP serving, and open-loop evaluation.

| entry point | purpose |
|---|---|
| `policy.py` | `Tau0VLAPolicy.from_checkpoint(...).infer(payload)` |
| `server.py` | HTTP policy server |
| `openloop.py` | local checkpoint evaluation |
| `openloop_with_server.py` | evaluation through a running server |

## Policy server

```bash
python -m deploy.server --model <checkpoint>
```

For a multi-route checkpoint:

```bash
python -m deploy.server --model <checkpoint> --route <route>
```

The server exposes:

- `POST /act` — canonical `{prompt, images, state, meta}` payload to a
  name-keyed action dictionary
- `POST /act_lerobot_bytes` — A2D SDK payload to a flat action chunk
- `GET /health` — health check

The checkpoint stores its robot adapter, camera keys, transforms, and
normalization contract. An adapter can be selected explicitly with
`--adapter <dotted.package>`.

## Action order

Unified outputs are restored by semantic slot name. For the flat G1 SDK
endpoint, `adapters/g1/deploy_io.py` maps the output to:

```text
[left_arm × 7, right_arm × 7, left_gripper, right_gripper]
```

Confirm this order matches the target SDK before hardware deployment.

## Open-loop evaluation

```bash
python deploy/openloop.py --ckpt <checkpoint> --no-plot
```

Add plots or select another route/config when needed:

```bash
python deploy/openloop.py \
    --ckpt <checkpoint> \
    --route <route> \
    --config <config-name> \
    --out-dir <output-dir>
```

To evaluate through HTTP:

```bash
python deploy/openloop_with_server.py \
    --server-url http://127.0.0.1:10088 \
    --config-module <config-module> \
    --no-plot
```
