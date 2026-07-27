# Adapter template

```bash
cp -r src/tau0_vla/adapters/_template \
      src/tau0_vla/adapters/my_robot
grep -rn 'YOUR_' src/tau0_vla/adapters/my_robot
```

| file | purpose |
|---|---|
| `layout.py` | dataset fields, RobotConfig, cameras, and native dimensions |
| `deploy_io.py` | SDK state, image, and action mapping |

The template starts with joint control. For datasets that provide native EEF
fields, declare those fields directly in the adapter.

After filling the template:

1. Export the adapter classes from `__init__.py`.
2. Register the robot in `tau0_vla/data/robots/__init__.py`.
3. Use the adapter from a `configs/<task>/data.py`.

For unified 40D training, follow
[`../g1/layout.py`](../g1/layout.py) to define a `_UnifiedMixin` variant with
`_unified_registry_key`, then add its
[`dim_registry.json`](../../data/assets/dim_registry.json) entry.

Before hardware deployment, make sure `build_sdk_action_perm` returns the
column order expected by the robot SDK.
