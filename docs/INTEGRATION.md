# Core API integration

The module does not load or run a video backbone. An adapter must supply four current shallow states, the deeper suffix outputs when refreshed, current output heads and the original solver.

```python
from jedcache import CachePolicy, ResidualCacheBank, direction_coefficients

policy = CachePolicy('joint', warmup_steps=2, max_reuse=1)
bank = ResidualCacheBank()
bank.begin_task('unique-edit-id')
coefficients = direction_coefficients(target_scale=10.0, source_scale=5.0)
```

For real frozen inference, load a valid development-fitted state with `policy.load_state_dict(state)` and set the development-selected `policy.joint_threshold` before editing. Experimental states are not supplied in this code-only release. With no fitted model, the policy safely refreshes on every step.

At step `t`, evaluate the four current prefixes to obtain `shallow_by_branch`:

```python
features = bank.feature_values(shallow_by_branch, coefficients)
refresh, diagnostics = policy.decide(t, total_steps, bank, features)
```

For a refresh, run that branch's deep suffix to obtain `H_b`, use its current output head on `H_b`, and call `bank.refresh(b, h_b, H_b - h_b, t)`. For reuse, use the current head on `h_b + bank.residual(b)` and call `bank.reused(b)`. Combine current branch outputs with the signed coefficients, then use the unchanged solver. All four current prefixes and output heads still run at every step.

Use a fresh bank or unique task key for every edit. Warmup, missing residuals, missing fit or one prior reuse forces refresh. A prediction equal to the threshold permits reuse; a strictly greater prediction triggers the error guard. Do not call `observe_refresh(..., update_models=True)` during frozen evaluation. `CausalRidge` retains its historical class name but need not be fitted online.

The synthetic tests validate control flow, separate storage, the guidance equation and reset behavior. They are not calibrated experiments and do not establish paper-level speed or quality.
