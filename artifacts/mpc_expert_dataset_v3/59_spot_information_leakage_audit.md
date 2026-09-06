# SpotGPU2026 Information Leakage Audit

- Deployable files checked: 8.
- Label files physically separate: 4.
- Privileged-future files physically separate: 4.
- Forbidden columns found in deployable region: `[]`.
- `true_duration` and true completion are under `simulator_only/` only.
- H1/H4 actions and objectives are under `labels/` and require explicit loader opt-in.
- +15/+30/+45/+60 workload, price and carbon are under `privileged_future/` only.
- Default loader manifest exposes only `deployable_current/`.
- Transformer forecast columns/checkpoints: absent.

Result: **PASS**.
