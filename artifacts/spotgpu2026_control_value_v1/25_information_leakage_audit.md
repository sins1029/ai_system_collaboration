
# Information Leakage Audit

- H1 Oracle future state accesses: 0 (required 0).
- H1 Oracle future rows: 0 (required 0).
- H4 Oracle future state accesses: 17673.
- H4 uses exactly four future offsets per state and five deterministic DC rows per offset.
- Workload source hash shared: 17BDE4F43208422843339BAAECF6BD2F2360ABB5A9C35EC7972C55474D234BFA.
- Capacity, energy signals, SLA, true duration, estimated duration, order, and initial state are shared and preflight-checked.
- true_duration is read only when applying actions to simulator transit/running state.
- estimated_duration is the only duration supplied to the optimizer.
- Transformer code path used: NO.
- BC code path used: NO.
- RL code path used: NO.
- Silent fallback label used: NO.
- Regional energy data provenance: SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL; not Alibaba2026 measured energy data.
