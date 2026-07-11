# Deployment Review

## Decision

Proceed with a two-track local layout:

1. Mainline implementation in this repository.
2. External repositories cloned under `references/external_repos/` as read-only
   references.

The external repositories are ignored by Git and must not become the mainline
development surface.

## Reference Repositories

Section 4.2 recommended order:

1. SustainDC: `https://github.com/HewlettPackard/dc-rl.git`
2. CompOpt: `https://github.com/HewlettPackard/compopt.git`
3. Alibaba Cluster Data: `https://github.com/alibaba/clusterdata.git`
4. CarbonScaler: `https://github.com/umassos/CarbonScaler.git`
5. SustainCluster: `https://github.com/HewlettPackard/sustain-cluster.git`

Local target directory:

```text
references/external_repos/
```

Clone helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/clone_reference_repos.ps1
```

## Mainline MVP

The first implementation follows Appendix B:

- 24 h horizon
- 15 min time step
- Single datacenter
- No storage, no multi-datacenter routing, no grid/VPP extension
- Baseline dispatch and price-aware temporal shifting

Run:

```powershell
python scripts/run_mvp.py
```

Generated outputs go to `results/` and are excluded from Git.

## Current Network Note

The first attempted clone of SustainDC failed twice because the environment
could not connect to `github.com:443`. The project structure and clone script
are ready; run the helper when GitHub network access is available.

