# Reference Repositories

Section 4.2 of the technical route recommends five repositories for reading
and local experiments. They are cloned into `references/external_repos/` and
ignored by Git.

Recommended order:

1. `sustaindc`: https://github.com/HewlettPackard/dc-rl.git
2. `compopt`: https://github.com/HewlettPackard/compopt.git
3. `clusterdata`: https://github.com/alibaba/clusterdata.git
4. `CarbonScaler`: https://github.com/umassos/CarbonScaler.git
5. `sustain-cluster`: https://github.com/HewlettPackard/sustain-cluster.git

Usage rule:

- Read README, examples, data managers, environment APIs, and model structure.
- Do not develop long-term project code inside these cloned repositories.
- Port only the necessary concepts into the mainline modules after review.

