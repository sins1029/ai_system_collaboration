# Information Leakage Audit

| check | passed | evidence |
| --- | --- | --- |
| true_duration excluded from deployable observation | True | whitelist |
| estimated_duration train-only | True | sample after train |
| memory model train-only | True | Alibaba2020 train prefix |
| bandwidth model train-only | True | Alibaba2020 train prefix |
| no Oracle future | True | schema scan |
| no teacher action leakage | True | schema scan |
| no H1/H4 action leakage | True | schema scan |
| same raw task -> same canonical task | True | repeat hash |
| provenance complete | True | schema fields |
| request_scope respected | True | PER_JOB identity |
| GPU model preserved | True | row equality |
| HP/Spot preserved | True | enum |
| forbidden deployable tokens absent | True | deployable scan |
