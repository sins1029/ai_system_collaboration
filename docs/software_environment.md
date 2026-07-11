# Software Environment

This project follows section 12.1 of the technical route.

## Runtime

- Python 3.11 is configured locally under `.runtime/python311`.
- The active project environment is `.venv311`.
- The previous `.venv` may exist from earlier setup, but `.venv311` is the
  environment to use for this project.

## Dependency Groups

- `requirements.txt`: data analysis and plotting
- `requirements-optimization.txt`: mathematical optimization
- `requirements-rl.txt`: Gymnasium, Stable-Baselines3, and TensorBoard
- `requirements-optional.txt`: W&B and RLlib for later-stage experiments

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
```

Activate:

```powershell
.\.venv311\Scripts\Activate.ps1
$env:MPLCONFIGDIR="$PWD\.cache\matplotlib"
```

Verify:

```powershell
python scripts/verify_env.py
python scripts/run_mvp.py
```

## Notes

- Gurobi is not installed by default. Use HiGHS first, as recommended by the
  technical route.
- RLlib and W&B are optional and intentionally kept outside the default setup.
- Large datasets and generated results stay outside Git.
- Matplotlib uses `.cache/matplotlib` because the default user cache directory
  may be unavailable in the sandboxed desktop environment.
