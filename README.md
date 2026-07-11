# Data Center Thermal-Electric Environment

This repository contains a reusable single-data-center thermo-electric control
environment derived from the project technical route. The main runtime is the
installable `datacenter_env` package; external signal generation and experiment
matrix construction remain outside that package.

## Install

```powershell
.\.venv311\Scripts\python.exe -m pip install -e .
.\.venv311\Scripts\python.exe -c "import datacenter_env; print(datacenter_env.__version__)"
```

Python 3.11 or newer is required.

## Package Controller And SQLite

```python
from datacenter_env import DataCenterSystem, RunMetadata, SQLiteRunStore

provider = build_signal_provider(...)  # Created outside datacenter_env.
system = DataCenterSystem.from_config(
    config,
    store=SQLiteRunStore("data/db/single_center.sqlite"),
)
system.start_run(RunMetadata(controller_name=config.controller_name))

for index, current_input in enumerate(provider):
    window = provider.window(index, system.controller.max_forecast_steps)
    system.step(current_input=current_input, forecast_window=window)

summary = system.finish_run()
```

## Disable Persistence

```python
from datacenter_env import DataCenterSystem, NullRunStore

system = DataCenterSystem.from_config(config, store=NullRunStore())
```

`NullRunStore` keeps the same run lifecycle without writing to disk. It is
intended for tests, repeated training episodes, and external agents.

## External Agent

```python
from datacenter_env import DataCenterAction, DataCenterEnvironment

env = DataCenterEnvironment.from_config(config)
observation = env.reset(seed=42)
action = DataCenterAction(cooling_target_kw=140.0)
result = env.step(action=action, current_input=current_input)
```

This lower-level interface bypasses the built-in controller while preserving
thermal constraints, actuator dynamics, plant physics, measurement, accounting,
and diagnostics.

## Commands

```powershell
.\.venv311\Scripts\python.exe scripts/run_single_center.py
.\.venv311\Scripts\python.exe scripts/inspect_database.py
.\.venv311\Scripts\python.exe scripts/compare_single_center_runs.py
.\.venv311\Scripts\python.exe scripts/run_robustness_experiments.py
.\.venv311\Scripts\python.exe scripts/compare_robustness_experiments.py
.\.venv311\Scripts\python.exe scripts/analyze_closed_loop_robustness.py
.\.venv311\Scripts\python.exe scripts/run_tests.py
```

## Structure

- `src/datacenter_env/`: installable environment package
- `external_signals/`: synthetic/CSV signal ownership and forecast slicing
- `experiments/`: thin experiment and matrix orchestration
- `scripts/`: command-line entry points and reports
- `configs/`: physical, controller, condition, and external-signal configuration
- `tests/`: legacy regression and package architecture tests
- `references/`: read-only external reference repositories

The legacy `models/`, `optimization/`, `evaluation/`, and `storage/` imports are
temporarily retained for compatibility. New integrations should import only the
documented top-level `datacenter_env` API. See
`docs/package_architecture.md` for package boundaries and extension points.
