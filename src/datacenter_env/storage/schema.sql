PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    time_step_minutes INTEGER NOT NULL,
    signal_source TEXT,
    metadata_json TEXT,
    random_seed INTEGER,
    timezone TEXT,
    start_timestamp TEXT,
    end_timestamp TEXT,
    number_of_steps INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS input_timeseries (
    dataset_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    online_load REAL NOT NULL,
    batch_load REAL NOT NULL,
    price REAL NOT NULL,
    carbon REAL NOT NULL,
    outdoor_temp REAL NOT NULL,
    renewable REAL NOT NULL,
    online_workload REAL,
    batch_workload REAL,
    electricity_price REAL,
    carbon_intensity_kg_per_kwh REAL,
    outdoor_temperature_c REAL,
    renewable_power_kw REAL,
    PRIMARY KEY (dataset_id, timestamp),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    scenario TEXT NOT NULL,
    controller_name TEXT,
    condition_name TEXT,
    scheduler_name TEXT,
    cooling_controller_name TEXT,
    task_dataset_id TEXT,
    interface_type TEXT,
    environment_id TEXT,
    reward_config_json TEXT,
    observation_config_json TEXT,
    invalid_action_policy TEXT,
    candidate_order TEXT,
    episode_seed INTEGER,
    actuator_mode TEXT,
    mismatch_scenario TEXT,
    measurement_mode TEXT,
    dataset_id INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    git_commit TEXT,
    config_json TEXT NOT NULL,
    scenario_config_json TEXT,
    plant_config_json TEXT,
    prediction_config_json TEXT,
    package_version TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    error_message TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE TABLE IF NOT EXISTS simulation_results (
    run_id INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    online_load REAL NOT NULL,
    batch_load REAL NOT NULL,
    batch_service REAL NOT NULL,
    backlog REAL NOT NULL,
    total_load REAL NOT NULL,
    prev_temp_c REAL,
    price REAL NOT NULL,
    carbon REAL NOT NULL,
    outdoor_temp REAL NOT NULL,
    renewable REAL NOT NULL,
    renewable_kw REAL,
    online_workload REAL,
    batch_workload REAL,
    electricity_price REAL,
    carbon_intensity_kg_per_kwh REAL,
    outdoor_temperature_c REAL,
    renewable_available_kw REAL,
    renewable_used_kw REAL,
    renewable_curtailed_kw REAL,
    it_power_kw REAL NOT NULL,
    heat_kw REAL,
    proposed_cooling_kw REAL,
    constrained_cooling_kw REAL,
    applied_cooling_kw REAL,
    cooling_command_kw REAL,
    cooling_kw REAL NOT NULL,
    control_movement_kw REAL,
    proposed_control_movement_kw REAL,
    applied_control_movement_kw REAL,
    actuator_tracking_error_kw REAL,
    actuator_ramp_limited INTEGER,
    actuator_ramp_up_limited INTEGER,
    actuator_ramp_down_limited INTEGER,
    actuator_alpha REAL,
    actuator_induced_thermal_infeasibility INTEGER,
    cooling_min_feasible_kw REAL,
    cooling_max_feasible_kw REAL,
    cooling_constraint_intervention INTEGER,
    cooling_constraint_intervention_kw REAL,
    cooling_constraint_intervention_energy_kwh REAL,
    cooling_power_kw REAL NOT NULL,
    p_aux_kw REAL,
    total_power_kw REAL NOT NULL,
    grid_power_kw REAL NOT NULL,
    temp_c REAL NOT NULL,
    true_temperature_c REAL,
    measured_temperature_c REAL,
    controller_measured_temperature_c REAL,
    predicted_next_temperature_c REAL,
    actual_next_temperature_c REAL,
    one_step_temperature_prediction_error_c REAL,
    temp_min_c REAL,
    temp_setpoint_c REAL,
    temp_max_c REAL NOT NULL,
    temp_deadband_c REAL,
    temp_control_target_c REAL,
    precool_floor_c REAL,
    precooling_active INTEGER,
    power_balance_error REAL,
    renewable_balance_error REAL,
    thermal_balance_error REAL,
    temperature_is_finite INTEGER,
    temperature_within_physical_range INTEGER,
    temperature_step_change REAL,
    below_min_temperature INTEGER,
    above_max_temperature INTEGER,
    temperature_violation INTEGER,
    temperature_deviation_from_setpoint_c REAL,
    thermal_infeasible INTEGER,
    optimizer_attempted INTEGER,
    optimizer_success INTEGER,
    optimizer_failure INTEGER,
    optimizer_fallback INTEGER,
    prediction_infeasibility INTEGER,
    actuator_infeasibility INTEGER,
    fallback_reason TEXT,
    optimizer_candidate_evaluations INTEGER,
    objective_total REAL,
    objective_energy_cost REAL,
    objective_carbon REAL,
    objective_temperature REAL,
    objective_control_movement REAL,
    invalid_value_count INTEGER,
    negative_power_count INTEGER,
    workload_mode TEXT,
    cpu_utilization REAL,
    gpu_utilization REAL,
    memory_utilization REAL,
    waiting_task_count INTEGER,
    running_task_count INTEGER,
    completed_task_count INTEGER,
    at_risk_task_count INTEGER,
    PRIMARY KEY (run_id, step_index),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_metrics (
    run_id INTEGER NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (run_id, metric),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_input_timeseries (
    run_id INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    workload_fraction REAL NOT NULL,
    electricity_price_per_kwh REAL NOT NULL,
    carbon_intensity_kg_per_kwh REAL NOT NULL,
    outdoor_temperature_c REAL NOT NULL,
    renewable_power_kw REAL NOT NULL,
    PRIMARY KEY (run_id, step_index),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tasks (
    dataset_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    arrival_time TEXT NOT NULL,
    duration_steps INTEGER NOT NULL,
    cpu_cores REAL NOT NULL,
    gpu_units REAL NOT NULL,
    memory_gb REAL NOT NULL,
    deadline_time TEXT NOT NULL,
    priority INTEGER NOT NULL,
    "deferrable" INTEGER NOT NULL,
    PRIMARY KEY (dataset_id, task_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_task_events (
    run_id INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    task_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    details_json TEXT,
    FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_task_outcomes (
    run_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    final_status TEXT NOT NULL,
    first_start_time TEXT,
    completion_time TEXT,
    wait_steps INTEGER NOT NULL,
    deferral_count INTEGER NOT NULL,
    sla_violated INTEGER NOT NULL,
    lateness_minutes REAL,
    resource_blocked_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, task_id),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_agent_decisions (
    run_id INTEGER NOT NULL,
    decision_step_index INTEGER NOT NULL,
    simulation_step_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    candidate_task_id TEXT,
    action INTEGER NOT NULL,
    action_legal INTEGER NOT NULL,
    action_mask_json TEXT NOT NULL,
    decision_reward REAL NOT NULL,
    simulation_reward REAL NOT NULL,
    total_reward REAL NOT NULL,
    reward_components_json TEXT NOT NULL,
    PRIMARY KEY (run_id, decision_step_index),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_gym_episode_summaries (
    run_id INTEGER PRIMARY KEY,
    episode_reward REAL NOT NULL,
    decision_steps INTEGER NOT NULL,
    simulation_steps INTEGER NOT NULL,
    invalid_actions INTEGER NOT NULL,
    terminated INTEGER NOT NULL,
    truncated INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_input_timeseries_timestamp
    ON input_timeseries(timestamp);

CREATE INDEX IF NOT EXISTS idx_simulation_results_timestamp
    ON simulation_results(timestamp);

CREATE INDEX IF NOT EXISTS idx_run_task_events_task
    ON run_task_events(run_id, task_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_run_agent_decisions_step
    ON run_agent_decisions(run_id, simulation_step_index, decision_step_index);
