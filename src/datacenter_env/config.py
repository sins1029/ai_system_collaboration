from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from datacenter_env.exceptions import ConfigurationError


IDEAL_SCALES = {
    "thermal_capacity_scale": 1.0,
    "heat_transfer_scale": 1.0,
    "cooling_effectiveness_scale": 1.0,
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class DataCenterSystemConfig:
    datacenter: Mapping[str, Any]
    optimization: Mapping[str, Any]
    controller_name: str = "baseline"
    condition_name: str = "ideal"
    step_minutes: int = 15
    actuator: Mapping[str, Any] | None = None
    plant_parameters: Mapping[str, Any] | None = None
    prediction_parameters: Mapping[str, Any] | None = None
    measurement: Mapping[str, Any] | None = None
    mismatch_scenario: str = "perfect_model"
    database_path: str | None = None

    def __post_init__(self) -> None:
        if self.controller_name not in {"baseline", "heuristic", "finite_horizon"}:
            raise ConfigurationError(f"unknown controller: {self.controller_name}")
        if int(self.step_minutes) <= 0:
            raise ConfigurationError("step_minutes must be positive")
        object.__setattr__(self, "datacenter", _freeze(self.datacenter))
        object.__setattr__(self, "optimization", _freeze(self.optimization))
        object.__setattr__(
            self,
            "actuator",
            _freeze(self.actuator or {"mode": "ideal", "initial_applied_cooling_kw": 0.0}),
        )
        object.__setattr__(self, "plant_parameters", _freeze(self.plant_parameters or IDEAL_SCALES))
        default_prediction = self.prediction_parameters
        if default_prediction is None:
            default_prediction = self.optimization.get("finite_horizon", {}).get(
                "prediction", IDEAL_SCALES
            )
        object.__setattr__(self, "prediction_parameters", _freeze(default_prediction))
        object.__setattr__(
            self,
            "measurement",
            _freeze(self.measurement or {"mode": "none", "temperature_noise_std_c": 0.0, "seed": 0}),
        )
        self._validate()

    @classmethod
    def from_dict(
        cls,
        datacenter: Mapping[str, Any],
        optimization: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "DataCenterSystemConfig":
        return cls(datacenter=datacenter, optimization=optimization or {}, **kwargs)

    @classmethod
    def coerce(cls, config: "DataCenterSystemConfig | Mapping[str, Any]") -> "DataCenterSystemConfig":
        if isinstance(config, cls):
            return config
        if "datacenter" not in config:
            raise ConfigurationError("mapping config must contain a datacenter section")
        return cls(
            datacenter=config["datacenter"],
            optimization=config.get("optimization", {}),
            controller_name=str(config.get("controller_name", "baseline")),
            condition_name=str(config.get("condition_name", "ideal")),
            step_minutes=int(config.get("step_minutes", 15)),
            actuator=config.get("actuator"),
            plant_parameters=config.get("plant_parameters"),
            prediction_parameters=config.get("prediction_parameters"),
            measurement=config.get("measurement"),
            mismatch_scenario=str(config.get("mismatch_scenario", "perfect_model")),
            database_path=(
                str(config["database_path"])
                if config.get("database_path") is not None
                else None
            ),
        )

    @property
    def step_hours(self) -> float:
        return self.step_minutes / 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "datacenter": _thaw(self.datacenter),
            "optimization": _thaw(self.optimization),
            "controller_name": self.controller_name,
            "condition_name": self.condition_name,
            "step_minutes": self.step_minutes,
            "actuator": _thaw(self.actuator),
            "plant_parameters": _thaw(self.plant_parameters),
            "prediction_parameters": _thaw(self.prediction_parameters),
            "measurement": _thaw(self.measurement),
            "mismatch_scenario": self.mismatch_scenario,
            "database_path": self.database_path,
        }

    def _validate(self) -> None:
        try:
            thermal = self.datacenter["thermal"]
            cooling = self.datacenter["cooling"]
            if float(thermal["min_temp_c"]) >= float(thermal["max_temp_c"]):
                raise ConfigurationError("minimum temperature must be below maximum temperature")
            if float(cooling["max_cooling_kw"]) <= 0:
                raise ConfigurationError("maximum cooling must be positive")
            if min(float(value) for value in self.plant_parameters.values()) <= 0:
                raise ConfigurationError("plant parameter scales must be positive")
            if min(float(value) for value in self.prediction_parameters.values()) <= 0:
                raise ConfigurationError("prediction parameter scales must be positive")
        except KeyError as error:
            raise ConfigurationError(f"missing configuration field: {error}") from error
