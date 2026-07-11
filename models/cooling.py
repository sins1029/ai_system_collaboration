from __future__ import annotations


def cop(outdoor_temp_c: float, cop_base: float, cop_temp_slope: float, cop_min: float) -> float:
    return max(cop_min, cop_base - cop_temp_slope * outdoor_temp_c)


def cooling_power_kw(cooling_kw: float, outdoor_temp_c: float, config: dict) -> float:
    effective_cop = cop(
        outdoor_temp_c=outdoor_temp_c,
        cop_base=float(config["cop_base"]),
        cop_temp_slope=float(config["cop_temp_slope"]),
        cop_min=float(config["cop_min"]),
    )
    return max(cooling_kw, 0.0) / effective_cop

