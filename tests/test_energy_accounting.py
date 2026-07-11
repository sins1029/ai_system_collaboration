from __future__ import annotations

import unittest

from evaluation.accounting import carbon_emissions_kg, energy_cost, renewable_allocation
from models.datacenter import DatacenterModel


class EnergyAccountingTest(unittest.TestCase):
    def test_cost_and_carbon_match_hand_calculation(self) -> None:
        self.assertAlmostEqual(energy_cost(100.0, 0.5, 0.25), 12.5)
        self.assertAlmostEqual(carbon_emissions_kg(100.0, 0.4, 0.25), 10.0)

    def test_renewable_allocation_conserves_power(self) -> None:
        used, curtailed = renewable_allocation(300.0, 220.0)
        self.assertEqual(used, 220.0)
        self.assertEqual(curtailed, 80.0)
        self.assertAlmostEqual(300.0, used + curtailed)

    def test_plant_never_exports_negative_grid_power(self) -> None:
        config = {
            "capacity": {"max_load": 1.0},
            "power": {"p_idle_kw": 80.0, "p_peak_kw": 260.0, "p_aux_kw": 12.0},
            "thermal": {
                "thermal_resistance_c_per_kw": 0.18,
                "thermal_capacitance_kwh_per_c": 55.0,
            },
            "cooling": {
                "max_cooling_kw": 280.0, "cop_base": 4.0,
                "cop_temp_slope": 0.035, "cop_min": 2.2,
            },
        }
        result = DatacenterModel(config, 0.25).plant_step(
            24.0, 0.2, 0.1, 50.0, 20.0, 10_000.0
        )
        self.assertGreaterEqual(result.grid_power_kw, 0.0)
        self.assertAlmostEqual(
            result.renewable_available_kw,
            result.renewable_used_kw + result.renewable_curtailed_kw,
        )


if __name__ == "__main__":
    unittest.main()
