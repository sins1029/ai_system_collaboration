from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from experiments.mvp import load_yaml
from models.datacenter import DatacenterModel
from models.measurement import TemperatureMeasurement
from models.prediction import ThermalPredictionModel


ROOT = Path(__file__).resolve().parents[1]


class MeasurementAndPredictionTest(unittest.TestCase):
    def test_measurement_noise_is_seed_reproducible(self) -> None:
        config = {"mode": "gaussian", "temperature_noise_std_c": 0.2, "seed": 7}
        first = TemperatureMeasurement(config)
        second = TemperatureMeasurement(config)
        third = TemperatureMeasurement({**config, "seed": 8})
        first_values = [first.measure_temperature_c(24.0) for _ in range(5)]
        second_values = [second.measure_temperature_c(24.0) for _ in range(5)]
        third_values = [third.measure_temperature_c(24.0) for _ in range(5)]
        self.assertEqual(first_values, second_values)
        self.assertNotEqual(first_values, third_values)

    def test_plant_and_prediction_parameters_are_isolated(self) -> None:
        dc = load_yaml(ROOT / "configs" / "datacenter.yaml")
        plant_parameters = {
            "thermal_capacity_scale": 1.0,
            "heat_transfer_scale": 1.0,
            "cooling_effectiveness_scale": 1.0,
        }
        prediction_parameters = deepcopy(plant_parameters)
        plant = DatacenterModel(dc, 0.25, plant_parameters)
        predictor = ThermalPredictionModel(dc, prediction_parameters, 0.25)
        original_plant = plant.plant_step(24.0, 0.4, 0.1, 100.0, 20.0, 0.0).temp_c
        prediction_parameters["thermal_capacity_scale"] = 2.0
        unchanged_prediction = predictor.predict_step(24.0, 170.0, 100.0, 20.0)
        repeated_prediction = predictor.predict_step(24.0, 170.0, 100.0, 20.0)
        self.assertEqual(unchanged_prediction, repeated_prediction)
        self.assertEqual(
            original_plant,
            plant.plant_step(24.0, 0.4, 0.1, 100.0, 20.0, 0.0).temp_c,
        )

    def test_prediction_error_sign_is_predicted_minus_actual(self) -> None:
        dc = load_yaml(ROOT / "configs" / "datacenter.yaml")
        perfect = ThermalPredictionModel(
            dc,
            {
                "thermal_capacity_scale": 1.0,
                "heat_transfer_scale": 1.0,
                "cooling_effectiveness_scale": 1.0,
            },
            0.25,
        )
        biased = ThermalPredictionModel(
            dc,
            {
                "thermal_capacity_scale": 1.05,
                "heat_transfer_scale": 0.95,
                "cooling_effectiveness_scale": 1.05,
            },
            0.25,
        )
        actual = perfect.predict_step(24.0, 170.0, 100.0, 20.0)
        predicted = biased.predict_step(24.0, 170.0, 100.0, 20.0)
        passive_kw = (24.0 - 20.0) / 0.18
        expected_actual = 24.0 + 0.25 / 55.0 * (170.0 - 100.0 - passive_kw)
        expected_predicted = 24.0 + 0.25 / (55.0 * 1.05) * (
            170.0 * 0.95 - 100.0 * 1.05 - passive_kw
        )
        self.assertAlmostEqual(predicted - actual, expected_predicted - expected_actual)
        self.assertNotEqual(predicted - actual, 0.0)


if __name__ == "__main__":
    unittest.main()
