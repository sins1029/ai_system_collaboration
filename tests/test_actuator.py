from __future__ import annotations

import unittest

from models.actuator import CoolingActuator


class CoolingActuatorTest(unittest.TestCase):
    def test_ideal_applies_target_exactly(self) -> None:
        actuator = CoolingActuator(
            {"mode": "ideal", "initial_applied_cooling_kw": 0.0}, 15.0, 280.0
        )
        result = actuator.step(150.0, actuator.initial_state())
        self.assertEqual(result.applied_cooling_kw, 150.0)
        self.assertEqual(result.tracking_error_kw, 0.0)

    def test_rate_limits_up_and_down_separately(self) -> None:
        actuator = CoolingActuator(
            {
                "mode": "rate_limited", "initial_applied_cooling_kw": 100.0,
                "max_ramp_up_kw_per_step": 30.0, "max_ramp_down_kw_per_step": 50.0,
            },
            15.0,
            280.0,
        )
        up = actuator.step(250.0, actuator.initial_state())
        down = actuator.step(0.0, up.state)
        self.assertEqual(up.applied_cooling_kw, 130.0)
        self.assertTrue(up.ramp_up_limited)
        self.assertEqual(down.applied_cooling_kw, 80.0)
        self.assertTrue(down.ramp_down_limited)

    def test_first_order_approaches_step_without_instant_jump(self) -> None:
        actuator = CoolingActuator(
            {
                "mode": "first_order", "initial_applied_cooling_kw": 0.0,
                "time_constant_minutes": 30.0,
            },
            15.0,
            280.0,
        )
        first = actuator.step(200.0, actuator.initial_state())
        second = actuator.step(200.0, first.state)
        self.assertGreater(first.applied_cooling_kw, 0.0)
        self.assertLess(first.applied_cooling_kw, 200.0)
        self.assertGreater(second.applied_cooling_kw, first.applied_cooling_kw)
        self.assertLess(second.applied_cooling_kw, 200.0)
        self.assertGreater(first.alpha, 0.0)
        self.assertLessEqual(first.alpha, 1.0)

    def test_delayed_mode_uses_explicit_queue(self) -> None:
        actuator = CoolingActuator(
            {"mode": "delayed", "initial_applied_cooling_kw": 0.0, "delay_steps": 1},
            15.0,
            280.0,
        )
        first = actuator.step(120.0, actuator.initial_state())
        second = actuator.step(60.0, first.state)
        self.assertEqual(first.applied_cooling_kw, 0.0)
        self.assertEqual(second.applied_cooling_kw, 120.0)

    def test_actuator_infeasibility_is_explicit(self) -> None:
        actuator = CoolingActuator(
            {
                "mode": "rate_limited", "initial_applied_cooling_kw": 0.0,
                "max_ramp_up_kw_per_step": 30.0, "max_ramp_down_kw_per_step": 50.0,
            },
            15.0,
            280.0,
        )
        _, infeasible = actuator.actuator_aware_target(
            250.0, actuator.initial_state(), 200.0, 280.0
        )
        self.assertTrue(infeasible)


if __name__ == "__main__":
    unittest.main()
