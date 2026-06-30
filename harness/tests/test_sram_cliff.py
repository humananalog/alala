"""Unit tests for SRAM cliff detection."""

from __future__ import annotations

import unittest

from sram_cliff import ContextStepResult, context_lengths, detect_sram_cliff


def _step(
    context_length: int,
    tps: float,
    ane: float | None = None,
    power: float | None = None,
    memory: float = 1.0,
) -> ContextStepResult:
    return ContextStepResult(
        context_length=context_length,
        tokens_per_second_sustained=tps,
        tokens_per_second=tps,
        tokens_generated=100,
        ane_utilization_pct=ane,
        sustained_power_w=power,
        peak_memory_gb=memory,
        energy_joules=10.0,
        experiment_id=f"ctx{context_length}",
        powermetrics_log_path=f"logs/ctx{context_length}.txt",
    )


class SramCliffTests(unittest.TestCase):
    def test_context_lengths(self) -> None:
        self.assertEqual(context_lengths(2048), [512, 1024, 2048])
        self.assertEqual(context_lengths(512), [512])

    def test_detect_cliff_on_throughput_drop(self) -> None:
        steps = [
            _step(512, 20.0, ane=50.0, power=10.0, memory=1.0),
            _step(1024, 12.0, ane=45.0, power=11.0, memory=1.2),
        ]
        self.assertEqual(detect_sram_cliff(steps), 1024)

    def test_detect_cliff_with_monotonic_memory_rise(self) -> None:
        steps = [
            _step(512, 9.65, memory=4.70),
            _step(1024, 6.40, memory=4.92),
        ]
        self.assertEqual(detect_sram_cliff(steps), 1024)

    def test_no_cliff_if_drop_below_threshold(self) -> None:
        steps = [
            _step(512, 20.0, ane=50.0, power=10.0, memory=1.0),
            _step(1024, 18.0, ane=49.0, power=10.0, memory=1.0),
        ]
        self.assertIsNone(detect_sram_cliff(steps))


if __name__ == "__main__":
    unittest.main()