"""Aggregate powermetrics samples into harness metrics."""

from __future__ import annotations

from dataclasses import dataclass

from powermetrics_log import PowerSample, sample_energy_joules


@dataclass
class PowerSummary:
    energy_joules: float
    energy_ane_joules: float
    energy_cpu_joules: float
    energy_gpu_joules: float
    sustained_power_w: float | None
    ane_utilization_pct: float | None
    sample_count: int


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_power_samples(samples: list[PowerSample], steady_window: int = 60) -> PowerSummary:
    if not samples:
        return PowerSummary(0.0, 0.0, 0.0, 0.0, None, None, 0)

    steady = samples[-steady_window:] if len(samples) >= steady_window else samples
    energy_joules = sum(sample_energy_joules(sample) for sample in samples)
    energy_cpu = sum(sample.cpu_mj for sample in samples) / 1000.0
    energy_gpu = sum(sample.gpu_mj for sample in samples) / 1000.0
    energy_ane = sum(sample.ane_mj for sample in samples) / 1000.0
    sustained_power_w = _mean([sample.package_mw / 1000.0 for sample in steady])

    ane_utilization_pct = None
    if energy_ane > 0:
        denom = energy_cpu + energy_gpu + energy_ane
        if denom > 0:
            ane_utilization_pct = (energy_ane / denom) * 100.0

    return PowerSummary(
        energy_joules=energy_joules,
        energy_ane_joules=energy_ane,
        energy_cpu_joules=energy_cpu,
        energy_gpu_joules=energy_gpu,
        sustained_power_w=sustained_power_w,
        ane_utilization_pct=ane_utilization_pct,
        sample_count=len(samples),
    )