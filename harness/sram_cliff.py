"""SRAM cliff sweep logic for Phase 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CONTEXT_SWEEP = (512, 1024, 2048, 4096, 8192, 16384)
CLIFF_DROP_THRESHOLD = 0.30


@dataclass
class ContextStepResult:
    context_length: int
    tokens_per_second_sustained: float
    tokens_per_second: float
    tokens_generated: int
    ane_utilization_pct: float | None
    sustained_power_w: float | None
    peak_memory_gb: float
    energy_joules: float
    experiment_id: str
    powermetrics_log_path: str


def context_lengths(max_context: int) -> list[int]:
    return [length for length in CONTEXT_SWEEP if length <= max_context]


def detect_sram_cliff(steps: list[ContextStepResult]) -> int | None:
    if len(steps) < 2:
        return None

    for index in range(1, len(steps)):
        previous = steps[index - 1]
        current = steps[index]
        if previous.tokens_per_second_sustained <= 0:
            continue

        drop_ratio = (
            previous.tokens_per_second_sustained - current.tokens_per_second_sustained
        ) / previous.tokens_per_second_sustained
        if drop_ratio < CLIFF_DROP_THRESHOLD:
            continue

        ane_drop = False
        if (
            previous.ane_utilization_pct is not None
            and current.ane_utilization_pct is not None
            and current.ane_utilization_pct < previous.ane_utilization_pct
        ):
            ane_drop = True

        power_rise = False
        if (
            previous.sustained_power_w is not None
            and current.sustained_power_w is not None
            and current.sustained_power_w > previous.sustained_power_w * 1.05
        ):
            power_rise = True

        memory_rise = current.peak_memory_gb > previous.peak_memory_gb * 1.02
        memory_monotonic = current.peak_memory_gb > previous.peak_memory_gb

        if ane_drop or power_rise or memory_rise or memory_monotonic:
            return current.context_length

    return None


def build_context_record(
    *,
    run_id: str,
    step: ContextStepResult,
    hardware_brand: str,
    model: str,
    temp_start_c: float | None,
    temp_steady_state_c: float | None,
    peak_temp_c: float | None,
    timestamp: str,
    quality_rate: float = 1.0,
) -> dict[str, Any]:
    u_task = step.tokens_generated * quality_rate
    ipj = u_task / step.energy_joules if step.energy_joules > 0 else None

    return {
        "timestamp": timestamp,
        "experiment_id": step.experiment_id,
        "run_id": run_id,
        "task_type": "sram_cliff",
        "model": model,
        "context_length": step.context_length,
        "energy_joules": round(step.energy_joules, 4),
        "energy_ane_joules": None,
        "energy_cpu_orchestration_joules": None,
        "energy_dequant_joules": None,
        "tokens_generated": step.tokens_generated,
        "tokens_per_second": round(step.tokens_per_second, 3),
        "tokens_per_second_sustained": round(step.tokens_per_second_sustained, 3),
        "ane_utilization_pct": step.ane_utilization_pct,
        "peak_memory_gb": round(step.peak_memory_gb, 3),
        "temp_start_c": temp_start_c,
        "peak_temp_c": peak_temp_c,
        "temp_steady_state_c": temp_steady_state_c,
        "time_to_throttle_s": None,
        "sustained_power_w": (
            round(step.sustained_power_w, 3) if step.sustained_power_w is not None else None
        ),
        "u_task_score": round(quality_rate, 3),
        "ipj": round(ipj, 6) if ipj is not None else None,
        "hca_impact": None,
        "hardware": hardware_brand,
        "notes": "Phase 0 SRAM cliff context step; batch_size=1; MLX decode sweep.",
        "powermetrics_log_path": step.powermetrics_log_path,
    }


def build_summary_record(
    *,
    run_id: str,
    steps: list[ContextStepResult],
    l_cliff: int | None,
    hardware_brand: str,
    model: str,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "experiment_id": f"{run_id}_summary",
        "run_id": run_id,
        "task_type": "sram_cliff_summary",
        "model": model,
        "context_length": l_cliff,
        "l_cliff": l_cliff,
        "contexts_tested": [step.context_length for step in steps],
        "tokens_per_second_sustained_by_context": {
            str(step.context_length): round(step.tokens_per_second_sustained, 3) for step in steps
        },
        "hardware": hardware_brand,
        "notes": (
            f"SRAM cliff L_cliff={l_cliff} (>=30% sustained throughput drop vs prior step)."
            if l_cliff is not None
            else "No SRAM cliff detected within tested context lengths."
        ),
        "powermetrics_log_path": None,
    }