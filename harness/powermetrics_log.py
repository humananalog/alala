"""Parse powermetrics plist output into structured samples."""

from __future__ import annotations

import plistlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PowerSample:
    timestamp: datetime | None
    elapsed_ns: int
    cpu_mj: float
    gpu_mj: float
    ane_mj: float
    package_mw: float
    thermal_pressure: str
    gpu_active_ratio: float
    temp_c: float | None


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _extract_temperature_c(sample: dict[str, Any]) -> float | None:
    candidates: list[float] = []

    smc = sample.get("smc")
    if isinstance(smc, dict):
        sensors = smc.get("sensors") or smc.get("keys")
        if isinstance(sensors, list):
            for entry in sensors:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", "")).lower()
                if "temp" not in name and "therm" not in name:
                    continue
                value = entry.get("value")
                if value is not None:
                    candidates.append(float(value))

    for key in ("die_temperature", "package_temperature", "cpu_die_temp"):
        if key in sample and sample[key] is not None:
            candidates.append(float(sample[key]))

    if not candidates:
        return None
    return max(candidates)


def parse_sample(sample: dict[str, Any]) -> PowerSample | None:
    processor = sample.get("processor")
    if not isinstance(processor, dict):
        return None

    gpu = sample.get("gpu") if isinstance(sample.get("gpu"), dict) else {}
    timestamp = sample.get("timestamp")
    if isinstance(timestamp, datetime):
        parsed_ts: datetime | None = timestamp
    else:
        parsed_ts = None

    idle_ratio = _as_float(gpu.get("idle_ratio"), 1.0)
    return PowerSample(
        timestamp=parsed_ts,
        elapsed_ns=int(sample.get("elapsed_ns", 0) or 0),
        cpu_mj=_as_float(processor.get("cpu_energy")),
        gpu_mj=_as_float(processor.get("gpu_energy")),
        ane_mj=_as_float(processor.get("ane_energy")),
        package_mw=_as_float(processor.get("combined_power")),
        thermal_pressure=str(sample.get("thermal_pressure", "Unknown")),
        gpu_active_ratio=max(0.0, 1.0 - idle_ratio),
        temp_c=_extract_temperature_c(sample),
    )


def parse_powermetrics_file(path: Path) -> list[PowerSample]:
    raw = path.read_bytes()
    if not raw:
        return []

    samples: list[PowerSample] = []
    for blob in raw.split(b"\x00"):
        if not blob.strip():
            continue
        try:
            parsed = plistlib.loads(blob)
        except (plistlib.InvalidFileException, ValueError, OSError):
            continue
        if not isinstance(parsed, dict):
            continue
        sample = parse_sample(parsed)
        if sample is not None:
            samples.append(sample)
    return samples


def sample_energy_joules(sample: PowerSample) -> float:
    return (sample.cpu_mj + sample.gpu_mj + sample.ane_mj) / 1000.0


def sample_interval_seconds(sample: PowerSample) -> float:
    if sample.elapsed_ns <= 0:
        return 1.0
    return sample.elapsed_ns / 1_000_000_000.0