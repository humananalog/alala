#!/usr/bin/env python3
"""Alalā Phase 0 energy measurement harness for Mac Mini M4 24 GB.

Spawns powermetrics, integrates CPU/GPU/ANE power, logs JSONL per
IPJ_Measurement_Protocol_Alalā.md. Use --dry-run off-hardware for structure checks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"

MODES = (
    "thermal_baseline",
    "sram_cliff",
    "kv_comparison",
    "orchestration",
    "ane_utilization",
    "thermal_ipj_curve",
    "meta_tax",
    "memory_spill",
)

# Regexes for powermetrics text output (Apple Silicon).
_RE_CPU_POWER = re.compile(r"CPU Power:\s+([\d.]+)\s+mW", re.I)
_RE_GPU_POWER = re.compile(r"GPU Power:\s+([\d.]+)\s+mW", re.I)
_RE_ANE_POWER = re.compile(r"ANE Power:\s+([\d.]+)\s+mW", re.I)
_RE_PACKAGE_POWER = re.compile(r"Package Power:\s+([\d.]+)\s+mW", re.I)
_RE_CPU_TEMP = re.compile(r"CPU die temperature:\s+([\d.]+)\s+C", re.I)
_RE_GPU_TEMP = re.compile(r"GPU die temperature:\s+([\d.]+)\s+C", re.I)


@dataclass
class PowerSample:
    elapsed_s: float
    cpu_mw: float = 0.0
    gpu_mw: float = 0.0
    ane_mw: float = 0.0
    package_mw: float = 0.0
    temp_c: float | None = None


@dataclass
class EnergyTotals:
    cpu_joules: float = 0.0
    gpu_joules: float = 0.0
    ane_joules: float = 0.0
    total_joules: float = 0.0
    samples: list[PowerSample] = field(default_factory=list)

    @property
    def sustained_power_w(self) -> float:
        if not self.samples:
            return 0.0
        dt = self.samples[-1].elapsed_s - self.samples[0].elapsed_s
        if dt <= 0:
            return 0.0
        return self.total_joules / dt

    def peak_temp_c(self) -> float | None:
        temps = [s.temp_c for s in self.samples if s.temp_c is not None]
        return max(temps) if temps else None

    def steady_state_temp_c(self, tail_fraction: float = 0.2) -> float | None:
        temps = [s.temp_c for s in self.samples if s.temp_c is not None]
        if not temps:
            return None
        n = max(1, int(len(temps) * tail_fraction))
        tail = temps[-n:]
        return sum(tail) / len(tail)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def check_platform(*, dry_run: bool) -> None:
    if dry_run:
        return
    if platform.system() != "Darwin":
        sys.exit("ERROR: harness requires macOS on physical Mac Mini M4 24 GB (use --dry-run off-hardware).")
    machine = platform.machine().lower()
    if machine not in ("arm64", "aarch64"):
        sys.exit(f"ERROR: expected Apple Silicon (arm64), got {machine!r}.")
    ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3)
    if ram_gb < 20 or ram_gb > 28:
        sys.exit(
            f"ERROR: expected ~24 GB unified memory, detected {ram_gb:.1f} GB. "
            "Refusing to run outside target envelope."
        )


def parse_powermetrics_chunk(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pattern in (
        ("cpu_mw", _RE_CPU_POWER),
        ("gpu_mw", _RE_GPU_POWER),
        ("ane_mw", _RE_ANE_POWER),
        ("package_mw", _RE_PACKAGE_POWER),
    ):
        m = pattern.search(text)
        if m:
            out[key] = float(m.group(1))
    for key, pattern in (("temp_c", _RE_CPU_TEMP), ("temp_c", _RE_GPU_TEMP)):
        m = pattern.search(text)
        if m:
            out[key] = float(m.group(1))
            break
    return out


def integrate_sample(totals: EnergyTotals, sample: PowerSample, dt_s: float) -> None:
    if dt_s <= 0:
        return
    totals.cpu_joules += sample.cpu_mw * dt_s / 1000.0
    totals.gpu_joules += sample.gpu_mw * dt_s / 1000.0
    totals.ane_joules += sample.ane_mw * dt_s / 1000.0
    pkg = sample.package_mw or (sample.cpu_mw + sample.gpu_mw + sample.ane_mw)
    totals.total_joules += pkg * dt_s / 1000.0
    totals.samples.append(sample)


class PowermetricsCollector:
    """Spawn powermetrics and parse streaming samples."""

    def __init__(self, interval_ms: int = 1000, dry_run: bool = False) -> None:
        self.interval_ms = interval_ms
        self.dry_run = dry_run
        self._proc: subprocess.Popen[str] | None = None
        self._raw_lines: list[str] = []

    def start(self) -> None:
        if self.dry_run:
            return
        cmd = [
            "powermetrics",
            "-i",
            str(self.interval_ms),
            "--samplers",
            "cpu_power,gpu_power,ane_power,thermal",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            sys.exit("ERROR: powermetrics not found. Run on macOS with Xcode CLT installed.")
        except PermissionError:
            sys.exit("ERROR: powermetrics permission denied. Re-run with sudo.")

    def stop(self) -> str:
        if self.dry_run:
            return ""
        if self._proc is None:
            return ""
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        return "\n".join(self._raw_lines)

    def collect_for(
        self,
        duration_s: float,
        *,
        max_temp_c: float | None,
        on_sample: Callable[[PowerSample], None] | None = None,
    ) -> EnergyTotals:
        totals = EnergyTotals()
        t0 = time.monotonic()
        last_t = t0
        buffer: list[str] = []

        if self.dry_run:
            while time.monotonic() - t0 < duration_s:
                elapsed = time.monotonic() - t0
                dt = elapsed - (last_t - t0)
                last_t = time.monotonic()
                sample = PowerSample(
                    elapsed_s=elapsed,
                    cpu_mw=4200.0,
                    gpu_mw=300.0,
                    ane_mw=800.0,
                    package_mw=5300.0,
                    temp_c=42.0 + elapsed * 0.05,
                )
                integrate_sample(totals, sample, dt)
                if on_sample:
                    on_sample(sample)
                if max_temp_c is not None and sample.temp_c and sample.temp_c > max_temp_c:
                    raise ThermalAbort(sample.temp_c, max_temp_c)
                time.sleep(min(1.0, self.interval_ms / 1000.0))
            return totals

        if self._proc is None or self._proc.stdout is None:
            sys.exit("ERROR: powermetrics process not started.")

        while time.monotonic() - t0 < duration_s:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    sys.exit(
                        "ERROR: powermetrics exited early. Try: sudo python harness/m4_energy_harness.py ..."
                    )
                continue
            self._raw_lines.append(line.rstrip("\n"))
            buffer.append(line)
            if line.strip() == "" or "Sampled system activity" in line:
                chunk = "".join(buffer)
                parsed = parse_powermetrics_chunk(chunk)
                if parsed:
                    elapsed = time.monotonic() - t0
                    dt = elapsed - (last_t - t0)
                    last_t = time.monotonic()
                    sample = PowerSample(
                        elapsed_s=elapsed,
                        cpu_mw=parsed.get("cpu_mw", 0.0),
                        gpu_mw=parsed.get("gpu_mw", 0.0),
                        ane_mw=parsed.get("ane_mw", 0.0),
                        package_mw=parsed.get("package_mw", 0.0),
                        temp_c=parsed.get("temp_c"),
                    )
                    integrate_sample(totals, sample, dt)
                    if on_sample:
                        on_sample(sample)
                    if max_temp_c is not None and sample.temp_c and sample.temp_c > max_temp_c:
                        raise ThermalAbort(sample.temp_c, max_temp_c)
                buffer = []

        return totals


class ThermalAbort(Exception):
    def __init__(self, temp_c: float, limit_c: float) -> None:
        self.temp_c = temp_c
        self.limit_c = limit_c
        super().__init__(f"temperature {temp_c:.1f}°C exceeded safe limit {limit_c:.1f}°C")


def cpu_sustained_load(stop: threading.Event) -> None:
    """Placeholder sustained load until MLX/ANE decode workloads are integrated."""
    x = 1.0001
    while not stop.is_set():
        for _ in range(10_000):
            x *= 1.0000001
        if x > 2.0:
            x = 1.0001


def run_with_load(duration_s: float, collector: PowermetricsCollector, max_temp_c: float | None) -> EnergyTotals:
    stop = threading.Event()
    thread = threading.Thread(target=cpu_sustained_load, args=(stop,), daemon=True)
    thread.start()
    try:
        return collector.collect_for(duration_s, max_temp_c=max_temp_c)
    finally:
        stop.set()
        thread.join(timeout=2)


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def base_record(experiment_id: str, mode: str, **extra: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "experiment_id": experiment_id,
        "benchmark_name": mode,
        "powermetrics_log_path": str(LOGS_DIR / f"{experiment_id}.powermetrics.txt"),
    }
    rec.update(extra)
    return rec


def energy_fields(totals: EnergyTotals, *, orchestration_fraction: float = 0.0) -> dict[str, Any]:
    orch = totals.cpu_joules * orchestration_fraction
    return {
        "energy_joules": round(totals.total_joules, 4),
        "energy_ane_joules": round(totals.ane_joules, 4),
        "energy_cpu_orchestration_joules": round(orch, 4),
        "sustained_power_w": round(totals.sustained_power_w, 4),
        "peak_temp_c": totals.peak_temp_c(),
        "temp_steady_state_c": totals.steady_state_temp_c(),
    }


def run_thermal_baseline(args: argparse.Namespace) -> dict[str, Any]:
    experiment_id = args.experiment_id or f"thermal_baseline_{uuid.uuid4().hex[:8]}"
    collector = PowermetricsCollector(dry_run=args.dry_run)
    collector.start()
    try:
        totals = run_with_load(args.duration, collector, args.max_temp_c)
    except ThermalAbort as exc:
        sys.exit(f"THERMAL ABORT: {exc}")
    finally:
        raw = collector.stop()
        raw_path = LOGS_DIR / f"{experiment_id}.powermetrics.txt"
        if raw:
            raw_path.write_text(raw, encoding="utf-8")
        elif args.dry_run:
            raw_path.write_text("# dry-run: no powermetrics on this host\n", encoding="utf-8")

    temps = [s.temp_c for s in totals.samples if s.temp_c is not None]
    record = base_record(
        experiment_id,
        "thermal_baseline",
        task_type="thermal",
        temp_start_c=temps[0] if temps else None,
        time_to_throttle_s=None,
        thermal_headroom_c=(args.max_temp_c - temps[-1]) if args.max_temp_c and temps else None,
        thermal_envelope_valid=True,
        u_task_score=1.0,
        ipj=round(1.0 / totals.total_joules, 6) if totals.total_joules > 0 else None,
        notes="thermal_baseline; workload=cpu_sustained_stub pending MLX integration",
        **energy_fields(totals),
    )
    out = LOGS_DIR / f"{experiment_id}.jsonl"
    write_jsonl(out, record)
    return record


def run_stub_mode(args: argparse.Namespace, mode: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Shared path for modes awaiting MLX workload integration."""
    experiment_id = args.experiment_id or f"{mode}_{uuid.uuid4().hex[:8]}"
    collector = PowermetricsCollector(dry_run=args.dry_run)
    collector.start()
    try:
        totals = run_with_load(args.duration, collector, args.max_temp_c)
    except ThermalAbort as exc:
        sys.exit(f"THERMAL ABORT: {exc}")
    finally:
        raw = collector.stop()
        raw_path = LOGS_DIR / f"{experiment_id}.powermetrics.txt"
        if raw:
            raw_path.write_text(raw, encoding="utf-8")
        elif args.dry_run:
            raw_path.write_text(f"# dry-run stub for {mode}\n", encoding="utf-8")

    record = base_record(
        experiment_id,
        mode,
        task_type=mode,
        thermal_envelope_valid=True,
        u_task_score=1.0,
        notes=f"{mode}; workload stub — integrate MLX/ANE path on physical M4",
        **energy_fields(totals, orchestration_fraction=0.15 if mode in ("orchestration", "ane_utilization") else 0.0),
        **extra,
    )
    out = LOGS_DIR / f"{experiment_id}.jsonl"
    write_jsonl(out, record)
    return record


def run_sram_cliff(args: argparse.Namespace) -> dict[str, Any]:
    contexts = [512, 1024, 2048, 4096, 8192]
    if args.max_context:
        contexts = [c for c in contexts if c <= args.max_context]
    records = []
    for ctx in contexts:
        rec = run_stub_mode(
            args,
            "sram_cliff",
            {
                "context_length": ctx,
                "tokens_per_second_sustained": None,
                "tokens_generated": 0,
            },
        )
        records.append(rec)
    return records[-1]


def run_kv_comparison(args: argparse.Namespace) -> dict[str, Any]:
    return run_stub_mode(
        args,
        "kv_comparison",
        {
            "context_length": args.context,
            "energy_dequant_joules": None,
            "kv_path": "fp16_vs_int4_pending",
        },
    )


MODE_HANDLERS: dict[str, Callable[[argparse.Namespace], Any]] = {
    "thermal_baseline": run_thermal_baseline,
    "sram_cliff": run_sram_cliff,
    "kv_comparison": run_kv_comparison,
    "orchestration": lambda a: run_stub_mode(
        a, "orchestration", {"orchestration_tax_pct": None}
    ),
    "ane_utilization": lambda a: run_stub_mode(
        a,
        "ane_utilization",
        {"ane_compute_fraction_pct": None, "ane_utilization_pct": None, "orchestration_tax_pct": None},
    ),
    "thermal_ipj_curve": lambda a: run_stub_mode(
        a,
        "thermal_ipj_curve",
        {"thermal_headroom_c": None, "ipj_degradation_pct": None, "window_minutes": a.duration / 60},
    ),
    "meta_tax": lambda a: run_stub_mode(
        a,
        "meta_tax",
        {
            "energy_meta_total_joules": None,
            "energy_saved_subsequent_joules": None,
            "net_ipj_delta": None,
        },
    ),
    "memory_spill": lambda a: run_stub_mode(
        a,
        "memory_spill",
        {
            "context_length": a.context,
            "working_set_mb": None,
            "energy_spill_joules_per_token": None,
            "energy_recompute_joules_per_token": None,
        },
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Alalā M4 energy measurement harness (Phase 0).")
    p.add_argument("--mode", choices=MODES, required=True, help="Benchmark or gap-closing experiment mode.")
    p.add_argument("--duration", type=float, default=120.0, help="Load duration in seconds.")
    p.add_argument("--context", type=int, default=2048, help="Context length for decode modes.")
    p.add_argument("--max-context", type=int, default=None, help="Max context for sram_cliff sweep.")
    p.add_argument("--iterations", type=int, default=50, help="Iterations (reserved for decode loops).")
    p.add_argument("--experiment-id", type=str, default=None, help="Override experiment ID.")
    p.add_argument(
        "--max-temp-c",
        type=float,
        default=None,
        help="Safe sustained temperature limit (°C). Abort if exceeded.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate samples off-hardware; for structure validation only.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_dirs()
    check_platform(dry_run=args.dry_run)

    handler = MODE_HANDLERS[args.mode]
    record = handler(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"\nWrote logs to {LOGS_DIR}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
