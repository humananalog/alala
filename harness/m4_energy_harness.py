#!/usr/bin/env python3
"""Alalā Phase 0 energy measurement harness for Mac Mini M4 24 GB."""

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

from workloads import (
    HAS_MLX,
    WorkloadResult,
    cpu_sustained_load,
    mlx_context_scaled_load,
    mlx_matmul_sustained,
    resolve_workload_name,
    run_in_thread,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    "setup_check",
)

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
        if len(self.samples) < 2:
            return 0.0
        dt = self.samples[-1].elapsed_s - self.samples[0].elapsed_s
        return self.total_joules / dt if dt > 0 else 0.0

    def peak_temp_c(self) -> float | None:
        temps = [s.temp_c for s in self.samples if s.temp_c is not None]
        return max(temps) if temps else None

    def steady_state_temp_c(self, tail_fraction: float = 0.2) -> float | None:
        temps = [s.temp_c for s in self.samples if s.temp_c is not None]
        if not temps:
            return None
        n = max(1, int(len(temps) * tail_fraction))
        return sum(temps[-n:]) / n

    def idle_power_w(self, head_fraction: float = 0.5) -> float | None:
        if len(self.samples) < 2:
            return None
        n = max(1, int(len(self.samples) * head_fraction))
        head = self.samples[:n]
        dt = head[-1].elapsed_s - head[0].elapsed_s
        if dt <= 0:
            return None
        joules = sum(
            (s.package_mw or s.cpu_mw + s.gpu_mw + s.ane_mw) * 0.001
            for s in head[1:]
        )
        return joules / dt if dt > 0 else None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs(*subpaths: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULTS_DIR
    for part in subpaths:
        target = target / part
        target.mkdir(parents=True, exist_ok=True)
    return target


def check_platform(*, dry_run: bool) -> None:
    if dry_run:
        return
    if platform.system() != "Darwin":
        sys.exit("ERROR: requires macOS on physical Mac Mini M4 24 GB (use --dry-run off-hardware).")
    if platform.machine().lower() not in ("arm64", "aarch64"):
        sys.exit(f"ERROR: expected Apple Silicon, got {platform.machine()!r}.")
    ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3)
    if ram_gb < 20 or ram_gb > 28:
        sys.exit(f"ERROR: expected ~24 GB unified memory, detected {ram_gb:.1f} GB.")


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
    for pattern in (_RE_CPU_TEMP, _RE_GPU_TEMP):
        m = pattern.search(text)
        if m:
            out["temp_c"] = float(m.group(1))
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
    def __init__(self, interval_ms: int = 1000, dry_run: bool = False) -> None:
        self.interval_ms = interval_ms
        self.dry_run = dry_run
        self._proc: subprocess.Popen[str] | None = None
        self._raw_lines: list[str] = []

    def start(self) -> None:
        if self.dry_run:
            return
        cmd = ["powermetrics", "-i", str(self.interval_ms), "--samplers", "cpu_power,gpu_power,ane_power,thermal"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except FileNotFoundError:
            sys.exit("ERROR: powermetrics not found.")
        except PermissionError:
            sys.exit("ERROR: powermetrics permission denied. Re-run with sudo.")

    def stop(self) -> str:
        if self.dry_run or self._proc is None:
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
        load_start_temp: float = 42.0,
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
                    cpu_mw=1800.0 if elapsed < duration_s * 0.3 else 4200.0,
                    gpu_mw=200.0 if elapsed < duration_s * 0.3 else 1200.0,
                    ane_mw=100.0 if elapsed < duration_s * 0.3 else 900.0,
                    package_mw=2100.0 if elapsed < duration_s * 0.3 else 5300.0,
                    temp_c=load_start_temp + elapsed * 0.02,
                )
                integrate_sample(totals, sample, dt)
                if max_temp_c and sample.temp_c and sample.temp_c > max_temp_c:
                    raise ThermalAbort(sample.temp_c, max_temp_c)
                time.sleep(min(1.0, self.interval_ms / 1000.0))
            return totals

        if self._proc is None or self._proc.stdout is None:
            sys.exit("ERROR: powermetrics not started.")
        while time.monotonic() - t0 < duration_s:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    sys.exit("ERROR: powermetrics exited early. Use sudo.")
                continue
            self._raw_lines.append(line.rstrip("\n"))
            buffer.append(line)
            if line.strip() == "" or "Sampled system activity" in line:
                parsed = parse_powermetrics_chunk("".join(buffer))
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
                    if max_temp_c and sample.temp_c and sample.temp_c > max_temp_c:
                        raise ThermalAbort(sample.temp_c, max_temp_c)
                buffer = []
        return totals


class ThermalAbort(Exception):
    def __init__(self, temp_c: float, limit_c: float) -> None:
        super().__init__(f"temperature {temp_c:.1f}°C exceeded safe limit {limit_c:.1f}°C")
        self.temp_c = temp_c
        self.limit_c = limit_c


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def energy_fields(totals: EnergyTotals, *, orch_frac: float = 0.0) -> dict[str, Any]:
    return {
        "energy_joules": round(totals.total_joules, 4),
        "energy_ane_joules": round(totals.ane_joules, 4),
        "energy_cpu_orchestration_joules": round(totals.cpu_joules * orch_frac, 4),
        "sustained_power_w": round(totals.sustained_power_w, 4),
        "idle_power_w": round(totals.idle_power_w() or 0.0, 4) or None,
        "peak_temp_c": totals.peak_temp_c(),
        "temp_steady_state_c": totals.steady_state_temp_c(),
    }


def run_session(
    args: argparse.Namespace,
    *,
    experiment_id: str,
    phase: str,
    duration_s: float,
    workload_fn: Callable[..., WorkloadResult] | None = None,
    workload_kwargs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    collector = PowermetricsCollector(dry_run=args.dry_run)
    collector.start()
    workload_result = WorkloadResult()
    stop = threading.Event()
    thread: threading.Thread | None = None

    if workload_fn:
        kw = workload_kwargs or {}

        def _run() -> None:
            nonlocal workload_result
            workload_result = workload_fn(stop, **kw)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    try:
        totals = collector.collect_for(duration_s, max_temp_c=args.max_temp_c)
    except ThermalAbort as exc:
        sys.exit(f"THERMAL ABORT: {exc}")
    finally:
        stop.set()
        if thread:
            thread.join(timeout=3)
        raw = collector.stop()
        raw_path = LOGS_DIR / f"{experiment_id}.powermetrics.txt"
        if raw:
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(f"\n# phase={phase}\n{raw}\n")
        elif args.dry_run:
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(f"# dry-run phase={phase}\n")

    temps = [s.temp_c for s in totals.samples if s.temp_c is not None]
    orch_frac = 0.2 if args.mode in ("orchestration", "ane_utilization") else 0.0
    total_j = totals.total_joules or 1e-9
    record: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "experiment_id": experiment_id,
        "benchmark_name": args.mode,
        "phase": phase,
        "duration_s": duration_s,
        "task_type": args.mode,
        "temp_start_c": temps[0] if temps else None,
        "thermal_headroom_c": (args.max_temp_c - temps[-1]) if args.max_temp_c and temps else None,
        "thermal_envelope_valid": True,
        "tokens_generated": workload_result.tokens_generated,
        "forward_passes": workload_result.forward_passes,
        "tokens_per_second_sustained": workload_result.tokens_per_second_sustained,
        "powermetrics_log_path": str(LOGS_DIR / f"{experiment_id}.powermetrics.txt"),
        "workload": workload_result.notes,
        "mlx_available": HAS_MLX,
        "notes": workload_result.notes,
        **energy_fields(totals, orch_frac=orch_frac),
    }
    if args.mode == "ane_utilization" and workload_result.forward_passes:
        ane_frac = totals.ane_joules / total_j * 100
        record["ane_utilization_pct"] = round(ane_frac, 2)
        record["ane_compute_fraction_pct"] = round(ane_frac, 2)
        record["orchestration_tax_pct"] = round(totals.cpu_joules * orch_frac / total_j * 100, 2)
    if extra:
        record.update(extra)
    write_jsonl(LOGS_DIR / f"{experiment_id}.jsonl", record)
    return record


def pick_workload(args: argparse.Namespace):
    name = resolve_workload_name(args.workload)
    if name == "mlx":
        return mlx_matmul_sustained, {}
    return cpu_sustained_load, {}


def run_setup_check(args: argparse.Namespace) -> dict[str, Any]:
    args.duration = max(args.duration, 30.0)
    eid = args.experiment_id or f"setup_{uuid.uuid4().hex[:8]}"
    wl, kw = pick_workload(args)
    rec = run_session(args, experiment_id=eid, phase="setup_check", duration_s=args.duration, workload_fn=wl, workload_kwargs=kw)
    write_jsonl(LOGS_DIR / "setup_log.jsonl", rec)
    return rec


def run_thermal_baseline(args: argparse.Namespace) -> dict[str, Any]:
    eid = args.experiment_id or f"thermal_baseline_{uuid.uuid4().hex[:8]}"
    out_dir = ensure_dirs("thermal_baseline")
    records: list[dict[str, Any]] = []

    if args.idle_duration > 0:
        idle = run_session(args, experiment_id=eid, phase="idle", duration_s=args.idle_duration)
        records.append(idle)

    wl, kw = pick_workload(args)
    load = run_session(
        args,
        experiment_id=eid,
        phase="sustained_load",
        duration_s=args.duration,
        workload_fn=wl,
        workload_kwargs=kw,
        extra={"u_task_score": 1.0, "time_to_throttle_s": None},
    )
    load["ipj"] = round(1.0 / max(load.get("energy_joules") or 1.0, 1e-9), 6)
    records.append(load)

    summary = {
        "timestamp": utc_now_iso(),
        "experiment_id": eid,
        "benchmark_name": "thermal_baseline",
        "phases": records,
        "safe_sustained_power_w": load.get("sustained_power_w"),
        "temp_steady_state_c": load.get("temp_steady_state_c"),
        "idle_power_w": records[0].get("idle_power_w") if args.idle_duration > 0 else None,
    }
    (out_dir / f"{eid}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def detect_sram_cliff(steps: list[dict[str, Any]]) -> int | None:
    rates = [(s["context_length"], s.get("tokens_per_second_sustained") or 0.0) for s in steps]
    for i in range(1, len(rates)):
        prev, curr = rates[i - 1][1], rates[i][1]
        if prev > 0 and curr <= prev * 0.7:
            return rates[i][0]
    return None


def run_sram_cliff(args: argparse.Namespace) -> dict[str, Any]:
    contexts = [512, 1024, 2048, 4096, 8192]
    if args.max_context:
        contexts = [c for c in contexts if c <= args.max_context]
    eid = args.experiment_id or f"sram_cliff_{uuid.uuid4().hex[:8]}"
    out_dir = ensure_dirs("sram_cliff")
    steps: list[dict[str, Any]] = []
    per_ctx = max(15.0, args.duration / max(len(contexts), 1))

    for ctx in contexts:
        if args.dry_run:
            tps = round(40.0 * (512 / ctx) ** 0.55, 2)
            rec = run_session(
                args,
                experiment_id=eid,
                phase=f"context_{ctx}",
                duration_s=per_ctx,
                extra={"context_length": ctx, "tokens_per_second_sustained": tps},
            )
        else:
            rec = run_session(
                args,
                experiment_id=eid,
                phase=f"context_{ctx}",
                duration_s=per_ctx,
                workload_fn=mlx_context_scaled_load,
                workload_kwargs={"context_length": ctx},
                extra={"context_length": ctx},
            )
        steps.append(rec)

    cliff = detect_sram_cliff(steps)
    summary = {
        "timestamp": utc_now_iso(),
        "experiment_id": eid,
        "benchmark_name": "sram_cliff",
        "steps": steps,
        "l_cliff": cliff,
        "sram_cliff_context_length": cliff,
    }
    (out_dir / f"{eid}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_thermal_ipj_curve(args: argparse.Namespace) -> dict[str, Any]:
    eid = args.experiment_id or f"thermal_ipj_curve_{uuid.uuid4().hex[:8]}"
    window_s = args.window_s
    windows: list[dict[str, Any]] = []
    elapsed = 0.0
    first_ipj: float | None = None
    wl, kw = pick_workload(args)

    while elapsed < args.duration:
        chunk = min(window_s, args.duration - elapsed)
        rec = run_session(
            args,
            experiment_id=eid,
            phase=f"window_{int(elapsed)}s",
            duration_s=chunk,
            workload_fn=wl,
            workload_kwargs=kw,
            extra={"window_start_s": elapsed, "thermal_headroom_c": None},
        )
        ipj = 1.0 / max(rec.get("energy_joules") or 1.0, 1e-9)
        rec["ipj"] = round(ipj, 6)
        if first_ipj is None:
            first_ipj = ipj
        if first_ipj and first_ipj > 0:
            rec["ipj_degradation_pct"] = round((1 - ipj / first_ipj) * 100, 2)
        windows.append(rec)
        elapsed += chunk

    summary = {
        "timestamp": utc_now_iso(),
        "experiment_id": eid,
        "benchmark_name": "thermal_ipj_curve",
        "windows": windows,
        "duration_s": args.duration,
        "ipj_degradation_pct_final": windows[-1].get("ipj_degradation_pct") if windows else None,
    }
    ensure_dirs("thermal_ipj_curve")
    (RESULTS_DIR / "thermal_ipj_curve" / f"{eid}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_kv_comparison(args: argparse.Namespace) -> dict[str, Any]:
    eid = args.experiment_id or f"kv_comparison_{uuid.uuid4().hex[:8]}"
    wl, kw = pick_workload(args)
    fp16 = run_session(args, experiment_id=eid, phase="fp16", duration_s=args.duration, workload_fn=wl, workload_kwargs=kw, extra={"kv_path": "fp16"})
    int4 = run_session(args, experiment_id=eid, phase="int4", duration_s=args.duration, workload_fn=wl, workload_kwargs=kw, extra={"kv_path": "int4_fused"})
    dequant = max(0.0, (int4.get("energy_joules") or 0) - (fp16.get("energy_joules") or 0))
    summary = {
        "experiment_id": eid,
        "benchmark_name": "kv_comparison",
        "context_length": args.context,
        "fp16": fp16,
        "int4": int4,
        "energy_dequant_joules": round(dequant, 4),
        "notes": "kv paths use same MLX stub until fused int4 KV kernel integrated",
    }
    ensure_dirs("kv_comparison")
    (RESULTS_DIR / "kv_comparison" / f"{eid}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stub(args: argparse.Namespace, mode: str, extra: dict[str, Any]) -> dict[str, Any]:
    wl, kw = pick_workload(args)
    eid = args.experiment_id or f"{mode}_{uuid.uuid4().hex[:8]}"
    return run_session(args, experiment_id=eid, phase=mode, duration_s=args.duration, workload_fn=wl, workload_kwargs=kw, extra=extra)


MODE_HANDLERS = {
    "setup_check": run_setup_check,
    "thermal_baseline": run_thermal_baseline,
    "sram_cliff": run_sram_cliff,
    "kv_comparison": run_kv_comparison,
    "orchestration": lambda a: run_stub(a, "orchestration", {"orchestration_tax_pct": None}),
    "ane_utilization": lambda a: run_stub(a, "ane_utilization", {}),
    "thermal_ipj_curve": run_thermal_ipj_curve,
    "meta_tax": lambda a: run_stub(
        a,
        "meta_tax",
        {"energy_meta_total_joules": None, "energy_saved_subsequent_joules": None, "net_ipj_delta": None},
    ),
    "memory_spill": lambda a: run_stub(
        a,
        "memory_spill",
        {"context_length": a.context, "working_set_mb": None, "energy_spill_joules_per_token": None},
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Alalā M4 energy measurement harness (Phase 0).")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--duration", type=float, default=120.0, help="Sustained load duration (seconds).")
    p.add_argument("--idle-duration", type=float, default=0.0, help="Idle monitoring before load (thermal_baseline).")
    p.add_argument("--window-s", type=float, default=300.0, help="Window size for thermal_ipj_curve (seconds).")
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--max-context", type=int, default=None)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--experiment-id", type=str, default=None)
    p.add_argument("--max-temp-c", type=float, default=None)
    p.add_argument("--workload", choices=("auto", "cpu", "mlx"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_dirs()
    check_platform(dry_run=args.dry_run)
    record = MODE_HANDLERS[args.mode](args)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"\nWrote logs to {LOGS_DIR}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
