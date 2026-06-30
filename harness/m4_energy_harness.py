#!/usr/bin/env python3
"""Phase 0 M4 energy measurement harness for Alalā."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from decode_client import DecodeRunner
from env import ENV_FILE, LOGS_DIR, REPO_ROOT, RESULTS_DIR, load_repo_env, sudo_password
from errors import HarnessError
from metrics import summarize_power_samples
from models import resolve_model_id
from powermetrics_log import (
    PowerSample,
    parse_powermetrics_file,
    sample_energy_joules,
    sample_interval_seconds,
)
from sram_cliff import (
    ContextStepResult,
    build_context_record,
    build_summary_record,
    context_lengths,
    detect_sram_cliff,
)
from workloads import SustainedLoad, cpu_spin_load, mlx_matmul_load

MODES = ("thermal_baseline", "sram_cliff", "kv_comparison", "orchestration")
EXPECTED_MEMORY_BYTES = 24 * 1024**3
MEMORY_TOLERANCE_BYTES = 512 * 1024**2
THERMAL_PRESSURE_ABORT = frozenset({"Heavy", "Critical"})


@dataclass(frozen=True)
class HardwareInfo:
    brand: str
    memory_bytes: int
    is_apple_silicon: bool
    is_m4: bool
    is_24gb: bool


@dataclass
class ThermalSummary:
    idle_power_w: float | None
    sustained_power_w: float | None
    temp_start_c: float | None
    peak_temp_c: float | None
    temp_steady_state_c: float | None
    time_to_throttle_s: float | None
    energy_joules: float
    energy_ane_joules: float
    energy_cpu_joules: float
    energy_gpu_joules: float
    ane_utilization_pct: float | None
    sample_count: int
    aborted: bool
    abort_reason: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def can_run_powermetrics() -> bool:
    return os.geteuid() == 0 or bool(sudo_password())


def detect_hardware() -> HardwareInfo:
    brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    is_apple_silicon = brand.startswith("Apple ")
    is_m4 = "M4" in brand
    is_24gb = abs(memory_bytes - EXPECTED_MEMORY_BYTES) <= MEMORY_TOLERANCE_BYTES
    return HardwareInfo(
        brand=brand,
        memory_bytes=memory_bytes,
        is_apple_silicon=is_apple_silicon,
        is_m4=is_m4,
        is_24gb=is_24gb,
    )


def require_target_hardware(info: HardwareInfo) -> None:
    if sys.platform != "darwin":
        raise HarnessError("Alalā harness requires macOS on physical Apple Silicon.")
    if not info.is_apple_silicon:
        raise HarnessError(f"Unsupported CPU: {info.brand!r}. Target is Apple Silicon M4.")
    if not info.is_m4:
        raise HarnessError(
            f"Unsupported chip: {info.brand!r}. Phase 0 measurements require Mac Mini M4 24 GB."
        )
    if not info.is_24gb:
        gib = info.memory_bytes / (1024**3)
        raise HarnessError(
            f"Unsupported memory: {gib:.1f} GiB. Phase 0 requires 24 GiB unified memory."
        )


def ensure_powermetrics_available() -> None:
    if shutil.which("powermetrics") is None:
        raise HarnessError("powermetrics not found. Install Xcode CLT or run on macOS.")
    if not can_run_powermetrics():
        raise HarnessError(
            "powermetrics requires root. Either:\n"
            "  sudo python harness/m4_energy_harness.py --mode thermal_baseline --duration 120\n"
            "or set SUDO_PASSWORD in the repo .env file (gitignored)."
        )


class PowerMetricsSession:
    def __init__(self, output_path: Path, interval_ms: int = 1000) -> None:
        self.output_path = output_path
        self.interval_ms = interval_ms
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()

        command = [
            "powermetrics",
            "--samplers",
            "cpu_power,gpu_power,ane_power,thermal",
            "-f",
            "plist",
            "-i",
            str(self.interval_ms),
            "-n",
            "-1",
            "-o",
            str(self.output_path),
        ]
        stdin_payload: bytes | None = None
        if os.geteuid() != 0:
            password = sudo_password()
            if not password:
                raise HarnessError("SUDO_PASSWORD is required in .env when not running as root.")
            command = ["sudo", "-S", *command]
            stdin_payload = f"{password}\n".encode("utf-8")

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if stdin_payload else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if stdin_payload is not None and self._process.stdin is not None:
            self._process.stdin.write(stdin_payload)
            self._process.stdin.close()
        time.sleep(1.5)
        if self._process.poll() is not None:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace").strip()
            raise HarnessError(f"powermetrics failed to start: {stderr or 'unknown error'}")

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2.0)
        self._process = None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _temperatures(samples: list[PowerSample]) -> list[float]:
    return [s.temp_c for s in samples if s.temp_c is not None]


def read_macmon_cpu_temp() -> float | None:
    if shutil.which("macmon") is None:
        return None
    try:
        result = subprocess.run(
            ["macmon", "pipe", "-s", "1", "-i", "500"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        payload = json.loads(lines[0])
        return float(payload["temp"]["cpu_temp_avg"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, subprocess.TimeoutExpired):
        return None


def _resolve_temperatures(samples: list[PowerSample], external_temps: list[float]) -> list[float]:
    powermetrics_temps = _temperatures(samples)
    if powermetrics_temps:
        return powermetrics_temps
    return external_temps


def _detect_steady_state_window(samples: list[PowerSample], window: int = 60) -> list[PowerSample]:
    if len(samples) < window:
        return samples

    temps = _temperatures(samples)
    if len(temps) >= window:
        for start in range(0, len(samples) - window + 1):
            window_samples = samples[start : start + window]
            window_temps = [s.temp_c for s in window_samples if s.temp_c is not None]
            if len(window_temps) < max(10, window // 2):
                continue
            if max(window_temps) - min(window_temps) <= 2.0:
                return window_samples

    return samples[-window:]


def _detect_time_to_throttle(load_samples: list[PowerSample]) -> float | None:
    if len(load_samples) < 5:
        return None

    pressures = [s.thermal_pressure for s in load_samples]
    for idx, pressure in enumerate(pressures):
        if pressure in THERMAL_PRESSURE_ABORT:
            return float(idx * sample_interval_seconds(load_samples[idx]))

    powers = [s.package_mw for s in load_samples]
    peak = max(powers)
    if peak <= 0:
        return None

    for idx in range(5, len(powers)):
        trailing = powers[max(0, idx - 5) : idx + 1]
        if _mean(trailing) is not None and _mean(trailing) < peak * 0.9:
            return float(idx * sample_interval_seconds(load_samples[idx]))
    return None


def summarize_thermal_baseline(
    samples: list[PowerSample],
    idle_seconds: int,
    interval_seconds: float,
    aborted: bool,
    abort_reason: str | None,
    external_temps: list[float] | None = None,
) -> ThermalSummary:
    if not samples:
        raise HarnessError("No powermetrics samples captured. Check sudo access and sampler support.")

    idle_count = min(len(samples), max(1, int(idle_seconds / interval_seconds)))
    idle_samples = samples[:idle_count]
    load_samples = samples[idle_count:] or samples

    idle_power_w = _mean([s.package_mw / 1000.0 for s in idle_samples])
    steady_samples = _detect_steady_state_window(load_samples)
    sustained_power_w = _mean([s.package_mw / 1000.0 for s in steady_samples])

    all_temps = _resolve_temperatures(samples, external_temps or [])
    idle_temp_count = min(len(all_temps), max(1, idle_seconds)) if external_temps else idle_count
    load_temp_slice = all_temps[idle_temp_count:] if external_temps and len(all_temps) > idle_temp_count else all_temps[idle_count:]
    load_temps = load_temp_slice[-60:] if load_temp_slice else []
    temp_start_c = all_temps[0] if all_temps else None
    peak_temp_c = max(all_temps) if all_temps else None
    temp_steady_state_c = _mean(load_temps) if load_temps else peak_temp_c

    energy_joules = sum(sample_energy_joules(s) for s in samples)
    energy_cpu_joules = sum(s.cpu_mj for s in samples) / 1000.0
    energy_gpu_joules = sum(s.gpu_mj for s in samples) / 1000.0
    energy_ane_joules = sum(s.ane_mj for s in samples) / 1000.0

    ane_utilization_pct = None
    ane_total = energy_ane_joules
    if ane_total > 0:
        denom = energy_cpu_joules + energy_gpu_joules + ane_total
        if denom > 0:
            ane_utilization_pct = (ane_total / denom) * 100.0

    return ThermalSummary(
        idle_power_w=idle_power_w,
        sustained_power_w=sustained_power_w,
        temp_start_c=temp_start_c,
        peak_temp_c=peak_temp_c,
        temp_steady_state_c=temp_steady_state_c,
        time_to_throttle_s=_detect_time_to_throttle(load_samples),
        energy_joules=energy_joules,
        energy_ane_joules=energy_ane_joules,
        energy_cpu_joules=energy_cpu_joules,
        energy_gpu_joules=energy_gpu_joules,
        ane_utilization_pct=ane_utilization_pct,
        sample_count=len(samples),
        aborted=aborted,
        abort_reason=abort_reason,
    )


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_thermal_record(
    experiment_id: str,
    summary: ThermalSummary,
    powermetrics_log_path: Path,
    duration_s: int,
    idle_seconds: int,
    load_kind: str,
    hardware: HardwareInfo,
    temp_source: str = "powermetrics",
) -> dict[str, Any]:
    ipj = None
    if summary.energy_joules > 0 and not summary.aborted:
        ipj = 1.0 / summary.energy_joules

    return {
        "timestamp": utc_now_iso(),
        "experiment_id": experiment_id,
        "task_type": "thermal_baseline",
        "model": None,
        "context_length": None,
        "duration_s": duration_s,
        "idle_seconds": idle_seconds,
        "load_kind": load_kind,
        "hardware": hardware.brand,
        "memory_gib": round(hardware.memory_bytes / (1024**3), 2),
        "energy_joules": round(summary.energy_joules, 4),
        "energy_ane_joules": round(summary.energy_ane_joules, 4),
        "energy_cpu_orchestration_joules": round(summary.energy_cpu_joules, 4),
        "energy_gpu_joules": round(summary.energy_gpu_joules, 4),
        "energy_dequant_joules": None,
        "tokens_generated": None,
        "tokens_per_second": None,
        "tokens_per_second_sustained": None,
        "ane_utilization_pct": (
            round(summary.ane_utilization_pct, 2) if summary.ane_utilization_pct is not None else None
        ),
        "idle_power_w": round(summary.idle_power_w, 3) if summary.idle_power_w is not None else None,
        "temp_start_c": round(summary.temp_start_c, 2) if summary.temp_start_c is not None else None,
        "peak_temp_c": round(summary.peak_temp_c, 2) if summary.peak_temp_c is not None else None,
        "temp_steady_state_c": (
            round(summary.temp_steady_state_c, 2) if summary.temp_steady_state_c is not None else None
        ),
        "time_to_throttle_s": summary.time_to_throttle_s,
        "sustained_power_w": (
            round(summary.sustained_power_w, 3) if summary.sustained_power_w is not None else None
        ),
        "u_task_score": 0.0 if summary.aborted else 1.0,
        "ipj": round(ipj, 6) if ipj is not None else None,
        "hca_impact": None,
        "sample_count": summary.sample_count,
        "aborted": summary.aborted,
        "abort_reason": summary.abort_reason,
        "notes": (
            "Phase 0 thermal baseline. Interim sustained load uses MLX GPU matmul until "
            "ANE-first decode workload is integrated."
            if load_kind == "mlx_matmul"
            else f"Phase 0 thermal baseline. Temperature source: {temp_source}."
        ),
        "powermetrics_log_path": str(powermetrics_log_path.relative_to(REPO_ROOT)),
    }


def select_load(load_kind: str) -> SustainedLoad:
    if load_kind == "cpu_spin":
        return cpu_spin_load()
    if load_kind == "mlx_matmul":
        try:
            return mlx_matmul_load()
        except ImportError as exc:
            raise HarnessError(
                "MLX is required for --load mlx_matmul. Install mlx or use --load cpu_spin."
            ) from exc
    raise HarnessError(f"Unknown load kind: {load_kind!r}")


def run_thermal_baseline(args: argparse.Namespace, hardware: HardwareInfo) -> int:
    ensure_powermetrics_available()

    if args.idle_seconds >= args.duration:
        raise HarnessError("--idle-seconds must be less than --duration.")

    experiment_id = f"thermal_baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    powermetrics_path = LOGS_DIR / f"{experiment_id}.powermetrics.txt"
    jsonl_path = LOGS_DIR / f"{experiment_id}.jsonl"
    result_dir = RESULTS_DIR / "thermal_baseline" / experiment_id

    session = PowerMetricsSession(powermetrics_path, interval_ms=args.interval_ms)
    load = select_load(args.load)
    aborted = False
    abort_reason: str | None = None
    macmon_temps: list[float] = []

    print(f"[alala] experiment_id={experiment_id}")
    print(f"[alala] logging powermetrics to {powermetrics_path}")
    print(
        f"[alala] thermal_baseline: idle={args.idle_seconds}s load={args.duration - args.idle_seconds}s "
        f"load_kind={args.load}"
    )

    session.start()
    load_started = False
    start_time = time.monotonic()

    try:
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= args.duration:
                break

            if not load_started and elapsed >= args.idle_seconds:
                print("[alala] starting sustained load")
                load.start()
                load_started = True

            macmon_temp = read_macmon_cpu_temp()
            if macmon_temp is not None:
                macmon_temps.append(macmon_temp)

            if args.temp_threshold is not None and elapsed >= args.idle_seconds:
                samples = parse_powermetrics_file(powermetrics_path)
                temps = _resolve_temperatures(samples, macmon_temps)
                if temps and temps[-1] >= args.temp_threshold:
                    aborted = True
                    abort_reason = f"package temperature {temps[-1]:.1f}C exceeded threshold {args.temp_threshold:.1f}C"
                    print(f"[alala] ABORT: {abort_reason}")
                    break

                if samples and samples[-1].thermal_pressure in THERMAL_PRESSURE_ABORT:
                    aborted = True
                    abort_reason = f"thermal pressure {samples[-1].thermal_pressure}"
                    print(f"[alala] ABORT: {abort_reason}")
                    break

            time.sleep(1.0)
    finally:
        load.stop()
        session.stop()

    samples = parse_powermetrics_file(powermetrics_path)
    summary = summarize_thermal_baseline(
        samples=samples,
        idle_seconds=args.idle_seconds,
        interval_seconds=args.interval_ms / 1000.0,
        aborted=aborted,
        abort_reason=abort_reason,
        external_temps=macmon_temps,
    )
    temp_source = (
        "powermetrics"
        if _temperatures(samples)
        else "macmon (powermetrics smc sampler unavailable on this host)"
    )
    record = build_thermal_record(
        experiment_id=experiment_id,
        summary=summary,
        powermetrics_log_path=powermetrics_path,
        duration_s=args.duration,
        idle_seconds=args.idle_seconds,
        load_kind=args.load,
        hardware=hardware,
        temp_source=temp_source,
    )

    write_jsonl(jsonl_path, record)
    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(powermetrics_path, result_dir / powermetrics_path.name)
    shutil.copy2(jsonl_path, result_dir / jsonl_path.name)
    (result_dir / "summary.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[alala] wrote {jsonl_path}")
    print(f"[alala] copied artifacts to {result_dir}")
    print(
        "[alala] sustained_power_w="
        f"{record['sustained_power_w']} temp_steady_state_c={record['temp_steady_state_c']}"
    )
    return 1 if aborted else 0


def run_sram_cliff(args: argparse.Namespace, hardware: HardwareInfo) -> int:
    ensure_powermetrics_available()

    if args.steady_window >= args.step_duration:
        raise HarnessError("--steady-window must be less than --step-duration.")

    model_id = resolve_model_id(args.model)
    lengths = context_lengths(args.max_context)
    if not lengths:
        raise HarnessError(f"No context lengths to test with --max-context {args.max_context}.")

    temp_threshold = args.temp_threshold if args.temp_threshold is not None else 85.0
    run_id = f"sram_cliff_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    jsonl_path = LOGS_DIR / f"{run_id}.jsonl"
    result_dir = RESULTS_DIR / "sram_cliff" / run_id

    print(f"[alala] run_id={run_id}")
    print(f"[alala] model={model_id} contexts={lengths}")
    print(
        f"[alala] step_duration={args.step_duration}s steady_window={args.steady_window}s "
        f"temp_threshold={temp_threshold}C"
    )

    runner = DecodeRunner(model_id)
    step_results: list[ContextStepResult] = []
    aborted = False
    abort_reason: str | None = None

    for context_length in lengths:
        step_id = f"{run_id}_ctx{context_length}"
        powermetrics_path = LOGS_DIR / f"{step_id}.powermetrics.txt"
        session = PowerMetricsSession(powermetrics_path, interval_ms=args.interval_ms)
        macmon_temps: list[float] = []

        print(f"[alala] context_length={context_length} powermetrics={powermetrics_path.name}")

        macmon_temp = read_macmon_cpu_temp()
        if macmon_temp is not None:
            macmon_temps.append(macmon_temp)
            if macmon_temp >= temp_threshold:
                aborted = True
                abort_reason = (
                    f"temperature {macmon_temp:.1f}C exceeded threshold {temp_threshold:.1f}C"
                )
                print(f"[alala] ABORT: {abort_reason}")
                break

        session.start()
        try:
            decode_result = runner.run_context_step(
                context_length=context_length,
                duration_s=args.step_duration,
                steady_window_s=args.steady_window,
                decode_tokens=args.decode_tokens,
            )
        finally:
            session.stop()

        post_temp = read_macmon_cpu_temp()
        if post_temp is not None:
            macmon_temps.append(post_temp)

        if args.cooldown_seconds > 0 and context_length != lengths[-1]:
            print(f"[alala] cooldown {args.cooldown_seconds}s before next context")
            time.sleep(args.cooldown_seconds)

        if aborted:
            print(f"[alala] ABORT: {abort_reason}")
            break

        samples = parse_powermetrics_file(powermetrics_path)
        power = summarize_power_samples(samples)
        all_temps = _resolve_temperatures(samples, macmon_temps)
        temp_start_c = all_temps[0] if all_temps else None
        peak_temp_c = max(all_temps) if all_temps else None
        temp_steady_state_c = _mean(all_temps[-args.steady_window :]) if all_temps else None

        step = ContextStepResult(
            context_length=context_length,
            tokens_per_second_sustained=decode_result.tokens_per_second_sustained,
            tokens_per_second=decode_result.tokens_per_second,
            tokens_generated=decode_result.tokens_generated,
            ane_utilization_pct=(
                round(power.ane_utilization_pct, 2) if power.ane_utilization_pct is not None else None
            ),
            sustained_power_w=power.sustained_power_w,
            peak_memory_gb=decode_result.peak_memory_gb,
            energy_joules=power.energy_joules,
            experiment_id=step_id,
            powermetrics_log_path=str(powermetrics_path.relative_to(REPO_ROOT)),
        )
        step_results.append(step)

        record = build_context_record(
            run_id=run_id,
            step=step,
            hardware_brand=hardware.brand,
            model=args.model,
            temp_start_c=round(temp_start_c, 2) if temp_start_c is not None else None,
            temp_steady_state_c=round(temp_steady_state_c, 2) if temp_steady_state_c is not None else None,
            peak_temp_c=round(peak_temp_c, 2) if peak_temp_c is not None else None,
            timestamp=utc_now_iso(),
        )
        record["energy_ane_joules"] = round(power.energy_ane_joules, 4)
        record["energy_cpu_orchestration_joules"] = round(power.energy_cpu_joules, 4)
        record["energy_gpu_joules"] = round(power.energy_gpu_joules, 4)
        write_jsonl(jsonl_path, record)

        print(
            f"[alala] ctx={context_length} "
            f"tps_sustained={step.tokens_per_second_sustained:.2f} "
            f"ane_util={step.ane_utilization_pct} peak_mem_gb={step.peak_memory_gb:.2f}"
        )

    l_cliff = detect_sram_cliff(step_results)
    summary = build_summary_record(
        run_id=run_id,
        steps=step_results,
        l_cliff=l_cliff,
        hardware_brand=hardware.brand,
        model=args.model,
        timestamp=utc_now_iso(),
    )
    write_jsonl(jsonl_path, summary)

    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jsonl_path, result_dir / jsonl_path.name)
    for step in step_results:
        src = REPO_ROOT / step.powermetrics_log_path
        if src.exists():
            shutil.copy2(src, result_dir / src.name)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[alala] wrote {jsonl_path}")
    print(f"[alala] copied artifacts to {result_dir}")
    print(f"[alala] L_cliff={l_cliff}")
    return 1 if aborted else 0


def _run_decode_benchmark(
    *,
    run_id: str,
    label: str,
    model_id: str,
    context_length: int,
    duration_s: int,
    steady_window_s: int,
    decode_tokens: int,
    hardware: HardwareInfo,
    args: argparse.Namespace,
    kv_bits: int | None = None,
    orchestration_delay_ms: int = 0,
) -> tuple[dict[str, Any], Path]:
    step_id = f"{run_id}_{label}"
    powermetrics_path = LOGS_DIR / f"{step_id}.powermetrics.txt"
    session = PowerMetricsSession(powermetrics_path, interval_ms=args.interval_ms)
    runner = DecodeRunner(model_id)

    print(f"[alala] {label}: context={context_length} kv_bits={kv_bits} orch_delay_ms={orchestration_delay_ms}")
    session.start()
    try:
        decode_result = runner.run_context_step(
            context_length=context_length,
            duration_s=duration_s,
            steady_window_s=steady_window_s,
            decode_tokens=decode_tokens,
            kv_bits=kv_bits,
            orchestration_delay_ms=orchestration_delay_ms,
        )
    finally:
        session.stop()

    samples = parse_powermetrics_file(powermetrics_path)
    power = summarize_power_samples(samples)
    macmon_temps: list[float] = []
    temp = read_macmon_cpu_temp()
    if temp is not None:
        macmon_temps.append(temp)
    all_temps = _resolve_temperatures(samples, macmon_temps)

    u_task = decode_result.tokens_generated
    ipj = u_task / power.energy_joules if power.energy_joules > 0 else None
    cpu_ratio = power.energy_cpu_joules / power.energy_joules if power.energy_joules > 0 else None

    record: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "experiment_id": step_id,
        "run_id": run_id,
        "task_type": label,
        "model": args.model,
        "context_length": context_length,
        "kv_bits": kv_bits,
        "orchestration_delay_ms": orchestration_delay_ms,
        "energy_joules": round(power.energy_joules, 4),
        "energy_ane_joules": round(power.energy_ane_joules, 4),
        "energy_cpu_orchestration_joules": round(power.energy_cpu_joules, 4),
        "energy_gpu_joules": round(power.energy_gpu_joules, 4),
        "energy_dequant_joules": None,
        "cpu_orchestration_ratio": round(cpu_ratio, 4) if cpu_ratio is not None else None,
        "tokens_generated": decode_result.tokens_generated,
        "tokens_per_second": round(decode_result.tokens_per_second, 3),
        "tokens_per_second_sustained": round(decode_result.tokens_per_second_sustained, 3),
        "ane_utilization_pct": (
            round(power.ane_utilization_pct, 2) if power.ane_utilization_pct is not None else None
        ),
        "peak_memory_gb": round(decode_result.peak_memory_gb, 3),
        "temp_start_c": round(all_temps[0], 2) if all_temps else None,
        "peak_temp_c": round(max(all_temps), 2) if all_temps else None,
        "temp_steady_state_c": round(_mean(all_temps[-steady_window_s:]), 2) if all_temps else None,
        "sustained_power_w": (
            round(power.sustained_power_w, 3) if power.sustained_power_w is not None else None
        ),
        "u_task_score": 1.0,
        "ipj": round(ipj, 6) if ipj is not None else None,
        "hardware": hardware.brand,
        "powermetrics_log_path": str(powermetrics_path.relative_to(REPO_ROOT)),
    }
    return record, powermetrics_path


def run_kv_comparison(args: argparse.Namespace, hardware: HardwareInfo) -> int:
    ensure_powermetrics_available()
    model_id = resolve_model_id(args.model)
    duration_s = args.step_duration
    steady_window_s = min(args.steady_window, duration_s - 1)
    context_length = args.context
    run_id = f"kv_comparison_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    jsonl_path = LOGS_DIR / f"{run_id}.jsonl"
    result_dir = RESULTS_DIR / "kv_comparison" / run_id

    print(f"[alala] run_id={run_id} context={context_length} model={model_id}")

    fp16_record, fp16_pm = _run_decode_benchmark(
        run_id=run_id,
        label="kv_fp16",
        model_id=model_id,
        context_length=context_length,
        duration_s=duration_s,
        steady_window_s=steady_window_s,
        decode_tokens=args.decode_tokens,
        hardware=hardware,
        args=args,
        kv_bits=None,
    )
    write_jsonl(jsonl_path, fp16_record)

    if args.cooldown_seconds > 0:
        print(f"[alala] cooldown {args.cooldown_seconds}s before int4 path")
        time.sleep(args.cooldown_seconds)

    int4_record, int4_pm = _run_decode_benchmark(
        run_id=run_id,
        label="kv_int4",
        model_id=model_id,
        context_length=context_length,
        duration_s=duration_s,
        steady_window_s=steady_window_s,
        decode_tokens=args.decode_tokens,
        hardware=hardware,
        args=args,
        kv_bits=4,
    )
    int4_record["energy_dequant_joules"] = round(
        max(0.0, int4_record["energy_joules"] - fp16_record["energy_joules"]), 4
    )
    write_jsonl(jsonl_path, int4_record)

    ipj_delta = None
    if fp16_record["ipj"] is not None and int4_record["ipj"] is not None:
        ipj_delta = round(int4_record["ipj"] - fp16_record["ipj"], 6)

    summary = {
        "timestamp": utc_now_iso(),
        "experiment_id": f"{run_id}_summary",
        "run_id": run_id,
        "task_type": "kv_comparison_summary",
        "model": args.model,
        "context_length": context_length,
        "fp16_ipj": fp16_record["ipj"],
        "int4_ipj": int4_record["ipj"],
        "ipj_delta": ipj_delta,
        "energy_dequant_joules": int4_record["energy_dequant_joules"],
        "fp16_tokens_per_second_sustained": fp16_record["tokens_per_second_sustained"],
        "int4_tokens_per_second_sustained": int4_record["tokens_per_second_sustained"],
        "hardware": hardware.brand,
        "notes": "FP16 vs fused int4 KV at context below L_cliff.",
        "powermetrics_log_path": None,
    }
    write_jsonl(jsonl_path, summary)

    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jsonl_path, result_dir / jsonl_path.name)
    for path in (fp16_pm, int4_pm):
        shutil.copy2(path, result_dir / path.name)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[alala] ipj_delta={ipj_delta} energy_dequant_joules={int4_record['energy_dequant_joules']}")
    print(f"[alala] wrote {jsonl_path}")
    return 0


def run_orchestration(args: argparse.Namespace, hardware: HardwareInfo) -> int:
    ensure_powermetrics_available()
    model_id = resolve_model_id(args.model)
    duration_s = args.step_duration
    steady_window_s = min(args.steady_window, duration_s - 1)
    context_length = min(args.context, 512)
    delay_ms = max(1, int(1000 / max(args.iterations, 1)))
    run_id = f"orchestration_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    jsonl_path = LOGS_DIR / f"{run_id}.jsonl"
    result_dir = RESULTS_DIR / "orchestration" / run_id

    print(f"[alala] run_id={run_id} context={context_length} orchestration_delay_ms={delay_ms}")

    tight_record, tight_pm = _run_decode_benchmark(
        run_id=run_id,
        label="orch_tight",
        model_id=model_id,
        context_length=context_length,
        duration_s=duration_s,
        steady_window_s=steady_window_s,
        decode_tokens=args.decode_tokens,
        hardware=hardware,
        args=args,
        orchestration_delay_ms=0,
    )
    write_jsonl(jsonl_path, tight_record)

    if args.cooldown_seconds > 0:
        print(f"[alala] cooldown {args.cooldown_seconds}s before orchestrated path")
        time.sleep(args.cooldown_seconds)

    orch_record, orch_pm = _run_decode_benchmark(
        run_id=run_id,
        label="orch_delayed",
        model_id=model_id,
        context_length=context_length,
        duration_s=duration_s,
        steady_window_s=steady_window_s,
        decode_tokens=args.decode_tokens,
        hardware=hardware,
        args=args,
        orchestration_delay_ms=delay_ms,
    )
    write_jsonl(jsonl_path, orch_record)

    cpu_delta = orch_record["energy_cpu_orchestration_joules"] - tight_record["energy_cpu_orchestration_joules"]
    summary = {
        "timestamp": utc_now_iso(),
        "experiment_id": f"{run_id}_summary",
        "run_id": run_id,
        "task_type": "orchestration_summary",
        "model": args.model,
        "context_length": context_length,
        "orchestration_delay_ms": delay_ms,
        "tight_cpu_orchestration_ratio": tight_record["cpu_orchestration_ratio"],
        "orchestrated_cpu_orchestration_ratio": orch_record["cpu_orchestration_ratio"],
        "energy_cpu_orchestration_joules_delta": round(cpu_delta, 4),
        "tight_tokens_per_second_sustained": tight_record["tokens_per_second_sustained"],
        "orchestrated_tokens_per_second_sustained": orch_record["tokens_per_second_sustained"],
        "hardware": hardware.brand,
        "notes": "Tight MLX decode loop vs Python-delayed agent-style dispatch.",
        "powermetrics_log_path": None,
    }
    write_jsonl(jsonl_path, summary)

    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jsonl_path, result_dir / jsonl_path.name)
    for path in (tight_pm, orch_pm):
        shutil.copy2(path, result_dir / path.name)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[alala] cpu_orch_ratio tight={tight_record['cpu_orchestration_ratio']} delayed={orch_record['cpu_orchestration_ratio']}")
    print(f"[alala] wrote {jsonl_path}")
    return 0


def run_unimplemented_mode(mode: str) -> int:
    raise HarnessError(
        f"Mode {mode!r} is not implemented yet. "
        "thermal_baseline is available now; other modes follow Week 1 tasks W1-03+."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alalā Phase 0 M4 energy measurement harness")
    parser.add_argument("--mode", choices=MODES, required=True, help="Harness benchmark mode")
    parser.add_argument("--duration", type=int, default=120, help="Total run duration in seconds")
    parser.add_argument(
        "--idle-seconds",
        type=int,
        default=30,
        help="Initial idle monitoring period before sustained load starts",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=1000,
        help="powermetrics sample interval in milliseconds (1 Hz default)",
    )
    parser.add_argument(
        "--temp-threshold",
        type=float,
        default=None,
        help="Abort if package temperature exceeds this sustained threshold (Celsius)",
    )
    parser.add_argument(
        "--load",
        choices=("mlx_matmul", "cpu_spin"),
        default="mlx_matmul",
        help="Interim sustained load generator for thermal_baseline",
    )
    parser.add_argument("--model", default="baseline", help="Model name (used by future modes)")
    parser.add_argument("--max-context", type=int, default=8192, help="SRAM cliff sweep upper bound")
    parser.add_argument("--context", type=int, default=512, help="Fixed context length for KV/orchestration")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations for KV/orchestration modes")
    parser.add_argument(
        "--step-duration",
        type=int,
        default=90,
        help="Per-context decode duration for sram_cliff (seconds)",
    )
    parser.add_argument(
        "--steady-window",
        type=int,
        default=60,
        help="Tail window for sustained tokens/s in sram_cliff",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=32,
        help="Max decode tokens per round in sram_cliff",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=120,
        help="Idle cooldown between sram_cliff context steps",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        hardware = detect_hardware()
        require_target_hardware(hardware)
        print(f"[alala] target hardware: {hardware.brand}, {hardware.memory_bytes / (1024**3):.0f} GiB")

        if args.mode == "thermal_baseline":
            return run_thermal_baseline(args, hardware)
        if args.mode == "sram_cliff":
            return run_sram_cliff(args, hardware)
        if args.mode == "kv_comparison":
            return run_kv_comparison(args, hardware)
        if args.mode == "orchestration":
            return run_orchestration(args, hardware)
        return run_unimplemented_mode(args.mode)
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())