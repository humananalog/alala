#!/usr/bin/env python3
"""Phase 1 ANE residency benchmark — measure forward-pass compute placement on M4.

Why ANE residency matters on Alalā:
- The ANE sits close to on-chip SRAM (~28–30 MB); keeping matmul/attention there minimizes
  unified-memory traffic (Phase 0 L_cliff=1024 showed steep IPJ loss once working sets spill).
- GPU-heavy MLX decode drew ~0% ANE power in Phase 0; routing to ANE should improve Joules/token
  and preserve thermal headroom (steady-state ~82–86 °C with throttling in ~5 s).

This script compares MLX (GPU baseline) vs Core ML (.mlpackage) at context 512 and 1024,
logging powermetrics + Phase 0-compatible JSONL for IPJ gating.

Examples:
    # MLX real decode (KV inside mlx_lm)
    python phase1/ane_residency_benchmark.py --backend mlx --decode --context 512,1024

    # Core ML stateful decode with explicit KV cache hand-off
    python phase1/ane_residency_benchmark.py \\
        --backend coreml --decode --context 512,1024

    # Legacy prefill-only proxy (no KV)
    python phase1/ane_residency_benchmark.py \\
        --backend coreml --model models/qwen2.5-0.5b-ane.mlpackage --context 512,1024
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE1_DIR = REPO_ROOT / "phase1"
HARNESS_DIR = REPO_ROOT / "harness"
MODELS_DIR = REPO_ROOT / "models"
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(PHASE1_DIR))

from coreml_instrumentation import (  # noqa: E402
    COMPUTE_UNIT_CHOICES,
    dump_load_report,
    load_coreml_model,
    log_load_info,
    log_runtime_environment,
)
from kv_decode import (  # noqa: E402
    run_coreml_decode_loop,
    run_coreml_prefill_proxy,
    run_mlx_decode_loop,
)
from decode_client import DecodeRunner  # noqa: E402
from env import LOGS_DIR, RESULTS_DIR  # noqa: E402
from errors import HarnessError  # noqa: E402
from metrics import summarize_power_samples  # noqa: E402
from m4_energy_harness import (  # noqa: E402
    PowerMetricsSession,
    read_macmon_cpu_temp,
    utc_now_iso,
    write_jsonl,
)
from powermetrics_log import parse_powermetrics_file  # noqa: E402

DEFAULT_CONTEXTS = (512, 1024)
DEFAULT_TEMP_THRESHOLD_C = 85.0
DEFAULT_DECODE_TEMP_THRESHOLD_C = 88.0
DEFAULT_STEP_DURATION_S = 60
DEFAULT_STEADY_WINDOW_S = 30
DEFAULT_DECODE_TOKENS = 32
DEFAULT_COOLDOWN_S = 60
DEFAULT_INTERVAL_MS = 1000
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_COREML_CONTEXT_SIZE = 1024
DEFAULT_COREML_PREFILL_KV = MODELS_DIR / "qwen2.5-0.5b-prefill-kv.mlpackage"
DEFAULT_COREML_DECODE_KV = MODELS_DIR / "qwen2.5-0.5b-decode-kv.mlpackage"
DEFAULT_COREML_DECODE_PT = MODELS_DIR / "qwen2.5-0.5b-decode-kv.pt"
DEFAULT_COREML_PREFILL_ONLY = MODELS_DIR / "qwen2.5-0.5b-ane.mlpackage"

PHASE0_MLX_BASELINE: dict[int, dict[str, Any]] = {
    512: {"tokens_per_second_sustained": 9.65, "ane_utilization_pct": 0.0},
    1024: {"tokens_per_second_sustained": 6.40, "ane_utilization_pct": 0.0},
}


@dataclass
class StepMetrics:
    context_length: int
    backend: str
    model_path: str
    tokens_generated: int
    tokens_per_second: float
    tokens_per_second_sustained: float
    peak_memory_gb: float | None
    energy_joules: float
    energy_ane_joules: float
    energy_cpu_joules: float
    energy_gpu_joules: float
    ane_utilization_proxy: float | None
    sustained_power_w: float | None
    temp_start_c: float | None
    temp_steady_state_c: float | None
    peak_temp_c: float | None
    thermal_warning: str | None
    experiment_id: str
    powermetrics_log_path: str
    decode_mode: bool
    kv_cache_active: bool
    decode_runtime: str | None
    compute_units: str | None = None
    kv_cache_mode: str | None = None
    ring_size: int | None = None
    re_prefill_count: int | None = None
    kv_io_bytes_per_step: int | None = None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_contexts(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _resolve_temperatures(samples, macmon_temps: list[float]) -> list[float]:
    pm = [s.temp_c for s in samples if s.temp_c is not None]
    return pm if pm else macmon_temps


def _thermal_check(temp_c: float | None, threshold_c: float) -> tuple[str | None, str | None]:
    """Return (abort_reason, warning). Abort on peak >= threshold; warn on steady exceed."""
    if temp_c is None:
        return None, None
    if temp_c >= threshold_c:
        return f"temperature {temp_c:.1f}C exceeded threshold {threshold_c:.1f}C", None
    return None, None


def _steady_thermal_warning(temp_steady: float | None, threshold_c: float) -> str | None:
    if temp_steady is not None and temp_steady > threshold_c:
        return f"steady-state {temp_steady:.1f}C > {threshold_c:.1f}C envelope"
    return None


def resolve_coreml_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None, Path | None, str]:
    """Return (prefill_kv, decode_kv, decode_pt, prefill_only, model_label) for Core ML backend."""
    if args.decode:
        prefill_kv = Path(args.coreml_prefill_kv) if args.coreml_prefill_kv else DEFAULT_COREML_PREFILL_KV
        decode_kv = Path(args.coreml_decode_kv) if args.coreml_decode_kv else DEFAULT_COREML_DECODE_KV
        decode_pt = Path(args.coreml_decode_pt) if args.coreml_decode_pt else DEFAULT_COREML_DECODE_PT
        if not prefill_kv.exists():
            raise HarnessError(
                f"Core ML prefill KV model not found: {prefill_kv}. "
                "Run: phase1/.venv/bin/python phase1/coreml_kv_convert.py"
            )
        if not decode_kv.exists() and not decode_pt.exists():
            raise HarnessError(
                f"Core ML decode artifact not found: {decode_kv} or {decode_pt}. "
                "Run: phase1/.venv/bin/python phase1/coreml_kv_convert.py"
            )
        decode_label = decode_kv.name if decode_kv.exists() else decode_pt.name
        label = f"{prefill_kv.name}+{decode_label}"
        return prefill_kv, decode_kv, decode_pt, None, label

    model_path = Path(args.model) if args.model else DEFAULT_COREML_PREFILL_ONLY
    if not model_path.exists():
        raise HarnessError(f"Core ML model not found: {model_path}")
    return None, None, None, model_path, str(model_path)


def run_workload(
    *,
    backend: str,
    decode_mode: bool,
    mlx_runner: DecodeRunner | None,
    prefill_kv_path: Path | None,
    decode_kv_path: Path | None,
    decode_pt_path: Path | None,
    prefill_only_path: Path | None,
    context_length: int,
    coreml_max_ctx: int,
    duration_s: int,
    steady_window_s: int,
    decode_tokens: int,
    compute_units: str = "all",
    log_model_load: bool = True,
    kv_cache_mode: str = "linear",
    ring_size: int = 512,
) -> tuple[int, float, float, float | None, bool, str | None, dict[str, object]]:
    """Run one benchmark step; returns tokens, tps, tps_sustained, peak_mem, kv_cache_active, decode_runtime."""
    if backend == "mlx":
        assert mlx_runner is not None
        if decode_mode:
            result = run_mlx_decode_loop(
                mlx_runner,
                context_length=context_length,
                duration_s=duration_s,
                steady_window_s=steady_window_s,
                decode_tokens=decode_tokens,
            )
        else:
            result = run_mlx_decode_loop(
                mlx_runner,
                context_length=context_length,
                duration_s=duration_s,
                steady_window_s=steady_window_s,
                decode_tokens=decode_tokens,
            )
        meta = {
            "kv_cache_mode": getattr(result, "kv_cache_mode", None),
            "ring_size": getattr(result, "ring_size", None),
            "re_prefill_count": getattr(result, "re_prefill_count", None),
            "kv_io_bytes_per_step": getattr(result, "kv_io_bytes_per_step", None),
        }
        return (
            result.tokens_generated,
            result.tokens_per_second,
            result.tokens_per_second_sustained,
            result.peak_memory_gb,
            result.kv_cache_active,
            result.decode_runtime,
            meta,
        )

    if decode_mode:
        assert prefill_kv_path is not None
        result = run_coreml_decode_loop(
            prefill_path=prefill_kv_path,
            decode_path=decode_kv_path or DEFAULT_COREML_DECODE_KV,
            decode_pt_path=decode_pt_path,
            context_length=context_length,
            max_ctx=coreml_max_ctx,
            duration_s=duration_s,
            steady_window_s=steady_window_s,
            decode_tokens=decode_tokens,
            compute_units=compute_units,
            log_model_load=log_model_load,
            kv_cache_mode=kv_cache_mode,
            ring_size=ring_size,
        )
    else:
        assert prefill_only_path is not None
        result = run_coreml_prefill_proxy(
            prefill_only_path,
            context_length=context_length,
            trace_context_size=coreml_max_ctx,
            duration_s=duration_s,
            steady_window_s=steady_window_s,
            compute_units=compute_units,
            log_model_load=log_model_load,
        )
    meta = {
        "kv_cache_mode": getattr(result, "kv_cache_mode", None),
        "ring_size": getattr(result, "ring_size", None),
        "re_prefill_count": getattr(result, "re_prefill_count", None),
        "kv_io_bytes_per_step": getattr(result, "kv_io_bytes_per_step", None),
    }
    return (
        result.tokens_generated,
        result.tokens_per_second,
        result.tokens_per_second_sustained,
        result.peak_memory_gb,
        result.kv_cache_active,
        result.decode_runtime,
        meta,
    )


def build_step_record(
    *,
    run_id: str,
    step: StepMetrics,
    hardware: str,
    timestamp: str,
) -> dict[str, Any]:
    ipj = step.tokens_generated / step.energy_joules if step.energy_joules > 0 else None
    baseline = PHASE0_MLX_BASELINE.get(step.context_length, {})
    base_tps = baseline.get("tokens_per_second_sustained")
    delta_tps = None
    if base_tps:
        delta_tps = round(((step.tokens_per_second_sustained - base_tps) / base_tps) * 100.0, 2)

    return {
        "timestamp": timestamp,
        "experiment_id": step.experiment_id,
        "run_id": run_id,
        "task_type": "ane_residency",
        "backend": step.backend,
        "model_path": step.model_path,
        "model": step.model_path,
        "context_length": step.context_length,
        "energy_joules": round(step.energy_joules, 4),
        "energy_ane_joules": round(step.energy_ane_joules, 4),
        "energy_cpu_orchestration_joules": round(step.energy_cpu_joules, 4),
        "energy_gpu_joules": round(step.energy_gpu_joules, 4),
        "energy_dequant_joules": None,
        "tokens_generated": step.tokens_generated,
        "tokens_per_second": round(step.tokens_per_second, 3),
        "tokens_per_second_sustained": round(step.tokens_per_second_sustained, 3),
        "ane_utilization_pct": (
            round(step.ane_utilization_proxy, 2) if step.ane_utilization_proxy is not None else None
        ),
        "ane_utilization_proxy": (
            round(step.ane_utilization_proxy, 2) if step.ane_utilization_proxy is not None else None
        ),
        "peak_memory_gb": round(step.peak_memory_gb, 3) if step.peak_memory_gb is not None else None,
        "temp_start_c": step.temp_start_c,
        "peak_temp_c": step.peak_temp_c,
        "temp_steady_state_c": step.temp_steady_state_c,
        "thermal_warning": step.thermal_warning,
        "time_to_throttle_s": None,
        "sustained_power_w": (
            round(step.sustained_power_w, 3) if step.sustained_power_w is not None else None
        ),
        "u_task_score": 1.0,
        "ipj": round(ipj, 6) if ipj is not None else None,
        "hca_impact": None,
        "hardware": hardware,
        "delta_tps_sustained_pct_vs_phase0": delta_tps,
        "decode_mode": step.decode_mode,
        "kv_cache_active": step.kv_cache_active,
        "decode_runtime": step.decode_runtime,
        "compute_units": step.compute_units,
        "kv_cache_mode": step.kv_cache_mode,
        "ring_size": step.ring_size,
        "re_prefill_count": step.re_prefill_count,
        "kv_io_bytes_per_step": step.kv_io_bytes_per_step,
        "notes": (
            f"Phase 1 ANE residency; backend={step.backend}; "
            f"decode_mode={step.decode_mode}; kv_cache_active={step.kv_cache_active}; "
            f"decode_runtime={step.decode_runtime}; compute_units={step.compute_units}; "
            f"kv_cache_mode={step.kv_cache_mode}; ring_size={step.ring_size}; "
            f"re_prefill_count={step.re_prefill_count}; "
            "ane_utilization_proxy = ane_energy / (cpu+gpu+ane) from powermetrics."
        ),
        "powermetrics_log_path": step.powermetrics_log_path,
    }


def print_summary(steps: list[StepMetrics], aborted: bool, abort_reason: str | None) -> None:
    print("\n=== ANE Residency Summary ===")
    for step in steps:
        mode = "decode+kv" if step.kv_cache_active else ("decode" if step.decode_mode else "prefill_proxy")
        runtime = f" runtime={step.decode_runtime}" if step.decode_runtime else ""
        cu = f" compute_units={step.compute_units}" if step.compute_units else ""
        print(
            f"  ctx={step.context_length} backend={step.backend} mode={mode}{runtime}{cu} "
            f"tps_peak={step.tokens_per_second:.2f} tps_sustained={step.tokens_per_second_sustained:.2f} "
            f"ane_proxy={step.ane_utilization_proxy}% temp_steady={step.temp_steady_state_c}C"
        )
        if step.thermal_warning:
            print(f"    WARN: {step.thermal_warning}")
    if aborted:
        print(f"  ABORTED: {abort_reason}")
    else:
        ane_vals = [s.ane_utilization_proxy for s in steps if s.ane_utilization_proxy is not None]
        best = max(ane_vals) if ane_vals else None
        print(f"  best_ane_utilization_proxy: {best}%")


def run_benchmark(args: argparse.Namespace) -> int:
    if args.steady_window >= args.step_duration:
        raise HarnessError("--steady-window must be less than --step-duration")

    backend = args.backend
    contexts = parse_contexts(args.context)
    hardware = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()

    prefill_kv_path: Path | None = None
    decode_kv_path: Path | None = None
    decode_pt_path: Path | None = None
    prefill_only_path: Path | None = None
    if backend == "coreml":
        prefill_kv_path, decode_kv_path, decode_pt_path, prefill_only_path, model_label = resolve_coreml_paths(
            args
        )
    else:
        model_label = args.model or DEFAULT_MLX_MODEL

    run_id = f"ane_residency_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    jsonl_path = LOGS_DIR / f"{run_id}.jsonl"
    result_dir = RESULTS_DIR / "ane_residency" / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[alala] run_id={run_id}")
    print(
        f"[alala] backend={backend} decode={args.decode} model={model_label} "
        f"contexts={contexts} temp_threshold={args.temp_threshold}C "
        f"compute_units={args.compute_units} kv_cache_mode={args.kv_cache_mode} "
        f"ring_size={args.ring_size}"
    )

    if backend == "coreml":
        log_runtime_environment()
        load_infos = []
        if args.decode and prefill_kv_path and decode_kv_path:
            for role, path in (("prefill_kv", prefill_kv_path), ("decode_kv", decode_kv_path)):
                if path.exists():
                    _, info = load_coreml_model(path, role=role, compute_units=args.compute_units)
                    load_infos.append(info)
                    log_load_info(info)
        elif prefill_only_path:
            _, info = load_coreml_model(
                prefill_only_path,
                role="prefill_only_proxy",
                compute_units=args.compute_units,
            )
            load_infos.append(info)
            log_load_info(info)
        dump_load_report(
            load_infos,
            result_dir / "coreml_load_report.json",
            environment=log_runtime_environment(),
        )

    mlx_runner = DecodeRunner(model_label) if backend == "mlx" else None
    steps: list[StepMetrics] = []
    aborted = False
    abort_reason: str | None = None

    for context_length in contexts:
        step_id = f"{run_id}_ctx{context_length}"
        pm_path = LOGS_DIR / f"{step_id}.powermetrics.txt"
        macmon_temps: list[float] = []

        pre_temp = read_macmon_cpu_temp()
        if pre_temp is not None:
            macmon_temps.append(pre_temp)
            abort_reason, _ = _thermal_check(pre_temp, args.temp_threshold)
            if abort_reason:
                aborted = True
                print(f"[alala] ABORT: {abort_reason}")
                break

        session = PowerMetricsSession(pm_path, interval_ms=args.interval_ms)
        session.start()
        try:
            tokens, tps, tps_sustained, peak_mem, kv_active, decode_runtime, kv_meta = run_workload(
                backend=backend,
                decode_mode=args.decode,
                mlx_runner=mlx_runner,
                prefill_kv_path=prefill_kv_path,
                decode_kv_path=decode_kv_path,
                decode_pt_path=decode_pt_path,
                prefill_only_path=prefill_only_path,
                context_length=context_length,
                coreml_max_ctx=args.coreml_context_size,
                duration_s=args.step_duration,
                steady_window_s=args.steady_window,
                decode_tokens=args.decode_tokens,
                compute_units=args.compute_units,
                log_model_load=False,
                kv_cache_mode=args.kv_cache_mode,
                ring_size=args.ring_size,
            )
        finally:
            session.stop()

        post_temp = read_macmon_cpu_temp()
        if post_temp is not None:
            macmon_temps.append(post_temp)

        samples = parse_powermetrics_file(pm_path)
        power = summarize_power_samples(samples, steady_window=args.steady_window)
        temps = _resolve_temperatures(samples, macmon_temps)
        temp_start = temps[0] if temps else None
        peak_temp = max(temps) if temps else None
        temp_steady = _mean(temps[-args.steady_window :]) if temps else None

        if not aborted:
            abort_reason, _ = _thermal_check(peak_temp, args.temp_threshold)
            if abort_reason:
                aborted = True

        thermal_warning = _steady_thermal_warning(temp_steady, args.temp_threshold)

        step = StepMetrics(
            context_length=context_length,
            backend=backend,
            model_path=model_label,
            tokens_generated=tokens,
            tokens_per_second=tps,
            tokens_per_second_sustained=tps_sustained,
            peak_memory_gb=peak_mem,
            energy_joules=power.energy_joules,
            energy_ane_joules=power.energy_ane_joules,
            energy_cpu_joules=power.energy_cpu_joules,
            energy_gpu_joules=power.energy_gpu_joules,
            ane_utilization_proxy=power.ane_utilization_pct,
            sustained_power_w=power.sustained_power_w,
            temp_start_c=round(temp_start, 2) if temp_start is not None else None,
            temp_steady_state_c=round(temp_steady, 2) if temp_steady is not None else None,
            peak_temp_c=round(peak_temp, 2) if peak_temp is not None else None,
            thermal_warning=thermal_warning,
            experiment_id=step_id,
            powermetrics_log_path=str(pm_path.relative_to(REPO_ROOT)),
            decode_mode=args.decode,
            kv_cache_active=kv_active,
            decode_runtime=decode_runtime,
            compute_units=args.compute_units if backend == "coreml" else None,
            kv_cache_mode=kv_meta.get("kv_cache_mode"),  # type: ignore[arg-type]
            ring_size=kv_meta.get("ring_size"),  # type: ignore[arg-type]
            re_prefill_count=kv_meta.get("re_prefill_count"),  # type: ignore[arg-type]
            kv_io_bytes_per_step=kv_meta.get("kv_io_bytes_per_step"),  # type: ignore[arg-type]
        )
        steps.append(step)
        write_jsonl(jsonl_path, build_step_record(run_id=run_id, step=step, hardware=hardware, timestamp=utc_now_iso()))
        shutil.copy2(pm_path, result_dir / pm_path.name)

        if thermal_warning:
            print(f"[alala] WARN ctx={context_length}: {thermal_warning}")
        print(
            f"[alala] ctx={context_length} tps_sustained={tps_sustained:.2f} "
            f"ane_proxy={step.ane_utilization_proxy}% temp={temp_steady}C"
        )

        if aborted:
            break
        if args.cooldown_seconds > 0 and context_length != contexts[-1]:
            time.sleep(args.cooldown_seconds)

    summary = {
        "timestamp": utc_now_iso(),
        "experiment_id": f"{run_id}_summary",
        "run_id": run_id,
        "task_type": "ane_residency_summary",
        "backend": backend,
        "model_path": model_label,
        "contexts_tested": [s.context_length for s in steps],
        "ane_utilization_proxy_by_context": {str(s.context_length): s.ane_utilization_proxy for s in steps},
        "tokens_per_second_sustained_by_context": {
            str(s.context_length): round(s.tokens_per_second_sustained, 3) for s in steps
        },
        "aborted": aborted,
        "abort_reason": abort_reason,
        "decode_mode": args.decode,
        "kv_cache_active": all(s.kv_cache_active for s in steps) if steps else False,
        "compute_units": args.compute_units if backend == "coreml" else None,
        "hardware": hardware,
        "powermetrics_log_path": None,
    }
    write_jsonl(jsonl_path, summary)
    shutil.copy2(jsonl_path, result_dir / jsonl_path.name)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print_summary(steps, aborted, abort_reason)
    print(f"[alala] wrote {jsonl_path}")
    return 1 if aborted else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1 ANE residency benchmark.")
    parser.add_argument("--backend", choices=("mlx", "coreml"), default="mlx")
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Stateful autoregressive decode with KV cache (vs prefill-only proxy for coreml)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="MLX HF id, or Core ML prefill-only .mlpackage (legacy proxy without --decode)",
    )
    parser.add_argument(
        "--coreml-prefill-kv",
        default=None,
        help="Prefill KV .mlpackage for --decode (default: models/qwen2.5-0.5b-prefill-kv.mlpackage)",
    )
    parser.add_argument(
        "--coreml-decode-kv",
        default=None,
        help="Decode KV .mlpackage for --decode (default: models/qwen2.5-0.5b-decode-kv.mlpackage)",
    )
    parser.add_argument(
        "--coreml-decode-pt",
        default=None,
        help="TorchScript decode fallback .pt (default: models/qwen2.5-0.5b-decode-kv.pt)",
    )
    parser.add_argument("--context", default="512,1024", help="Comma-separated context lengths")
    parser.add_argument(
        "--coreml-context-size",
        type=int,
        default=DEFAULT_COREML_CONTEXT_SIZE,
        help="Max traced context / KV cache length for Core ML models",
    )
    parser.add_argument("--step-duration", type=int, default=DEFAULT_STEP_DURATION_S)
    parser.add_argument("--steady-window", type=int, default=DEFAULT_STEADY_WINDOW_S)
    parser.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    parser.add_argument("--temp-threshold", type=float, default=DEFAULT_TEMP_THRESHOLD_C)
    parser.add_argument("--cooldown-seconds", type=int, default=DEFAULT_COOLDOWN_S)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    parser.add_argument(
        "--compute-units",
        choices=COMPUTE_UNIT_CHOICES,
        default="all",
        help="Core ML compute unit preference (all, cpu_and_ne, cpu_and_gpu, cpu_only)",
    )
    parser.add_argument(
        "--kv-cache-mode",
        choices=("linear", "ring"),
        default="linear",
        help="KV cache update strategy for Core ML decode (ring = fixed-size circular buffer)",
    )
    parser.add_argument(
        "--ring-size",
        type=int,
        default=512,
        help="Ring buffer capacity when --kv-cache-mode=ring (default: 512)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Short profiling run (20s step, 15s steady, ctx 512 only) then exit",
    )
    parser.add_argument("--coreml-verbose", action="store_true", help="Set COREML_VERBOSE=1 for profiling")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.profile:
        args.backend = "coreml"
        args.decode = True
        args.context = "512"
        args.step_duration = 20
        args.steady_window = 15
        args.cooldown_seconds = 0
    if args.decode and args.temp_threshold == DEFAULT_TEMP_THRESHOLD_C:
        args.temp_threshold = DEFAULT_DECODE_TEMP_THRESHOLD_C
    if args.backend == "coreml" and not args.decode and not args.model:
        args.model = str(DEFAULT_COREML_PREFILL_ONLY)
    if args.backend == "mlx" and args.model is None:
        args.model = DEFAULT_MLX_MODEL
    if args.coreml_verbose:
        import os

        os.environ.setdefault("COREML_VERBOSE", "1")
    try:
        return run_benchmark(args)
    except HarnessError as exc:
        print(f"[alala] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())