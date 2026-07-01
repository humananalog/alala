#!/usr/bin/env python3
"""Short Core ML decode profiling session with compute-plan and powermetrics capture."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE1_DIR = REPO_ROOT / "phase1"
HARNESS_DIR = REPO_ROOT / "harness"
MODELS_DIR = REPO_ROOT / "models"
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(PHASE1_DIR))

from dataclasses import asdict

from coreml_instrumentation import (  # noqa: E402
    dump_load_report,
    load_coreml_model,
    log_load_info,
    log_runtime_environment,
)
from env import LOGS_DIR, RESULTS_DIR  # noqa: E402
from kv_decode import run_coreml_decode_loop  # noqa: E402
from metrics import summarize_power_samples  # noqa: E402
from m4_energy_harness import PowerMetricsSession, read_macmon_cpu_temp, utc_now_iso  # noqa: E402
from powermetrics_log import parse_powermetrics_file  # noqa: E402

DEFAULT_PREFILL = MODELS_DIR / "qwen2.5-0.5b-prefill-kv.mlpackage"
DEFAULT_DECODE = MODELS_DIR / "qwen2.5-0.5b-decode-kv.mlpackage"
DEFAULT_PREFILL_ONLY = MODELS_DIR / "qwen2.5-0.5b-ane.mlpackage"


def _hardware() -> str:
    return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()


def profile_compute_units(
    *,
    prefill_path: Path,
    decode_path: Path,
    prefill_only_path: Path,
    compute_units: str,
    context_length: int,
    max_ctx: int,
    profile_tokens: int,
    steady_window_s: int,
    coreml_verbose: bool,
) -> dict:
    if coreml_verbose:
        os.environ.setdefault("COREML_VERBOSE", "1")

    env = log_runtime_environment()
    load_infos = []

    print(f"\n[profile] === compute_units={compute_units} ctx={context_length} ===")
    for role, path in (
        ("prefill_kv", prefill_path),
        ("decode_kv", decode_path),
        ("prefill_only_proxy", prefill_only_path),
    ):
        _, info = load_coreml_model(path, role=role, compute_units=compute_units, capture_compute_plan=True)
        load_infos.append(info)
        log_load_info(info)

    duration_s = max(10, profile_tokens // 2 + 5)
    pm_path = LOGS_DIR / f"ane_placement_profile_{compute_units}_ctx{context_length}.powermetrics.txt"
    session = PowerMetricsSession(pm_path, interval_ms=500)
    pre_temp = read_macmon_cpu_temp()
    t0 = time.monotonic()
    session.start()
    try:
        result = run_coreml_decode_loop(
            prefill_path=prefill_path,
            decode_path=decode_path,
            context_length=context_length,
            max_ctx=max_ctx,
            duration_s=duration_s,
            steady_window_s=steady_window_s,
            decode_tokens=profile_tokens,
            compute_units=compute_units,
            log_model_load=False,
        )
    finally:
        session.stop()
    elapsed = time.monotonic() - t0
    post_temp = read_macmon_cpu_temp()

    samples = parse_powermetrics_file(pm_path)
    power = summarize_power_samples(samples, steady_window=steady_window_s)

    summary = {
        "compute_units": compute_units,
        "context_length": context_length,
        "profile_tokens_target": profile_tokens,
        "duration_s": duration_s,
        "elapsed_s": round(elapsed, 3),
        "tokens_generated": result.tokens_generated,
        "tokens_per_second": round(result.tokens_per_second, 3),
        "tokens_per_second_sustained": round(result.tokens_per_second_sustained, 3),
        "decode_runtime": result.decode_runtime,
        "ane_utilization_proxy_pct": power.ane_utilization_pct,
        "energy_joules": round(power.energy_joules, 4),
        "energy_ane_joules": round(power.energy_ane_joules, 4),
        "energy_cpu_joules": round(power.energy_cpu_joules, 4),
        "energy_gpu_joules": round(power.energy_gpu_joules, 4),
        "temp_start_c": pre_temp,
        "temp_end_c": post_temp,
        "powermetrics_log_path": str(pm_path.relative_to(REPO_ROOT)),
        "environment": env,
    }
    print(
        f"[profile] tps_sustained={summary['tokens_per_second_sustained']} "
        f"ane_proxy={summary['ane_utilization_proxy_pct']}% "
        f"runtime={summary['decode_runtime']}"
    )
    return {"summary": summary, "load_infos": load_infos}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile Core ML decode ANE placement.")
    parser.add_argument("--prefill-kv", type=Path, default=DEFAULT_PREFILL)
    parser.add_argument("--decode-kv", type=Path, default=DEFAULT_DECODE)
    parser.add_argument("--prefill-only", type=Path, default=DEFAULT_PREFILL_ONLY)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-ctx", type=int, default=1024)
    parser.add_argument("--profile-tokens", type=int, default=30)
    parser.add_argument("--steady-window", type=int, default=15)
    parser.add_argument(
        "--compute-units",
        default="all,cpu_and_ne",
        help="Comma-separated compute unit modes to test",
    )
    parser.add_argument("--coreml-verbose", action="store_true")
    args = parser.parse_args(argv)

    run_id = f"ane_placement_profile_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    out_dir = RESULTS_DIR / "ane_placement_profile" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for cu in [part.strip() for part in args.compute_units.split(",") if part.strip()]:
        try:
            exp = profile_compute_units(
                prefill_path=args.prefill_kv,
                decode_path=args.decode_kv,
                prefill_only_path=args.prefill_only,
                compute_units=cu,
                context_length=args.context,
                max_ctx=args.max_ctx,
                profile_tokens=args.profile_tokens,
                steady_window_s=args.steady_window,
                coreml_verbose=args.coreml_verbose,
            )
            experiments.append(exp)
        except Exception as exc:
            print(f"[profile] FAILED compute_units={cu}: {exc}", file=sys.stderr)
            experiments.append({"summary": {"compute_units": cu, "error": str(exc)}, "load_infos": []})

    all_infos = [info for exp in experiments for info in exp.get("load_infos", [])]
    env = experiments[0]["summary"].get("environment", {}) if experiments else {}
    dump_load_report(all_infos, out_dir / "compute_plan.json", environment=env)

    report = {
        "timestamp": utc_now_iso(),
        "run_id": run_id,
        "hardware": _hardware(),
        "reference_run": "ane_residency_20260701T010929Z_830681e7",
        "reference_metrics": {
            "compute_units": "all",
            "tokens_per_second_sustained": 7.45,
            "ane_utilization_proxy_pct": 0.11435237710644898,
            "decode_runtime": "coreml",
        },
        "experiments": [
            {
                "summary": exp["summary"],
                "load_infos": [asdict(info) for info in exp.get("load_infos", [])],
            }
            for exp in experiments
        ],
        "instruments_instructions": [
            "Open Instruments.app → Core ML template.",
            "Select the qwen2.5-0.5b-decode-kv process during a 30-token decode loop.",
            "Inspect 'Core ML Performance' and 'Neural Engine' lanes for op placement.",
            "Compare against qwen2.5-0.5b-ane.mlpackage prefill-only proxy at ctx 512.",
        ],
    }
    report_path = out_dir / "profile_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n[profile] wrote {report_path}")
    print("[profile] Instruments: Core ML template on decode process (see profile_report.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())