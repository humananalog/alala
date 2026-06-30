#!/usr/bin/env python3
"""Validate Phase 0 JSONL measurement artifacts per IPJ_Measurement_Protocol_Alalā.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_ALL = ("timestamp", "experiment_id", "benchmark_name", "powermetrics_log_path")
REQUIRED_IPJ = (
    "energy_joules",
    "temp_start_c",
    "thermal_envelope_valid",
)
THERMAL_FIELDS = ("peak_temp_c", "temp_steady_state_c", "sustained_power_w")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def validate_record(rec: dict[str, Any], *, require_m4: bool) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_ALL:
        if key not in rec or rec[key] in (None, ""):
            errors.append(f"missing {key}")
    for key in REQUIRED_IPJ:
        if key not in rec:
            errors.append(f"missing {key}")
    pm = rec.get("powermetrics_log_path")
    if pm and require_m4:
        p = Path(str(pm))
        if not p.exists():
            errors.append(f"powermetrics log missing: {p}")
        elif p.stat().st_size < 10:
            errors.append(f"powermetrics log too small: {p}")
        elif "# dry-run" in p.read_text(encoding="utf-8", errors="ignore")[:200]:
            errors.append(f"dry-run log not valid for M4 claims: {p}")
    if rec.get("ipj") is not None and not rec.get("thermal_envelope_valid"):
        errors.append("ipj set but thermal_envelope_valid is false")
    return errors


def validate_file(path: Path, *, require_m4: bool = False) -> list[str]:
    if not path.exists():
        return [f"file not found: {path}"]
    errors: list[str] = []
    for i, rec in enumerate(load_jsonl(path)):
        for err in validate_record(rec, require_m4=require_m4):
            errors.append(f"line {i + 1}: {err}")
    return errors


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Validate Alalā measurement JSONL artifacts.")
    p.add_argument("paths", nargs="+", type=Path, help="JSONL files to validate")
    p.add_argument("--require-m4", action="store_true", help="Reject dry-run logs; require real powermetrics files")
    args = p.parse_args(argv)
    all_errors: list[str] = []
    for path in args.paths:
        all_errors.extend(validate_file(path, require_m4=args.require_m4))
    if all_errors:
        for e in all_errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"PASSED: {len(args.paths)} artifact(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
