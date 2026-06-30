#!/usr/bin/env python3
"""Mark a Phase 0 criterion validated in results/measurement_status.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

STATUS_PATH = Path(__file__).resolve().parents[1] / "results" / "measurement_status.json"

CRITERION_KEYS = {
    "thermal_baseline": "thermal_baseline",
    "sram_cliff": "sram_cliff",
    "kv_comparison": "kv_comparison",
    "orchestration": "orchestration_overhead",
    "orchestration_overhead": "orchestration_overhead",
    "ane_utilization": "e1_ane_utilization",
    "e1": "e1_ane_utilization",
    "thermal_ipj_curve": "e2_thermal_ipj_curve",
    "e2": "e2_thermal_ipj_curve",
    "meta_tax": "e3_meta_tax",
    "e3": "e3_meta_tax",
    "memory_spill": "e4_memory_spill",
    "e4": "e4_memory_spill",
}


def main() -> int:
    p = argparse.ArgumentParser(description="Mark M4 measurement criterion validated.")
    p.add_argument("--criterion", required=True, choices=sorted(set(CRITERION_KEYS.keys())))
    p.add_argument("--jsonl", required=True, type=Path, help="Validated JSONL artifact path")
    args = p.parse_args()

    key = CRITERION_KEYS[args.criterion]
    data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    if key not in data["criteria"]:
        raise SystemExit(f"Unknown criterion key: {key}")

    data["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data["criteria"][key]["m4_validated"] = True
    data["criteria"][key]["artifact_path"] = str(args.jsonl.resolve())
    STATUS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Marked {key} validated → {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
