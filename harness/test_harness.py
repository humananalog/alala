#!/usr/bin/env python3
"""Smoke tests for m4_energy_harness (dry-run; no M4 required)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "harness" / "m4_energy_harness.py"
VALIDATE = ROOT / "harness" / "validate_artifact.py"


class HarnessSmokeTest(unittest.TestCase):
    def _run(self, *extra: str) -> dict:
        cmd = [sys.executable, str(HARNESS), "--dry-run", *extra]
        out = subprocess.check_output(cmd, cwd=ROOT, text=True)
        return json.loads(out)

    def test_setup_check(self) -> None:
        rec = self._run("--mode", "setup_check", "--duration", "2")
        self.assertEqual(rec["benchmark_name"], "setup_check")
        self.assertIn("energy_joules", rec)

    def test_thermal_baseline(self) -> None:
        rec = self._run("--mode", "thermal_baseline", "--duration", "2", "--idle-duration", "1")
        self.assertIn("phases", rec)
        self.assertEqual(len(rec["phases"]), 2)

    def test_kv_comparison_distinct_paths(self) -> None:
        rec = self._run("--mode", "kv_comparison", "--duration", "2")
        self.assertIn("fp16", rec)
        self.assertIn("int4", rec)
        self.assertNotEqual(rec["fp16"].get("workload"), rec["int4"].get("workload"))

    def test_sram_cliff_detection(self) -> None:
        rec = self._run("--mode", "sram_cliff", "--duration", "20", "--max-context", "2048")
        self.assertIn("steps", rec)

    def test_artifact_validation(self) -> None:
        rec = self._run("--mode", "setup_check", "--duration", "2", "--experiment-id", "test_val_001")
        log = ROOT / "logs" / "test_val_001.jsonl"
        self.assertTrue(log.exists())
        subprocess.check_call([sys.executable, str(VALIDATE), str(log)], cwd=ROOT)


if __name__ == "__main__":
    unittest.main()
