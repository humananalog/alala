#!/usr/bin/env bash
# Phase 0 execution kickoff — physical Mac Mini M4 24 GB only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Alalā Phase 0 Kickoff ==="
echo "Platform: $(uname -s) $(uname -m)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: Run on physical Mac Mini M4 24 GB (macOS)." >&2
  exit 1
fi

python3 -m pip install -q -r requirements.txt
./verify.sh

echo "--- Step 1: setup_check (30s) ---"
sudo python3 harness/m4_energy_harness.py --mode setup_check --duration 30

echo "--- Step 2: thermal_baseline (idle 10min + sustained 10min) ---"
echo "Ensure machine idled 10+ minutes before continuing."
sudo python3 harness/m4_energy_harness.py --mode thermal_baseline --idle-duration 600 --duration 600

echo "=== Kickoff steps 1–2 complete ==="
echo "Validate: python3 harness/validate_artifact.py --require-m4 logs/<experiment_id>.jsonl"
echo "Mark done: python3 harness/mark_validated.py --criterion thermal_baseline --jsonl logs/<experiment_id>.jsonl"
echo "Continue Week 1: docs/Phase0_AI_Coder_Task_List.md"
