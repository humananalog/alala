# Phase 1 — Core ML Runtime Profiling Guide

Manual profiling on Mac Mini M4 using **Instruments** to understand planner vs runtime ANE placement gaps (e.g. 44% compute-plan ANE but ~3% powermetrics proxy).

## Prerequisites

- Xcode / Instruments.app installed
- Phase 1 venv + models exported locally under `models/`
- Powermetrics access (benchmark harness uses `SUDO_PASSWORD` from repo `.env`)

## Quick benchmark for profiling target

Run a sustained decode session the profiler can attach to:

```bash
# Mask int4 (higher ANE proxy ~2.9%)
PYTHONPATH=phase1 phase1/.venv/bin/python phase1/ane_residency_benchmark.py \
  --backend coreml --decode --context 512 \
  --coreml-decode-kv models/qwen2.5-0.5b-decode-kv-torch-export-int4.mlpackage \
  --compute-units all --step-duration 120 --steady-window 60

# Scatter int4 clean (higher throughput ~48 t/s, lower ANE proxy)
PYTHONPATH=phase1 phase1/.venv/bin/python phase1/ane_residency_benchmark.py \
  --backend coreml --decode --context 512 \
  --coreml-prefill-kv models/qwen2.5-0.5b-prefill-kv-int4.mlpackage \
  --coreml-decode-kv models/qwen2.5-0.5b-decode-kv-torch-export-int4-clean.mlpackage \
  --compute-units all --step-duration 120 --steady-window 60
```

Note the `run_id` printed (`ane_residency_YYYYMMDDTHHMMSSZ_<hash>`). Artifacts land in `logs/` and `results/ane_residency/<run_id>/`.

## Attach Instruments

### 1. Launch Instruments

1. Open **Instruments.app** (Xcode → Open Developer Tool → Instruments).
2. Choose template **Core ML** (add **Metal System Trace** as a second instrument if GPU fallback is suspected).
3. Target process: **Choose Profile** → select the running `Python` process hosting the benchmark (filter by command line containing `ane_residency_benchmark.py`).

Alternatively: Product → Profile in Xcode is not required; attach to an already-running benchmark started from Terminal.

### 2. Record window

1. Start the benchmark in Terminal first (120 s step gives a long steady window).
2. In Instruments, click **Record** within 5–10 s of benchmark start.
3. Stop recording after the steady window (or when the benchmark prints `ANE Residency Summary`).

### 3. Core ML template — what to inspect

| Lane / view | Look for |
|-------------|----------|
| **Core ML Performance** | Per-model load time, op execution breakdown |
| **Neural Engine** activity | Non-zero NE tile usage during decode steps (not just prefill) |
| **Operation placement** | Ops falling back to GPU/CPU; compare decode vs prefill model names |
| **Scheduler decisions** | Dynamic-shape or unsupported-op fallback messages |

**Decode-specific checks (mask int4 vs scatter int4):**

- Mask path: `equal`, `tile`, `mul` chains around KV slot write — often GPU-preferred despite high ANE *plan* fraction.
- Scatter path: `scatter` along seq dim — compute plan shows **0% ANE / ~44% GPU** for scatter-int4-clean; runtime ANE proxy ~0.36%.
- KV I/O: large `keyCache` / `valueCache` tensor copies each step — may dominate wall time regardless of matmul placement.

### 4. Metal System Trace (optional)

Use when Core ML shows GPU fallback:

- **AGX/Apple M4 GPU** command buffers during decode loop
- Buffer allocation spikes → possible unified-memory pressure (see Phase 0 SRAM cliff)
- Correlate GPU command bursts with powermetrics `energy_gpu_joules`

### 5. Correlate with powermetrics + JSONL

Each benchmark writes:

| Artifact | Path | Use |
|----------|------|-----|
| Powermetrics log | `logs/ane_residency_<run_id>_ctx512.powermetrics.txt` | ANE/CPU/GPU joules, power W |
| JSONL row | `logs/ane_residency_<run_id>.jsonl` | `tokens_per_second_sustained`, `ane_utilization_proxy`, energy fields |
| Compute plan | `results/ane_residency/<run_id>/coreml_load_report.json` | `ane_preferred_fraction` at load time |

**Correlation workflow:**

1. Note `timestamp` and `run_id` from JSONL.
2. Align Instruments record start with benchmark steady window (`steady_window_s` after step start).
3. Compare:
   - `ane_preferred_fraction` (compile-time plan) vs Instruments NE activity (runtime)
   - `ane_utilization_proxy` from powermetrics vs NE lane duty cycle
4. If plan says 44% ANE but Instruments shows NE idle during decode, suspect KV I/O + mask/scatter ops on GPU/CPU.

### 6. Environment flags (optional deeper logs)

```bash
export COREML_VERBOSE=1
export E5RT_LOG_LEVEL=debug
```

Re-run a short profile (`--step-duration 20`). Logs may show `ANECCompile() FAILED` for some subgraphs under memory pressure — cross-check with `MLComputePlan` and `ComputeUnit.ALL` vs `CPU_AND_GPU`.

## CLI compute plan (no GUI)

```bash
PYTHONPATH=phase1 phase1/.venv/bin/python -c "
from pathlib import Path
from coreml_instrumentation import load_coreml_model, log_load_info
_, info = load_coreml_model(
    Path('models/qwen2.5-0.5b-decode-kv-torch-export-int4.mlpackage'),
    role='decode_kv', compute_units='all', capture_compute_plan=True)
log_load_info(info)
"
```

## Findings log (2026-07-01)

Automated CLI profiling only (Instruments GUI not captured in CI):

| Model | Plan ANE % | Runtime ANE proxy | Sust. t/s | Notes |
|-------|------------|-------------------|-----------|-------|
| mask int4 decode | 44.1% | 2.90% | 27.73 | `equal`/`tile`/`concat` in graph |
| scatter int4 clean | **0%** | **0.36%** | **48.60** | `scatter` op; GPU-plan dominant |
| prefill-kv int4 | 29.1% | — | — | ANE plan slightly below fp16 31% |

**Takeaway:** Graph cleanup via `scatter` removes ~468 ops and improves throughput, but regresses ANE placement and runtime ANE energy. Instruments session recommended to confirm NE idle during scatter decode steps.