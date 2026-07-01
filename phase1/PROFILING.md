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

## CLI analysis (when Instruments unavailable)

This host has Command Line Tools only (`xctrace` not installed). Use these scripts as an Instruments-equivalent baseline:

```bash
# Per-op MLComputePlan breakdown (placed vs unplaced, GPU-only ops, categories)
PYTHONPATH=phase1 phase1/.venv/bin/python phase1/compute_plan_analysis.py \
  models/qwen2.5-0.5b-decode-kv-torch-export-int4.mlpackage \
  --output results/compute_plan_analysis/mask_int4_decode.json

# 5 s ANE-proxy windows from powermetrics plist
PYTHONPATH=phase1 phase1/.venv/bin/python phase1/powermetrics_timeseries.py \
  logs/ane_residency_<run_id>_ctx512.powermetrics.txt --window-s 5 \
  --output results/compute_plan_analysis/<run_id>_ts.json
```

`COREML_VERBOSE=1` on a 2-step decode produced no scheduler lines on this build — rely on `MLComputePlan` + powermetrics windows for gap analysis.

## Findings log (2026-07-01)

### Instruments status

| Item | Result |
|------|--------|
| `xctrace` | **Unavailable** (CLT only; full Xcode required for Core ML + Metal System Trace) |
| CLI substitute | `compute_plan_analysis.py` + `powermetrics_timeseries.py` |
| Manual follow-up | Attach Instruments per sections above on physical M4 with Xcode |

### Runtime summary

| Model | Plan ANE % | Runtime ANE proxy | Sust. t/s | Notes |
|-------|------------|-------------------|-----------|-------|
| mask int4 decode | 44.1% | 2.90% | 27.73 | Run `1b69eca7` |
| mask int4 decode (30 s correlation) | 44.1% | 2.03% | 28.13 | Run `5fe0d68c` |
| scatter int4 clean | **0%** | **0.36%** | **48.60** | `scatter` GPU-plan dominant; Run `6f90882a` |
| prefill-kv int4 | 29.1% | — | — | ANE plan slightly below fp16 31% |

### Mask int4: compute plan vs runtime (CLI)

**Artifact:** `results/compute_plan_analysis/mask_int4_decode.json`

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total ops | 3697 | Includes 1821 `const` + 171 int4 dequant |
| Placed ops | 1705 | Only these have device preference |
| ANE % (all ops) | **44.1%** | Matches `coreml_load_report.json` |
| ANE % (placed only) | **95.6%** | Planner strongly prefers ANE for executable ops |
| GPU-preferred (placed) | 75 (4.4%) | Small fraction at compile time |

**GPU-only ops (no ANE path):** `greater_equal` (4), `select` (4), `gather` (3), `equal` (1) — mask-control chain, not bulk matmul.

**GPU despite ANE support (scheduler may still pick GPU):** `mul` (13), `add` (7), `linear` (7), `slice_by_index` (6), `concat` (2), `tile` (1).

**Category placement (mask int4):**

| Category | Total ops | ANE-preferred % (of placed) |
|----------|-----------|----------------------------|
| SDPA attention | 24 | **100%** |
| Linear (int4) | 169 | **95.3%** |
| Mask KV (`equal`/`tile`/`gather`/…) | 108 | **88.0%** |
| Cache rebuild (`concat`) | 96 | **97.9%** |
| int4 dequant (`constexpr_blockwise_shift_scale`) | 171 | **unplaced** |

**Conclusion:** The 44% → ~3% gap is **not** mainly “ANE-eligible matmuls compiled to GPU.” It is dominated by (1) **const/dequant ops inflating the plan denominator**, (2) **~12.6 MB KV tensor I/O per decode step** outside `MLComputePlan`, and (3) **Python/CPU orchestration + GPU memcpy** (646 J GPU + 421 J CPU vs 32 J ANE in run `1b69eca7`).

### Powermetrics time-series correlation (mask int4, `1b69eca7`, 5 s windows)

**Artifact:** `results/compute_plan_analysis/mask_int4_powermetrics_ts.json`

| Phase (window index) | ANE proxy | Dominant energy | Likely activity |
|----------------------|-----------|-----------------|-----------------|
| 0–2 | 0–0.1% | CPU ~25–30 J | Model load, prefill setup |
| 3, 7 | **4–17%** | Mixed | Prefill ANE bursts / decode model hand-off |
| 9–11 | 1.1–1.6% | **GPU ~100 J** | GPU shader/cache warmup spike |
| 12–18 (steady decode) | **2.7–5.6%** | GPU 40–60 J, CPU 13–40 J | Sustained decode loop |

Steady-state ANE proxy **p50 = 2.66%**, mean **3.23%** — aligns with aggregate **2.9%** in JSONL. Higher-ANE windows correlate with **lower relative GPU joules** (e.g. window 7: 17% ANE, 1.4 J GPU vs windows 9–11: ~100 J GPU).

**Correlation run `5fe0d68c` (30 s):** 2.03% ANE proxy, 28.13 t/s — same pattern; artifact `mask_int4_correlation_5fe0d68c_ts.json`.

### Expected Instruments confirmation (manual)

On Xcode-equipped M4, expect Core ML template to show:

1. **NE tile activity** in steady decode (low duty cycle, not idle) — consistent with 2–5% powermetrics proxy.
2. **Large buffer copies** at `predict()` boundaries for `keyCache`/`valueCache` I/O — not attributed to individual MIL ops.
3. **GPU command bursts** during windows 9–11 warmup and steady decode — matches powermetrics GPU dominance.
4. **Scatter int4 clean:** NE largely idle; `scatter` on GPU — confirms planner 0% ANE is honored at runtime.

**Takeaway:** Mask path ANE plan is real but **wall-time and energy are structurally capped** by explicit KV I/O and host orchestration. Scatter path trades remaining ANE for throughput. **Hybrid architecture** remains the recommended split unless KV state is moved in-package (MLState) without losing ATEN placement.