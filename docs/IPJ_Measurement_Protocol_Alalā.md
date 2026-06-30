# IPJ Measurement Protocol — Alalā

**Version**: 1.4  
**Purpose**: Define how Intelligence per Joule (IPJ) is measured, logged, and used as a gating criterion on **Mac Mini M4 24 GB** physical hardware.

**Execution constraint**: All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold.

## 1. Formal Definition

$$
\text{IPJ} = \frac{\mathbb{E}[U(\text{task})]}{\mathbb{E}[J]}
$$

Where:
- \( U(\text{task}) \) = Composite utility of completing the task (see §2)
- \( J \) = Energy consumed (joules), measured on the **physical M4** under controlled sustained thermal conditions

**M4 grounding**: On unified memory, \( J \) includes energy for data movement (unified memory ↔ ANE on-chip SRAM spills), dequantization, CPU orchestration between ANE invocations, and ANE/GPU/CPU compute — not compute alone. **Sustained IPJ under thermal steady state takes precedence over peak IPJ measured before throttling.**

## 2. Composite Utility \( U(\text{task}) \)

\( U(\text{task}) \) includes:
1. **Task Success** (correctness, quality, usefulness)
2. **Verification Quality** (HFPS from HCA spec)
3. **Future Value** (how much this outcome improves future self-improvement or capability)
4. **Efficiency** (tokens per joule, latency, etc.)

Each dimension is scored on a 0–1 scale and combined with weights that can evolve over time (initially conservative).

## 2.1 Phase 0 Operational Definition (Executable)

For Phase 0 micro-benchmarks on **physical Mac Mini M4 24 GB**, IPJ is computed as:

$$
\text{IPJ}_{\text{phase0}} = \frac{U_{\text{phase0}}(\text{task})}{J_{\text{measured}}}
$$

Where:

| Term | Operational meaning on M4 |
|------|---------------------------|
| \( U_{\text{phase0}}(\text{task}) \) | **Useful cognitive work completed** — e.g. tokens generated that pass a fixed quality/correctness threshold, or successful completion of a named micro-benchmark (thermal baseline, SRAM cliff sweep, KV comparison, orchestration profile) |
| \( J_{\text{measured}} \) | **Total joules** integrated from `powermetrics` (CPU + GPU + ANE domains) over the measured interval, on the real M4 under controlled sustained thermal conditions (idle 10+ min before load; run until steady state or safe duration cap) |

**Phase 0 simplification**: For decode benchmarks, \( U_{\text{phase0}} = \text{tokens\_generated} \times q \) where \( q \in [0,1] \) is a fixed quality pass rate (default \( q = 1 \) for deterministic micro-benchmark kernels with known-good output). For non-token benchmarks, \( U_{\text{phase0}} = 1 \) per successful benchmark completion.

**Denominator decomposition** (required where applicable):

```text
J_measured = J_ane + J_cpu_orchestration + J_dequant + J_other
```

- `J_dequant`: incremental joules of int4/int8 KV dequant path vs FP16 baseline at same context (Benchmark 3)
- `J_cpu_orchestration`: CPU domain joules during graph dispatch, Python agent loop, token bookkeeping minus attributed ANE idle gaps (Benchmark 4)

**Invalid without artifacts**: No IPJ\(_{phase0}\) value may be published without:
1. Raw `powermetrics` log file (`logs/<experiment_id>.powermetrics.txt`)
2. Thermal record: `temp_start_c`, `temp_steady_state_c`, `time_to_throttle_s` (or null), `sustained_power_w`
3. JSONL row per `IPJ_Measurement_Protocol_Alalā.md` §3.4

### 2.2 SRAM Cliff Detection (Phase 0)

Methodology (Benchmark 2):
1. Fix model, batch size = 1, ANE-first routing.
2. Sweep context length \( L \in \{512, 1024, 2048, 4096, 8192, \ldots\} \).
3. At each \( L \), record **sustained** `tokens_per_second_sustained` at thermal steady state.
4. **SRAM cliff** = smallest \( L \) where sustained throughput drops ≥30% vs. previous step **and** ANE utilization drops or unified-memory bandwidth proxy rises.

Document cliff \( L_{\text{cliff}} \) in JSONL and Program Board.

### 2.3 Harness Execution (`m4_energy_harness.py`)

The protocol maps directly to harness modes (see `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`):

| Harness `--mode` | Benchmark | IPJ numerator | Required logs |
|------------------|-----------|---------------|---------------|
| `thermal_baseline` | 1 | 1 (successful run) | powermetrics 1 Hz; thermal curve |
| `sram_cliff` | 2 | tokens × q per context step | per-step powermetrics + cliff detection |
| `kv_comparison` | 3 | tokens × q; FP16 vs int4 delta | `energy_dequant_joules` delta |
| `orchestration` | 4 | tokens × q or op count | `energy_cpu_orchestration_joules` |
| `ane_utilization` | E1 | forward passes × q | `ane_compute_fraction_pct`, orchestration tax |
| `thermal_ipj_curve` | E2 | tokens × q per time window | IPJ vs. thermal headroom time series |
| `meta_tax` | E3 | net useful work delta | `energy_meta_total_joules`, `net_ipj_delta` |
| `memory_spill` | E4 | tokens × q per context tier | spill vs. recompute joules/token |

Harness responsibilities (implementation spec):
- Spawn `powermetrics` subprocess; write raw log to `logs/<experiment_id>.powermetrics.txt`
- Emit one JSONL summary line per run to `logs/<experiment_id>.jsonl`
- Abort if package temperature exceeds configurable safe sustained threshold (default: human-set on first thermal baseline)
- Never run on non-M4 hosts (detect Apple Silicon + 24 GB)

### 2.4 Gap-Closing Experiments (E1–E4) — IPJ Requirements

These extend Phase 0 Week 1 tasks (`Phase0_AI_Coder_Task_List.md` § Phase 0 Extended). All run on **physical Mac Mini M4 24 GB**; stop if temperature exceeds safe sustained threshold.

#### E1 – ANE Real Utilization Baseline
- **First-class IPJ components**: `ane_compute_fraction_pct` (% forward-pass wall time on ANE), `ane_utilization_pct`, `energy_cpu_orchestration_joules`, `orchestration_tax_pct` (= orchestration joules / total joules).
- IPJ\(_{phase0}\) for E1 uses tokens or forward-pass count as numerator; denominator must separate ANE vs. orchestration joules.
- **Gate**: publish `ane_compute_fraction_pct` alongside every IPJ claim for ANE-first workloads.

#### E2 – Sustained Thermal + IPJ Degradation Curve
- **Thermal headroom is an explicit independent variable**: log `thermal_headroom_c` (margin to safe sustained threshold) per time window.
- IPJ is **only valid within the measured safe sustained envelope** — annotate each JSONL row with `thermal_envelope_valid: true|false`.
- Required series: `ipj` vs. elapsed minutes; flag `ipj_degradation_pct` from first steady-state window to post-throttle window.
- **Gate**: do not compare IPJ across runs unless thermal headroom conditions are stated and comparable.

#### E3 – Closed-Loop Meta-Tax Measurement
- **Full meta-overhead accounting** required in denominator:
  - `energy_meta_propose_joules`, `energy_meta_evaluate_joules`, `energy_meta_apply_joules`
  - `energy_meta_total_joules` = sum of above
  - `energy_saved_subsequent_joules` = baseline joules − post-change joules over amortization window
  - `net_ipj_delta` = (Δ useful work / Δ joules) accounting for meta total
- **Gate**: self-improvement cadence blocked until `net_ipj_delta` > 0 at sustained thermal conditions.

#### E4 – Memory Pressure & Spill Cost
- Log `working_set_mb`, `context_length`, unified-memory bandwidth utilization proxy (if available).
- **Spill vs. recompute**: `energy_spill_joules_per_token`, `energy_recompute_joules_per_token`, `spill_vs_recompute_delta`.
- Attribute spill energy to data movement component in §3.3 when ANE on-chip SRAM (~28–30 MB) is exceeded.
- **Gate**: hierarchical memory design changes require E4 numbers before adoption.

### 2.5 Lab Principle (Non-Negotiable)

**No IPJ claim is accepted without raw `powermetrics` + thermal logs and a clear statement of thermal headroom conditions** (`temp_start_c`, `temp_steady_state_c`, `thermal_headroom_c`, safe sustained threshold used, `thermal_envelope_valid`).

## 3. Measurement Infrastructure (M4)

### 3.1 Energy Logging — Physical M4 Only
- **Primary tool**: `powermetrics` on the Mac Mini M4 (CPU + GPU + ANE domain power; package/die temperature) — requires local execution, often `sudo`
- **No simulation or remote hosts** for IPJ claims; cloud or x86 numbers are not transferable due to unified memory coherence and ANE SRAM physics
- **Secondary** (higher precision when available): External measurement (e.g. Joulescope-style) for calibration runs only
- Logging frequency: Every experiment + continuous background monitoring during sustained (5–10 min) thermal steady-state runs

### 3.2 Thermal Headroom (First-Class)
Every measurement run must log:
- Start temperature (°C) before load
- Steady-state temperature after thermal plateau
- Time-to-throttle (seconds from load start to first DVFS-induced throughput drop, if any)
- Safe sustained power envelope (W) — the power level maintainable without exceeding safe sustained temperature

**Rule**: Thermal headroom and sustained IPJ take precedence over peak throughput.

### 3.3 Energy Accounting Components (M4)
IPJ denominators must be decomposable where possible:
| Component | What to measure | Why on M4 |
|-----------|-----------------|-----------|
| ANE compute | ANE domain joules via `powermetrics` | Primary efficient path (~6.6 TFLOPS/W when fed) |
| CPU orchestration | CPU domain joules during graph dispatch, Python loops, token bookkeeping | Minimized but non-zero; must not be hidden in ANE totals |
| Dequantization | Incremental joules for int4/int8 KV dequant vs FP16 baseline | Spill avoidance can win; dequant cost can erode gains — measure both |
| Data movement | Proxy via bandwidth-saturated runs or memory-tier telemetry | Unified memory traffic dominates when SRAM budget exceeded |

### 3.4 Structured Logging Format (JSONL)

```json
{
  "timestamp": "2026-06-30T08:00:00Z",
  "experiment_id": "phase0_sram_cliff_001",
  "task_type": "decode",
  "model": "baseline",
  "context_length": 2048,
  "energy_joules": 12.34,
  "energy_ane_joules": 8.1,
  "energy_cpu_orchestration_joules": 2.4,
  "energy_dequant_joules": 0.9,
  "tokens_generated": 256,
  "tokens_per_second": 18.7,
  "tokens_per_second_sustained": 16.2,
  "ane_utilization_pct": 72.4,
  "temp_start_c": 42.1,
  "peak_temp_c": 68.2,
  "temp_steady_state_c": 65.8,
  "time_to_throttle_s": null,
  "sustained_power_w": 18.5,
  "u_task_score": 0.87,
  "ipj": 0.021,
  "hca_impact": 0.05,
  "notes": "Fused int4 KV test",
  "powermetrics_log_path": "logs/phase0_sram_cliff_001.powermetrics.txt",
  "thermal_headroom_c": 8.4,
  "thermal_envelope_valid": true,
  "ane_compute_fraction_pct": 74.2,
  "orchestration_tax_pct": 18.5
}
```

**Rule**: No IPJ claim is valid without raw `powermetrics` logs + thermal data attached to the result (file path or inline archive) **and** stated thermal headroom conditions (§2.5).

### 3.5 Artifact Validation (`harness/validate_artifact.py`)

Before publishing any IPJ value:

```bash
python harness/validate_artifact.py logs/<experiment_id>.jsonl
python harness/validate_artifact.py --require-m4 logs/<experiment_id>.jsonl  # physical M4 only
```

Track criterion closure in `results/measurement_status.json`.

## 4. Gating Rules

| Decision | IPJ Requirement | HCA Requirement |
|----------|-----------------|-----------------|
| Accept self-improvement change | Marginal IPJ > 0 | HCA impact ≥ 0 |
| Keep change long-term | Sustained IPJ improvement | No degradation |
| Rollback | Significant IPJ regression | HCA degradation |

## 5. Reporting

Every week, a summary report must be generated showing:
- Average IPJ across representative workloads
- ANE utilization trends
- Thermal and power behavior
- Any self-improvement cycles attempted and their IPJ/HCA outcomes

This protocol makes IPJ a first-class, auditable metric rather than a vague aspiration.
