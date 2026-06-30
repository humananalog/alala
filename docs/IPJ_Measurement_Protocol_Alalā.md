# IPJ Measurement Protocol — Alalā

**Version**: 1.1  
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
  "powermetrics_log_path": "logs/phase0_sram_cliff_001.powermetrics.txt"
}
```

**Rule**: No IPJ claim is valid without raw `powermetrics` logs + thermal data attached to the result (file path or inline archive).

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
