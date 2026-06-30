# IPJ Measurement Protocol — Alalā

**Version**: 1.0  
**Purpose**: Define how Intelligence per Joule (IPJ) is measured, logged, and used as a gating criterion.

## 1. Formal Definition

$$
\text{IPJ} = \frac{\mathbb{E}[U(\text{task})]}{\mathbb{E}[J]}
$$

Where:
- \( U(\text{task}) \) = Composite utility of completing the task
- \( J \) = Energy consumed (joules)

## 2. Composite Utility \( U(\text{task}) \)

\( U(\text{task}) \) includes:
1. **Task Success** (correctness, quality, usefulness)
2. **Verification Quality** (HFPS from HCA spec)
3. **Future Value** (how much this outcome improves future self-improvement or capability)
4. **Efficiency** (tokens per joule, latency, etc.)

Each dimension is scored on a 0–1 scale and combined with weights that can evolve over time (initially conservative).

## 3. Measurement Infrastructure

### 3.1 Energy Logging
- Primary tool: `powermetrics` (CPU + GPU + ANE power)
- Secondary (higher precision when available): External measurement (e.g. Joulescope-style)
- Logging frequency: Every experiment + continuous background monitoring during long runs

### 3.2 Structured Logging Format (JSONL)

```json
{
  "timestamp": "2026-06-30T08:00:00Z",
  "experiment_id": "phase0_sram_cliff_001",
  "task_type": "decode",
  "model": "baseline",
  "context_length": 2048,
  "energy_joules": 12.34,
  "tokens_generated": 256,
  "tokens_per_second": 18.7,
  "ane_utilization_pct": 72.4,
  "peak_temp_c": 68.2,
  "u_task_score": 0.87,
  "ipj": 0.021,
  "hca_impact": 0.05,
  "notes": "Fused int4 KV test"
}
```
