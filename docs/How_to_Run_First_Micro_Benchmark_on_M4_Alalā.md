# How to Run the First Micro-Benchmark on M4 — Alalā

**Version**: 1.0  
**Target**: Mac Mini M4 24GB  
**Purpose**: Establish baseline measurement capability for IPJ, ANE utilization, thermal behavior, and SRAM limits.

## Prerequisites

- macOS with `powermetrics` access (usually requires `sudo` or proper entitlements).
- Python 3.11+ with `numpy`, `matplotlib` (optional for plotting).
- The energy logging harness: `harness/m4_energy_harness.py`.
- Stable thermal state (machine idled for 10+ minutes).

## Step-by-Step Instructions

### 1. Prepare Environment

```bash
cd alala
mkdir -p experiments logs harness results
cp harness/m4_energy_harness.py harness/
chmod +x harness/m4_energy_harness.py
```

### 2. Run Thermal Baseline (Recommended First Step)

```bash
python harness/m4_energy_harness.py --mode thermal_baseline --duration 120
```

**Expected Output:**
- A JSONL log in `logs/thermal_baseline_YYYYMMDD_HHMMSS.jsonl`
- Key metrics: idle power, sustained load power, peak temperature, time to thermal steady state.

### 3. Run ANE SRAM Cliff Characterization

```bash
python harness/m4_energy_harness.py --mode sram_cliff --model baseline --max_context 8192
```

**Expected Output:**
- Throughput (tokens/s) vs context length plot or table.
- Approximate context length where performance drops ~30% (SRAM cliff indicator).

### 4. Run Fused int4 KV vs FP16 Comparison

```bash
python harness/m4_energy_harness.py --mode kv_comparison --context 2048 --iterations 50
```

**Expected Output:**
- Energy per token and tokens/second for both paths.
- Rough IPJ delta between FP16 and fused int4 KV.

## Failure Modes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `powermetrics` permission denied | Insufficient privileges | Run with `sudo` or configure proper entitlements |
| Very noisy energy readings | Machine under other load | Close other apps, wait for thermal stabilization |
| No visible SRAM cliff | Workload too small or not ANE-bound | Increase context length or use larger model |
| Harness crashes | Missing Python dependencies | `pip install numpy matplotlib` |

## Success Criteria

- You can reliably log power, temperature, and performance metrics.
- You have identified the approximate SRAM cliff point.
- You have quantified a measurable difference between FP16 and fused int4 KV paths.

## Next Steps

Once these baselines are established, move to Week 2 tasks in `Phase0_AI_Coder_Task_List.md`.
