# How to Run the First Micro-Benchmark on M4 — Alalā

**Version**: 1.1  
**Target**: Mac Mini M4 24 GB (physical hardware only)  
**Purpose**: Establish baseline measurement capability for sustained IPJ, ANE utilization, thermal behavior, and SRAM limits.

**Execution constraint**: All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold.

**Rule**: No IPJ claim is valid without raw `powermetrics` logs + thermal data attached to the result (`IPJ_Measurement_Protocol_Alalā.md` §2.1).

## Prerequisites

- **Physical Mac Mini M4 24 GB** — no simulation, no remote SSH to non-M4 hosts, no cloud instances
- macOS with `powermetrics` access (usually requires `sudo`)
- Python 3.11+ with `numpy` (optional `matplotlib` for plotting)
- Energy logging harness: `harness/m4_energy_harness.py` (Phase 0 implementation)
- Stable thermal state: machine idled 10+ minutes before any benchmark
- Safe sustained temperature threshold: set from first `thermal_baseline` run (document in Program Board)

## IPJ for Phase 0 (Operational)

Per `IPJ_Measurement_Protocol_Alalā.md` §2.1:

$$\text{IPJ}_{\text{phase0}} = \frac{\text{useful work completed}}{J_{\text{measured}}}$$

- **Numerator**: tokens generated (× quality pass rate \( q \)) or successful benchmark completion
- **Denominator**: joules from `powermetrics` (CPU + GPU + ANE) at thermal steady state
- **Decompose** where applicable: `energy_ane_joules`, `energy_cpu_orchestration_joules`, `energy_dequant_joules`

## Step-by-Step Instructions

### 1. Prepare Environment

```bash
cd alala
mkdir -p experiments logs harness results
# harness/m4_energy_harness.py — implement per IPJ protocol §2.3
chmod +x harness/m4_energy_harness.py
```

### 2. Run Thermal Baseline (Required First)

Establishes safe sustained power envelope before other benchmarks.

```bash
sudo python harness/m4_energy_harness.py --mode thermal_baseline --duration 600
```

**Expected Output:**
- `logs/thermal_baseline_<id>.powermetrics.txt` (raw, required)
- `logs/thermal_baseline_<id>.jsonl` with: `temp_start_c`, `temp_steady_state_c`, `time_to_throttle_s`, `sustained_power_w`
- **Stop** if temperature exceeds safe sustained threshold; record threshold in Program Board

### 3. Run ANE SRAM Cliff Characterization

ANE-first routing; batch size = 1.

```bash
sudo python harness/m4_energy_harness.py --mode sram_cliff --model baseline --max_context 8192
```

**Expected Output:**
- Per-context JSONL rows + attached powermetrics logs
- Documented \( L_{\text{cliff}} \): context length where sustained throughput drops ≥30% (SRAM cliff)
- Plot or table: sustained `tokens_per_second` vs context length

### 4. Run Fused int4 KV vs FP16 Comparison

Includes dequantization energy accounting.

```bash
sudo python harness/m4_energy_harness.py --mode kv_comparison --context 2048 --iterations 50
```

**Expected Output:**
- FP16 and int4 runs each with powermetrics log
- `energy_dequant_joules` delta, sustained IPJ delta (not peak burst)
- Reject config if dequant + spill erodes theoretical bandwidth savings

### 5. Run Orchestration Overhead Profile

Measures CPU energy/latency outside ANE execution.

```bash
sudo python harness/m4_energy_harness.py --mode orchestration --iterations 100
```

**Expected Output:**
- `energy_cpu_orchestration_joules` vs `energy_ane_joules`
- ANE utilization under ANE-first routing vs GPU-default (if comparable)

## Failure Modes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `powermetrics` permission denied | Insufficient privileges | Run with `sudo` |
| Very noisy energy readings | Other load or not thermally stable | Close apps; idle 10+ min; use sustained metrics |
| No visible SRAM cliff | Workload too small or GPU-routed | Increase context; enforce ANE-first |
| Temperature too high | Exceeded safe sustained envelope | **Stop**; reduce duration or precision; re-run thermal baseline |
| Harness crashes | Missing deps or not on M4 | `pip install numpy`; verify Apple Silicon 24 GB |
| IPJ without powermetrics file | Incomplete logging | Re-run; attach `powermetrics_log_path` |

## Validate Artifacts (required before IPJ claims)

```bash
python harness/validate_artifact.py logs/<experiment_id>.jsonl
python harness/validate_artifact.py --require-m4 logs/<experiment_id>.jsonl
```

Update `results/measurement_status.json` with `m4_validated: true` and artifact path after each criterion closes.

## Success Criteria (Phase 0)

Measurable M4 numbers (not estimates):
- Thermal baseline curve and safe sustained power envelope logged
- SRAM cliff context length \( L_{\text{cliff}} \) documented with ≥30% sustained throughput drop
- int4 vs FP16 IPJ delta including `energy_dequant_joules`
- CPU orchestration overhead quantified (`energy_cpu_orchestration_joules`)

## Next Steps

Update `OSLab_Program_Board.md` with results. Proceed to Week 2 tasks in `Phase0_AI_Coder_Task_List.md`.
