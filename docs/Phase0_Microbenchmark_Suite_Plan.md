# Phase 0 Microbenchmark Suite Plan — Alalā

**Version**: 1.1  
**Purpose**: Define the minimal set of micro-benchmarks needed to characterize **Mac Mini M4 24 GB** silicon for Alalā development.

**Execution constraint**: All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold.

**Priority**: Thermal headroom and sustained IPJ take precedence over peak throughput. Benchmarks establish safe operating envelopes before optimization claims.

## Objectives

1. Establish reliable baselines for **sustained** IPJ, ANE utilization, thermal behavior, and SRAM limits on physical M4 hardware.
2. Quantify the gains (or lack thereof) from fused low-precision KV cache, including **dequantization energy cost**.
3. Identify the practical SRAM cliff (context length where throughput drops ~30% due to ANE on-chip SRAM exhaustion → unified memory spill).
4. Measure ANE vs. CPU orchestration energy and latency overhead.

## Benchmark 1: Thermal & Power Baseline

**Goal**: Map M4 idle vs sustained-load power and thermal rise curve; establish safe sustained power envelope before other benchmarks.

**Method** (physical M4 only):
- Idle 10+ minutes (ambient stable; log start temp)
- Run sustained ANE-first decode workload 5–10 minutes until thermal steady state
- Log `powermetrics` CPU/GPU/ANE power and package temperature every second
- Record time-to-throttle if DVFS reduces throughput

**Key Metrics**:
- Idle power (W)
- Sustained load power at thermal steady state (W) — not peak burst
- Time to thermal steady state (s)
- Peak and steady-state temperature (°C)
- Safe sustained power envelope (W maintainable without exceeding safe threshold)

## Benchmark 2: ANE SRAM Cliff Characterization

**Goal**: Find the context length where decode throughput drops ~30% because the active working set (KV + activations + weights) exceeds ~28–30 MB ANE on-chip SRAM and spills to unified memory.

**Method** (ANE-first routing; batch size = 1):
- Run decode workloads with increasing context lengths (512 → 8192+)
- Keep model and routing fixed; **ANE is default path**
- Measure sustained tokens/s and energy per token at thermal steady state
- Attach raw `powermetrics` + thermal logs per run

**SRAM cliff detection**: Plot tokens/s vs context length; cliff = first context length where sustained throughput drops ≥30% relative to the prior step, coinciding with measurable unified-memory bandwidth saturation or ANE utilization drop.

**Key Output**:
- Documented SRAM cliff context length (tokens) for the baseline model on M4 24 GB

## Benchmark 3: Fused int4 KV vs FP16 Comparison

**Goal**: Quantify IPJ delta from low-precision KV on M4, **including dequantization energy**.

**Method**:
- Same workload and context length (below SRAM cliff if possible; repeat above cliff to measure spill interaction)
- Run FP16 KV baseline (ANE-first)
- Run fused int4 KV + on-the-fly dequantization (ANE-first)
- Compare: sustained tokens/s, joules/token (total + dequant component), IPJ
- Log thermal steady-state conditions for both runs

**Success Criteria**:
- Repeatable improvement in IPJ or tokens/s at **similar or lower sustained energy**; reject configs where dequant + spill erode theoretical bandwidth savings.

## Benchmark 4: Dispatch & Orchestration Overhead

**Goal**: Measure CPU orchestration time and energy outside ANE execution on M4 unified memory — Python dispatch, graph compilation gaps, token bookkeeping.

**Method**:
- Compare end-to-end latency and joules vs pure ANE kernel time (where measurable via profiling)
- Profile a minimal agent loop: isolate CPU domain joules and wall time between ANE invocations
- Contrast ANE-first path vs GPU-default path for same op

**Key Insight**:
Orchestration overhead that idles ANE tiles (power gating) can dominate IPJ; minimize only after measuring.

## Execution Order

1. Thermal Baseline (establishes safe operating region)
2. SRAM Cliff Characterization
3. Fused KV Comparison
4. Dispatch Overhead (can be done in parallel with others)

## Logging Requirements

All benchmarks must produce structured JSONL logs (see `IPJ_Measurement_Protocol_Alalā.md`) with at minimum:
- timestamp, benchmark_name, key metrics
- hardware state: start/steady-state temperature, sustained power, ANE utilization
- energy breakdown: total, ANE, CPU orchestration, dequantization (where applicable)
- `powermetrics_log_path` — **no result valid without attached raw log**

Harness target: `harness/m4_energy_harness.py` (Phase 0 implementation).

### Harness ↔ Protocol Mapping

| `--mode` | Protocol section | Primary outputs |
|----------|------------------|-----------------|
| `thermal_baseline` | IPJ §2.1, Benchmark 1 | `sustained_power_w`, thermal curve, safe envelope |
| `sram_cliff` | IPJ §2.2, Benchmark 2 | `L_cliff`, per-context JSONL + powermetrics |
| `kv_comparison` | IPJ §2.1 denominator, Benchmark 3 | `energy_dequant_joules`, int4 vs FP16 IPJ delta |
| `orchestration` | IPJ §2.1, Benchmark 4 | `energy_cpu_orchestration_joules`, ANE vs CPU split |

See `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md` for exact commands.

## Success Gate for Phase 0

By the end of Phase 0 we must have **measured M4 numbers** (not estimates):
- Thermal baseline curve and safe sustained power envelope
- Documented SRAM cliff context length (~30% throughput drop)
- Quantified int4 vs FP16 IPJ delta including dequantization cost
- Measured CPU orchestration overhead (joules + latency) for baseline routing
