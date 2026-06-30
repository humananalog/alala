# Phase 0 Microbenchmark Suite Plan — Alalā

**Version**: 1.0  
**Purpose**: Define the minimal set of micro-benchmarks needed to characterize the M4 for Alalā development.

## Objectives

1. Establish reliable baselines for IPJ, ANE utilization, thermal behavior, and SRAM limits.
2. Quantify the gains (or lack thereof) from fused low-precision KV cache.
3. Identify the practical SRAM cliff on real workloads.
4. Measure dispatch and orchestration overhead.

## Benchmark 1: Thermal & Power Baseline

**Goal**: Understand idle vs sustained load power and thermal rise curve.

**Method**:
- Idle for 10+ minutes
- Run sustained decode workload for 5–10 minutes
- Log power (CPU/GPU/ANE) and temperature every second

**Key Metrics**:
- Idle power
- Sustained load power
- Time to thermal steady state
- Peak temperature under load

## Benchmark 2: ANE SRAM Cliff Characterization

**Goal**: Find the context length where performance drops due to SRAM limits.

**Method**:
- Run decode workloads with increasing context lengths (512 → 8192+)
- Keep batch size = 1 and model fixed
- Measure tokens/second and energy per token

**Key Output**:
- Approximate context length where throughput drops ~25–30%

## Benchmark 3: Fused int4 KV vs FP16 Comparison

**Goal**: Quantify the benefit of low-precision KV cache.

**Method**:
- Same workload and context length
- Run once with FP16 KV cache
- Run once with fused int4 KV + on-the-fly dequantization
- Compare: tokens/s, energy/token, IPJ

**Success Criteria**:
- Clear, repeatable improvement in at least one dimension (preferably IPJ or tokens/s with similar energy).

## Benchmark 4: Dispatch & Orchestration Overhead

**Goal**: Measure how much time/energy is spent outside the ANE.

**Method**:
- Compare end-to-end latency vs pure ANE kernel time (where measurable)
- Profile Python-level overhead in a simple agent loop

**Key Insight**:
This helps decide how much effort to invest in reducing orchestration.

## Execution Order

1. Thermal Baseline (establishes safe operating region)
2. SRAM Cliff Characterization
3. Fused KV Comparison
4. Dispatch Overhead (can be done in parallel with others)

## Logging Requirements

All benchmarks must produce structured JSONL logs with at minimum:
- timestamp
- benchmark_name
- key metrics
- hardware state (temperature, power)
- notes

## Success Gate for Phase 0

By the end of Phase 0 we must have:
- Reliable thermal/power baseline
- Identified approximate SRAM cliff
- Quantified benefit (or lack of benefit) from fused low-precision KV
- Clear understanding of current orchestration overhead
