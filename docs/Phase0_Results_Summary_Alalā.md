# Phase 0 Results Summary — Alalā on Mac Mini M4 24 GB

**Date**: 2026-06-30 (execution) / 2026-07-01 (synthesis)  
**Hardware**: Mac Mini M4 24 GB unified memory  
**Status**: Phase 0 COMPLETE — Measurement infrastructure validated on real silicon

## Executive Summary

Phase 0 successfully established the full measurement harness (`m4_energy_harness.py` + `powermetrics` integration) and executed the four core microbenchmarks on the target M4 hardware. All claims are backed by raw `powermetrics` logs and thermal data per the IPJ Measurement Protocol.

**Key Measured Outcomes**:
- Thermal steady-state ~82.7–86.5 °C with throttling onset in ~5 s under decode load.
- SRAM cliff begins at **L_cliff = 1024** context (33.7% sustained throughput drop from 9.65 → 6.40 t/s).
- Orchestration overhead is low: **3.7–4.3%** of total energy even with Python-style dispatch.
- int4 KV adds only **+5.55 J** dequant energy (~0.5% overhead); IPJ nearly neutral (ΔIPJ = −0.0028).
- Current MLX workloads route primarily to GPU; ANE utilization reported ~0% — major Phase 1 priority.

## Detailed Results

### 1. Thermal Baseline
- Start: 46.75 °C
- Steady-state: 82.72 °C | Peak: 83.75 °C
- Throttling after 5.03 s
- Sustained power: 11.85 W
- Proposed safe envelope for long runs: **≤ 85 °C** steady-state with active monitoring.

**Physics implication**: Thermal headroom is limited. Any sustained agentic workload must incorporate duty-cycling or power capping.

### 2. SRAM Cliff Characterization
Context sweep (batch=1, sustained thermal state):

| Context | Sust. t/s | Drop     | Steady Power | Peak Temp | IPJ    |
|---------|-----------|----------|--------------|-----------|--------|
| 512     | 9.65      | —        | 16.86 W      | 82.45 °C  | 0.550  |
| 1024    | 6.40      | **−33.7%** | 17.46 W    | 82.84 °C  | 0.314  |
| 2048    | 3.73      | −41.7%   | 17.06 W      | 85.25 °C  | 0.167  |
| 4096    | 2.13      | −42.8%   | 17.16 W      | 86.50 °C  | 0.079  |
| 8192    | 1.60      | −25.0%   | 17.04 W      | 81.01 °C  | 0.035  |

**L_cliff = 1024**. This is the first context length where sustained throughput drops ≥30%. Degradation continues with each doubling — classic unified memory spill once on-chip ANE SRAM (~28–30 MB working set) is exceeded.

**Architecture implication**: Long-context agent work will be severely memory-bandwidth bound. Hierarchical KV, recent-token ring buffers in fast memory, or aggressive summarization/compression are required.

### 3. KV Cache Quantization (int4 vs FP16)
Short-context decode (below cliff):
- FP16: 9.633 t/s | 1062.9 J | IPJ 0.542
- int4: 9.867 t/s (+2.4%) | 1068.45 J | IPJ 0.539
- Incremental dequant energy: **+5.546 J**

Dequant cost is low but currently erodes the theoretical win. Fusion is partially effective. Opportunity remains to improve the kernel path.

### 4. Orchestration Overhead
- Tight MLX loop vs delayed Python dispatch: CPU orchestration energy only **39–45 J** (~3.7–4.3% of total).
- Throughput and IPJ almost identical between tight and delayed modes.

**Positive result**: Agent-style orchestration tax is manageable on this hardware + MLX stack. We can keep irregular control-plane work on CPU without destroying sustained IPJ.

## Physics-Grounded Lessons for Alalā

1. **Thermal headroom is first-class**. Design all long-running loops with explicit thermal monitoring and back-off.
2. **Memory hierarchy dominates efficiency** beyond ~1 k context. The model architecture and KV system must be co-designed around the SRAM cliff.
3. **Orchestration is not the primary enemy** — current overhead is acceptable.
4. **ANE residency is currently near zero** in MLX decode paths. Phase 1 must prioritize CoreML conversion or ANE-mapped MLX kernels to meet the "route as much forward pass as possible to ANE" mandate.
5. **Self-improvement loop can start small**. With low orchestration tax and repeatable IPJ measurement, we can implement a minimal "propose → benchmark → accept if sustained IPJ gain + HCA" cycle in Phase 1.

## Phase 1 Entry Criteria (Recommended)

- Fix cliff detector (done in this commit).
- Achieve measurable ANE utilization > 60% on a small seeding model (target: 350M–1B class distilled model converted via CoreML or ANE-friendly MLX path).
- Define safe operating region: context ≤ 1024 (or paged KV) + thermal duty cycle.
- Implement first bounded self-improvement micro-scaffold with IPJ gating.

All raw `powermetrics` logs and JSONL artifacts remain in `logs/` and `results/`.

---
*This summary is the canonical reference for Phase 0 outcomes. All future decisions on model architecture, KV system, and self-improvement must be justified against these measured constraints.*