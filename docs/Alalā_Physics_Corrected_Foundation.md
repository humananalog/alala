# Alalā Physics-Corrected Foundation (v2)

**Date**: 2026-06-30  
**Status**: Authoritative living document — supersedes earlier high-level physics sections where conflicting.  
**Repository**: https://github.com/humananalog/alala

This document incorporates the rigorous first-principles diagnosis and corrections for building Alalā on **Mac Mini M4, 24 GB unified memory**.

## 0. M4 Silicon Realities (Non-Negotiable)

All design choices in Alalā must be traceable to these measured or well-characterized M4 properties:

| Property | M4 Reality | Design Implication |
|----------|------------|-------------------|
| **Unified memory** | CPU, GPU, and ANE share one 24 GB LPDDR5X pool (~100–120 GB/s effective bandwidth) with full cache coherence — **no PCIe copy tax** between accelerators | Pointer sharing is cheap; **bytes moved** (DRAM ↔ on-chip SRAM) dominate energy, not cross-device marshaling |
| **ANE on-chip SRAM** | ~28–30 MB fast SRAM per ANE tile; spill to unified memory when working set exceeds budget | KV cache + activations + weights must be budgeted; FP16 KV at long context forces DRAM spills |
| **ANE power gating** | Hard gating when idle; ~6.6 TFLOPS/W at sustained ANE utilization vs. much lower effective efficiency when underfed or idle | **ANE-first routing is the default**; idle gaps and dispatch latency waste the efficiency advantage |
| **Thermal envelope + DVFS** | Mac Mini M4 is thermally constrained under sustained ANE+CPU load; frequency and power scale down as die temperature rises | **Thermal headroom is a first-class optimization variable** — sustained useful work per joule under the thermal envelope **beats theoretical peak throughput** |
| **Data movement vs. compute** | On M4, memory bandwidth and spill traffic often dominate decode energy; fused int4 KV + register-level dequantization wins because it reduces bytes moved, not because int4 math is inherently faster | Minimize orchestration that triggers extra passes, copies, or materialization of full-precision tensors |

**Execution constraint**: All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold.

## 1. Core Diagnosis (Physics-Grounded Weaknesses)

The original framing had several systemic issues that violate M4 physics realities:

1. **Non-operational IPJ Metric**  
   "Useful cognitive work per joule" was not formalized. Without a measurable $U(\text{task})$ and IPJ = $\mathbb{E}[U]/\mathbb{E}[J]$, self-improvement cannot be validated or amortized.

2. **Insufficient ANE Co-Design**  
   Standard frameworks (MLX, llama.cpp) default to GPU paths on M4 unified memory. ANE utilization is often near zero in hybrid runs because graphs are not compiled for ANE tiles. Sequential decode with CPU orchestration between tokens destroys the ANE's efficiency advantage (~6.6 TFLOPS/W with hard power gating when fed continuously).

3. **KV Cache & Memory Ignore SRAM Physics**  
   FP16 KV cache + working sets > ~28–30 MB force spills from ANE on-chip SRAM into unified memory on every access — bandwidth-bound and high-energy on M4. No fused int4 KV with register-level dequantization (proven faster than FP16 on Apple Silicon because it cuts bytes moved across the SRAM cliff, not because int4 ALU is faster).

4. **Orchestration Overhead Not Minimized**  
   Agent loops multiply expensive forward passes across unified memory. Irregular control flow stays on CPU by necessity, but Python/CPU orchestration between ANE invocations is not measured or minimized — each gap risks ANE power-down. Verification re-uses heavy LLM calls instead of lightweight ANE/AMX operators.

5. **Thermal Headroom Treated as Afterthought**  
   M4 DVFS throttling under sustained ANE+CPU load reduces both throughput and IPJ, yet no closed-loop scheduler uses real `powermetrics` temperature/power feedback to modulate batch size, precision, recursion depth, or self-modification budget. Peak benchmark numbers taken before thermal steady state are misleading.

6. **Seeding Model & Self-Improvement Not Co-Designed for ANE**  
   Starting from standard dense transformers without 1×1 conv rewrites, fixed-shape enforcement, SRAM tiling, or delta-compilation paths wastes the hardware's strengths.

7. **No Calibrated First-Principles Energy Model**  
   Roofline + power + thermal + dispatch costs were not measured and modeled before major architectural decisions.

## 2. Corrected Foundational Principles

### 2.1 Operational IPJ Metric (Highest Priority)

Define:
$$
\text{IPJ} = \frac{\mathbb{E}[U(\text{task})]}{\mathbb{E}[J]}
$$

Where $U(\text{task})$ is a **composite utility** that includes:
- Immediate task success (reasoning, coding, planning accuracy)
- Verification quality (HFPS from HCA)
- **Future self-improvement delta** (amortized gain from the improvement cycle)

### 2.2 ANE-First + SRAM Budgeting (Default Path)

- **ANE routing is the default** for all compute-bound, regular tensor ops — GPU is fallback only when ANE compilation fails.
- Keep active working sets (KV cache + activations + weights) under ~28–30 MB ANE on-chip SRAM when possible; budget spill cost to unified memory explicitly.
- Use fused low-precision KV cache (int4/int8) with register-level or tile-level dequantization; account dequantization energy in IPJ (see `IPJ_Measurement_Protocol_Alalā.md`).
- Apply SRAM-aware tiling and recomputation where unified-memory spills would otherwise dominate energy.

### 2.3 Thermal Headroom as First-Class Variable

**Thermal headroom and sustained IPJ take precedence over peak throughput.**

- Monitor real die/package temperature and per-domain power via `powermetrics` on the physical M4 — not simulators or remote hosts.
- Log start temperature, steady-state temperature, time-to-throttle, and safe sustained power envelope for every benchmark.
- Reduce batch size, precision, or recursion depth when thermal headroom is low (DVFS active).
- Accept lower peak tokens/s if it yields higher **sustained** useful work per joule under the M4 thermal envelope.

### 2.4 Minimal Orchestration (Measured Overhead)

- Keep irregular control flow and decision logic on CPU — unavoidable on M4.
- Push regular, compute-heavy work to compiled ANE-friendly graphs to keep ANE power gating from idling tiles.
- Minimize Python-level orchestration in the hot path; **CPU orchestration energy and latency must be measured** (Benchmark 4 in `Phase0_Microbenchmark_Suite_Plan.md`) and subtracted from IPJ claims.

## 3. Phase Overview (Corrected)

**Phase 0**: ANE Characterization & Measurement Infrastructure (Current)
- Micro-benchmarks for SRAM cliff, fused KV gains, thermal behavior, and dispatch overhead.
- Build reliable energy logging harness.

**Phase 1**: Baseline Model + Compiler Passes
- Implement shape specialization, fused low-precision KV, and basic SRAM tiling.

**Phase 2**: Advanced Compiler Co-Design + Self-Improvement
- Full SRAM-aware tiling, pipelining, and bounded meta-controller.

**Phase 3**: Production Hardening
- Thermal-aware scheduler, robust verification, and long-horizon agent workflows.

## 4. Success Gates (Physics-First)

- Phase 0 Gate: Reliable measurement of IPJ, ANE utilization, and thermal behavior.
- Phase 1 Gate: Measurable decode IPJ improvement from compiler passes.
- Phase 2 Gate: Sustained ANE utilization > 70% on representative workloads with positive marginal IPJ on self-improvement cycles.

This foundation replaces earlier optimistic assumptions with constraints grounded in the actual physics of the M4.
