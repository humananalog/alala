# Alalā Physics-Corrected Foundation (v2)

**Date**: 2026-06-30  
**Status**: Authoritative living document — supersedes earlier high-level physics sections where conflicting.  
**Repository**: https://github.com/humananalog/alala

This document incorporates the rigorous first-principles diagnosis and corrections for building Alalā on M4 24 GB.

## 1. Core Diagnosis (Physics-Grounded Weaknesses)

The original framing had several systemic issues that violate M4 physics realities:

1. **Non-operational IPJ Metric**  
   "Useful cognitive work per joule" was not formalized. Without a measurable $U(\text{task})$ and IPJ = $\mathbb{E}[U]/\mathbb{E}[J]$, self-improvement cannot be validated or amortized.

2. **Insufficient ANE Co-Design**  
   Standard frameworks (MLX, llama.cpp) are GPU-heavy by default. ANE utilization is often near zero in hybrid runs. Sequential decode destroys the ANE's efficiency advantage (~6.6 TFLOPS/W with hard power gating).

3. **KV Cache & Memory Ignore SRAM Physics**  
   FP16 KV cache + working sets > ~30 MB cause DRAM spills on the ANE's limited on-chip SRAM. No fused int4 KV with register-level dequantization (proven faster than FP16 on Apple Silicon due to bandwidth savings).

4. **Orchestration Overhead Not Minimized**  
   Agent loops multiply expensive forward passes. Irregular work is not kept on CPU at absolute minimum. Verification re-uses heavy LLM calls instead of lightweight ANE/AMX operators.

5. **Thermal Headroom Treated as Afterthought**  
   No closed-loop scheduler using real temperature/power feedback to modulate batch size, precision, recursion depth, or self-modification budget.

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

### 2.2 ANE-First + SRAM Budgeting

- Route all compute-bound operations to the ANE by default.
- Keep active working sets (KV cache + activations + weights) under ~28–30 MB when possible.
- Use fused low-precision KV cache (int4/int8) with register-level or tile-level dequantization.
- Apply SRAM-aware tiling and recomputation where DRAM spills would otherwise occur.

### 2.3 Thermal-Aware Scheduling

Thermal headroom is a first-class scheduling variable:
- Monitor real temperature via `powermetrics`.
- Reduce batch size, precision, or recursion depth when thermal headroom is low.
- Accept lower peak speed if it enables significantly better sustained performance and lower energy per useful output.

### 2.4 Minimal Orchestration

- Keep irregular control flow and decision logic on CPU.
- Push regular, compute-heavy work to compiled ANE-friendly graphs.
- Minimize Python-level orchestration in the hot path.

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
