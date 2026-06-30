# Revised Phase 0–2 Systems & Compiler Plan — Alalā

**Version**: 2.0  
**Date**: 2026-06-30  
**Purpose**: Updated systems and compiler roadmap based on physics-first constraints and M4 realities.

## Overall Philosophy

We treat the M4 as a heterogeneous system with real physical constraints (ANE efficiency, SRAM limits, thermal headroom, dispatch characteristics). Our goal is not to match cloud-scale peak performance, but to extract maximum **sustained useful cognitive work per joule**.

We follow an "xAI-style" low-level optimization mindset, adapted to the realities of Apple Silicon:
- Remove unnecessary abstraction where it costs performance.
- Make data movement and orchestration cost visible and minimized.
- Co-design the compiler/runtime with the hardware instead of fighting it.

## Phase 0: Characterization (Current)

**Focus**: Understand the actual behavior of the M4 under realistic workloads.

**Key Activities**:
- Thermal and power baseline
- SRAM cliff characterization
- Fused low-precision KV gains measurement
- Dispatch and orchestration overhead profiling

**Exit Gate**: Reliable measurement infrastructure + quantified understanding of the main bottlenecks.

## Phase 1: Baseline Compiler Passes

**Focus**: Implement the highest-impact compiler passes identified in Phase 0.

**Priority Passes**:
1. **Shape Specialization + Fixed-Shape Enforcement** (highest immediate impact)
2. **Fused Low-Precision KV + Attention**
3. **Basic SRAM-Aware Tiling**

**Success Criteria**:
- Measurable improvement in decode IPJ or tokens/s from the passes.
- Clear reduction in data movement compared to baseline.

## Phase 2: Advanced Co-Design + Self-Improvement

**Focus**: Combine compiler advances with bounded self-improvement.

**Key Work**:
- Full SRAM-aware tiling + selective recomputation
- Compute + prefetch pipelining (double buffering)
- Lightweight meta-controller with HCA + IPJ gating
- First closed-loop thermal-aware scheduling

**Success Criteria**:
- Sustained ANE utilization significantly above baseline on representative workloads.
- Self-improvement cycles that deliver positive marginal IPJ while respecting HCA.

## Phase 3: Production Hardening (Future)

**Focus**: Make the system robust for long-horizon agentic work.

**Planned Work**:
- Robust memory manager with rollback
- Advanced verification and process supervision
- Long-running agent workflows with stable IPJ

## Cross-Cutting Principles

| Principle | How It Shows Up |
|-----------|-----------------|
| ANE-First | Default routing of compute-bound ops to ANE |
| SRAM Budgeting | Active working sets kept under ~28–30 MB when possible |
| Minimal Orchestration | Irregular work on CPU, regular work compiled for ANE |
| Measurement-Driven | No major change without before/after IPJ and utilization data |
| Thermal Awareness | Temperature is a first-class scheduling variable |

This plan replaces earlier optimistic assumptions with a grounded, measurement-first approach that respects the actual physics of the M4.
