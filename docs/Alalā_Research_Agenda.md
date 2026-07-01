# Alalā Research Agenda

**Status**: Active (July 2026)  
**Purpose**: Track research threads, hypotheses, experiments, and prioritization for high-risk exploration toward hardware-co-designed on-device intelligence.

This document is intentionally lightweight and will evolve rapidly as we learn.

## Research Philosophy

- Deep and broad exploration with high risk tolerance.
- First-principles grounding in M4 / Apple Silicon constraints (ANE characteristics, unified memory, thermal/power limits, data movement costs).
- Focus on breakthroughs rather than incremental gains on existing stacks.
- Fast iteration and intellectual honesty — kill or pivot ideas quickly when evidence demands it.

## Current Research Threads

### 1. KV Cache / State Architecture Redesign (Highest Priority)

**Core Question**:  
Can we design a KV/state mechanism that is significantly more compatible with ANE constraints (static shapes, reduced dynamic I/O, better state handling) than standard growing causal mask + dynamic updates?

**Key Hypotheses**:
- Reducing dynamic shape changes and per-step KV I/O volume can unlock substantially higher effective ANE utilization.
- Ring buffer, sliding window, or paged KV representations may be more ANE-friendly than traditional growing caches.

**Early Experiment Ideas**:
- Implement and benchmark a ring-buffer / sliding window KV cache for decode.
- Test static-shape KV cache with fixed maximum context.
- Explore paged or chunked KV approaches adapted to ANE.

**Status**: In progress (Exp 1 complete — **Iterate**)  
**Priority**: Highest  
**Notes**:

- Literature review (July 2026): [`KV_Cache_Techniques_for_NPUs_and_Apple_Silicon.md`](KV_Cache_Techniques_for_NPUs_and_Apple_Silicon.md)
- Synthesis + experiments: [`KV_Cache_Redesign_Synthesis_and_Experiments.md`](KV_Cache_Redesign_Synthesis_and_Experiments.md)
- **Exp 1 — Ring buffer KV (2026-07-01)**: **38.3 t/s**, **6.7% ANE proxy**, **IPJ 3.23** vs mask int4 linear 27.7 t/s / 2.9% / 1.32 (`fc860526`). **0 re-prefills**; GPU J −97%. Decision: **Iterate** — ring512 int4 is default decode path; next: quality gate + consolidated MLState.

### 2. Disaggregated / Hybrid Execution Architecture

**Core Question**:  
Can a deliberately designed hybrid system (ANE for prefill/structured work + GPU/MLX for decode/dynamic work) deliver better overall sustained useful work per joule than trying to force everything onto one backend?

**Key Hypotheses**:
- Smart routing between execution paths, combined with efficient KV handoff via unified memory, can outperform both pure-ANE and pure-GPU approaches.
- Thermal and power awareness should be part of the routing logic.

**Early Experiment Ideas**:
- Build a thin execution layer that can switch between current best variants (mask int4 vs scatter clean).
- Measure combined efficiency and thermal behavior under different workloads.
- Explore automatic or heuristic-based path selection.

**Status**: Not started  
**Priority**: High  
**Notes**:

### 3. Lower-Level ANE Access and Runtime Techniques

**Core Question**:  
Can we achieve meaningfully better ANE utilization and efficiency by going beyond standard Core ML abstractions (inspired by approaches like Orion)?

**Key Hypotheses**:
- Direct or lighter-weight control over MIL generation, program caching, and scheduling can reduce overheads that Core ML currently imposes.
- Custom operator implementations or graph transformations may unlock better placement.

**Early Experiment Ideas**:
- Explore lighter custom MIL generation for critical decode paths.
- Investigate delta compilation or program caching techniques.
- Profile and reduce IOSurface / dispatch overhead where possible.

**Status**: Not started  
**Priority**: Medium-High  
**Notes**:

### 4. KV Cache Quantization and Compression

**Core Question**:  
Can aggressive KV cache quantization (int4/int8 and beyond) deliver significant efficiency gains on Apple Silicon unified memory without major quality loss?

**Key Hypotheses**:
- On unified memory architectures, certain KV quantization schemes can be throughput-positive or near-zero cost.
- Fused compressed-domain attention may further reduce memory movement.

**Early Experiment Ideas**:
- Implement and benchmark int4 KV cache (various schemes) on current best decode paths.
- Explore fused int4 SDPA kernels via Metal where beneficial.

**Status**: Not started  
**Priority**: Medium  
**Notes**:

### 5. Speculative Decoding and Parallel Techniques

**Core Question**:  
Can speculative decoding, KV runahead, or other parallelization techniques meaningfully reduce the number of expensive serial decode steps?

**Key Hypotheses**:
- Reducing the total number of decode steps can improve effective efficiency even if per-step ANE utilization remains moderate.
- These techniques can be combined with hybrid execution.

**Early Experiment Ideas**:
- Implement basic speculative decoding on top of current best paths.
- Explore KV-Runahead style parallel KV population for prefill.

**Status**: Not started  
**Priority**: Medium  
**Notes**:

## Prioritization Criteria

When deciding what to work on next, consider:
- Leverage on sustained useful work per joule
- Alignment with ANE / M4 physical constraints
- Learning value (even negative results are useful)
- Feasibility within solo research mode
- Potential to unlock new capability regimes

## How to Use This Document

- Add new threads as they emerge.
- Update status, notes, and priority regularly.
- Kill or deprioritize threads when evidence shows low return.
- Link to relevant experiment results and NOTES.md entries.

## Next Actions

- [ ] Finalize initial prioritization
- [ ] Start first high-priority thread (KV Cache / State redesign)
- [ ] Schedule regular review points to kill/pivot directions