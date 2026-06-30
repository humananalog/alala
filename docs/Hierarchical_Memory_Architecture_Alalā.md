# Hierarchical Memory Architecture — Alalā

**Version**: 1.1  
**Goal**: Define a memory system that respects **Mac Mini M4 24 GB** unified memory and ANE SRAM limits, minimizes data movement energy, and supports efficient self-improvement.

**Execution constraint**: All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold.

## Design Principles

1. **ANE on-chip SRAM is precious** — ~28–30 MB per ANE tile; exceeding this spills to unified memory (LPDDR5X, ~100–120 GB/s) on every hot-path access.
2. **Data movement dominates energy on M4 decode** — Unified memory has no PCIe tax between CPU/GPU/ANE, but **bytes moved** (especially SRAM ↔ DRAM) cost joules; minimize movement over peak FLOPS.
3. **Recompute vs Store** — On M4, recomputing a tile in ANE SRAM is often cheaper than storing to unified memory and reloading across the SRAM cliff.
4. **Tiered memory on single pool** — CPU, GPU, and ANE share one coherent 24 GB unified memory pool plus hardware SLC and ANE-local SRAM.

## Memory Tiers (Mac Mini M4 24 GB)

| Tier | Size | Latency | Bandwidth | M4 Notes |
|------|------|---------|-----------|----------|
| ANE on-chip SRAM | ~28–30 MB | Very Low | Very High | **Primary target** for hot working sets; spill → unified memory is the SRAM cliff |
| Unified Memory (LPDDR5X) | 24 GB | Medium | ~100–120 GB/s | Shared by CPU, GPU, ANE with full coherence — no PCIe copy between accelerators |
| SLC (system last-level cache) | Several MB | Low | Very High | Hardware-managed; do not rely on it to fix SRAM budget violations |
| Storage (SSD) | Hundreds of GB | High | Low | Checkpoints only; loading to unified memory has energy cost |

**24 GB working-set pressure**: Model weights + KV cache + activations + harness overhead must fit in unified memory; long-context KV growth competes with compilation buffers and measurement harness.

## Recommended Architecture

### Hot Path (Decode / Agent Loop) — ANE-First
- **Default route**: compiled ANE graphs; keep current layer weights + KV cache slice in ANE on-chip SRAM when possible.
- Use **fused low-precision KV** (int4/int8) with register-level dequantization; budget dequant energy in IPJ.
- Apply **SRAM-aware tiling** so no single op exceeds ~28–30 MB; sustained throughput under thermal envelope > burst peak past cliff.

### Cold Path (Self-Improvement / Training)
- Accept higher DRAM usage.
- Use recomputation aggressively for activations.
- Stream weights from DRAM → ANE SRAM using double-buffering / prefetching.

### Long-term Storage
- Model checkpoints and improvement history stored on SSD.
- Only load what is needed for the current phase of work.

## Key Techniques

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| **Fused KV Cache** | Store KV in int4/int8, dequantize on the fly | Decode phase (default) |
| **SRAM Tiling + Recomputation** | Split operations so working set fits in SRAM | When DRAM spills would occur |
| **Double Buffering / Prefetch** | Overlap computation of layer N with loading of layer N+1 | Long context or large models |
| **Paged / On-demand Loading** | Only load experts or layers that are actually used | Mixture-of-Experts or very large models |
| **Activation Recomputation** | Recompute activations instead of storing them | Memory-bound training or long agent trajectories |

## Memory Manager Responsibilities

The memory manager (to be implemented) should:
- Track ANE SRAM vs unified memory usage against ~28–30 MB budget.
- Decide when to tile, recompute, or spill — spill is a last resort with logged energy cost.
- Provide telemetry for IPJ accounting (bytes moved, dequant joules, orchestration overhead).
- Support rollback of memory layout changes.

**Thermal coupling**: Sustained high-bandwidth unified memory traffic raises package temperature and triggers DVFS; memory layout choices must be evaluated at thermal steady state, not cold-start peaks.

This architecture treats memory as a first-class optimization target rather than an afterthought.
