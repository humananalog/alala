# Hierarchical Memory Architecture — Alalā

**Version**: 1.0  
**Goal**: Define a memory system that respects M4 SRAM limits, minimizes data movement, and supports efficient self-improvement.

## Design Principles

1. **SRAM is precious** — Keep hot working sets inside the ~28–30 MB on-chip SRAM when possible.
2. **Data movement has cost** — Every byte moved between DRAM and fast memory costs energy and time.
3. **Recompute vs Store** — Sometimes it is cheaper to recompute than to store and reload.
4. **Tiered memory** — Exploit unified memory + SLC cache + ANE SRAM hierarchy.

## Memory Tiers (M4)

| Tier | Size | Latency | Bandwidth | Notes |
|------|------|---------|-----------|-------|
| ANE SRAM | ~28–32 MB | Very Low | Very High | Primary target for hot working sets |
| Unified Memory (DRAM) | 24 GB | Medium | High | Main storage for model weights and KV cache |
| SLC Cache | Several MB | Low | Very High | Hardware-managed last-level cache |
| Storage (SSD) | Hundreds of GB | High | Low | For checkpoints and long-term storage |

## Recommended Architecture

### Hot Path (Decode / Agent Loop)
- Keep current layer weights + KV cache slice in ANE SRAM when possible.
- Use **fused low-precision KV** to reduce footprint.
- Apply **SRAM-aware tiling** so that no single operation exceeds SRAM budget.

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
- Track current SRAM and DRAM usage.
- Decide when to tile, recompute, or spill.
- Provide clear telemetry for IPJ accounting.
- Support rollback of memory layout changes.

This architecture treats memory as a first-class optimization target rather than an afterthought.
