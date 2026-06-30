# Memory Access Pattern Guidelines — Alalā

**Version**: 1.0  
**Purpose**: Define recommended memory access patterns that respect M4 hardware constraints (especially SRAM limits and data movement cost).

## Core Guidelines

### 1. Keep Hot Working Sets in Fast Memory
- The ANE has roughly 28–32 MB of fast on-chip SRAM.
- Try to keep the active working set (current layer weights + KV cache slice + activations) inside this budget whenever possible.
- When the working set exceeds SRAM, performance drops sharply due to DRAM spills.

### 2. Prefer Fused Low-Precision KV Cache
- Store KV cache in int4 or int8 when possible.
- Perform dequantization on-the-fly (register-level or tile-level) instead of materializing full FP16/BF16 KV.
- This is one of the highest-leverage techniques for staying within SRAM limits during decode.

### 3. Use Tiling + Selective Recomputation
- When a single operation would exceed SRAM, split it into tiles.
- Recompute activations when the cost of storing + reloading them exceeds the cost of recomputation.
- This is especially useful for long context or memory-heavy agent trajectories.

### 4. Exploit Compute + Prefetch Overlap
- Use double buffering or explicit prefetching when possible.
- While the ANE is computing layer N, the runtime should be loading/preparing layer N+1 or the next KV slice.
- This hides memory latency behind useful compute.

### 5. Minimize Irregular Memory Access
- Irregular access patterns (e.g., scattered expert selection in MoE, complex control flow) are expensive.
- Keep the hot path as regular and predictable as possible so the hardware prefetchers and ANE scheduler can work effectively.

### 6. Be Explicit About Data Movement
- Every time data moves between DRAM and fast memory, it costs energy and time.
- Make these movements visible in the runtime and compiler (log or profile them).
- Treat data movement reduction as a first-class optimization target (on par with compute optimization).

## Practical Rules of Thumb

| Workload Type | Recommended Strategy | Why |
|---------------|----------------------|-----|
| Short decode | Fused int4 KV + keep everything in SRAM | Highest efficiency |
| Long context decode | Tiled execution + selective recomputation | Avoids DRAM spills |
| Agent loop with tools | Keep recent context + tool state hot | Reduces repeated loading |
| Self-improvement / training | Aggressive recomputation + streaming | Memory is the bottleneck |

## Anti-Patterns to Avoid

- Loading entire large layers or full KV cache into DRAM and hoping the hardware caches it efficiently.
- Frequent small irregular memory accesses in the hot path.
- Storing large intermediate activations when recomputation is cheaper.
- Ignoring thermal effects of sustained high-bandwidth memory traffic.

## Measurement

Any memory access pattern change should be evaluated with:
- IPJ before and after
- ANE utilization
- Measured data movement volume (if possible)
- Thermal impact

This document should be updated as we learn more from Phase 0 micro-benchmarks.
