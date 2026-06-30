# Memory Access Pattern Guidelines — Alalā

**Version**: 1.1  
**Purpose**: Define recommended memory access patterns for **Mac Mini M4 24 GB** — ANE on-chip SRAM limits, unified memory bandwidth, and data-movement energy cost.

**Execution constraint**: All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold.

## Core Guidelines

### 1. Keep Hot Working Sets in ANE On-Chip SRAM
- The ANE has ~28–30 MB of fast on-chip SRAM per tile on M4.
- Keep the active working set (current layer weights + KV cache slice + activations) inside this budget whenever possible.
- When the working set exceeds ANE SRAM, every access spills to unified memory — throughput drops ~30% at the SRAM cliff and energy per token rises sharply.

### 2. Prefer Fused Low-Precision KV Cache (Account Dequant Energy)
- Store KV cache in int4 or int8 when possible on M4.
- Perform dequantization on-the-fly (register-level or tile-level) instead of materializing full FP16/BF16 KV in unified memory.
- **Measure dequantization joules** — theoretical bandwidth savings mean nothing if dequant + orchestration erodes IPJ.
- Highest-leverage technique for staying within ANE SRAM during decode.

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

### 6. Be Explicit About Data Movement on Unified Memory
- On M4, CPU/GPU/ANE share one coherent pool — pointer sharing is cheap, but **moving bytes** (especially across the SRAM cliff) costs energy and time.
- Log or profile bytes moved between ANE SRAM and unified memory; treat data-movement reduction as a first-class optimization target alongside ANE utilization.
- **ANE-first routing** minimizes unnecessary copies; CPU orchestration that materializes tensors in unified memory must be measured and minimized.

## Practical Rules of Thumb

| Workload Type | Recommended Strategy | Why |
|---------------|----------------------|-----|
| Short decode | Fused int4 KV + keep everything in SRAM | Highest efficiency |
| Long context decode | Tiled execution + selective recomputation | Avoids unified-memory spills past ANE SRAM cliff |
| Agent loop with tools | Keep recent context + tool state hot | Reduces repeated loading |
| Self-improvement / training | Aggressive recomputation + streaming | Memory is the bottleneck |

## Anti-Patterns to Avoid

- Loading entire large layers or full KV cache into unified memory and hoping SLC hides ANE SRAM overflow.
- GPU-default routing when the op compiles to ANE — wastes ANE power-gating efficiency.
- Frequent small irregular memory accesses in the hot path (breaks prefetch, idles ANE).
- Storing large intermediate activations when recomputation in ANE SRAM is cheaper.
- Ignoring thermal effects of sustained unified-memory bandwidth (DVFS throttling reduces sustained IPJ).
- Claiming IPJ without `powermetrics` logs and thermal steady-state data.

## Measurement (Physical M4)

Any memory access pattern change must be evaluated on the physical Mac Mini M4 with:
- Sustained IPJ before and after (not peak burst)
- ANE utilization under thermal steady state
- Measured data movement volume or SRAM cliff shift (if possible)
- Thermal headroom: start temp, steady-state temp, time-to-throttle

This document should be updated as we learn more from Phase 0 micro-benchmarks.
