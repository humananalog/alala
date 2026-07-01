# Phase 1 — ANE-First Execution & Seeding Model

**Status**: Starting (2026-07-01)
**Goal**: Maximize fraction of forward pass routed to ANE on M4 while maintaining sustainable IPJ and thermal headroom. First milestone: measurable ANE utilization on a small capable model under L_cliff=1024.

## Physics Constraints from Phase 0
- L_cliff = 1024 context (33.7% throughput drop)
- Thermal steady-state ~82.7–86.5 °C with throttling in ~5 s
- Orchestration tax is low (~4%)
- Current MLX decode paths show ~0% ANE utilization (GPU fallback)
- Dequant cost for int4 is low but not yet winning

## Phase 1 Principles
1. Route as much of the forward pass and verification as possible to the ANE.
2. Accept lower peak speed if it delivers better sustained IPJ and thermal headroom.
3. Memory system design must keep hot KV/activations inside or near ANE SRAM budget.
4. All major decisions gated by real M4 measurements (powermetrics + thermal + ANE attribution).

## First Experiment (This Week)
Select and convert a small seeding model (≤ ~1B params) to CoreML/ANE-friendly path.
Measure:
- Real ANE utilization % (or best proxy)
- Sustained tokens/s at context 512 and 1024 under thermal steady-state
- Energy per token and IPJ
- Thermal behavior vs current MLX GPU baseline

Success gate: Measurable ANE residency + sustained IPJ within 10% of current MLX path (or better).

## Candidate Seeding Models (ranked for first try)
1. Qwen2.5-0.5B-Instruct (strong instruction following, good convertibility)
2. Phi-3.5-mini / Phi-3-mini-128k (excellent reasoning per size)
3. Gemma-2-2B-IT (quantized) — slightly larger, test later
4. Custom distilled 350M model (if needed for maximum ANE mapping)

Start with #1 or #2.

## Next After First Experiment
- KV memory system design that respects L_cliff
- Minimal self-improvement scaffold with IPJ + HCA gating
- Thermal-aware scheduler