# Alalā — Vision and Strategy (v2 — High-Risk Research Mode)

**Status**: Active Research Phase (July 2026 onward)  
**Risk Posture**: High risk, high potential reward  
**Team**: Solo (human + Grok as primary collaborator)

## Core North Star

Build the most capable on-device intelligence possible on hardware like the Mac Mini M4 24GB, by designing model architectures and inference systems that are **co-optimized with the physical constraints and strengths of Apple Silicon** — particularly the Apple Neural Engine — rather than retrofitting existing transformer designs.

Success is measured primarily by **sustained useful cognitive work per joule** over long periods, combined with strong self-improving and agentic capabilities.

## Why This Shift

After extensive experimentation with current models and Core ML tooling, it has become clear that:

- Standard decoder-only transformers + autoregressive decode have fundamental mismatches with ANE characteristics (dynamic shapes, state management, data movement costs).
- Incremental optimization of existing stacks yields diminishing returns on ANE utilization during decode.
- The highest-leverage path forward is to explore more fundamental redesigns in model components (especially KV cache / state mechanisms) and inference execution strategies.

## Strategic Posture

- **Deep and broad exploration**: We are willing to pursue unconventional ideas, accept high failure rates, and change direction when evidence demands it.
- **First-principles grounding**: Every major direction must be justified by the actual physics and constraints of the target hardware (ANE shape/operator limitations, unified memory behavior, thermal/power envelopes, data movement costs).
- **Solo research mode**: Fast iteration, low process overhead, high intellectual honesty.
- **Nothing to lose mindset**: We prioritize learning and breakthrough potential over short-term polished results.

## Current Phase: Fundamental Redesign Exploration

**Primary research vectors** (not exhaustive, subject to change):

1. **KV Cache / State Architecture Redesign**
   - Static shapes, ring buffers, paged/sliding window approaches
   - Reduced dynamic masking and I/O per step
   - Better alignment with ANE-friendly memory access patterns

2. **Inference Execution Strategies**
   - Hybrid / disaggregated designs (ANE + GPU/MLX) designed from the start
   - Lower-level ANE access patterns and custom scheduling
   - Smart thermal/power-aware routing between execution paths

3. **Model Architecture Adaptations**
   - Components co-designed with ANE constraints (operator choices, shape regularity, state handling)
   - Potential distillation or architectural modifications for better hardware fit

4. **Self-Improvement Substrate**
   - Designing the efficiency layer to enable richer, longer-horizon self-improving and agentic loops

## Success Criteria (Longer-term)

- Demonstrable improvement in sustained useful work per joule compared to strong baselines on similar hardware.
- Evidence that the new designs unlock capabilities or efficiency regimes that are difficult with conventional approaches.
- Clear, documented understanding of the hardware-model co-design space for this class of NPUs.

## Risk Acceptance

This is explicitly a high-risk research effort. Many directions will fail or yield negative results. Progress may be non-linear. The goal is to thoroughly explore the possibility space rather than optimize within known local maxima.