# OSLab Program Board — Alalā

**Version**: 1.0  
**Purpose**: Single source of truth for current status, risks, decisions, and progress.

## Current Phase
**Phase 0 – Pre-hardware-measurement** (docs audit in progress, harness implementation next)

**Prior label**: ANE Characterization & Measurement Infrastructure  
**Status**: Documentation audit — Task 1 complete (foundational physics grounding)  
**Started**: 2026-06-30  
**Target Completion**: Week 2 of Phase 0 (after M4 harness + first measurements)

## Task 1 Audit (2026-06-30) — Foundational & Measurement Docs
Strengthened M4 silicon physics grounding in:
- `Alalā_Physics_Corrected_Foundation.md` — §0 M4 silicon realities table; thermal headroom first-class; ANE-first default; sustained IPJ > peak
- `IPJ_Measurement_Protocol_Alalā.md` — physical M4-only measurement; thermal/energy decomposition; powermetrics log requirement
- `Phase0_Microbenchmark_Suite_Plan.md` — SRAM cliff methodology; dequant energy; orchestration benchmark; harness reference
- `Hierarchical_Memory_Architecture_Alalā.md` — unified memory coherence; ANE SRAM spill physics; 24 GB pressure
- `Memory_Access_Pattern_Guidelines_Alalā.md` — M4-specific access patterns; anti-patterns; sustained measurement
- `Alalā_Core_Invariant_Specification_HCA.md` — HCA tied to sustained IPJ measurement discipline

**Principle added across docs**: Thermal headroom and sustained IPJ take precedence over peak throughput.

## Active Tasks (as of today)
- W1-01: Environment & Logging Setup — In Progress
- W1-02: Thermal Baseline — Not Started
- W1-03: ANE SRAM Cliff Characterization — Not Started

## Key Risks (Top 5)
1. **Measurement noise** — `powermetrics` can be noisy; may need external validation.
2. **Thermal instability** — M4 thermals can vary significantly with ambient conditions.
3. **SRAM cliff hard to isolate** — May be confounded by other bottlenecks.
4. **Grok Build over-optimism** — Risk of underestimating difficulty of low-level optimization.
5. **Documentation drift** — Risk that code and docs diverge.

## Recent Decisions
- 2026-06-30: Adopted strict "no placeholder content" policy for all documentation.
- 2026-06-30: Prioritized measurement infrastructure before any major compiler work.

## Blockers
None currently.

## Next Milestone
Complete Week 1 tasks and produce first set of reliable baseline measurements (IPJ, ANE utilization, thermal behavior, SRAM cliff).

## Notes
This board must be updated by Grok Build after every significant task or discovery.
