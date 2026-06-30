# OSLab Program Board — Alalā

**Version**: 1.0  
**Purpose**: Single source of truth for current status, risks, decisions, and progress.

## Current Phase
**Phase 0**: ANE Characterization & Measurement Infrastructure

**Status**: In Progress  
**Started**: 2026-06-30  
**Target Completion**: Week 2 of Phase 0

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
