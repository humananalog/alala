# Phase 0 Week 1–2 Task Breakdown — Alalā

**Version**: 1.0  
**Purpose**: Detailed, executable task list for the first two weeks of Phase 0.

## Week 1: Foundation & Thermal Characterization

### Day 1–2: Environment Setup
- Create full directory structure (`experiments/`, `logs/`, `harness/`, `results/`, `checkpoints/`)
- Install and verify dependencies
- Copy and test the energy logging harness
- Verify `powermetrics` access and basic logging works

**Acceptance Criteria**: Harness can log power + temperature for 60 seconds without errors.

### Day 3–4: Thermal Baseline
- Run idle + sustained load thermal characterization
- Document power draw and temperature curves
- Identify safe sustained operating region (temperature and power)

**Acceptance Criteria**: Clear thermal rise curve + steady-state temperature documented and logged.

### Day 5–7: ANE SRAM Cliff Characterization
- Run decode workloads with increasing context lengths
- Identify approximate context length where performance drops ~25–30%
- Document results with graphs/tables

**Acceptance Criteria**: Reproducible SRAM cliff point identified.

## Week 2: KV Cache & Overhead Measurement

### Day 8–10: Fused int4 KV vs FP16 Comparison
- Implement or integrate fused low-precision KV path
- Run controlled comparison at fixed context length (e.g. 2048)
- Measure tokens/s, energy per token, and rough IPJ

**Acceptance Criteria**: Quantified difference between the two paths documented.

### Day 11–12: Dispatch & Orchestration Overhead
- Profile time and energy spent outside the ANE (Python overhead, data movement, etc.)
- Identify biggest sources of overhead in a simple agent loop

**Acceptance Criteria**: Clear breakdown of where time/energy is being spent.

### Day 13–14: Integration & Reporting
- Consolidate all Week 1–2 results into a short report
- Update `OSLab_Program_Board.md` with findings and risks
- Decide on priorities for the rest of Phase 0

**Acceptance Criteria**: First Phase 0 progress report completed and logged.

## Notes for Grok Build

- Update the Program Board after every major task or discovery.
- If any task takes significantly longer than expected, flag it early.
- Prioritize measurement quality over speed in these early weeks.
