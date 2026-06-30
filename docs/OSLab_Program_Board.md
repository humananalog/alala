# OSLab Program Board — Alalā

**Version**: 1.1  
**Purpose**: Single source of truth for current status, risks, decisions, and progress.

## Current Phase

**Phase 0 – Pre-hardware-measurement** (docs audit complete, harness implementation next)

**Prior label**: ANE Characterization & Measurement Infrastructure  
**Started**: 2026-06-30  
**Target**: Harness implementation on physical Mac Mini M4 24 GB, then Week 1–2 measurements

**Readiness**: Documentation audit Tasks 1–3 complete. `harness/m4_energy_harness.py` not yet implemented — **blocker for hardware runs**.

## Phase 0 Success Criteria (Measurable M4 Numbers)

All criteria require raw `powermetrics` logs + thermal data per `IPJ_Measurement_Protocol_Alalā.md` §2.1.

| Criterion | Target | Source benchmark |
|-----------|--------|------------------|
| Thermal baseline curve | Idle + sustained load power (W), `temp_steady_state_c`, safe sustained envelope documented | Benchmark 1 / `thermal_baseline` |
| SRAM cliff context length | \( L_{\text{cliff}} \) where sustained throughput drops ≥30% | Benchmark 2 / `sram_cliff` |
| int4 vs FP16 IPJ delta | Repeatable `IPJ_phase0` delta including `energy_dequant_joules` | Benchmark 3 / `kv_comparison` |
| Sustained ANE utilization | ANE utilization % at thermal steady state under ANE-first routing (baseline TBD from measurement) | Benchmarks 2–4 |
| Orchestration overhead | `energy_cpu_orchestration_joules` / total joules ratio documented | Benchmark 4 / `orchestration` |

**Governing principle**: Thermal headroom and sustained IPJ take precedence over peak throughput.

## Documentation Audit Log (2026-06-30)

### Task 1 — Foundational physics grounding
- `Alalā_Physics_Corrected_Foundation.md` — §0 M4 silicon realities; thermal first-class; ANE-first default
- `IPJ_Measurement_Protocol_Alalā.md`, `Phase0_Microbenchmark_Suite_Plan.md`, memory docs, HCA — M4-specific phrasing

### Task 2 — Operational IPJ
- `IPJ_Measurement_Protocol_Alalā.md` §2.1–§2.3: executable IPJ\(_{phase0}\), SRAM cliff method, harness spec
- `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`: physical-M4-only execution guide
- `harness/README.md`: four harness modes

### Task 3 — Program Board + Risk Register
- Phase 0 success criteria table (above)
- `Risk_Register.md`: Phase 0 risks R02–R06 (SRAM cliff, thermal, 24 GB pressure, orchestration, dequant)

## Active Tasks (as of today)
- W1-00: Docs audit (Tasks 1–3) — **Complete**
- W1-01: Implement `harness/m4_energy_harness.py` — **Next**
- W1-02: Thermal Baseline on physical M4 — Blocked on harness
- W1-03: ANE SRAM Cliff Characterization — Blocked on harness

## Key Risks (Top 5 — see `Risk_Register.md`)
1. **R02** Thermal throttling under sustained ANE+CPU load
2. **R03** SRAM cliff impact on long-context decode
3. **R05** ANE utilization gaps due to orchestration
4. **R06** Dequantization energy eroding int4 gains
5. **R04** 24 GB working-set pressure

## Recent Decisions
- 2026-06-30: Thermal headroom and sustained IPJ take precedence over peak throughput.
- 2026-06-30: No IPJ claim without raw powermetrics + thermal artifacts.
- 2026-06-30: ANE-first routing is default; measure CPU orchestration before minimizing.
- 2026-06-30: Adopted strict "no placeholder content" policy for all documentation.

## Blockers
- `harness/m4_energy_harness.py` not implemented (required before physical M4 runs).

## Next Milestone
Implement harness per IPJ protocol §2.3; run thermal baseline on physical Mac Mini M4 24 GB.

## Human Review Flags
_None yet — re-check after Tasks 4–5 cross-consistency pass._

## Notes
This board must be updated by Grok Build after every significant task or discovery.
