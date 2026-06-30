# OSLab Program Board — Alalā

**Version**: 1.2  
**Purpose**: Single source of truth for current status, risks, decisions, and progress.

## Current Phase

**Phase 0 – Pre-hardware-measurement** (docs audit **complete**, harness implementation next)

Gap-closing experiments E1–E4 defined as decision gates. Awaiting harness implementation on physical M4.

**Prior label**: ANE Characterization & Measurement Infrastructure  
**Started**: 2026-06-30  
**Target**: Harness implementation on physical Mac Mini M4 24 GB, then Week 1–2 measurements

**Readiness (2026-06-30)**:
- Documentation audit Tasks 1–5 **complete**
- `./verify.sh` passing
- `harness/m4_energy_harness.py` **not implemented** — blocker for physical M4 runs
- Next action: implement harness per `IPJ_Measurement_Protocol_Alalā.md` §2.3

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

## Methodology Gap Closure (2026-06-30)

Four over-optimistic assumptions are now **testable hypotheses** with minimal decision-gate experiments on **physical Mac Mini M4 24 GB**. All require raw `powermetrics` + thermal logs and stated thermal headroom (`IPJ_Measurement_Protocol_Alalā.md` §2.5). **Stop if temperature exceeds safe sustained threshold.**

| Experiment | Closes assumption | Decision gate | Status |
|------------|-------------------|---------------|--------|
| **E1** ANE Real Utilization | ANE-first routing covers forward pass | Redesign if `ane_compute_fraction_pct` < 50% or orchestration tax > 40% | Defined – awaiting harness |
| **E2** Thermal + IPJ Curve | Sustained IPJ stable as thermal headroom shrinks | Redesign if sustained IPJ degrades ≥20% post-throttle | Defined – awaiting harness |
| **E3** Meta-Tax | Self-improvement pays for itself in joules | Block cadence if `net_ipj_delta` ≤ 0 | Defined – awaiting harness |
| **E4** Memory Spill Cost | Hierarchical memory beats spill at realistic working sets | Redesign if spill joules/token > recompute | Defined – awaiting harness |

**Blocked on**: `harness/m4_energy_harness.py` modes `ane_utilization`, `thermal_ipj_curve`, `meta_tax`, `memory_spill` (see IPJ protocol §2.3–§2.4).

**Posture**: Measure first on the single M4; redesign early if a gate fails. No model architecture or self-improvement cadence until applicable E-gates pass.

**Docs**: `Phase0_AI_Coder_Task_List.md` § Phase 0 Extended; `Risk_Register.md` R-ANE-01 … R-MEM-04.

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

### Task 4 — Cross-consistency
- `Project_Index_Alalā.md`: full 19-doc navigation hub, operational IPJ, harness modes
- Terminology alignment: unified-memory spills, ≥30% SRAM cliff, execution constraint on planning docs

### Task 5 — Execution guidance
- `OSLab_Execution_Playbook.md` v2.1: physical M4 only, powermetrics required, thermal safety, ANE-first

## Human Review Flags (post-audit)
| Doc | Flag | Reason |
|-----|------|--------|
| `Compiler_Passes_Skeleton_Alalā.md` | Expected Benefit numbers unmeasured | Pass benefits are hypotheses until Phase 1 M4 IPJ validation |
| `Alalā_Improvement_Playbook.md` | Marginal IPJ not cross-linked to §2.1 | Self-improvement gating should cite operational IPJ after Phase 0 |
| `Meta_Controller_Skeleton_Alalā.md` | Threshold constants unspecified | Controller thresholds should be set from Phase 0 measured baselines |
| Phase 0 success criteria | Sustained ANE utilization % TBD | Target % intentionally deferred until thermal baseline on hardware |

## Active Tasks (as of today)
- W1-00: Docs audit (Tasks 1–5) — **Complete**
- W1-01: Implement `harness/m4_energy_harness.py` — **Next**
- W1-02: Thermal Baseline on physical M4 — Blocked on harness
- W1-03: ANE SRAM Cliff Characterization — Blocked on harness

## Key Risks (Top 5 — see `Risk_Register.md`)
1. **R-GAP-01** Low real ANE forward-pass coverage (mitigation: E1)
2. **R-GAP-02** Thermal throttling erodes sustained IPJ (mitigation: E2)
3. **R-GAP-03** Meta-tax exceeds marginal gains (mitigation: E3)
4. **R-GAP-04** Working-set pressure and SRAM spill cost (mitigation: E4)
5. **R03** SRAM cliff impact on long-context decode

## Recent Decisions
- 2026-06-30: Thermal headroom and sustained IPJ take precedence over peak throughput.
- 2026-06-30: No IPJ claim without raw powermetrics + thermal artifacts.
- 2026-06-30: ANE-first routing is default; measure CPU orchestration before minimizing.
- 2026-06-30: Added E1–E4 gap-closing decision-gate experiments; risks R-ANE-01 … R-MEM-04.

## Blockers
- `harness/m4_energy_harness.py` not implemented (required before physical M4 runs).

## Next Milestone
Implement harness per IPJ protocol §2.3; run thermal baseline on physical Mac Mini M4 24 GB.

## Human Review Flags
_See table above (post-audit)._

## Notes
This board must be updated by Grok Build after every significant task or discovery.
