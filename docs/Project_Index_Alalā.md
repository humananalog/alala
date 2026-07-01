# Alalā / OSLab — Project Index & Navigation

**Status**: Authoritative entry point (as of July 2026)
**Repository**: https://github.com/humananalog/alala  
**Current phase**: **Phase 2 — Fundamental Co-Design Research (Option C)** — high-risk exploration per [`Alalā_Vision_and_Strategy.md`](Alalā_Vision_and_Strategy.md). Phase 0–1 complete; measurements and Core ML artifacts remain baselines.

**Execution Model**: Grok Build (local AI coding agent on the physical Mac Mini M4 24 GB) is the primary executor (~98%). All documentation is structured to be directly followable by an AI coder.

**Execution constraint** (all benchmark/measurement docs): All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold.

This document is the single source of truth for navigating the entire Alalā documentation set (**23 indexed docs** in `docs/`).

## 1. Quick Start for Grok Build

**New to the repo (Day 1):**

1. Read this entire index first.
2. Read [`Alalā_Vision_and_Strategy.md`](Alalā_Vision_and_Strategy.md) — current north star and research posture.
3. Read [`Alalā_Research_Agenda.md`](Alalā_Research_Agenda.md) — active research threads, priorities, and backlog.
4. Read `AI_Coder_Rules_Guidelines_Alalā.md` (mandatory rules).
5. Read `Alalā_Physics_Corrected_Foundation.md` (M4 silicon realities, §0).
6. Read `Alalā_Core_Invariant_Specification_HCA.md` (non-violable constraints).
7. Open `OSLab_Program_Board.md` — note Phase 2 status, strategic shift, success criteria.
8. Read `Phase0_Results_Summary_Alalā.md` for measured Phase 0 baselines; `phase1/NOTES.md` for Phase 1 Core ML findings.
9. Use `IPJ_Measurement_Protocol_Alalā.md` §2.1 for operational IPJ; log all experiments; update Program Board and Research Agenda.

**Phase 2 active (July 2026 onward):**

1. Ground new directions in measured Phase 0–1 constraints (thermal, SRAM cliff, ANE placement gaps).
2. Pick work from [`Alalā_Research_Agenda.md`](Alalā_Research_Agenda.md) — update thread status and notes as experiments run.
3. Pursue co-design vectors in `Alalā_Vision_and_Strategy.md` — KV/state, inference execution, architecture adaptations.
4. Preserve measurement discipline: `powermetrics`, JSONL, Program Board + Research Agenda updates after significant discoveries.

**Phase 1 reference (complete):**

1. `Phase1_ANE_First_Strategy.md` and Program Board Phase 1 results section.
2. `phase1/README.md` — Core ML conversion and ANE residency benchmarks (baseline tooling).

## 2. Document Hierarchy (All 23 Docs)

### Vision & Strategy (Phase 2)

- [`docs/Alalā_Vision_and_Strategy.md`](Alalā_Vision_and_Strategy.md) — v2 north star, high-risk research posture, co-design vectors, success criteria.
- [`docs/Alalā_Research_Agenda.md`](Alalā_Research_Agenda.md) — **active backlog**: research threads, hypotheses, priorities, next actions.

### Phase 0

- [`docs/Phase0_Results_Summary_Alalā.md`](Phase0_Results_Summary_Alalā.md) — Measured results, physics interpretation, and Phase 1 entry criteria.

### Phase 1

- [`docs/Phase1_ANE_First_Strategy.md`](Phase1_ANE_First_Strategy.md) — ANE-first execution plan, seeding model selection, and first experiment success gate.

### Core / Authoritative (7)
| Document | Purpose | Status |
|----------|---------|--------|
| `Alalā_Vision_and_Strategy.md` | Phase 2 vision, co-design research vectors, risk acceptance | **Active** (July 2026) |
| `Alalā_Physics_Corrected_Foundation.md` | M4 silicon realities (§0), ANE-first, thermal headroom, sustained IPJ > peak | Complete v2 |
| `Alalā_Core_Invariant_Specification_HCA.md` | Human Cooperation Attractor — constitutional constraint | Complete |
| `OSLab_Program_Board.md` | Live phase status, success criteria, audit log, blockers | Active v1.3+ (Phase 2) |
| `Alalā_Research_Agenda.md` | Research threads, hypotheses, prioritization, next actions | **Active** (July 2026) |
| `OSLab_Execution_Playbook.md` | Day-to-day program execution on physical M4 | Complete |
| `IPJ_Measurement_Protocol_Alalā.md` | Operational IPJ\(_{phase0}\), powermetrics + thermal artifacts required | Complete v1.2 |

### Execution & Planning (8)
| Document | Purpose | Status |
|----------|---------|--------|
| `Phase0_AI_Coder_Task_List.md` | Numbered Phase 0 tasks for Grok Build | Complete |
| `Phase0_Microbenchmark_Suite_Plan.md` | Four M4 benchmarks + harness mode mapping | Complete v1.1 |
| `Phase0_Results_Summary_Alalā.md` | Canonical Phase 0 measured outcomes + Phase 1 entry criteria | Complete (2026-07-01) |
| `Phase1_ANE_First_Strategy.md` | ANE-first strategy, seeding model, first experiment results | Complete (2026-07-01) |
| `Phase0_Week1_2_Task_Breakdown.md` | Week 1–2 day-level breakdown | Complete |
| `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md` | Step-by-step physical M4 benchmark commands | Complete v1.1 |
| `Risk_Register.md` | Phase 0 risks: SRAM cliff, thermal, 24 GB, orchestration, dequant | Complete v1.1 |
| `Revised_Phase0_2_Systems_Plan_Alalā.md` | Systems & compiler roadmap Phases 0–2 | Complete v2 |

### Technical Architecture (3)
| Document | Purpose | Status |
|----------|---------|--------|
| `Hierarchical_Memory_Architecture_Alalā.md` | Unified memory + ANE SRAM tiers, 24 GB pressure | Complete v1.1 |
| `Memory_Access_Pattern_Guidelines_Alalā.md` | M4 access patterns, data-movement minimization | Complete v1.1 |
| `Compiler_Passes_Skeleton_Alalā.md` | Shape specialization, fused KV, SRAM tiling passes | Complete |

### Supporting / Meta (5)
| Document | Purpose | Status |
|----------|---------|--------|
| `AI_Coder_Rules_Guidelines_Alalā.md` | Non-negotiable agent rules | Complete |
| `Alalā_Improvement_Playbook.md` | Bounded self-improvement mechanics | Complete |
| `Alalā_Experimentation_Framework.md` | Experiment design and JSONL logging | Complete |
| `Meta_Controller_Skeleton_Alalā.md` | HCA + IPJ-gated meta-controller | Complete |
| `Project_Index_Alalā.md` | This navigation hub | Complete |

**Total**: 23 documents (21 prior + `Alalā_Vision_and_Strategy.md` + `Alalā_Research_Agenda.md`).

## 3. Key Concepts & Single Sources of Truth

| Concept | Authoritative Document | Short Definition |
|---------|------------------------|------------------|
| **IPJ** | `IPJ_Measurement_Protocol_Alalā.md` §2.1 | Phase 0: useful work / joules (`powermetrics` on physical M4 at thermal steady state) |
| **HCA** | `Alalā_Core_Invariant_Specification_HCA.md` | Human Cooperation Attractor — non-violable constitutional constraint |
| **ANE-First** | `Alalā_Physics_Corrected_Foundation.md` §2.2 | Default route for compute-bound ops; measure CPU orchestration overhead |
| **SRAM Budgeting** | `Hierarchical_Memory_Architecture_Alalā.md` | Active working sets < ~28–30 MB ANE on-chip SRAM; spill → unified memory |
| **Thermal Headroom** | `Alalā_Physics_Corrected_Foundation.md` §2.3 | First-class variable; sustained IPJ > peak throughput |
| **SRAM Cliff** | `Phase0_Results_Summary_Alalā.md` + `IPJ_Measurement_Protocol_Alalā.md` §2.2 | \( L_{\text{cliff}} = 1024 \) on M4; ≥30% sustained throughput drop |
| **Harness** | `harness/m4_energy_harness.py` + `How_to_Run_...` | Phase 0: thermal_baseline, sram_cliff, kv_comparison, orchestration |
| **Phase 1 ANE** | `Phase1_ANE_First_Strategy.md` + `phase1/` | Core ML conversion + ANE residency benchmark (baseline) |
| **Phase 2 vision** | `Alalā_Vision_and_Strategy.md` | Co-design research mode, north star, risk posture |
| **Research backlog** | `Alalā_Research_Agenda.md` | Active threads, priorities, experiment ideas |

## 4. How to Navigate

- **Vision / strategy (start here for Phase 2)** → `Alalā_Vision_and_Strategy.md`
- **Research backlog / what to work on next** → `Alalā_Research_Agenda.md`
- **Planning / architecture** → `Alalā_Physics_Corrected_Foundation.md`
- **Self-improvement** → HCA spec + `Alalā_Improvement_Playbook.md`
- **Daily work** → `OSLab_Execution_Playbook.md` + Program Board
- **Phase 0 results** → `Phase0_Results_Summary_Alalā.md`
- **Phase 1 ANE strategy** → `Phase1_ANE_First_Strategy.md`
- **Run Phase 0 benchmarks** → `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`
- **Run Phase 1 ANE benchmarks** → `phase1/README.md`
- **Measurement / IPJ** → `IPJ_Measurement_Protocol_Alalā.md`
- **Memory / SRAM** → `Hierarchical_Memory_Architecture_Alalā.md` + `Memory_Access_Pattern_Guidelines_Alalā.md`
- **Risks** → `Risk_Register.md` + Program Board

**This index will be kept up to date.** When in doubt, return here first.
