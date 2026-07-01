# Alalā / OSLab — Project Index & Navigation

**Status**: Authoritative entry point (as of 2026-07-01)
**Repository**: https://github.com/humananalog/alala  
**Current phase**: Phase 2 **ACTIVE** (2026-07-01) — Fundamental redesign research per `Alalā_Vision_and_Strategy.md`

**Execution Model**: Grok Build (local AI coding agent on the physical Mac Mini M4 24 GB) is the primary executor (~98%). All documentation is structured to be directly followable by an AI coder.

**Execution constraint** (all benchmark/measurement docs): All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold.

This document is the single source of truth for navigating the entire Alalā documentation set (**22 indexed docs** in `docs/`).

## 1. Quick Start for Grok Build

**New to the repo (Day 1):**

1. Read this entire index first.
2. Read `AI_Coder_Rules_Guidelines_Alalā.md` (mandatory rules).
3. Read `Alalā_Physics_Corrected_Foundation.md` (M4 silicon realities, §0).
4. Read `Alalā_Core_Invariant_Specification_HCA.md` (non-violable constraints).
5. Open `OSLab_Program_Board.md` — note current phase, success criteria, blockers.
6. Read `Alalā_Vision_and_Strategy.md` for current research direction and risk posture.
7. Read `Phase0_Results_Summary_Alalā.md` for measured Phase 0 baselines.
8. Use `IPJ_Measurement_Protocol_Alalā.md` §2.1 for operational IPJ; log all experiments; update Program Board.

**Phase 2 active (2026-07-01):**

1. Read `Alalā_Vision_and_Strategy.md` and Program Board Phase 2 section.
2. Use Phase 1 artifacts (`phase1/`, `Phase1_ANE_First_Strategy.md`) as measured baselines only.
3. Commit `logs/` and `results/` artifacts with benchmark changes.

## 2. Document Hierarchy (All 22 Docs)

### Vision & Strategy

- [`docs/Alalā_Vision_and_Strategy.md`](Alalā_Vision_and_Strategy.md) — North star, strategic posture, Phase 2 research vectors, risk acceptance.

### Phase 0

- [`docs/Phase0_Results_Summary_Alalā.md`](Phase0_Results_Summary_Alalā.md) — Measured results, physics interpretation, and Phase 1 entry criteria.

### Phase 1

- [`docs/Phase1_ANE_First_Strategy.md`](Phase1_ANE_First_Strategy.md) — ANE-first execution plan, seeding model selection, and first experiment success gate.

### Core / Authoritative (6)
| Document | Purpose | Status |
|----------|---------|--------|
| `Alalā_Vision_and_Strategy.md` | North star, Phase 2 research direction, risk posture | Complete v2 (2026-07-01) |
| `Alalā_Physics_Corrected_Foundation.md` | M4 silicon realities (§0), ANE-first, thermal headroom, sustained IPJ > peak | Complete v2 |
| `Alalā_Core_Invariant_Specification_HCA.md` | Human Cooperation Attractor — constitutional constraint | Complete |
| `OSLab_Program_Board.md` | Live phase status, success criteria, audit log, blockers | Complete v1.3 (Phase 2 active) |
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

**Total**: 22 documents (6 + 8 + 3 + 5 = 22).

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
| **Phase 1 ANE** | `Phase1_ANE_First_Strategy.md` + `phase1/` | Core ML conversion + ANE residency benchmark |

## 4. How to Navigate

- **Vision / strategy** → `Alalā_Vision_and_Strategy.md`
- **Planning / architecture** → `Alalā_Physics_Corrected_Foundation.md`
- **Self-improvement** → HCA spec + `Alalā_Improvement_Playbook.md`
- **Daily work** → `OSLab_Execution_Playbook.md` + Program Board
- **Phase 0 results** → `Phase0_Results_Summary_Alalā.md`
- **Phase 1 ANE baselines** → `Phase1_ANE_First_Strategy.md` + `phase1/README.md`
- **Phase 2 research** → `Alalā_Vision_and_Strategy.md` + Program Board Phase 2 section
- **Run Phase 0 benchmarks** → `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`
- **Run Phase 1 ANE benchmarks** → `phase1/README.md`
- **Measurement / IPJ** → `IPJ_Measurement_Protocol_Alalā.md`
- **Memory / SRAM** → `Hierarchical_Memory_Architecture_Alalā.md` + `Memory_Access_Pattern_Guidelines_Alalā.md`
- **Risks** → `Risk_Register.md` + Program Board

**This index will be kept up to date.** When in doubt, return here first.
