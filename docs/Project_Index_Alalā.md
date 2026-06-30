# Alalā / OSLab — Project Index & Navigation

**Status**: Authoritative entry point (as of 2026-06-30)  
**Repository**: https://github.com/humananalog/alala

**Execution Model**: Grok Build (local AI coding agent on the M4) is the primary executor of this plan (~98%). All documentation is structured to be directly followable by an AI coder.

This document is the single source of truth for navigating the entire Alalā documentation set.

## 1. Quick Start for Grok Build (Day 1)

**For Grok Build**:
1. Read this entire index first.
2. Read `AI_Coder_Rules_Guidelines_Alalā.md` (mandatory rules you must follow).
3. Read `Alalā_Physics_Corrected_Foundation.md` (governing physics principles and constraints).
4. Read `Alalā_Core_Invariant_Specification_HCA.md` (non-violable constraints).
5. Open `OSLab_Program_Board.md` and note current status + risks.
6. Follow `Phase0_AI_Coder_Task_List.md` (explicit numbered tasks).
7. Always log experiments and update the Program Board after completing tasks.

## 2. Document Hierarchy (Read in This Order)

### Core / Authoritative Documents
| Priority | Document | Purpose | Status | Read When |
|----------|----------|---------|--------|-----------|
| 1 | `Alalā_Physics_Corrected_Foundation.md` | Physics constraints, corrected principles, and phased plan | Complete | Before any major decision |
| 2 | `Alalā_Core_Invariant_Specification_HCA.md` | Non-violable Human Cooperation Attractor | Complete | Before any self-improvement work |
| 3 | `OSLab_Program_Board.md` | Live status, risks, decisions, and progress | Complete | Daily / Weekly |
| 4 | `OSLab_Execution_Playbook.md` | How to actually run the program day-to-day | Complete | When starting work |
| 5 | `IPJ_Measurement_Protocol_Alalā.md` | How to measure Intelligence per Joule | Complete | When running experiments |

### Execution & Planning
| Document | Purpose | Status |
|----------|---------|--------|
| `Phase0_AI_Coder_Task_List.md` | Concrete weekly tasks for Phase 0 | Complete |
| `Phase0_Microbenchmark_Suite_Plan.md` | Detailed first experiments | Complete |
| `Phase0_Week1_2_Task_Breakdown.md` | Weekly task breakdown | Complete |
| `Risk_Register.md` | Living risk management | Complete |
| `Revised_Phase0_2_Systems_Plan_Alalā.md` | Systems & compiler focused plan | Complete |

### Technical Architecture
| Document | Purpose | Status |
|----------|---------|--------|
| `Hierarchical_Memory_Architecture_Alalā.md` | Memory system design | Complete |
| `Memory_Access_Pattern_Guidelines_Alalā.md` | How to optimize memory access & SLC usage | Complete |
| `Compiler_Passes_Skeleton_Alalā.md` | Compiler pass definitions and skeletons | Complete |

### Supporting Documents
| Document | Purpose | Status |
|----------|---------|--------|
| `Alalā_Improvement_Playbook.md` | Self-improvement mechanics | Complete |
| `Alalā_Experimentation_Framework.md` | How to design and run experiments | Complete |
| `Meta_Controller_Skeleton_Alalā.md` | Bounded self-improvement controller | Complete |
| `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md` | Step-by-step benchmark execution guide | Complete |

## 3. Key Concepts & Single Sources of Truth

| Concept | Authoritative Document | Short Definition |
|---------|------------------------|------------------|
| **IPJ** | `IPJ_Measurement_Protocol_Alalā.md` | Intelligence per Joule = E[U(task)] / E[J] |
| **HCA** | `Alalā_Core_Invariant_Specification_HCA.md` | Human Cooperation Attractor — non-violable constitutional constraint |
| **ANE-First** | `Alalā_Physics_Corrected_Foundation.md` | All compute-bound ops routed to ANE where possible |
| **SRAM Budgeting** | `Hierarchical_Memory_Architecture_Alalā.md` | Keep active working sets < ~28–30 MB |
| **Thermal Awareness** | `Alalā_Physics_Corrected_Foundation.md` | Thermal headroom is a first-class scheduling variable |

## 4. How to Navigate

- **Planning / Architecture decisions** → Start with `Alalā_Physics_Corrected_Foundation.md`
- **Self-improvement or training decisions** → Start with HCA spec + Improvement Playbook
- **Daily work / experiments** → Start with Execution Playbook + Program Board
- **Measurement** → Start with IPJ Measurement Protocol
- **Risks or blockers** → Check Risk Register + Program Board

**This index will be kept up to date.** When in doubt, return here first.
