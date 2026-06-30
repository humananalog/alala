# Alalā – Hardware-Aware Self-Improving AI on M4

**Goal**: Build an open-source system capable of effective GLM-5.2 level intelligence while maximizing **Intelligence per Joule** on a Mac Mini M4 24GB.

We work strictly from first principles, with deep co-design between the model architecture, compiler/runtime, and the physical constraints of Apple Silicon (ANE, unified memory, SRAM limits, thermal headroom, power gating).

## Core Principles

- **Physics First**: Data movement, SRAM budgeting, thermal headroom, and ANE efficiency are first-class constraints.
- **Measurement-Driven**: All major decisions are gated by real hardware measurements (IPJ, utilization, energy).
- **Bounded Self-Improvement**: Improvement loops are constrained by measurable IPJ gains and the Human Cooperation Attractor (HCA).
- **AI-Native Execution**: Designed to be primarily executed by Grok Build (local AI coding agent) with clear rules and task lists.

## Repository Structure

- `docs/` — Core documentation (physics, architecture, execution, measurement, Phase 0 tasks)
- `verify.sh` — Pre-commit verification script (run before every commit)
- Key entry points:
  - [`docs/Project_Index_Alalā.md`](docs/Project_Index_Alalā.md) — Main navigation hub (19 documents)
  - [`docs/AI_Coder_Rules_Guidelines_Alalā.md`](docs/AI_Coder_Rules_Guidelines_Alalā.md) — Rules for Grok Build
  - [`docs/OSLab_Program_Board.md`](docs/OSLab_Program_Board.md) — Live status and progress
  - [`docs/Phase0_AI_Coder_Task_List.md`](docs/Phase0_AI_Coder_Task_List.md) — Explicit Phase 0 tasks

## Getting Started (for Grok Build)

1. Read [`docs/Project_Index_Alalā.md`](docs/Project_Index_Alalā.md)
2. Read [`docs/AI_Coder_Rules_Guidelines_Alalā.md`](docs/AI_Coder_Rules_Guidelines_Alalā.md)
3. Read [`docs/Alalā_Physics_Corrected_Foundation.md`](docs/Alalā_Physics_Corrected_Foundation.md)
4. Open [`docs/OSLab_Program_Board.md`](docs/OSLab_Program_Board.md)
5. Follow [`docs/Phase0_AI_Coder_Task_List.md`](docs/Phase0_AI_Coder_Task_List.md)

Before committing any change:

```bash
./verify.sh
```

## Status

**Documentation**: All 19 Project Index documents are populated (as of 2026-06-30).

**Phase 0**: Ready to begin — measurement harness and experiment directories still to be implemented on the M4.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

---

*This project is developed with heavy use of local AI coding agents.*
