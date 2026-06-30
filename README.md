# Alalā – Hardware-Aware Self-Improving AI on M4

**Goal**: Build an open-source system capable of effective GLM-5.2 level intelligence while maximizing **Intelligence per Joule** on a Mac Mini M4 24GB.

We work strictly from first principles, with deep co-design between the model architecture, compiler/runtime, and the physical constraints of Apple Silicon (ANE, unified memory, SRAM limits, thermal headroom, power gating).

## Core Principles

- **Physics First**: Data movement, SRAM budgeting, thermal headroom, and ANE efficiency are first-class constraints.
- **Measurement-Driven**: All major decisions are gated by real hardware measurements (IPJ, utilization, energy).
- **Bounded Self-Improvement**: Improvement loops are constrained by measurable IPJ gains and the Human Cooperation Attractor (HCA).
- **AI-Native Execution**: Designed to be primarily executed by Grok Build (local AI coding agent) with clear rules and task lists.

## Repository Structure

- `AGENTS.md` – Instructions for Cursor / cloud AI coding agents
- `.cursor/rules/` – Cursor project rules (always-on + scoped)
- `docs/` – All core documentation (physics, architecture, execution, measurement, etc.)
- `harness/` – Phase 0 measurement harness (M4)
- `experiments/`, `logs/`, `results/`, `checkpoints/` – Phase 0 workflow directories
- Key entry points:
  - `docs/Project_Index_Alalā.md` – Main navigation hub
  - `docs/AI_Coder_Rules_Guidelines_Alalā.md` – Rules for AI coders
  - `docs/Phase0_AI_Coder_Task_List.md` – Explicit tasks for Phase 0

## Getting Started (Cursor / AI Agents)

1. Read `AGENTS.md`
2. Read `docs/AI_Coder_Rules_Guidelines_Alalā.md`
3. Read `docs/Project_Index_Alalā.md`
4. Follow `docs/Phase0_AI_Coder_Task_List.md`
5. Run `./verify.sh` before every commit

## Status

**Phase 0 execution kickoff ready** (v0.4.4). Harness, tests, validation, and docs complete. Next step: run `./experiments/phase0_kickoff.sh` on physical Mac Mini M4 24 GB with `sudo`.

```bash
pip install -r requirements.txt
./verify.sh
./experiments/phase0_kickoff.sh
```

## License

Apache 2.0 (see LICENSE file)

---

*This project is developed with heavy use of local AI coding agents.*
