# AGENTS.md — Alalā (Cursor / Cloud Agents)

Instructions for AI coding agents working in this repository.

## Project

Alalā is a physics-first, measurement-driven AI system for Apple Silicon M4 (24 GB).
North star: maximize **Intelligence per Joule (IPJ)** with bounded self-improvement under the **Human Cooperation Attractor (HCA)**.

## Before You Code

1. Read `docs/Project_Index_Alalā.md` (navigation hub for all 19 docs).
2. Read `docs/AI_Coder_Rules_Guidelines_Alalā.md` (non-negotiable rules).
3. Check `docs/OSLab_Program_Board.md` for current phase, tasks, and blockers.
4. Read the authoritative doc for your task area (physics, IPJ, memory, compiler, etc.).

## Repository Layout

```text
alala/
├── AGENTS.md              # This file — agent operational context
├── .cursor/rules/         # Cursor-scoped rules (.mdc)
├── .cursor/agents/        # Custom subagents (verifier, etc.)
├── docs/                  # Authoritative documentation (19 indexed docs)
├── harness/               # Measurement harness (Phase 0)
├── experiments/           # Experiment scripts and configs
├── logs/                  # JSONL + powermetrics experiment logs (tracked)
├── results/               # Benchmark outputs per run (tracked)
├── checkpoints/           # Rollback checkpoints (gitignored contents)
├── verify.sh              # Pre-commit verification — run before every commit
└── VERSION                # Repo version
```

## Commands

```bash
# Verify docs and repo structure (required before commit)
./verify.sh

# Phase 0 harness (when implemented on M4)
python harness/m4_energy_harness.py --help
```

## Commit Workflow

1. Make focused, minimal diffs — match existing conventions.
2. Run `./verify.sh` — must pass.
3. Update `docs/OSLab_Program_Board.md` after significant tasks.
4. Write commit messages: what changed and why (complete sentences).

## Non-Negotiable Rules

- **No placeholder content** — never commit `"Content of X.md"` or `"Binary zip content"` stubs.
- **Measurement first** — no performance claims without IPJ/utilization/energy numbers.
- **HCA compliance** — self-improvement changes need an HCA Impact Statement + marginal IPJ.
- **Physics constraints** — respect SRAM (~28–30 MB), ANE-first routing, thermal limits.
- **Real binaries only** — PNG/ZIP files must be valid binary, not ASCII placeholders.

## Decision Escalation

| Situation | Action |
|-----------|--------|
| >10% IPJ impact or architectural change | Update Program Board + ask human |
| Thermal or SRAM violation | Stop and redesign |
| Blocked >2 hours | Update Program Board + ask human |
| Unclear approach | Propose 2–3 options with trade-offs |

## Key Docs by Task

| Task | Read First |
|------|------------|
| Phase 0 experiments | `Phase0_AI_Coder_Task_List.md`, `Phase0_Microbenchmark_Suite_Plan.md` |
| Run benchmarks on M4 | `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md` |
| IPJ / logging | `IPJ_Measurement_Protocol_Alalā.md` |
| Memory / SRAM | `Hierarchical_Memory_Architecture_Alalā.md`, `Memory_Access_Pattern_Guidelines_Alalā.md` |
| Compiler work | `Compiler_Passes_Skeleton_Alalā.md`, `Revised_Phase0_2_Systems_Plan_Alalā.md` |
| Self-improvement | `Alalā_Improvement_Playbook.md`, `Meta_Controller_Skeleton_Alalā.md`, HCA spec |
| Experiments | `Alalā_Experimentation_Framework.md` |
| Risks | `Risk_Register.md` |

## Cursor-Specific

- Project rules live in `.cursor/rules/*.mdc` — follow them automatically.
- Use `@docs/Project_Index_Alalā.md` to pull navigation context into chat.
- Invoke `@verifier` subagent after completing multi-step tasks.
- Do not read or commit secrets; `.env*` is ignored.
- Commit Phase 0 `logs/` and `results/` artifacts (powermetrics + JSONL) with harness changes.

## Target Platform

- **Hardware**: Mac Mini M4, 24 GB unified memory
- **OS**: macOS with `powermetrics` access (often requires sudo)
- **Python**: 3.11+ (numpy, matplotlib optional for plotting)

## License

Apache 2.0 — see `LICENSE`.
