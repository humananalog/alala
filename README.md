<p align="center">
  <img src="assets/alala-img.jpg" alt="Alalā header: hardware-aware self-improving AI on M4" width="100%">
</p>

<h1 align="center">Alalā</h1>

<p align="center">
  <strong>Physics-first, measurement-driven AI for Apple Silicon M4. Goal: maximize Intelligence per Joule (IPJ).</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/humananalog/alala"><img src="https://img.shields.io/badge/Platform-Mac%20Mini%20M4%20(24%20GB)-black?logo=apple" alt="Platform: Mac Mini M4"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <img src="https://img.shields.io/badge/Phase%200-Complete-brightgreen" alt="Phase 0: Complete">
</p>

## About

**Alalā** is an open-source research and systems project for hardware-aware AI on Apple Silicon. The primary metric is **Intelligence per Joule (IPJ)**: useful work per joule, measured on real hardware, not from FLOPs or peak throughput alone.

Model architecture, compiler/runtime, and memory layout are co-designed around Mac Mini M4 constraints (24 GB unified memory): ANE residency, SRAM limits (~28–30 MB working sets), thermal headroom, and orchestration overhead. Self-improvement loops require measurable IPJ gains and compliance with the **Human Cooperation Attractor (HCA)**.

**Current status:** Phase 0 is complete. The M4 measurement harness runs on physical silicon. Raw `powermetrics` logs and JSONL artifacts live in `logs/` and `results/`.

## Phase 0 Highlights

Measured on Mac Mini M4 24 GB. Full write-up: [`docs/Phase0_Results_Summary_Alalā.md`](docs/Phase0_Results_Summary_Alalā.md).

| Metric | Result |
|--------|--------|
| SRAM cliff \(L_{\text{cliff}}\) | **1024** context tokens (33.7% sustained throughput drop) |
| Thermal steady state | **~82.7 °C** under sustained decode load |
| Orchestration overhead | **3.7–4.3%** of total energy (Python-style dispatch) |
| int4 KV dequant cost | **+5.55 J** (~0.5% overhead; ΔIPJ ≈ −0.0028 vs FP16) |
| Safe sustained envelope | **≤ 85 °C** with active thermal monitoring |

These numbers gate architectural decisions. No performance claim without attached `powermetrics` and thermal data per the [IPJ Measurement Protocol](docs/IPJ_Measurement_Protocol_Alalā.md).

## Features

- **M4 energy harness:** `harness/m4_energy_harness.py` with modes `thermal_baseline`, `sram_cliff`, `kv_comparison`, `orchestration`
- **IPJ-first logging:** structured JSONL plus raw `powermetrics` per experiment
- **20 indexed docs:** physics, HCA, memory, compiler, experimentation, Phase 0/1 planning
- **Agent workflow:** `AGENTS.md`, Cursor rules, explicit task lists for work on physical M4
- **Tracked artifacts:** benchmark logs and results in-repo for audit and comparison

## Requirements

| Requirement | Notes |
|-------------|-------|
| **Hardware** | Physical **Mac Mini M4, 24 GB** unified memory (no cloud or simulated substitutes for benchmarks) |
| **OS** | macOS with `powermetrics` access (typically requires `sudo`) |
| **Python** | 3.11+ |
| **Dependencies** | `numpy` (required); `matplotlib` (optional); MLX stack for decode workloads |
| **Secrets** | Optional `.env` with `SUDO_PASSWORD` and `MLX_PYTHON` (see [Harness README](harness/README.md)) |

## Quick Start

```bash
git clone https://github.com/humananalog/alala.git
cd alala

# Verify repository structure and documentation
./verify.sh

# Inspect harness options (physical M4 only for benchmarks)
python harness/m4_energy_harness.py --help
```

### Run a benchmark (M4 hardware)

Idle the machine **10+ minutes** before measuring. All modes write raw `powermetrics` logs to `logs/` and summaries to `results/`.

```bash
# Thermal baseline: power curve and safe sustained envelope
python harness/m4_energy_harness.py --mode thermal_baseline --duration 600 --idle-seconds 60

# SRAM cliff: sweep context lengths to find L_cliff
python harness/m4_energy_harness.py --mode sram_cliff --model baseline --max-context 8192

# KV comparison: FP16 vs int4 including dequant energy
python harness/m4_energy_harness.py --mode kv_comparison --context 512 --iterations 3

# Orchestration: CPU dispatch overhead vs tight MLX loop
python harness/m4_energy_harness.py --mode orchestration --context 512 --iterations 3
```

Step-by-step guide: [`docs/How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`](docs/How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md).

## Project Structure

```text
alala/
├── AGENTS.md              # Instructions for Cursor / cloud AI coding agents
├── assets/                # README and project assets
├── docs/                  # Authoritative documentation (20 indexed docs)
├── harness/               # Phase 0 M4 measurement harness
├── experiments/           # Experiment scripts and configs
├── logs/                  # JSONL + powermetrics experiment logs
├── results/               # Benchmark outputs per run
├── checkpoints/           # Rollback checkpoints (contents gitignored)
├── verify.sh              # Pre-commit verification
└── VERSION                # Repository version
```

## Documentation

Start at the [Project Index](docs/Project_Index_Alalā.md).

| Topic | Document |
|-------|----------|
| Navigation hub | [`docs/Project_Index_Alalā.md`](docs/Project_Index_Alalā.md) |
| Program status | [`docs/OSLab_Program_Board.md`](docs/OSLab_Program_Board.md) |
| Phase 0 results | [`docs/Phase0_Results_Summary_Alalā.md`](docs/Phase0_Results_Summary_Alalā.md) |
| IPJ protocol | [`docs/IPJ_Measurement_Protocol_Alalā.md`](docs/IPJ_Measurement_Protocol_Alalā.md) |
| Physics and M4 constraints | [`docs/Alalā_Physics_Corrected_Foundation.md`](docs/Alalā_Physics_Corrected_Foundation.md) |
| HCA spec | [`docs/Alalā_Core_Invariant_Specification_HCA.md`](docs/Alalā_Core_Invariant_Specification_HCA.md) |
| AI coder rules | [`docs/AI_Coder_Rules_Guidelines_Alalā.md`](docs/AI_Coder_Rules_Guidelines_Alalā.md) |
| Harness reference | [`harness/README.md`](harness/README.md) |

## Core Principles

- **Physics first:** data movement, SRAM budgeting, thermal headroom, ANE efficiency are constraints, not afterthoughts
- **Measurement-driven:** decisions need IPJ, utilization, and energy numbers from hardware
- **Bounded self-improvement:** loops need measurable IPJ gains and HCA compliance
- **ANE-first routing:** compute-bound ops default to ANE; measure orchestration overhead, do not assume it

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 0** | ANE characterization and measurement infrastructure | Complete |
| **Phase 1** | ANE residency (>60% utilization target), safe operating region, first IPJ-gated self-improvement scaffold | Next |

Phase 1 entry criteria: [`docs/Phase0_Results_Summary_Alalā.md`](docs/Phase0_Results_Summary_Alalā.md) (Phase 1 Entry Criteria section).

## Contributing

Read [`AGENTS.md`](AGENTS.md) and [`docs/AI_Coder_Rules_Guidelines_Alalā.md`](docs/AI_Coder_Rules_Guidelines_Alalā.md) first. Check [`docs/OSLab_Program_Board.md`](docs/OSLab_Program_Board.md) for current tasks and blockers.

1. Keep diffs focused; match existing naming and patterns
2. Run `./verify.sh` before every commit
3. Include `logs/` and `results/` artifacts with harness or benchmark changes
4. No placeholder stubs; no unmeasured performance claims
5. Follow `.cursor/rules/no-ai-markers.mdc` for prose (no em dashes, no stock AI phrasing)

For architectural changes or >10% IPJ impact, update the Program Board and open a discussion before merging.

## Authors

- **[Human Analog Ltd](https://github.com/humananalog)**
- **Lucius Stel**

## License

[Apache License 2.0](LICENSE). Copyright Human Analog Ltd.
