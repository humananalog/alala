<p align="center">
  <img src="assets/alala-img.jpg" alt="Alalā: Hardware-Aware Self-Improving AI models for M4 Silicon" width="100%">
</p>

<p align="center">
  <strong>Physics-first, measurement-driven AI for Apple Silicon M4. Goal: maximize Intelligence per Joule (IPJ).</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/humananalog/alala"><img src="https://img.shields.io/badge/Platform-Mac%20Mini%20M4%20(24%20GB)-black?logo=apple" alt="Platform: Mac Mini M4"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <img src="https://img.shields.io/badge/Phase%200-Complete-brightgreen" alt="Phase 0: Complete">
  <img src="https://img.shields.io/badge/Phase%201-Active-blue" alt="Phase 1: Active">
</p>

## About

**Alalā** is an open-source research and systems project for hardware-aware AI on Apple Silicon. The primary metric is **Intelligence per Joule (IPJ)**: useful work per joule, measured on real hardware, not from FLOPs or peak throughput alone.

Model architecture, compiler/runtime, and memory layout are co-designed around Mac Mini M4 constraints (24 GB unified memory): ANE residency, SRAM limits (~28–30 MB working sets), thermal headroom, and orchestration overhead. Self-improvement loops require measurable IPJ gains and compliance with the **Human Cooperation Attractor (HCA)**.

**Current status (2026-07-01):** Phase 0 is complete on physical M4 silicon. **Phase 1 is active**: first ANE residency experiment measured **38% ANE energy share** on a Core ML Qwen2.5-0.5B path vs ~0% on MLX GPU. Raw `powermetrics` logs and JSONL artifacts live in `logs/` and `results/`.

## Phase 0 Highlights

Measured on Mac Mini M4 24 GB. Full write-up: [`docs/Phase0_Results_Summary_Alalā.md`](docs/Phase0_Results_Summary_Alalā.md).

| Metric | Result |
|--------|--------|
| SRAM cliff \(L_{\text{cliff}}\) | **1024** context tokens (33.7% sustained throughput drop) |
| Thermal steady state | **~82.7 °C** under sustained decode load |
| Orchestration overhead | **3.7–4.3%** of total energy (Python-style dispatch) |
| int4 KV dequant cost | **+5.55 J** (~0.5% overhead; ΔIPJ ≈ −0.0028 vs FP16) |
| MLX ANE utilization | **~0%** (GPU-routed decode path) |
| Safe sustained envelope | **≤ 85 °C** with active thermal monitoring |

## Phase 1 Highlights (First ANE Experiment)

Measured 2026-07-01. Strategy: [`docs/Phase1_ANE_First_Strategy.md`](docs/Phase1_ANE_First_Strategy.md). Live status: [`docs/OSLab_Program_Board.md`](docs/OSLab_Program_Board.md).

| Backend | ctx | Sust. tok/s | ANE proxy | Temp steady | Run ID |
|---------|-----|-------------|-----------|-------------|--------|
| MLX 0.5B | 512 | **84.2** | ~0% | 81.9 °C | `ane_residency_20260701T001734Z_0bd0328f` |
| Core ML Qwen2.5-0.5B | 512 | 4.2 | **38.0%** | 62.9 °C | `ane_residency_20260701T002500Z_d1b410d0` |
| Core ML Qwen2.5-0.5B | 1024 | 3.9 | **11.7%** | 84.8 °C | `ane_residency_20260701T002500Z_d1b410d0` |

**Interpretation:** Core ML achieves measurable ANE residency; MLX remains faster but GPU-heavy. Core ML benchmark uses a prefill proxy (stateful KV decode is next). IPJ gate (within 10% of MLX) is **not yet met**.

These numbers gate architectural decisions. No performance claim without attached `powermetrics` and thermal data per the [IPJ Measurement Protocol](docs/IPJ_Measurement_Protocol_Alalā.md).

## Features

- **Phase 0 harness:** `harness/m4_energy_harness.py` — `thermal_baseline`, `sram_cliff`, `kv_comparison`, `orchestration`
- **Phase 1 ANE tooling:** `phase1/coreml_convert.py`, `phase1/ane_residency_benchmark.py`
- **IPJ-first logging:** structured JSONL plus raw `powermetrics` per experiment
- **22 indexed docs:** vision, physics, HCA, memory, compiler, Phase 0/1/2 planning and results
- **Agent workflow:** `AGENTS.md`, Cursor rules, explicit task lists for work on physical M4
- **Tracked artifacts:** benchmark logs and results in-repo for audit and comparison

## Requirements

| Requirement | Notes |
|-------------|-------|
| **Hardware** | Physical **Mac Mini M4, 24 GB** unified memory (no cloud or simulated substitutes for benchmarks) |
| **OS** | macOS with `powermetrics` access (typically requires `sudo`) |
| **Python** | 3.11+ for harness; **3.12 venv** recommended for Phase 1 Core ML (`phase1/requirements.txt`) |
| **Dependencies** | `numpy` (harness); MLX stack for decode workloads; `torch`, `transformers`, `coremltools` for Phase 1 |
| **Secrets** | Optional `.env` with `SUDO_PASSWORD` and `MLX_PYTHON` (see [Harness README](harness/README.md)) |

## Quick Start

```bash
git clone https://github.com/humananalog/alala.git
cd alala

./verify.sh
python harness/m4_energy_harness.py --help
```

### Phase 0 benchmarks (M4 hardware)

Idle the machine **10+ minutes** before measuring. All modes write raw `powermetrics` logs to `logs/` and summaries to `results/`.

```bash
python harness/m4_energy_harness.py --mode thermal_baseline --duration 600 --idle-seconds 60
python harness/m4_energy_harness.py --mode sram_cliff --model baseline --max-context 8192
python harness/m4_energy_harness.py --mode kv_comparison --context 512 --iterations 3
python harness/m4_energy_harness.py --mode orchestration --context 512 --iterations 3
```

Guide: [`docs/How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`](docs/How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md).

### Phase 1 ANE residency (M4 hardware)

```bash
# One-time: Phase 1 venv
uv venv phase1/.venv --python 3.12
uv pip install --python phase1/.venv/bin/python -r phase1/requirements.txt

# Convert Qwen2.5-0.5B to Core ML (torch.export path; ~1 GB mlpackage, gitignored)
phase1/.venv/bin/python phase1/coreml_convert.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output models/qwen2.5-0.5b-ane.mlpackage \
  --context-size 1024

# MLX baseline (system python3 + MLX_PYTHON in .env)
python3 phase1/ane_residency_benchmark.py --backend mlx

# Core ML benchmark (use phase1 venv — needs coremltools)
phase1/.venv/bin/python phase1/ane_residency_benchmark.py \
  --backend coreml \
  --model models/qwen2.5-0.5b-ane.mlpackage \
  --context 512,1024 \
  --coreml-context-size 1024
```

Details: [`phase1/README.md`](phase1/README.md).

## Project Structure

```text
alala/
├── AGENTS.md              # Instructions for Cursor / cloud AI coding agents
├── assets/                # README and project assets
├── docs/                  # Authoritative documentation (22 indexed docs)
├── harness/               # Phase 0 M4 measurement harness
├── phase1/                # Phase 1 Core ML conversion + ANE residency benchmarks
├── models/                # Converted .mlpackage output (gitignored; see .gitkeep)
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
| Phase 1 strategy | [`docs/Phase1_ANE_First_Strategy.md`](docs/Phase1_ANE_First_Strategy.md) |
| IPJ protocol | [`docs/IPJ_Measurement_Protocol_Alalā.md`](docs/IPJ_Measurement_Protocol_Alalā.md) |
| Physics and M4 constraints | [`docs/Alalā_Physics_Corrected_Foundation.md`](docs/Alalā_Physics_Corrected_Foundation.md) |
| HCA spec | [`docs/Alalā_Core_Invariant_Specification_HCA.md`](docs/Alalā_Core_Invariant_Specification_HCA.md) |
| AI coder rules | [`docs/AI_Coder_Rules_Guidelines_Alalā.md`](docs/AI_Coder_Rules_Guidelines_Alalā.md) |
| Phase 0 harness | [`harness/README.md`](harness/README.md) |
| Phase 1 tooling | [`phase1/README.md`](phase1/README.md) |

## Core Principles

- **Physics first:** data movement, SRAM budgeting, thermal headroom, ANE efficiency are constraints, not afterthoughts
- **Measurement-driven:** decisions need IPJ, utilization, and energy numbers from hardware
- **Bounded self-improvement:** loops need measurable IPJ gains and HCA compliance
- **ANE-first routing:** compute-bound ops default to ANE; Phase 1 now has measured Core ML residency signal

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 0** | ANE characterization and measurement infrastructure | **Complete** (2026-06-30) |
| **Phase 1** | ANE residency, Core ML seeding model, IPJ-gated iteration | **Active** — 38% ANE @ ctx 512 measured |
| **Phase 2** | Compiler passes, KV system, self-improvement scaffold | Planned |

**Phase 1 next:** stateful Core ML decode with KV cache, push ANE toward >60%, close IPJ gap vs MLX.

## Contributing

Read [`AGENTS.md`](AGENTS.md) and [`docs/AI_Coder_Rules_Guidelines_Alalā.md`](docs/AI_Coder_Rules_Guidelines_Alalā.md) first. Check [`docs/OSLab_Program_Board.md`](docs/OSLab_Program_Board.md) for current tasks and blockers.

1. Keep diffs focused; match existing naming and patterns
2. Run `./verify.sh` before every commit
3. Include `logs/` and `results/` artifacts with harness or benchmark changes
4. No placeholder stubs; no unmeasured performance claims
5. Follow `.cursor/rules/no-ai-markers.mdc` for prose

For architectural changes or >10% IPJ impact, update the Program Board and open a discussion before merging.

## Authors

- **[Human Analog Ltd](https://github.com/humananalog)**
- **Lucius Stel**

## License

[Apache License 2.0](LICENSE). Copyright Human Analog Ltd.