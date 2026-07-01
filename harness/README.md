# Harness

Measurement harness for Phase 0 M4 micro-benchmarks.

## Target file

`m4_energy_harness.py` — energy logging via `powermetrics`, thermal monitoring, benchmark modes.

## Modes (from docs)

- `thermal_baseline` — Benchmark 1; thermal curve + safe sustained envelope
- `sram_cliff` — Benchmark 2; SRAM cliff detection (≥30% sustained throughput drop)
- `kv_comparison` — Benchmark 3; FP16 vs int4 KV including `energy_dequant_joules`
- `orchestration` — Benchmark 4; CPU orchestration energy vs ANE

Each mode must emit raw `powermetrics` log + JSONL per `IPJ_Measurement_Protocol_Alalā.md` §2.1–§2.3.

## Docs

- `docs/How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`
- `docs/Phase0_Microbenchmark_Suite_Plan.md`
- `docs/IPJ_Measurement_Protocol_Alalā.md`

## Usage

```bash
python harness/m4_energy_harness.py --mode thermal_baseline --duration 600 --idle-seconds 60
python harness/m4_energy_harness.py --mode sram_cliff --model baseline --max-context 8192
```

Requires `SUDO_PASSWORD` in repo `.env` (or run with sudo). Optional `MLX_PYTHON` in `.env` if `mlx_lm` is not on the default interpreter.

Logs go to `logs/`; results to `results/`.

## Phase 1 (ANE residency)

Phase 1 benchmarks live in `phase1/` and reuse this harness's powermetrics patterns. See [`phase1/README.md`](../phase1/README.md).
