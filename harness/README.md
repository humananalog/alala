# Harness

Measurement harness for Phase 0 M4 micro-benchmarks.

## Target file

`m4_energy_harness.py` — energy logging via `powermetrics`, thermal monitoring, benchmark modes.

## Modes (from docs)

- `thermal_baseline` — Benchmark 1; thermal curve + safe sustained envelope
- `sram_cliff` — Benchmark 2; SRAM cliff detection (≥30% sustained throughput drop)
- `kv_comparison` — Benchmark 3; FP16 vs int4 KV including `energy_dequant_joules`
- `orchestration` — Benchmark 4; CPU orchestration energy vs ANE
- `ane_utilization` — E1; ANE compute fraction + orchestration tax
- `thermal_ipj_curve` — E2; sustained IPJ vs thermal headroom time series
- `meta_tax` — E3; meta-overhead joules + `net_ipj_delta`
- `memory_spill` — E4; spill vs recompute joules/token

Each mode must emit raw `powermetrics` log + JSONL per `IPJ_Measurement_Protocol_Alalā.md` §2.1–§2.5.

## Docs

- `docs/How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`
- `docs/Phase0_Microbenchmark_Suite_Plan.md`
- `docs/IPJ_Measurement_Protocol_Alalā.md`

## Usage

```bash
# On physical Mac Mini M4 24 GB (usually requires sudo for powermetrics)
sudo python harness/m4_energy_harness.py --mode thermal_baseline --duration 600

# Off-hardware structure check only
python harness/m4_energy_harness.py --dry-run --mode thermal_baseline --duration 30
```

Logs go to `logs/`; results to `results/`.
