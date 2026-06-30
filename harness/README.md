# Harness

Measurement harness for Phase 0 M4 micro-benchmarks.

## Target file

`m4_energy_harness.py` — energy logging via `powermetrics`, thermal monitoring, benchmark modes.

## Modes (from docs)

- `thermal_baseline`
- `sram_cliff`
- `kv_comparison`

## Docs

- `docs/How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`
- `docs/Phase0_Microbenchmark_Suite_Plan.md`
- `docs/IPJ_Measurement_Protocol_Alalā.md`

## Usage (when implemented)

```bash
python harness/m4_energy_harness.py --mode thermal_baseline --duration 120
```

Logs go to `logs/`; results to `results/`.
