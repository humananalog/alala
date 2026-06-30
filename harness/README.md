# Harness

Measurement harness for Phase 0 M4 micro-benchmarks.

## Target file

`m4_energy_harness.py` — energy logging via `powermetrics`, thermal monitoring, benchmark modes.

## Modes (from docs)

- `setup_check` — W1-01 powermetrics + logging validation (writes `logs/setup_log.jsonl`)
- `thermal_baseline` — Benchmark 1; optional `--idle-duration` then sustained load
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
# W1-01 setup check (30s minimum)
sudo python harness/m4_energy_harness.py --mode setup_check --duration 30

# W1-02 thermal baseline (idle + sustained)
sudo python harness/m4_energy_harness.py --mode thermal_baseline --idle-duration 600 --duration 600

# Off-hardware structure check only
python harness/m4_energy_harness.py --dry-run --mode thermal_baseline --duration 30
```

Logs go to `logs/`; results to `results/`.
