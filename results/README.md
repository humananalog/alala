# Results — Phase 0 M4 Measurements

Artifacts from `harness/m4_energy_harness.py` land here and in `logs/`.

## Validation

```bash
# After any run, validate JSONL structure:
python harness/validate_artifact.py logs/<experiment_id>.jsonl

# For M4 IPJ claims (rejects dry-run logs):
python harness/validate_artifact.py --require-m4 logs/<experiment_id>.jsonl
```

## Measurement status

Update `results/measurement_status.json` with `m4_validated: true` and artifact path after each criterion closes.

```bash
python harness/mark_validated.py --criterion thermal_baseline --jsonl logs/<experiment_id>.jsonl
```

## Layout

| Directory | Content |
|-----------|---------|
| `thermal_baseline/` | W1-02 summaries |
| `sram_cliff/` | W1-03 + L_cliff |
| `kv_comparison/` | W1-04 FP16 vs int4 |
| `thermal_ipj_curve/` | E2 time series |
