# Phase 1 — ANE Residency Tooling

Measurement and conversion tooling for Phase 1 ANE-first execution on Mac Mini M4 24 GB.

**Strategy doc:** [`docs/Phase1_ANE_First_Strategy.md`](../docs/Phase1_ANE_First_Strategy.md)  
**Program status:** [`docs/OSLab_Program_Board.md`](../docs/OSLab_Program_Board.md)

## Files

| File | Purpose |
|------|---------|
| `coreml_convert.py` | Hugging Face causal LM → Core ML `.mlpackage` (`torch.export` primary path) |
| `ane_residency_benchmark.py` | MLX vs Core ML comparison at ctx 512/1024 with powermetrics |
| `requirements.txt` | Pinned deps for conversion venv (Python 3.12) |

## Setup

```bash
cd /path/to/alala
uv venv phase1/.venv --python 3.12
uv pip install --python phase1/.venv/bin/python -r phase1/requirements.txt
```

Harness benchmarks still use repo `.env` for `SUDO_PASSWORD` (powermetrics) and `MLX_PYTHON` (MLX decode subprocess).

## Convert Qwen2.5-0.5B

```bash
phase1/.venv/bin/python phase1/coreml_convert.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output models/qwen2.5-0.5b-ane.mlpackage \
  --context-size 1024
```

Converted packages are **gitignored** (`models/*.mlpackage/`). Regenerate locally after clone.

**Note:** `torch.jit.trace` fails on Qwen2 ops; the script uses `torch.export` + `run_decompositions()` first.

## Run ANE residency benchmark

```bash
# MLX baseline (0.5B; ~0% ANE expected)
python3 phase1/ane_residency_benchmark.py --backend mlx

# Core ML (requires phase1 venv for coremltools)
phase1/.venv/bin/python phase1/ane_residency_benchmark.py \
  --backend coreml \
  --model models/qwen2.5-0.5b-ane.mlpackage \
  --context 512,1024 \
  --coreml-context-size 1024
```

Outputs: `logs/ane_residency_<run_id>.jsonl`, per-context `.powermetrics.txt`, `results/ane_residency/<run_id>/summary.json`.

## First measured results (2026-07-01)

| Backend | ctx | Sust. tok/s | ANE proxy |
|---------|-----|-------------|-----------|
| MLX 0.5B | 512 | 84.2 | ~0% |
| Core ML | 512 | 4.2 | **38.0%** |
| Core ML | 1024 | 3.9 | **11.7%** |

Core ML path is prefill-proxy only until stateful KV decode is implemented.

## Thermal safety

Default abort threshold: **85 °C** (`--temp-threshold`). Allow 60–90 s cooldown between context steps. Phase 0 decode sweeps used **88 °C** for 7B MLX loads.