# Phase 1 — ANE-First Execution & Seeding Model

**Status**: Active — first experiment complete (2026-07-01)
**Goal**: Maximize fraction of forward pass routed to ANE on M4 while maintaining sustainable IPJ and thermal headroom. First milestone: measurable ANE utilization on a small capable model under L_cliff=1024.

## Physics Constraints from Phase 0
- L_cliff = 1024 context (33.7% throughput drop)
- Thermal steady-state ~82.7–86.5 °C with throttling in ~5 s
- Orchestration tax is low (~4%)
- Current MLX decode paths show ~0% ANE utilization (GPU fallback)
- Dequant cost for int4 is low but not yet winning

## Phase 1 Principles
1. Route as much of the forward pass and verification as possible to the ANE.
2. Accept lower peak speed if it delivers better sustained IPJ and thermal headroom.
3. Memory system design must keep hot KV/activations inside or near ANE SRAM budget.
4. All major decisions gated by real M4 measurements (powermetrics + thermal + ANE attribution).

## First Experiment (This Week)
Select and convert a small seeding model (≤ ~1B params) to CoreML/ANE-friendly path.
Measure:
- Real ANE utilization % (or best proxy)
- Sustained tokens/s at context 512 and 1024 under thermal steady-state
- Energy per token and IPJ
- Thermal behavior vs current MLX GPU baseline

Success gate: Measurable ANE residency + sustained IPJ within 10% of current MLX path (or better).

## First Experiment Results (2026-07-01)

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` → Core ML via `torch.export` (`phase1/coreml_convert.py`)  
**Benchmark:** `phase1/ane_residency_benchmark.py` with powermetrics + macmon thermal

| Backend | ctx | Sust. tok/s | ANE proxy | Energy ANE (J) | Temp steady |
|---------|-----|-------------|-----------|----------------|-------------|
| MLX 0.5B | 512 | 84.2 | ~0% | 0 | 81.9 °C |
| Core ML | 512 | 4.2 | **38.0%** | 200 | 62.9 °C |
| Core ML | 1024 | 3.9 | **11.7%** | 172 | 84.8 °C |

**Runs:** `ane_residency_20260701T001734Z_0bd0328f` (MLX), `ane_residency_20260701T002500Z_d1b410d0` (Core ML)  
**Artifacts:** `logs/ane_residency_*.jsonl`, `results/ane_residency/<run_id>/`

**Gate assessment:**
- Measurable ANE residency: **yes** (38% @ ctx 512)
- ANE > 60% target: **no** (not yet)
- IPJ within 10% of MLX: **no** (prefill proxy; no KV decode)
- Thermal compliance: partial (MLX aborted at 92 °C; Core ML post-run flag at 87.7 °C)

**Next:** stateful Core ML decode + KV cache (Apple on-device Llama pattern); re-benchmark IPJ and ANE at ctx 512/1024.

## KV Cache Decode Results (2026-07-01)

**Artifacts:** `phase1/kv_decode.py`, `phase1/coreml_kv_convert.py`, `models/qwen2.5-0.5b-prefill-kv.mlpackage`, `models/qwen2.5-0.5b-decode-kv.pt`  
**Benchmark:** `phase1/ane_residency_benchmark.py --decode` (88 °C threshold, 60 s step / 30 s steady)

| Run | Backend | ctx | Sust. tok/s | ANE proxy | Decode runtime | Notes |
|-----|---------|-----|-------------|-----------|----------------|-------|
| `ane_residency_20260701T005044Z_d9ea7ae1` | MLX 0.5B | 512 | **106.7** | ~0% | mlx_lm KV | Baseline |
| `ane_residency_20260701T005414Z_011715de` | MLX 0.5B | 1024 | **57.6** | ~4% | mlx_lm KV | Thermal OK @ 82 °C |
| `ane_residency_20260701T005247Z_b8d6539e` | Core ML | 512 | **35.0** | 0.3% | TorchScript fallback | Prefill on Core ML ANE; decode on CPU TS |
| `ane_residency_20260701T005431Z_c4529935` | Core ML | 1024 | — | — | — | **OOM** on prefill-kv re-prefill (GPU) |

**vs prefill proxy (4.2 t/s @ ctx 512):** KV decode path is **~8× faster** (35 t/s).  
**vs MLX @ ctx 512:** Core ML hybrid is **~3× slower** (within 3–5× gate) but ANE attribution collapses because decode `.mlpackage` conversion is blocked.

**Blocker:** `coremltools.convert(TorchScript)` fails with `No matching select or slice` on dynamic KV cache index writes. TorchScript runtime works; Core ML decode package still needed for ANE residency during sustained decode.

**Next:** unblock Core ML decode conversion (MLState / export-friendly cache write), re-run ANE + IPJ at ctx 512; investigate ctx 1024 OOM on prefill-kv recycle.

## Candidate Seeding Models (ranked for first try)
1. Qwen2.5-0.5B-Instruct (strong instruction following, good convertibility)
2. Phi-3.5-mini / Phi-3-mini-128k (excellent reasoning per size)
3. Gemma-2-2B-IT (quantized) — slightly larger, test later
4. Custom distilled 350M model (if needed for maximum ANE mapping)

Start with #1 or #2.

## Next After First Experiment
- KV memory system design that respects L_cliff
- Minimal self-improvement scaffold with IPJ + HCA gating
- Thermal-aware scheduler