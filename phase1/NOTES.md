# Phase 1 — Core ML Model Interface Notes

## Current prefill-only package (`models/qwen2.5-0.5b-ane.mlpackage`)

**Exported:** 2026-07-01 via `torch.export` + `use_cache=False` wrapper.

| Direction | Name | Shape | Dtype |
|-----------|------|-------|-------|
| Input | `inputIds` | `(1, max_ctx)` — max_ctx=1024 in current artifact | int32 |
| Output | `logits` | `(1, max_ctx, vocab)` — vocab=151936 | float16 |

**No KV cache.** Each `predict()` recomputes attention over the full padded sequence (prefill proxy). This explains ~4 t/s vs MLX ~84 t/s.

## Architecture (Qwen2.5-0.5B)

24 layers, 2 KV heads, head_dim 64, max_ctx ≤ 1024, vocab 151936.

### Prefill step (`qwen2.5-0.5b-prefill-kv.mlpackage`)

| Direction | Name | Shape | Notes |
|-----------|------|-------|-------|
| In | `inputIds` | `(1, 1024)` | int32; active prompt tokens in `[0:prompt_len)` |
| Out | `logits` | `(1, 1024, 151936)` | fp16 |
| Out | `keyCache` | `(24, 1, 2, 1024, 64)` | fp16 stacked per-layer K |
| Out | `valueCache` | `(24, 1, 2, 1024, 64)` | fp16 stacked per-layer V |

Export: `torch.export` → `coremltools.convert` with `use_cache=True`.

### Decode step (`qwen2.5-0.5b-decode-kv.mlpackage`) — MLState (default)

| Direction | Name | Shape | Notes |
|-----------|------|-------|-------|
| In | `inputIds` | `(1, 1)` | int32 single new token |
| In | `causalMask` | `(1, 1, 1, end_step)` | fp16; `end_step = cache_seq_len + 1`; zeros = attend to all cached keys |
| State | `keyCache` | `(24, 1, 2, 1024, 64)` | fp16; updated in-place via `MLState` |
| State | `valueCache` | `(24, 1, 2, 1024, 64)` | fp16; updated in-place via `MLState` |
| Out | `logits` | `(1, 1, 151936)` | fp16 next-token logits |

**Hand-off logic** (`phase1/kv_decode.py`):

1. Prefill Core ML → `keyCache` / `valueCache` outputs.
2. `state = decode_model.make_state()`; `write_state("keyCache", cache.astype(fp32))` (Core ML requires fp32 copy for prefill hand-off).
3. Each decode step: `predict({inputIds, causalMask}, state=state)`; increment `cache_seq_len`.
4. Lazy-load decode model after prefill to avoid GPU OOM when both packages resident.

Export method: `torch.jit.trace` + `ct.StateType` following [HuggingFace Mistral7B export](https://github.com/huggingface/swift-transformers/blob/preview/Examples/Mistral7B/export.py).

**Export fixes required for Qwen2** (not needed for Mistral):

- `rotate_half`: constant `head_dim // 2` split (avoid dynamic `x.shape[-1] // 2` → coremltools `int` op failure).
- `repeat_kv`: constant `num_heads` reshape (avoid dynamic `num_key_value_heads * n_rep`).
- SDPA inputs: cast K/V/mask to query dtype (fp16) before `scaled_dot_product_attention`.
- `SliceUpdateQwen2SdpaAttention`: integer-slice KV cache updates via `causalMask.shape[-1]`.

### TorchScript fallback (`models/qwen2.5-0.5b-decode-kv.pt`)

Same 2-input interface (`inputIds`, `causalMask`); KV held in traced module buffers. Used when `.mlpackage` missing. Prior explicit I/O path (`cacheSeqLen` + cache tensors) is **deprecated**.

## Export failure history

| Attempt | Error | Root cause |
|---------|-------|------------|
| `StaticCache` + tensor index write | `No matching select or slice` | Dynamic tensor indexing in cache update |
| `torch.export` + dynamic slice | `Eq(1024, u0)` | Dynamic slicing with tensor bounds |
| MLState v1 (stock RoPE) | `only 0-dimensional arrays can be converted to Python scalars` @ `rotate_half` | `x.shape[-1] // 2` in RoPE |
| MLState v2 | SDPA dtype mismatch | K cache fp32 vs query fp16 |
| **MLState v3 (current)** | ✅ converts | Patched attention + Mistral-style 2-input forward |

## MLX decode baseline

`harness/decode_client.DecodeRunner` → subprocess `mlx_lm` `stream_generate` with real KV inside MLX. Benchmark `--decode --backend mlx` uses this path.

## Measured decode residency (2026-07-01, M4 24 GB)

| Run | Backend | Decode runtime | ctx | Sust. t/s | ANE proxy | Temp steady |
|-----|---------|----------------|-----|-----------|-----------|-------------|
| `ane_residency_20260701T005247Z_b8d6539e` | Core ML | TorchScript .pt | 512 | **35.0** | 0.3% | 83.4°C |
| `ane_residency_20260701T010929Z_830681e7` | Core ML | **MLState .mlpackage** | 512 | **7.45** | **0.11%** | 83.5°C |

**Status:** Core ML decode export **unblocked**; first end-to-end MLState decode measured. ANE residency **not recovered** (0.11% vs 38% prefill proxy). Root cause identified: **decode graph does not compile for ANE** (see ANE Placement Diagnosis below).

## Compute unit configuration (reference run `ane_residency_20260701T010929Z_830681e7`)

| Setting | Value |
|---------|-------|
| Prefill load | `ct.ComputeUnit.ALL` via `coreml_instrumentation.load_coreml_model()` |
| Decode load | `ct.ComputeUnit.ALL` (lazy, after prefill) |
| Env flags | None (`COREML_VERBOSE`, `E5RT_LOG_LEVEL`, etc. unset) |
| Decode runtime | `coreml` (MLState `.mlpackage`, not TorchScript) |
| Result | 7.45 t/s sustained, **0.11% ANE proxy**, 83.5°C steady |

Instrumentation added in `phase1/coreml_instrumentation.py`; benchmark logs load config + `MLComputePlan` device class fractions at startup.

## ANE Placement Diagnosis (2026-07-01)

`MLComputePlan` analysis (`phase1/ane_placement_profile.py`, run `ane_placement_profile_20260701T012500Z_e43e6053`):

| Model | Ops | ANE preferred | GPU preferred | Source dialect |
|-------|-----|---------------|---------------|----------------|
| `qwen2.5-0.5b-ane.mlpackage` (prefill proxy) | 2573 | **48.7%** | 0.2% | TorchExport::ATEN |
| `qwen2.5-0.5b-prefill-kv.mlpackage` | 4044 | **31.0%** | 8.3% | TorchExport::ATEN |
| `qwen2.5-0.5b-decode-kv.mlpackage` (MLState) | 5165 | **0.0%** | **44.0%** | TorchScript |

**Forced ANE compile (`ComputeUnit.CPU_AND_NE`) on decode:** fails at load with `MILCompilerForANE error: failed to compile ANE model using ANEF. ANECCompile() FAILED.` Prefill models compile fine under `CPU_AND_NE`.

### Hypotheses (ranked by likelihood)

1. **ANE compiler rejection of stateful MLState graph** — decode includes `ios18.read_state`, `ios18.slice_update`, dynamic `shape`/`gather`/`cast` chains tied to `causalMask` length; ANE compile fails outright when GPU is disallowed. Matches 0% ANE in compute plan + `CPU_AND_NE` load failure.
2. **TorchScript export dialect vs torch.export** — decode exported via `torch.jit.trace` (TorchScript); prefill via `torch.export` (ATEN). Prefill achieves 31–49% ANE placement; decode does not. Re-export decode with `torch.export` may improve eligibility.
3. **Dynamic `causalMask` rank-4 shape** (`end_step` RangeDim 1..1024) — forces runtime shape inference; GPU-preferred ops (`gather`, `slice_by_index`, `greater_equal`) dominate decode plan. Prefill uses fixed `(1, 1024)` input.
4. **MLState in-place cache mutation** — 24 layers × `slice_update` on 1 GB state tensors; ANE may refuse read-modify-write on large state. Prefill emits cache as outputs (no state).
5. **Graph size / working set** — 5165 ops + 24×(1024×64) state vs 2573 ops prefill-only; memory footprint may push planner to GPU even under `ALL`.
6. **SDPA + RoPE slice patterns** — `ios18.scaled_dot_product_attention` is ANE-capable in prefill (listed in ANE ops) but decode SDPA is GPU-only in plan, likely due to surrounding state/mask ops.

### Quick configuration experiments

| Experiment | t/s sustained | ANE proxy | Notes |
|------------|---------------|-----------|-------|
| Reference `830681e7` (`ALL`, ctx 512, 30s) | 7.45 | 0.11% | Baseline |
| Profile `e43e6053` (`ALL`, ctx 512, ~17s) | **17.83** | **0.41%** | Warmed loop; still ~0% ANE in plan |
| `CPU_AND_NE` decode load | — | — | **Load fails** (ANE compile error) |
| `COREML_VERBOSE=1` | not run | — | Use Instruments Core ML template for per-op confirmation |

**Conclusion:** Low ANE is not a benchmark artifact — Core ML's compile-time plan assigns **zero** decode ops to ANE and cannot compile the graph for ANE-only execution. Powermetrics 0.11% matches GPU+CPU dominated execution.

### Manual profiling (Instruments)

1. Instruments.app → **Core ML** template.
2. Run `phase1/ane_placement_profile.py` or `ane_residency_benchmark.py --profile --backend coreml --decode`.
3. Compare decode process op placement vs `qwen2.5-0.5b-ane.mlpackage` prefill proxy.

### torch.export decode experiment (2026-07-01)

**Export:** `phase1/coreml_kv_convert.py --mode decode_torch_export --max-ctx 1024`  
**Artifact:** `models/qwen2.5-0.5b-decode-kv-torch-export.mlpackage` ✅  
**Dialect:** `TorchExport::ATEN` (matches prefill-kv)

| Direction | Name | Shape | Notes |
|-----------|------|-------|-------|
| In | `inputIds` | `(1, 1)` | int32 single new token |
| In | `keyCache` | `(24, 1, 2, 1024, 64)` | fp16; from prefill output |
| In | `valueCache` | `(24, 1, 2, 1024, 64)` | fp16; from prefill output |
| In | `cachePosition` | `(1,)` | int32; index of next write slot |
| In | `causalMask` | `(1, 1, 1, 1024)` | fp16 fixed width; zeros = attend |
| Out | `logits` | `(1, 1, 151936)` | fp16 |
| Out | `keyCacheOut` | `(24, 1, 2, 1024, 64)` | fp16 updated cache (renamed to avoid I/O name collision) |
| Out | `valueCacheOut` | `(24, 1, 2, 1024, 64)` | fp16 updated cache |

**Workarounds required for `torch.export`:**

1. **No module-attribute mutation** — functional `SliceKVCache` wrapper per forward; out-of-place `torch.cat` per layer.
2. **No tensor slice bounds** — mask-based slot write (`idx == position`) instead of `k_cache[..., begin:end, :]`.
3. **`index_copy` unsupported in coremltools** — mask write pattern used instead (adds `equal`/`tile`/`mul` ops).
4. **Input/output name collision** — Core ML rejects `keyCache` as both input and output; outputs named `keyCacheOut` / `valueCacheOut`.
5. **Fixed `causalMask` width** — `max_ctx` constant shape; validity enforced by mask content + full-cache SDPA (not dynamic `end_step` slice).
6. **`cachePosition` drives RoPE** — attention patch uses `cache_position` tensor instead of `causalMask.shape[-1]`.

**Export failures encountered (resolved):**

| Stage | Error | Fix |
|-------|-------|-----|
| `torch.export` | `Mutating module attribute k_cache` | Functional cache wrapper, no `self.kv_cache.k_cache = …` |
| `torch.export` | `Dynamic slicing with Tensor arguments` | Mask-based slot write + fixed mask width |
| `torch.export` | `graph input … received a mutation` | Out-of-place layer `cat` instead of in-place cache writes |
| `coremltools` | `Unsupported fx node index_copy` | Mask-based slot write |
| `coremltools` | `keyCache` used as input and output | Rename outputs to `keyCacheOut` / `valueCacheOut` |

### Export method comparison (decode @ ctx 512, M4 24 GB)

| Method | Dialect | ANE plan % | GPU plan % | `CPU_AND_NE` | Sust. t/s | ANE proxy | Ops |
|--------|---------|------------|------------|--------------|-----------|-----------|-----|
| TorchScript `.pt` | TorchScript | — | — | — | **35.0** | 0.3% | — |
| MLState `.mlpackage` | TorchScript | **0.0%** | **44.0%** | ❌ ANE compile fail | 7.45 | 0.11% | 5165 |
| **torch.export `.mlpackage`** | **TorchExport::ATEN** | **44.8%** | **1.3%** | ⚠️ not confirmed (disk) | **7.93** | **0.067%** | 3687 |

Profile run: `ane_placement_profile_20260701T020740Z_bf783c54` (`ComputeUnit.ALL`, ctx 512, 30-token target).

**Key finding:** `torch.export` recovers **compile-time ANE placement** (44.8% vs 0% MLState) — hypothesis #2 confirmed. Runtime ANE energy proxy remains **~0%** (GPU joules dominate: 1158 J GPU vs 1.0 J ANE in profile). Planner vs runtime gap likely due to large KV tensor I/O per step and mask-update overhead (`equal`/`tile` chains).

### Recommendation (2026-07-01)

**Compute-plan ANE placement improved meaningfully (>15–20%).** Continue optimizing the torch.export decode path rather than abandoning to hybrid:

1. **int4 weight quant** post-export (Mistral-style) — reduce memory bandwidth + graph size.
2. **Graph cleanup** — replace mask-based slot write with ANE-friendlier scatter/slice_update if exportable; prune `equal`/`tile` chains.
3. **Runtime ANE validation** — Instruments Core ML template to confirm planner vs execution placement; re-run powermetrics after quant.
4. **Throughput** — investigate KV I/O cost (2× 24×1024×64 fp16 copies per step); consider slimmer cache layout or stateful hybrid only for cache storage.
5. **Re-test `CPU_AND_NE`** once disk cache cleared — MLState failed outright; torch.export may compile.

If runtime ANE proxy stays &lt;5% after quant + cleanup, pivot to **hybrid architecture** (Core ML ANE prefill + optimized non-CoreML decode for throughput).

## Tooling and commands

| Script | Purpose |
|--------|---------|
| `coreml_kv_convert.py` | Export prefill-kv + MLState decode + `--mode decode_torch_export` |
| `coreml_instrumentation.py` | Load models with logged `ComputeUnit` + `MLComputePlan` |
| `ane_residency_benchmark.py` | Full benchmark; `--compute-units`, `--profile` |
| `ane_placement_profile.py` | Short decode profile + compute-plan JSON |

```bash
# Export torch.export decode (explicit KV I/O, no MLState)
phase1/.venv/bin/python phase1/coreml_kv_convert.py \
  --mode decode_torch_export --output-dir models --max-ctx 1024

# Benchmark with instrumentation (writes coreml_load_report.json per run)
phase1/.venv/bin/python phase1/ane_residency_benchmark.py \
  --backend coreml --decode --context 512 --compute-units all

# Placement profile (compute plan + powermetrics)
PYTHONPATH=phase1 phase1/.venv/bin/python phase1/ane_placement_profile.py \
  --decode-kv models/qwen2.5-0.5b-decode-kv-torch-export.mlpackage \
  --max-ctx 1024 --compute-units all --context 512
```

## Tracked measurement artifacts (committed to repo)

| Run ID | Type | Key metrics | Paths |
|--------|------|-------------|-------|
| `ane_residency_20260701T010929Z_830681e7` | MLState decode benchmark | 7.45 t/s, 0.11% ANE | `logs/ane_residency_20260701T010929Z_830681e7*`, `results/ane_residency/ane_residency_20260701T010929Z_830681e7/` |
| `ane_placement_profile_20260701T012500Z_e43e6053` | ANE placement profile (MLState) | 17.83 t/s, 0.41% ANE; 0% ANE compute plan | `results/ane_placement_profile/ane_placement_profile_20260701T012500Z_e43e6053/`, `logs/ane_placement_profile_all_ctx512.powermetrics.txt` |
| `ane_placement_profile_20260701T020740Z_bf783c54` | torch.export decode profile | 7.93 t/s, 0.067% ANE proxy; **44.8% ANE compute plan** | `results/ane_placement_profile/ane_placement_profile_20260701T020740Z_bf783c54/`, `logs/ane_placement_profile_all_ctx512.powermetrics.txt` |
| `ane_residency_20260701T010854Z_a164b6cc` | Failed (pre-fix OOM) | GPU OOM loading both models | `logs/ane_residency_20260701T010854Z_a164b6cc_ctx512.powermetrics.txt` |

Local-only (not tracked): `models/*.mlpackage/`, `models/qwen2.5-0.5b-decode-kv.pt`.