# Phase 1 — Core ML Model Interface Notes

## Strategic Context

As of **July 2026**, Alalā has shifted to **high-risk, deep + broad exploration mode (Option C)** — fundamental hardware–model co-design rather than incremental Core ML optimization. Phase 1 notes and measurements remain authoritative baselines. See [`docs/Alalā_Vision_and_Strategy.md`](../docs/Alalā_Vision_and_Strategy.md).

**KV / state redesign prior art**: [`docs/KV_Cache_Techniques_for_NPUs_and_Apple_Silicon.md`](../docs/KV_Cache_Techniques_for_NPUs_and_Apple_Silicon.md) (literature review for Research Agenda thread 1).  
**Exp 1 (ring buffer)**: [`docs/KV_Cache_Redesign_Synthesis_and_Experiments.md`](../docs/KV_Cache_Redesign_Synthesis_and_Experiments.md) — **Iterate**; ring512 int4 default decode path.

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

### int4 weight quantization (2026-07-01)

**Tool:** `phase1/coreml_quantize.py` — post-export `linear_quantize_weights` with `OpLinearQuantizerConfig(dtype='int4')`.

```bash
phase1/.venv/bin/python phase1/coreml_quantize.py \
  --input models/qwen2.5-0.5b-decode-kv-torch-export.mlpackage \
  --output models/qwen2.5-0.5b-decode-kv-torch-export-int4.mlpackage
```

**KV cache handling:** `linear_quantize_weights` compresses **linear layer weights only**. Activation I/O (`keyCache`, `valueCache`, `cachePosition`, `causalMask`, `logits`, `keyCacheOut`, `valueCacheOut`) remain fp16 — no special exclusion needed. `op_selector` in `OptimizationConfig` is deprecated in coremltools ≥8; global linear config suffices.

**Limitations observed:**
- `RuntimeWarning: invalid value encountered in divide` on a subset of weights during quant (likely near-zero scales); model still validates and runs.
- Package size: **952 MB → 240 MB** (fp16 torch.export → int4).
- Placement profile decode loop **GPU OOM** when loading prefill + int4 decode + prefill-proxy simultaneously; 60 s residency benchmark (prefill + decode only) succeeds.
- Cleared **30 GB** `e5rt.e5bundlecache` required before quant save on disk-constrained host.

### torch.export decode: fp16 vs int4 comparison (ctx 512, M4 24 GB)

| Variant | ANE plan % | GPU plan % | Sust. t/s | ANE proxy | ANE J | GPU J | CPU J | Temp steady | Package |
|---------|------------|------------|-----------|-----------|-------|-------|-------|-------------|---------|
| torch.export fp16 | **44.8%** | 1.3% | 7.93 | 0.067% | 1.0 | 1158 | 314 | ~84°C | 952 MB |
| **torch.export int4** | **44.1%** | 2.0% | **27.73** | **2.90%** | **31.9** | **646** | 421 | **75.2°C** | **240 MB** |

Runs: fp16 profile `bf783c54`; int4 compute plan from `1b69eca7` load + 60 s benchmark `ane_residency_20260701T022853Z_1b69eca7`.

**Observations:** int4 quant closes much of the planner/runtime gap — ANE proxy **43×** fp16 (0.067% → 2.9%), throughput **3.5×** (7.93 → 27.73 t/s), GPU joules **−44%**, thermal headroom improved. Compute-plan ANE fraction unchanged (~44%). Still below TorchScript 35 t/s and MLX 106 t/s; ANE proxy well under 60% gate.

### Recommendation (post-int4, 2026-07-01)

**Runtime ANE proxy improved meaningfully; throughput increased substantially.** Continue Core ML torch.export + int4 path:

1. **Graph cleanup** — replace mask-based KV slot write (`equal`/`tile`/`mul`) with leaner update ops.
2. **Instruments Core ML template** — confirm per-op ANE execution vs compute plan.
3. **Quantize prefill-kv** — match decode int4 path for end-to-end bandwidth reduction.
4. **Profile with 2-model load only** — avoid triple-model OOM in `ane_placement_profile.py`.

If ANE proxy plateaus &lt;10% after graph cleanup, evaluate **hybrid** (ANE int4 prefill + optimized decode runtime) for throughput while preserving measured ANE energy on prefill.

### Graph cleanup — scatter KV write (2026-07-01)

**Problem:** Mask-based KV slot write (`arange` + `equal` + `expand` + `mul` + `add`) per layer, plus 24× `torch.cat` layer rebuilds → `equal`/`tile`/`concat`/`mul` heavy graph (3697 ops).

**Change:** `--mode decode_torch_export_clean` (or `--decode-kv-write scatter`):
- Replace mask write with `tensor.scatter(dim=2, index=cachePosition, src=token_kv)`
- Clone input cache once per forward; update layers via in-place `k_cache[layer_idx] = …` (no per-layer `cat`)

**Artifacts:**
- fp16 scatter: `models/qwen2.5-0.5b-decode-kv-torch-export-scatter.mlpackage`
- int4 clean: `models/qwen2.5-0.5b-decode-kv-torch-export-int4-clean.mlpackage`

**Op-count delta (int4 variants):**

| Op / metric | mask int4 | scatter int4 clean | Δ |
|-------------|-----------|-------------------|---|
| Total ops | 3697 | **3229** | −468 |
| `tile` | 96 | **0** | −96 |
| `ios18.concat` (layer cat) | 96 | **0** | −96 |
| `ios18.mul` (mask) | 362 | **266** | −96 |
| `equal` | present | **removed** | — |

**Runtime trade-off (unexpected):** scatter improves throughput but **regresses ANE placement**:

| Variant | ANE plan % | ANE proxy | Sust. t/s | ANE J | GPU J | CPU J | Temp |
|---------|------------|-----------|-----------|-------|-------|-------|------|
| mask int4 decode | **44.1%** | **2.90%** | 27.73 | 31.9 | 646 | 421 | 75°C |
| scatter int4 clean + prefill int4 | **0%** (decode) | **0.36%** | **48.60** | 3.7 | **122** | 900 | **64°C** |

Runs: mask `1b69eca7`; scatter clean `6f90882a`. Prefill int4 plan ANE **29.1%** (fp16 31.0%).

`scatter` compiles GPU-preferred in `MLComputePlan` despite fewer ops. Throughput exceeds TorchScript (35 t/s) but runtime ANE proxy falls below mask int4.

### prefill-kv int4 (2026-07-01)

```bash
phase1/.venv/bin/python phase1/coreml_quantize.py --model-role prefill \
  --input models/qwen2.5-0.5b-prefill-kv.mlpackage \
  --output models/qwen2.5-0.5b-prefill-kv-int4.mlpackage
```

Compute plan: **29.1% ANE** (4049 ops). KV cache outputs remain fp16.

### Plan vs runtime gap analysis — mask int4 (2026-07-01)

**Question:** Why does mask int4 show **44.1% ANE in `MLComputePlan`** but only **~2.9% runtime ANE proxy**?

**Tools:** `phase1/compute_plan_analysis.py`, `phase1/powermetrics_timeseries.py`, powermetrics + JSONL from runs `1b69eca7` (60 s) and `5fe0d68c` (30 s correlation). Instruments GUI unavailable on this host (no full Xcode); see `PROFILING.md` for manual protocol.

#### 1. Compute-plan denominator vs executable ops

Of 3697 total ops, **1992 are unplaced** in `MLComputePlan`:

| Unplaced op | Count | Role |
|-------------|-------|------|
| `const` | 1821 | Weights/scales; no runtime device |
| `constexpr_blockwise_shift_scale` | 171 | int4 dequant metadata |

Among **1705 placed** ops, **1630 (95.6%)** prefer ANE. The headline **44.1%** is `1630/3697` — it understates how ANE-heavy the *executable* graph is.

#### 2. Which ANE-eligible ops are GPU at compile time?

Only **75 placed ops (4.4%)** are GPU-preferred:

- **GPU-only (12 ops):** `greater_equal`, `select`, `gather`, `equal` — mask-index control for KV slot write. Too small to explain the gap alone.
- **GPU despite ANE support (63 ops):** scattered `mul`/`add`/`linear`/`slice_by_index`/`concat`/`tile` — scheduler *may* still run these on GPU at runtime even when plan says ANE.

Bulk compute (**24 SDPA @ 100% ANE**, **161/169 linear @ ANE**) is plan-compliant. Mask KV ops (**95/108 @ ANE**) are mostly ANE-planned, not GPU-fallback victims.

#### 3. What dominates wall time and energy (outside the plan)?

| Cost center | Evidence | Estimate / impact |
|-------------|----------|-------------------|
| **KV tensor I/O** | torch.export explicit `keyCache`/`valueCache` in+out each step | ~2 × 24×2×1024×64 × 2 B ≈ **12.6 MB/step**; at 28 t/s ≈ **350 MB/s** memcpy not in op counts |
| **Python orchestration** | `np.ascontiguousarray`, `predict()` marshalling, greedy argmax | **421 J CPU** (38%) in `1b69eca7` |
| **GPU execution + copies** | Powermetrics steady windows 12–18 | **646 J GPU** (59%); GPU joules spike ~100 J/5 s during warmup windows 9–11 |
| **ANE matmul/SDPA** | 31.9 J ANE (2.9%) | Real NE activity, but small vs I/O + GPU |

#### 4. Powermetrics correlation

5 s windows (`mask_int4_powermetrics_ts.json`, run `1b69eca7`):

- **Prefill/load (w0–2):** ANE ≈ 0%, CPU-dominated.
- **Hand-off bursts (w3, w7):** ANE **4–17%** — prefill/decode transitions.
- **Steady decode (w12–18):** ANE **2.7–5.6%** (p50 **2.66%**); GPU 40–60 J/window.
- **Higher ANE proxy ↔ lower GPU joules** in the same window (w7 vs w9–11).

Correlation re-run `5fe0d68c`: **2.03% ANE**, **28.13 t/s** — consistent.

#### 5. Root-cause ranking

1. **Structural: explicit out-of-place KV I/O** per decode step (functional torch.export requirement).
2. **Structural: host orchestration** (CPU energy, `predict()` boundaries).
3. **Scheduler/heuristic: GPU memcpy and warmup** — explains GPU-heavy windows despite ANE plan.
4. **Minor: mask-control GPU-only ops** — 12 ops; fixing mask alone won't close a 40-point plan/runtime gap.
5. **Not primary: dynamic shapes** — fixed `(1,1,1,1024)` mask and static export shapes.

Scatter clean confirms scheduler behavior: **`scatter` → 0% ANE plan → 0.36% runtime** with **48.6 t/s**. Replacing mask ops does not increase ANE; it removes ANE eligibility entirely.

#### 6. Quick wins (limited upside)

| Action | Expected effect | Effort |
|--------|-----------------|--------|
| ctx-specific decode package (512 not 1024) | −75% KV I/O bytes | Medium (re-export) |
| int4 prefill-kv in benchmark path | Faster hand-off, less fp16 prefill bandwidth | Done (`prefill-kv-int4`) |
| Remove per-layer `concat` without `scatter` | Unclear — scatter regressed ANE | High risk |
| `CPU_AND_NE` forced compile on mask int4 | Unknown; MLState failed | Low cost test |
| Instruments on M4 + Xcode | Confirm NE duty cycle + buffer copies | Manual |

None of these alone is likely to reach **60% ANE proxy** while keeping torch.export functional KV I/O.

### Instruments profiling

See **`phase1/PROFILING.md`** for Core ML + Metal System Trace attach workflow, CLI substitutes (`compute_plan_analysis.py`, `powermetrics_timeseries.py`), powermetrics correlation tables, and manual Instruments checklist.

### ctx-512 decode export experiment (2026-07-01)

**Hypothesis:** Halving KV tensor I/O (~12.6 MB → **6.3 MB**/step) via `max_ctx=512` decode export should improve throughput and efficiency.

**Export:**

```bash
# fp16 torch.export decode @ max_ctx=512 → …-ctx512.mlpackage
phase1/.venv/bin/python phase1/coreml_kv_convert.py \
  --mode decode_torch_export --max-ctx 512 --output-dir models

# int4 post-export
phase1/.venv/bin/python phase1/coreml_quantize.py \
  --input models/qwen2.5-0.5b-decode-kv-torch-export-ctx512.mlpackage \
  --output models/qwen2.5-0.5b-decode-kv-torch-export-int4-ctx512.mlpackage \
  --max-ctx 512
```

**Code changes:**

- `coreml_kv_convert.py`: `-ctx{max_ctx}` suffix on decode artifact when `max_ctx ≠ 1024`.
- `kv_decode.py`: infer `decode_max_ctx` from decode model `keyCache` shape; slice 1024 prefill caches to decode width; cap prompt to `decode_max_ctx - 1` when benchmark context would fill the cache.

**KV cache shapes:**

| Model | `keyCache` / `valueCache` | Bytes/step (in+out fp16) |
|-------|---------------------------|--------------------------|
| mask int4 ctx1024 | `(24, 1, 2, 1024, 64)` | ~12.6 MB |
| mask int4 ctx512 | `(24, 1, 2, 512, 64)` | ~**6.3 MB** |

**Compute plan (unexpected regression):**

| Artifact | ANE plan % | GPU plan % | Placed ANE % |
|----------|------------|------------|--------------|
| mask int4 ctx1024 | **44.1%** | 2.0% | **95.6%** |
| mask int4 ctx512 (fp16 + int4) | **0.0%** | **46.1%** | **0%** |

`max_ctx=512` export routes **all 1705 placed ops to GPU** — ANE placement from ctx1024 is **not preserved**. Runtime logs `ANECCompile() FAILED` on load. This is independent of int4 quant (fp16 ctx512 shows the same 0% plan).

**Benchmark comparison** (int4 prefill-kv + mask int4 decode, 60 s steady window):

| Run | Decode pkg | Bench ctx | Active prompt | Sust. t/s | ANE proxy | ANE J | GPU J | CPU J | Power W | Temp steady |
|-----|------------|-----------|---------------|-----------|-----------|-------|-------|-------|---------|-------------|
| `1b69eca7` | int4 ctx1024 | 512 | 512 | **27.73** | **2.90%** | 31.9 | 646 | 421 | — | 75.2°C |
| `bbc356a7` | int4 ctx512 | 512 | **511** (capped) | **1.33** | 1.94% | 48.1 | 1626 | 808 | 15.3 | 83.5°C |
| `649919bf` | int4 ctx512 | **256** | 256 | **47.37** | 0.78% | 7.7 | **261** | 720 | 7.4 | 78.3°C |
| `6f90882a` | scatter int4 ctx1024 | 512 | 512 | 48.60 | 0.36% | 3.7 | 122 | 900 | — | 64°C |

**Analysis:**

1. **KV I/O reduction works for throughput** when sustained decode is possible — `649919bf` achieves **47.4 t/s** (+71% vs ctx1024 mask int4) with **−60% GPU joules** (261 vs 646 J), consistent with halved tensor copies dominating GPU energy on the ctx1024 path.
2. **`bbc356a7` is invalid at ctx 512** — with `decode_max_ctx=512`, a 511-token prompt leaves **one** decode slot; the loop re-prefills almost every step → 1.33 t/s. Do not benchmark ctx512 decode at full context without a smaller prompt.
3. **ANE proxy regresses** on ctx512 (0.78% vs 2.9%) because compute plan drops to **0% ANE** — smaller KV export changes planner/compiler behavior, not just I/O volume.
4. **IPJ** on the valid ctx512 run (`649919bf`): **3.16** vs mask int4 ctx1024 **1.32** — better energy per token despite lower ANE share.

**Limitations:**

- Prefill remains **1024-wide** (`prefill-kv-int4`); caches are sliced on hand-off (small CPU cost, no re-export).
- ctx512 decode cannot serve **512-token prompts with 512 tokens of decode headroom** — `decode_max_ctx` caps total sequence length.
- ctx512 loses ANE compile-time placement entirely on this export.

### Recommendation (post ctx-512 experiment, 2026-07-01)

**Keep hybrid architecture; ring512 supersedes mask linear for sustained Core ML decode.**

| Path | Model | When to use |
|------|-------|-------------|
| **Default (Core ML decode)** | **ring int4 512** | **38.3 t/s**, **6.7% ANE**, IPJ **3.23**, 0 re-prefill (`fc860526`) |
| **Legacy linear** | mask int4 ctx1024 | 27.7 t/s, 2.9% ANE — superseded by ring for sustained loops |
| **Max throughput** | scatter int4 clean + int4 prefill | **48.6 t/s**, 0.36% ANE when ANE irrelevant |
| **ctx512 tensor (deprecated for ANE)** | mask int4 ctx512 @ ctx≤256 | 47.4 t/s, 0% ANE plan |

**Do not maintain ctx512 for ANE residency** — export regresses plan to 0% ANE. The KV I/O win is real but trades away the mask path's only ANE advantage.

### Ring buffer KV experiment (2026-07-01) — Exp 1 ✅ Iterate

**Hypothesis**: Fixed-size ring buffer (512) eliminates linear re-prefill thrash, bounds attended KV slots, improves sustained t/s and ANE proxy vs mask int4 linear — using same ctx1024 tensor I/O.

**Implementation**:
- `phase1/ring_buffer_kv.py` — sliding-window causal mask, re-prefill policy
- `kv_decode.py` — `--kv-cache-mode ring` (via `ane_residency_benchmark.py`)
- `coreml_kv_convert.py --ring-size 512` — bakes `cache_position % 512` for KV writes; logical position for RoPE
- Artifacts: `…-decode-kv-torch-export-ring512.mlpackage`, `…-int4-ring512.mlpackage`

**60 s benchmark @ ctx 512** (`ane_residency_20260701T102405Z_fc860526` vs baseline `1b69eca7`):

| Metric | mask int4 linear | ring int4 512 | Δ |
|--------|------------------|---------------|---|
| Sust. t/s | 27.73 | **38.30** | +38% |
| ANE proxy | 2.90% | **6.70%** | +131% |
| ANE J | 31.9 | 47.7 | +50% |
| GPU J | 646 | **20.6** | −97% |
| Total J | 1099 | **712** | −35% |
| IPJ | 1.32 | **3.23** | +145% |
| Temp steady | 75.2°C | **56.9°C** | −18°C |
| Re-prefills | ~3 | **0** | eliminated |
| KV I/O bytes/step | ~25 MB | ~25 MB | unchanged |
| ANE plan % | 44.1% | 44.0% | ~same |

**Verdict**: Hypothesis **supported** for throughput, ANE proxy, energy, thermal. KV tensor I/O size unchanged; wins from **no re-prefill** + **512-slot attention window** + shifted runtime energy (GPU → ANE/CPU).

**Surprises**: GPU joules collapsed despite full cache copies; ANE proxy >2× without MLState.

**Decision**: **Iterate** — ring512 default for sustained decode; next: quality gate, then consolidated MLState to drop explicit I/O.

```bash
phase1/.venv/bin/python phase1/ane_residency_benchmark.py \
  --backend coreml --decode --context 512 \
  --coreml-prefill-kv models/qwen2.5-0.5b-prefill-kv-int4.mlpackage \
  --coreml-decode-kv models/qwen2.5-0.5b-decode-kv-torch-export-int4-ring512.mlpackage \
  --kv-cache-mode ring --ring-size 512
```

**Next experiments:**

1. **Quality gate** — greedy match / perplexity vs linear full-cache.
2. **Consolidated MLState** (SqueezeBits 2-state) — remove explicit KV I/O.
3. Instruments on Xcode M4 to confirm per-op NE duty cycle on ring vs linear.

## Tooling and commands

| Script | Purpose |
|--------|---------|
| `coreml_kv_convert.py` | Export prefill-kv + MLState decode + `--mode decode_torch_export` |
| `coreml_quantize.py` | Post-export int4/int8 linear weight quantization (decode + prefill) |
| `PROFILING.md` | Instruments + CLI profiling workflow |
| `compute_plan_analysis.py` | Per-op MLComputePlan breakdown (placed vs unplaced) |
| `powermetrics_timeseries.py` | 5 s ANE-proxy windows from powermetrics logs |
| `coreml_instrumentation.py` | Load models with logged `ComputeUnit` + `MLComputePlan` |
| `ane_residency_benchmark.py` | Full benchmark; `--compute-units`, `--profile` |
| `ane_placement_profile.py` | Short decode profile + compute-plan JSON |

```bash
# Export torch.export decode (explicit KV I/O, no MLState)
phase1/.venv/bin/python phase1/coreml_kv_convert.py \
  --mode decode_torch_export --output-dir models --max-ctx 1024

# Ring-buffer decode (bakes position % ring_size for KV write)
phase1/.venv/bin/python phase1/coreml_kv_convert.py \
  --mode decode_torch_export --ring-size 512 --output-dir models
phase1/.venv/bin/python phase1/coreml_quantize.py \
  --input models/qwen2.5-0.5b-decode-kv-torch-export-ring512.mlpackage \
  --output models/qwen2.5-0.5b-decode-kv-torch-export-int4-ring512.mlpackage

# ctx-512 decode variant (halved KV I/O; artifact suffix -ctx512)
phase1/.venv/bin/python phase1/coreml_kv_convert.py \
  --mode decode_torch_export --max-ctx 512 --output-dir models
phase1/.venv/bin/python phase1/coreml_quantize.py \
  --input models/qwen2.5-0.5b-decode-kv-torch-export-ctx512.mlpackage \
  --output models/qwen2.5-0.5b-decode-kv-torch-export-int4-ctx512.mlpackage \
  --max-ctx 512

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
| `ane_placement_profile_20260701T020740Z_bf783c54` | torch.export fp16 decode profile | 7.93 t/s, 0.067% ANE proxy; **44.8% ANE compute plan** | `results/ane_placement_profile/ane_placement_profile_20260701T020740Z_bf783c54/` |
| `ane_residency_20260701T022853Z_1b69eca7` | mask int4 decode 60 s | **27.73 t/s**, **2.90% ANE**; 44.1% plan | `results/ane_residency/ane_residency_20260701T022853Z_1b69eca7/` |
| `ane_residency_20260701T024057Z_6f90882a` | scatter int4 clean + prefill int4 | **48.60 t/s**, **0.36% ANE**; 0% decode plan | `results/ane_residency/ane_residency_20260701T024057Z_6f90882a/` |
| `ane_residency_20260701T025617Z_5fe0d68c` | mask int4 correlation 30 s | **28.13 t/s**, **2.03% ANE** | `results/ane_residency/ane_residency_20260701T025617Z_5fe0d68c/`, `results/compute_plan_analysis/mask_int4_correlation_5fe0d68c_ts.json` |
| (analysis) | mask int4 compute plan CLI | 95.6% ANE of placed ops | `results/compute_plan_analysis/mask_int4_decode.json`, `mask_int4_powermetrics_ts.json` |
| `ane_residency_20260701T032704Z_649919bf` | mask int4 **ctx512** @ bench ctx 256 | **47.37 t/s**, 0.78% ANE; 0% plan | `results/ane_residency/ane_residency_20260701T032704Z_649919bf/` |
| `ane_residency_20260701T031953Z_bbc356a7` | mask int4 ctx512 @ ctx 512 (invalid) | 1.33 t/s — re-prefill thrash | `logs/ane_residency_20260701T031953Z_bbc356a7*` |
| (analysis) | ctx512 compute plan | **0% ANE** (GPU-only placed) | `results/compute_plan_analysis/mask_int4_ctx512_decode.json` |
| `ane_residency_20260701T010854Z_a164b6cc` | Failed (pre-fix OOM) | GPU OOM loading both models | `logs/ane_residency_20260701T010854Z_a164b6cc_ctx512.powermetrics.txt` |
| `ane_residency_20260701T102405Z_fc860526` | **ring int4 512** @ ctx 512 | **38.30 t/s**, **6.70% ANE**, 0 re-prefill; IPJ **3.23** | `results/ane_residency/ane_residency_20260701T102405Z_fc860526/` |

Local-only (not tracked): `models/*.mlpackage/`, `models/qwen2.5-0.5b-decode-kv.pt`.