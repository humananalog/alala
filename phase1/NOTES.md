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

**Status:** Core ML decode export **unblocked**; first end-to-end MLState decode measured. ANE residency **not recovered** (0.11% vs 38% prefill proxy). Throughput regression vs TorchScript (7.45 vs 35 t/s) — likely GPU/CPU routing + compile overhead on stateful 0.5B graph. Next: ANE placement profiling, int4 weight quant (Mistral export pattern), graph simplification.