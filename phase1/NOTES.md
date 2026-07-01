# Phase 1 — Core ML Model Interface Notes

## Current prefill-only package (`models/qwen2.5-0.5b-ane.mlpackage`)

**Exported:** 2026-07-01 via `torch.export` + `use_cache=False` wrapper.

| Direction | Name | Shape | Dtype |
|-----------|------|-------|-------|
| Input | `inputIds` | `(1, max_ctx)` — max_ctx=1024 in current artifact | int32 |
| Output | `logits` | `(1, max_ctx, vocab)` — vocab=151936 | float16 |

**No KV cache.** Each `predict()` recomputes attention over the full padded sequence (prefill proxy). This explains ~4 t/s vs MLX ~84 t/s.

## Target stateful decode interface (Qwen2.5-0.5B)

Architecture: 24 layers, 2 KV heads, head_dim 64, max_ctx ≤ 1024.

### Prefill step (`qwen2.5-0.5b-prefill-kv.mlpackage`)

| Direction | Name | Shape | Notes |
|-----------|------|-------|-------|
| In | `inputIds` | `(1, 1024)` | Prompt left-padded or right-padded; active tokens in `[0:prompt_len)` |
| Out | `logits` | `(1, 1024, 151936)` | Prompt logits |
| Out | `keyCache` | `(24, 1, 2, 1024, 64)` | Stacked per-layer K |
| Out | `valueCache` | `(24, 1, 2, 1024, 64)` | Stacked per-layer V |

`use_cache=True` on first forward; valid KV length = `prompt_len`.

### Decode step (`qwen2.5-0.5b-decode-kv.mlpackage`)

| Direction | Name | Shape | Notes |
|-----------|------|-------|-------|
| In | `inputIds` | `(1, 1)` | Single new token id |
| In | `keyCache` | `(24, 1, 2, 1024, 64)` | KV from prior step |
| In | `valueCache` | `(24, 1, 2, 1024, 64)` | KV from prior step |
| In | `cacheSeqLen` | `(1,)` int32 | Number of valid KV positions before this token |
| Out | `logits` | `(1, 1, 151936)` | Next-token logits |
| Out | `keyCache` | `(24, 1, 2, 1024, 64)` | Updated KV |
| Out | `valueCache` | `(24, 1, 2, 1024, 64)` | Updated KV |

**Hand-off logic:** After prefill, `cacheSeqLen = prompt_len`. Each decode step feeds `inputIds` for greedy token, passes caches + `cacheSeqLen`, receives updated caches; increment `cacheSeqLen` by 1. Stop at `max_ctx` or thermal abort.

### Apple reference pattern

macOS 15+ supports fused SDPA and optional `MLState` for KV. We use explicit cache I/O first (Apple blog “KV as model I/O”) for debuggability; migrate to `StateType` once stable.

### MLX decode baseline

`harness/decode_client.DecodeRunner` → subprocess `mlx_lm` `stream_generate` with real KV inside MLX. Benchmark `--decode --backend mlx` uses this path.

### Decode export status (2026-07-01)

- **Prefill-kv**: `torch.export` → Core ML ✅ (`models/qwen2.5-0.5b-prefill-kv.mlpackage`)
- **Decode-kv**: `torch.jit.trace` with `StaticCache` + slice cache update ✅ runtime
- **Decode-kv Core ML**: `coremltools.convert(traced)` ❌ `No matching select or slice` (dynamic tensor index in cache write)
- **Fallback**: `models/qwen2.5-0.5b-decode-kv.pt` (TorchScript) used by `decode.py` until Core ML decode converts
- Trace warmed at `cacheSeqLen=1023` so RoPE tables cover ctx ≤ 1024