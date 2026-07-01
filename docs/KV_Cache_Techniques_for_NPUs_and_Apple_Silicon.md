# KV Cache Techniques for NPUs and Apple Silicon

**Status**: Literature review (July 2026)  
**Purpose**: Inform Alalā’s KV Cache / State Architecture Redesign thread with high-signal prior art before prototyping.  
**Audience**: Phase 1 engineers and research threads in [`Alalā_Research_Agenda.md`](Alalā_Research_Agenda.md).

---

## Introduction & Scope

Key–value (KV) caching is the dominant mechanism for autoregressive decode: each new token attends to cached keys and values from prior tokens instead of recomputing them. On Apple Silicon, KV design is not a neutral implementation detail—it interacts directly with **ANE static-shape constraints**, **~28–30 MB on-chip SRAM budgets**, **unified-memory bandwidth**, and **Core ML state/I/O semantics**.

This review focuses on techniques that could improve **ANE compatibility**, **reduced data movement**, and **sustained efficiency (IPJ)** on M-series NPUs. We exclude server-centric designs (e.g., multi-GPU PagedAttention at datacenter scale) unless they offer transferable principles for on-device NPUs.

**Sources**: Apple ML research and Core ML docs; SqueezeBits Yetter / disaggregated inference; Orion direct-ANE work; NPUMoE; ANEMLL community stack; KV quantization literature; Alalā Phase 0/1 measured baselines ([`phase1/NOTES.md`](../phase1/NOTES.md)).

---

## Core Challenges of KV Cache on NPUs/ANE

### 1. Static shapes vs. growing context

The ANE compiler and runtime favor **fixed tensor shapes** and **predictable control flow**. Standard decode grows sequence length every step—flexible `causalMask` ranks, dynamic `end_step`, and tensor-indexed cache writes often route graphs to GPU or fail ANE compilation entirely.

**Alalā evidence**: MLState decode shows **0% ANE** in `MLComputePlan` and `ANECCompile() FAILED` under `CPU_AND_NE`. Mask-based `torch.export` recovers **44.1% ANE plan** but **~2.9% runtime ANE proxy**—planner ≠ execution.

### 2. KV I/O dominates when state is external

Apple’s Llama 3.1 Core ML post documents that KV-as-model-I/O requires copying full cache tensors each step; at 8B / ctx 8192 this can approach **~1 GB** of copies per iteration. Stateful `MLState` avoids host round-trips but introduces `slice_update` / `read_state` patterns that may not compile for ANE.

**Alalā evidence**: Explicit `keyCache`/`valueCache` in+out per decode step ≈ **12.6 MB** (ctx 1024) or **6.3 MB** (ctx 512)—~**350 MB/s** at 28 t/s, largely outside op-count-based compute plans.

### 3. Irregular ops are NPU-hostile

`scatter`, `gather`, dynamic `equal`/`select`, and per-token indexing are repeatedly cited as **GPU/CPU fallbacks** on ANE. NPUMoE generalizes this: unpredictable shapes and irregular ops break NPU offload.

**Alalā evidence**: Replacing mask write with `scatter` raised throughput to **48.6 t/s** but dropped ANE plan to **0%** and runtime proxy to **0.36%**.

### 4. State count and layout constraints

SqueezeBits reports ANE compile failures when **too many MLState tensors** exist (e.g., per-layer KV → 56 states). Mitigation: **two consolidated states** `(layers, heads, max_len, head_dim)`. Non-power-of-two inner dims (e.g., head_dim 80) can cause runtime errors; padding to powers of two is required.

### 5. SRAM cliff and bandwidth-bound decode

Apple’s ml-ane-transformers work notes Transformers become **bandwidth-bound** at short sequence lengths on ANE; KV + activations exceeding ~28–30 MB force DRAM spill. Phase 0 measured **L_cliff ≈ 1024** tokens on MLX decode.

### 6. Planner vs. runtime gap

High ANE *plan* fractions do not guarantee NE duty cycle. Structural costs—KV copies, `predict()` marshalling, IOSurface paths, GPU scheduler heuristics—can absorb wall time and energy even when matmul/SDPA ops prefer ANE.

---

## Existing Approaches

### Static / Fixed-Shape KV Cache

**Idea**: Export decode with fixed `max_ctx`, fixed mask width, and explicit slot index (`cachePosition`). Validity encoded in mask content, not dynamic tensor shapes.

| Pros | Cons |
|------|------|
| ANE-friendly; matches Core ML multifunction models | Wastes compute on padded slots if not windowed |
| Predictable memory; easy benchmarking | Separate packages per `max_ctx` (Alalā: ctx512 export **lost all ANE placement**) |
| Works with `torch.export` + int4 weights | Still pays full cache I/O if KV is explicit in/out |

**References**: Apple Llama baseline → KV-as-I/O; Alalā mask int4 torch.export path.

### Ring Buffer / Sliding Window KV Cache

**Idea**: Fixed-size cache; new tokens overwrite oldest slots (circular index). Attention uses local window (Gemma sliding-window, StreamingLLM “attention sinks,” Mistral SWA).

| Pros | Cons |
|------|------|
| **Bounded** KV size → stable shapes, lower I/O | Quality degrades beyond window without sinks / hierarchy |
| Natural fit for ANE static `max_ctx` | Ring index update must avoid `scatter`/`gather` on ANE |
| ANEMLL uses `infer_rotate` for Gemma-3 local KV | Heterogeneous layers (local + global KV) complicate consolidated state |

**References**: [StreamingLLM](https://arxiv.org/abs/2309.06180); ANEMLL Gemma-3 `infer_rotate`; Gemma-3 sliding-window configs.

### Paged / Chunked KV Cache

**Idea**: KV split into fixed-size **pages** or **chunks**; logical sequence maps to page table (vLLM PagedAttention, chunked prefill SDPA).

| Pros | Cons |
|------|------|
| Memory efficient for long context on GPU servers | Page tables + indirect indexing → scatter/gather |
| Chunked SDPA reduces peak activation memory | Apple’s `scaled_dot_product_attention_sliced_q` MIL pass helps **prefill**, not decode paging |
| Useful for unified-memory **capacity** planning | Poor ANE fit per NPUMoE and Alalā scatter regression |

**On-device adaptation**: Fixed chunk size with **static page indices** (no runtime allocator) may be testable, but high risk for ANE.

### KV Cache Quantization (int4/int8 and advanced schemes)

**Idea**: Store KV in low precision; dequantize at attention boundary (per-token or per-channel; residual FP16 window).

| Pros | Cons |
|------|------|
| **2–4×** KV bytes; critical past SRAM cliff | Per-step quant/dequant adds compute |
| Phase 0: int4 KV **+2.4% t/s**, IPJ nearly neutral on MLX | Core ML path: KV I/O still fp16 in current exports |
| KIVI, KVQuant, Quanto/HQQ backends | Fused int4 SDPA on ANE not yet proven in Alalā |

**References**: [Hugging Face KV quant blog](https://huggingface.co/blog/kv-cache-quantization); [KIVI](https://arxiv.org/abs/2402.02750); [KVQuant](https://arxiv.org/abs/2401.18079).

### Stateful KV Cache with MLState / Core ML

**Idea**: `MLState` tensors updated in-place via `slice_update`; host does not copy full cache each step.

| Pros | Cons |
|------|------|
| Eliminates output→input KV copy loop (Apple’s recommended path for GPU Llama) | Alalā Qwen2 MLState: **0% ANE**, compile fail |
| Two consolidated states (SqueezeBits) reduce state count | `read_state` + dynamic mask still problematic |
| Power-of-two padding + epsilon `add` tricks needed for ANE | TorchScript dialect worse than torch.export for placement |

**References**: [Apple Llama + MLState](https://machinelearning.apple.com/research/core-ml-on-device-llama); [Core ML stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html); SqueezeBits Yetter blog.

### Custom / Lower-Level KV Cache Implementations

**Orion** ([arXiv:2603.06728](https://arxiv.org/abs/2603.06728)): Bypasses Core ML; compiles MIL via private `_ANEClient`. **IOSurface** zero-copy I/O; weights baked at compile time; documents **20 ANE restrictions** (e.g., **concat banned**, 32 MB SRAM cliff). KV for decode likely still needs IOSurface-resident buffers—not automatic ANE residency.

**ANEMLL** ([github.com/anemll/anemll](https://github.com/anemll/anemll)): Production-oriented ANE LLM stack. Split **local/global KV** for Gemma-3; **ring rotation**; IOSurface-stable buffers; in-graph argmax to cut host transfer. Closest open “GOL-ANE”-style reference (no separate GOL-ANE paper found).

**NPUMoE** ([arXiv:2604.18788](https://arxiv.org/abs/2604.18788)): Static expert **tiers**, grouped execution, graph residency—principle: **offline shape calibration + static buckets** instead of dynamic routing. Analogous to fixed `max_ctx` tiers for KV.

**Yetter / disaggregated inference** ([SqueezeBits blog](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)): **ANE prefill + GPU decode**; 2-state KV; MIL `pow→mul` pass; sliced SDPA for long prefill. Decode stays GPU-fast; ANE wins TTFT.

---

## Key Papers, Projects, and Techniques

| Work | Link | Summary |
|------|------|---------|
| **Apple — On Device Llama 3.1** | [machinelearning.apple.com](https://machinelearning.apple.com/research/core-ml-on-device-llama) | KV-as-I/O vs **MLState**; flexible shapes; fused SDPA; int4 weights; GPU-targeted (~33 t/s decode M1 Max). KV I/O copy cost scales with ctx. |
| **Apple — ml-ane-transformers** | [machinelearning.apple.com](https://machinelearning.apple.com/research/neural-engine-transformers) | `(B,C,1,S)` layout, chunking, minimize copies; bandwidth-bound at short S; quantization reduces fetch cost. |
| **SqueezeBits — Yetter / disaggregated** | [blog.squeezebits.com](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176) | 2 consolidated KV states; power-of-2 dims; ANE prefill + GPU decode; MIL graph passes; sliced SDPA for long ctx compile. |
| **Orion** | [arXiv:2603.06728](https://arxiv.org/abs/2603.06728) | Direct ANE programming; IOSurface I/O; 20 MIL constraints; 170+ t/s GPT-2 124M on M4 Max; training via weight patch reload. |
| **NPUMoE** | [arXiv:2604.18788](https://arxiv.org/abs/2604.18788) | MoE on ANE via static tiers + grouped ops; dynamic scatter/gather on CPU/GPU fallback. |
| **ANEMLL** | [anemll.com](https://www.anemll.com/) / [GitHub](https://github.com/anemll/anemll) | ANE LLM inference; ring KV rotation; Gemma-3 local/global split; int4 QAT models. |
| **maderix ANE series** | [Substack](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine) | IOSurface, E5 bundles, queue depth—complements Orion characterization. |
| **Hugging Face — KV quant** | [blog](https://huggingface.co/blog/kv-cache-quantization) | Per-token int4 KV + residual cache; ~2.5× memory savings; works on MPS. |
| **KIVI** | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) | Asymmetric 2-bit KV; keys per-channel, values per-token. |
| **KVQuant** | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) | Extreme KV compression for million-token inference (GPU-oriented). |
| **StreamingLLM** | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | Attention sinks + sliding window enable infinite-length generation with bounded KV. |
| **vLLM PagedAttention** | [arXiv:2307.03294](https://arxiv.org/abs/2307.03294) | Paged KV blocks for GPU serving—principle only for ANE. |
| **Alalā Phase 0** | [`Phase0_Results_Summary_Alalā.md`](Phase0_Results_Summary_Alalā.md) | L_cliff ≈ 1024; int4 KV +2.4% t/s, IPJ neutral; ring buffer / hierarchy recommended. |
| **Alalā Phase 1** | [`phase1/NOTES.md`](../phase1/NOTES.md) | Mask int4: 27.7 t/s, 2.9% ANE; scatter: 48.6 t/s, 0% ANE; ctx512: 47.4 t/s, 0% ANE plan. |

---

## Implications & Recommended Directions for Alalā

### Which approaches to prototype first (ranked)

1. **Consolidated 2-tensor MLState + SqueezeBits export hygiene**  
   Re-export Qwen2 with `(24, 2, max_ctx, 64)` consolidated `key_cache`/`value_cache` states (not 48 per-layer states). Apply power-of-two head_dim padding (64 already OK), value-cache epsilon `add`, `pow→mul` MIL pass, and `torch.export` dialect. **Hypothesis**: Fixes ANE compile failure mode and removes ~12.6 MB/step explicit I/O—highest leverage for closing plan/runtime gap.

2. **Fixed-shape ring-buffer / sliding-window decode**  
   Bounded `max_ctx` with modular write index and static mask—ANEMLL `infer_rotate` pattern. Target **ANE-static** graphs without `scatter`. Pair with StreamingLLM-style sink tokens if quality drops. **Hypothesis**: Bounds KV working set for SRAM cliff *and* reduces bytes moved per step.

3. **Hybrid disaggregated execution (Yetter-style)**  
   ANE int4 **prefill-kv** (29% ANE plan, measured) + GPU path for decode (mask int4 or scatter int4 by workload). Unified-memory KV handoff without host copies. **Hypothesis**: Best sustained useful work per joule *today*, while KV redesign proceeds.

4. **KV int4 inside state or fused decode** (medium priority)  
   After state/I/O redesign stabilizes. Phase 0 shows near-neutral IPJ for int4 KV on MLX; bytes-cut should help ANE bandwidth-bound decode. Requires Core ML–exportable quant/dequant or Metal fused SDPA subgraph.

5. **Deprioritize: dynamic paged KV and scatter-based updates for ANE-first path**  
   Alalā and NPUMoE both show scatter/paging pushes graphs to GPU. Keep scatter variant as **throughput mode** only (48.6 t/s).

### Biggest open risks and unknowns

| Risk | Why it matters |
|------|----------------|
| **MLState may remain GPU-only for Qwen2** even with 2-state layout | SqueezeBits success on Llama/Qwen3 may not transfer; RoPE + GQA patterns differ. |
| **ctx-specific exports alter ANE placement** | ctx512 → 0% ANE plan despite halved I/O; compiler sensitivity unexplained. |
| **Ring buffer may fix I/O but not NE duty cycle** | 47.4 t/s ctx512 run still 0.78% ANE proxy with 0% plan. |
| **Quality vs. window** | Sliding-window + int4 KV may fail agent tasks needing full context. |
| **Orion/private API path** | High upside, HCA/portability concerns, maintenance cost. |
| **Consolidated state + heterogeneous layers** | Gemma-style local/global split may break single-tensor cache assumption. |

### Hybrid vs. pure-ANE: what this review changes

**Reinforces hybrid as the near-term production architecture**, not a failure mode:

- Yetter benchmarks: ANE wins **TTFT/prefill**; MLX/GPU wins **TPOT/decode**.
- Alalā measurements align: mask int4 **~28 t/s @ 2.9% ANE** vs scatter **48.6 t/s @ 0.36% ANE** vs MLX **~106 t/s**.
- Pure-ANE decode remains a **worthwhile research bet** if MLState + static ring KV achieves SqueezeBits-like Instruments reports (nearly all ops on ANE for Llama-3.2-1B decode).

**Revised mental model**:

```text
Prefill  → ANE-first (int4, sliced SDPA, consolidated KV state)
Decode   → Hybrid default (GPU/MLX or scatter Core ML for throughput)
Decode   → Pure-ANE target (2-state MLState + ring buffer + no scatter) — validate before committing
Paging   → Not for ANE v1
```

The KV redesign thread should **not** assume “fix mask ops” alone reaches 60% ANE proxy. Structural state/I/O redesign is the consensus across Apple, SqueezeBits, and Alalā measurements.

---

## Open Questions & Hypotheses Worth Testing

1. **H1**: Consolidated 2-state `torch.export` + MLState achieves **>0% ANE compile** and **>10% runtime ANE proxy** on Qwen2.5-0.5B decode where per-layer MLState failed.
2. **H2**: Ring-buffer write via `slice_update` at fixed indices compiles to ANE without `scatter`/`equal` chains.
3. **H3**: Halving KV bytes (int4 KV in state) improves runtime ANE proxy more than halving tensor dims (ctx512) because planner behavior differs from I/O volume.
4. **H4**: Sliding window 512 + sink tokens preserves ≥95% quality on Alalā eval suite while fitting KV in ANE SRAM budget.
5. **H5**: Hybrid handoff (ANE prefill state → MLX decode KV layout) costs **<5%** throughput vs pure MLX with **lower prefill joules**.
6. **H6**: `scaled_dot_product_attention_sliced_q` on prefill-only package raises max compile ctx without decode changes.
7. **H7**: Orion-style IOSurface KV buffers reduce `predict()` marshalling cost even within Core ML (without full private API).

---

## References

1. Apple Machine Learning Research. *On Device Llama 3.1 with Core ML* (2024). https://machinelearning.apple.com/research/core-ml-on-device-llama  
2. Apple Machine Learning Research. *Deploying Transformers on the Apple Neural Engine* (2022). https://machinelearning.apple.com/research/neural-engine-transformers  
3. Choi, J. *Disaggregated Inference on Apple Silicon: NPU prefill and GPU decode* (SqueezeBits, 2025). https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176  
4. Kumaresan, R. *Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference* (2026). https://arxiv.org/abs/2603.06728  
5. Benazir, A.; Lin, F. X. *Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs (NPUMoE)* (2026). https://arxiv.org/abs/2604.18788  
6. ANEMLL Project. https://github.com/anemll/anemll  
7. Turganbay, R. *Unlocking Longer Generation with Key-Value Cache Quantization* (Hugging Face, 2024). https://huggingface.co/blog/kv-cache-quantization  
8. Liu, Z. et al. *KIVI: Plug-and-play 2-bit KV Cache Quantization* (2024). https://arxiv.org/abs/2402.02750  
9. Hooper, C. et al. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* (2024). https://arxiv.org/abs/2401.18079  
10. Xiao, G. et al. *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)* (2023). https://arxiv.org/abs/2309.06180  
11. Kwon, W. et al. *Efficient Memory Management for LLM Serving with PagedAttention* (2023). https://arxiv.org/abs/2307.03294  
12. Alalā. *Phase 0 Results Summary* (2026). [`Phase0_Results_Summary_Alalā.md`](Phase0_Results_Summary_Alalā.md)  
13. Alalā. *Phase 1 Core ML Model Interface Notes* (2026). [`phase1/NOTES.md`](../phase1/NOTES.md)  
14. Apple. *Core ML Stateful Models* (coremltools). https://apple.github.io/coremltools/docs-guides/source/stateful-models.html  
15. maderix. *Inside the M4 Apple Neural Engine* (Substack). https://maderix.substack.com/p/inside-the-m4-apple-neural-engine  

---

*Next step*: Execute thread 1 experiments per [`Alalā_Research_Agenda.md`](Alalā_Research_Agenda.md), starting with consolidated MLState export and ring-buffer prototype branches.