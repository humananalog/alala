# KV Cache Redesign — Synthesis and Experiments

**Status**: Active (July 2026)  
**Thread**: Research Agenda #1 — KV Cache / State Architecture Redesign  
**Inputs**: [`KV_Cache_Techniques_for_NPUs_and_Apple_Silicon.md`](KV_Cache_Techniques_for_NPUs_and_Apple_Silicon.md), [`phase1/NOTES.md`](../phase1/NOTES.md), Phase 0 baselines  
**Purpose**: Turn literature findings into prioritized, measurable experiments on M4 24 GB.

---

## 1. Executive Summary

KV cache design on Apple Silicon is a **systems bottleneck**, not a minor optimization. Our Phase 1 measurements show that even when `MLComputePlan` assigns **~44% of decode ops to ANE**, runtime ANE energy proxy stays near **3%** because **~12.6 MB of KV tensor I/O per step**, host `predict()` orchestration, and GPU scheduler behavior dominate wall time. Per-layer MLState failed ANE compile entirely (**0% plan**, `ANECCompile() FAILED`). Scatter-based KV updates recover throughput (**48.6 t/s**) but **zero out ANE placement**.

The literature converges on a small set of actionable principles for NPUs:

1. **Fixed shapes** and **minimal irregular ops** (`scatter`, dynamic masks) are prerequisites for ANE decode.
2. **Consolidated state** (two KV tensors, not per-layer) and **in-place updates** (`MLState` / `slice_update`) eliminate the explicit I/O loop that hurts our torch.export path.
3. **Bounded KV** (ring buffer / sliding window) attacks both **SRAM cliff** (~28–30 MB) and **bytes moved per step**.
4. **Hybrid disaggregation** (ANE prefill + GPU decode) is the proven near-term architecture; pure-ANE decode is a **conditional** bet, not the default.

**Assessment**: KV cache redesign remains **high-leverage and highest-priority** for Alalā. It directly targets the structural gaps (I/O, state semantics, dynamic control flow) that mask tuning and int4 weights alone cannot fix. Success is not guaranteed—MLState may stay GPU-only for Qwen2—but the literature and SqueezeBits/ANEMLL precedents give enough concrete export patterns to produce a clear go/no-go signal within **1–2 weeks** of focused work.

---

## 2. Key Insights from Literature

Grouped by technique; emphasis on what transfers to M4 + ANE + Core ML.

### Static / fixed-shape vs dynamic KV

| Insight | Action for Alalā |
|---------|------------------|
| ANE requires **compile-time-known shapes**; growing `causalMask` ranks and tensor-indexed writes route to GPU. | Keep `max_ctx`, mask width, and `cachePosition` **fixed at export**; encode validity in mask *values*, not tensor ranks. |
| Explicit KV in+out works for export (`torch.export`) but copies **full cache every step** (Apple Llama I/O path). | Treat explicit I/O as **baseline only**; primary target is **stateful** decode. |
| **ctx-specific exports can change planner behavior** (our ctx512 → 0% ANE despite halved bytes). | Hold `max_ctx=1024` for ANE-placement experiments until MLState path is understood; do not conflate I/O reduction with placement wins. |

### Ring buffer and sliding window

| Insight | Action for Alalā |
|---------|------------------|
| Fixed-size cache with modular write index **bounds memory and I/O**; fits static ANE graphs. | Prototype ring write via **`slice_update` at computed static indices**, not `scatter`. |
| ANEMLL `infer_rotate` and Gemma sliding-window show production viability on ANE. | Start with **single window size** (512) on Qwen2.5-0.5B; defer heterogeneous local/global layers. |
| StreamingLLM **attention sinks** mitigate quality loss for bounded windows. | Add sink tokens (first 4) only if perplexity/regression tests fail. |

### KV cache quantization (int4/int8, unified memory)

| Insight | Action for Alalā |
|---------|------------------|
| int4 KV cuts memory **~2–4×** with acceptable quality (KIVI, HF Quanto path); Phase 0 MLX: **+2.4% t/s**, IPJ neutral. | **Defer** until state/I/O path is stable; current Core ML exports keep KV fp16. |
| Benefit on unified memory is **bytes moved**, not int4 ALU speed. | Prioritize int4 KV **inside MLState** once ANE compile succeeds—bandwidth win aligns with ANE-bound decode. |
| Per-step quant/dequant can slow decode at high batch; batch=1 on-device is fine. | Use residual FP16 window (128 tokens) if quant adds regressions. |

### Stateful KV (MLState / Core ML)

| Insight | Action for Alalā |
|---------|------------------|
| Apple recommends MLState to avoid KV copy loop; GPU Llama achieves large speedups vs KV-as-I/O. | Re-architect around **2 consolidated states**: `(layers, kv_heads, max_ctx, head_dim)`. |
| SqueezeBits: **>2 states per model** can fail ANE compile; per-layer KV → 56 states. | Replace 24× separate `keyCache`/`valueCache` states in current export. |
| Export hygiene: power-of-2 dims (head_dim 64 ✓), epsilon `add` before value `slice_update`, `pow→mul` for RMSNorm. | Port Yetter MIL passes into `coreml_kv_convert.py` conversion pipeline. |
| `torch.export` dialect > TorchScript for ANE placement (Alalā: 44.8% vs 0%). | MLState re-export must use **torch.export + `ct.StateType`**, not `jit.trace`. |

### Orion, ANEMLL, NPUMoE, disaggregated inference

| Project | Transferable lesson |
|---------|---------------------|
| **Yetter / SqueezeBits** | ANE wins prefill (TTFT); GPU wins decode (TPOT). **2-state KV + int4 weights** is the workable Core ML recipe. |
| **ANEMLL** | Ring rotation + IOSurface-stable buffers + in-graph argmax; community proof that ANE LLM decode is achievable with careful graph design. |
| **Orion** | IOSurface zero-copy and MIL constraint catalog (no `concat`, SRAM cliff); long-term escape hatch if Core ML ceiling is hit—not v1 experiment scope. |
| **NPUMoE** | Dynamic scatter/gather and unpredictable shapes → CPU/GPU fallback; **static tiers** instead of runtime routing. Analog: fixed `max_ctx` tiers, not paged KV. |

### Common pitfalls and success factors

**Pitfalls**

- Assuming high `MLComputePlan` ANE % ⇒ high runtime NE duty cycle.
- Using `scatter` / `gather` / mask `equal` chains for KV slot writes on ANE-first paths.
- Per-layer MLState proliferation.
- Shrinking `max_ctx` in export without re-validating ANE placement.
- Optimizing mask ops while **~12 MB/step KV I/O** remains.

**Success factors**

- Two consolidated KV states, `torch.export`, fused SDPA, int4 weights.
- Fixed shapes end-to-end; ring/window for bounded working set.
- Instruments / `MLComputePlan` + powermetrics **together** (plan + runtime).
- Hybrid fallback path always available for throughput benchmarks.

---

## 3. Implications for Alalā

### Feasibility of improving ANE utilization via KV redesign

**Moderate-to-high feasibility for meaningful improvement**, with caveats:

| Mechanism | Expected impact | Confidence |
|-----------|-----------------|------------|
| Remove explicit KV I/O (MLState) | Cuts ~350 MB/s memcpy + GPU joules; addresses #1 ranked root cause | Medium — SqueezeBits demonstrated on Llama/Qwen3; our per-layer MLState failed |
| Consolidated 2-state layout | Unblocks ANE **compile** (currently 0% + compile fail) | Medium |
| Ring/window bounded KV | Lowers bytes/step and SRAM pressure past L_cliff | Medium-high for throughput/IPJ; uncertain for ANE % |
| Eliminate scatter/mask irregular ops | Restores ANE eligibility (scatter → 0% plan) | High |
| Reach **60% runtime ANE proxy** on decode | Requires compile success *and* scheduler actually running SDPA on NE | Low until Exp 1–2 complete |

**Bottom line**: KV redesign is the **most plausible path** to closing the 44% plan → 3% runtime gap. Mask/graph micro-optimization alone is **insufficient** per our compute-plan analysis.

### Interaction with hybrid architecture

KV redesign and hybrid are **complementary**, not competing:

```text
KV redesign (thread 1)  →  Enables pure-ANE or ANE-heavy decode if successful
Hybrid (thread 2)       →  Production path while thread 1 validates; ANE prefill already ~29% plan
```

Literature and measurements both say: **ship hybrid now**, continue KV state work as the **upside option** that could shift decode onto ANE without sacrificing IPJ. If MLState succeeds, hybrid becomes “ANE prefill + ANE bounded-window decode”; if not, hybrid remains “ANE prefill + scatter/MLX decode.”

### Major risks and unknowns

1. **Qwen2-specific ANE rejection** of `slice_update` / `read_state` even with 2-state layout.
2. **Planner sensitivity** to `max_ctx` and graph size (ctx512 regression).
3. **Ring buffer write pattern** may still compile GPU-only without ANEMLL-level graph surgery.
4. **Quality** degradation under windowed KV for agent-style long-context tasks.
5. **Effort sink**: Weeks on Core ML export edge cases vs days of signal from hybrid baseline.

---

## 4. Proposed Experiments

Four experiments, ranked by leverage. All use Qwen2.5-0.5B-Instruct, M4 24 GB, existing `ane_residency_benchmark.py` + `compute_plan_analysis.py` toolchain, and IPJ/powermetrics protocol.

---

### Experiment 1: Ring Buffer KV Cache for Decode — **COMPLETED (2026-07-01)**

**Status**: ✅ Executed — **Iterate**

**Hypothesis**  
A fixed-size **ring buffer** (window=512) with logical `cachePosition` for RoPE, graph-baked `position % 512` for KV writes, and per-step **sliding-window causal masks** eliminates linear-mode **re-prefill thrashing**, bounds effective KV working set to 512 slots, and improves sustained throughput and runtime ANE proxy vs mask int4 linear decode — without requiring a smaller `max_ctx` export.

**Scope (as run)**

| In scope | Out of scope |
|----------|--------------|
| `phase1/ring_buffer_kv.py` — mask + re-prefill policy | MLState / consolidated 2-tensor state |
| `kv_decode.py` — `--kv-cache-mode ring` runtime path | Perplexity / quality suite |
| Export `--ring-size 512` in `coreml_kv_convert.py` (`remainder` in graph) | int4 KV inside cache tensors |
| int4 ring artifact + 60 s benchmark @ ctx 512 | ctx512 smaller-tensor export |

**Metrics measured**

- Sustained tokens/s, ANE proxy, CPU/GPU/ANE joules, IPJ, temperature
- `re_prefill_count`, `kv_io_bytes_per_step` (explicit in+out still full 1024 tensors)
- ANE compute plan at load (44.0% ANE)

**Results vs mask int4 linear baseline** (`1b69eca7` vs `fc860526`):

| Metric | mask int4 linear | ring int4 (512) | Δ |
|--------|------------------|-----------------|---|
| Sust. t/s @ ctx 512 | **27.73** | **38.30** | **+38%** |
| ANE proxy | **2.90%** | **6.70%** | **+131%** |
| ANE J | 31.9 | **47.7** | +50% (more NE work) |
| GPU J | **646** | **20.6** | **−97%** |
| CPU J | 421 | 643 | +53% |
| Total J | 1099 | **712** | **−35%** |
| IPJ | 1.32 | **3.23** | **+145%** |
| Temp steady | 75.2°C | **56.9°C** | −18°C |
| Tokens / 60 s | 1449 | **2299** | +59% |
| Re-prefill events | ~3 (implicit) | **0** | eliminated |
| KV I/O bytes/step | ~25 MB (in+out) | ~25 MB | unchanged |
| ANE plan % | 44.1% | 44.0% | ~same |

**Hypothesis verdict**: **Supported** on throughput, ANE proxy, energy, and thermal — **partially supported** on I/O (bytes/step unchanged; effective attended slots halved to 512).

**Key learnings**

1. **Re-prefill elimination** was the dominant win — linear mode re-ran full prefill every ~512 decode steps after prompt fill; ring mode sustained **2299 tokens** in 60 s vs **1449**.
2. **GPU joules collapsed** (646 → 21 J) despite identical explicit KV tensor I/O — runtime shifted toward ANE/CPU orchestration; per-step dynamic ring mask did not hurt placement (44% ANE plan retained).
3. **`remainder` op** in export (`cache_position % 512`) compiles without scatter-style GPU fallback.
4. **Surprise**: ANE proxy **6.7%** exceeds 2× baseline without MLState — ring policy + mask may help scheduler more than mask-only micro-optimizations.
5. **Not yet validated**: output quality beyond greedy bench tokens; long-context agent tasks beyond 512-window.

**Decision**: **Iterate** — adopt ring512 as default decode path for sustained benchmarks; next: quality gate (Exp 4), MLState to remove explicit I/O (Exp 2).

**Artifacts**

- Code: `phase1/ring_buffer_kv.py`, `kv_decode.py`, `coreml_kv_convert.py --ring-size 512`
- Models: `models/qwen2.5-0.5b-decode-kv-torch-export-ring512.mlpackage`, `…-int4-ring512.mlpackage`
- Run: `ane_residency_20260701T102405Z_fc860526` — `logs/`, `results/ane_residency/ane_residency_20260701T102405Z_fc860526/`

```bash
phase1/.venv/bin/python phase1/coreml_kv_convert.py --mode decode_torch_export --ring-size 512
phase1/.venv/bin/python phase1/coreml_quantize.py \
  --input models/qwen2.5-0.5b-decode-kv-torch-export-ring512.mlpackage \
  --output models/qwen2.5-0.5b-decode-kv-torch-export-int4-ring512.mlpackage
phase1/.venv/bin/python phase1/ane_residency_benchmark.py \
  --backend coreml --decode --context 512 \
  --coreml-prefill-kv models/qwen2.5-0.5b-prefill-kv-int4.mlpackage \
  --coreml-decode-kv models/qwen2.5-0.5b-decode-kv-torch-export-int4-ring512.mlpackage \
  --kv-cache-mode ring --ring-size 512
```

---

### Experiment 2: Consolidated 2-State MLState Export (SqueezeBits Pattern)

**Priority**: 2 — **next** (remove explicit KV I/O)

**Hypothesis**  
Consolidating per-layer KV into **two** `MLState` tensors `(24, 2, 1024, 64)` with `torch.export` + SqueezeBits export hygiene will eliminate ~25 MB/step explicit KV copies and push runtime ANE proxy toward plan levels (>10%).

**Scope**

| In scope | Out of scope |
|----------|--------------|
| Custom `Cache` stacking layers into 2 states | Paged KV |
| `torch.export` + `ct.StateType` | Orion / private API |
| Ring + MLState combination (follow-up) | Quality suite v1 |

**Success criteria**: ANE compile succeeds; runtime ANE proxy **>10%**; GPU J **−50%** vs ring explicit-I/O path.

**Estimated effort**: 3–5 days

---

### Experiment 3: Ring + Quality Gate (was Exp 4)

**Priority**: 3

**Hypothesis**  
512-token sliding window preserves ≥95% greedy token match vs linear full-cache on short prompts.

**Success criteria**: ≥95% match on 50 prompts; perplexity Δ ≤5%.

---

### Experiment 4: Structural KV I/O A/B (MLState vs Ring Explicit I/O)

**Priority**: 4 — after Exp 2 MLState export

**Hypothesis**: MLState in-place updates reduce remaining GPU/CPU overhead vs ring explicit I/O (ring already cut GPU J to ~21 J).

**Estimated effort**: 1–2 days

---

## 5. Recommendation

### Start first (updated post Exp 1)

**Adopt ring512 int4 decode** as the default sustained benchmark path (`--kv-cache-mode ring`). Exp 1 exceeded success gates: **+38% t/s**, **6.7% ANE proxy**, **3.23 IPJ**, **0 re-prefills**.

**Next experiment**: **Exp 2 (Consolidated MLState)** — attack remaining ~25 MB/step explicit KV I/O; ring mask/write pattern is proven compatible with 44% ANE plan.

**Then**: **Exp 3 (quality gate)** before agent-scale deployment of 512-window decode.

### Is this thread still highest-leverage?

**Yes — validated.** Ring buffer redesign delivered **>1.5× IPJ** and **>2× ANE proxy** vs mask int4 linear without ctx512 export regression. KV thread remains highest priority; MLState is the next structural step.

### Research agenda adjustments

| Item | Adjustment |
|------|------------|
| **Thread 1 status** | **`In progress`** — Exp 1 complete (Iterate) |
| **Default decode path** | ring int4 512 over mask int4 linear |
| **Thread 2 (Hybrid)** | Parallel; ring improves Core ML decode leg |
| **Thread 4 (KV quant)** | Defer int4 *KV tensors*; weight int4 + ring is current best |
| **Success gate** | **Met** for Exp 1 (IPJ 3.23 > 1.5× baseline; ANE 6.7% > 2× baseline) |

### Suggested next actions

1. Run Exp 3 quality gate (greedy match / short perplexity).
2. Start Exp 2 consolidated MLState export.
3. Make ring512 the default in `ane_residency_benchmark.py` docs/examples.

---

*Related*: [`KV_Cache_Techniques_for_NPUs_and_Apple_Silicon.md`](KV_Cache_Techniques_for_NPUs_and_Apple_Silicon.md) · [`Alalā_Research_Agenda.md`](Alalā_Research_Agenda.md) · [`phase1/NOTES.md`](../phase1/NOTES.md)