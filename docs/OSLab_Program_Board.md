# OSLab Program Board — Alalā

**Version**: 1.2
**Purpose**: Single source of truth for current status, risks, decisions, and progress.

## Current Phase

**Phase 1 – ANE-First Execution & Seeding Model** — **STARTING** (2026-07-01)  
**Strategy**: `Phase1_ANE_First_Strategy.md`  
**Phase 0** — **COMPLETE** (2026-06-30); synthesis: `Phase0_Results_Summary_Alalā.md`

**Prior label**: ANE Characterization & Measurement Infrastructure  
**Started**: 2026-06-30  
**Target**: Harness implementation on physical Mac Mini M4 24 GB, then Week 1–2 measurements

**Readiness (2026-06-30)**:
- Documentation audit Tasks 1–5 **complete**
- `./verify.sh` passing
- `harness/m4_energy_harness.py` **implemented** — `thermal_baseline` mode validated on hardware
- Thermal baseline run `thermal_baseline_20260630T144128Z_8947f4d5` complete (599 powermetrics samples + JSONL)
- All four harness modes implemented and run on physical M4 with powermetrics artifacts

## Phase 0 Success Criteria (Measurable M4 Numbers)

All criteria require raw `powermetrics` logs + thermal data per `IPJ_Measurement_Protocol_Alalā.md` §2.1.

| Criterion | Target | Source benchmark |
|-----------|--------|------------------|
| Thermal baseline curve | Idle + sustained load power (W), `temp_steady_state_c`, safe sustained envelope documented | Benchmark 1 / `thermal_baseline` | **Done** |
| SRAM cliff context length | \( L_{\text{cliff}} \) where sustained throughput drops ≥30% | Benchmark 2 / `sram_cliff` | **Done** — \( L_{\text{cliff}}=1024 \) |
| int4 vs FP16 IPJ delta | Repeatable `IPJ_phase0` delta including `energy_dequant_joules` | Benchmark 3 / `kv_comparison` | **Done** — ΔIPJ −0.0028 |
| Sustained ANE utilization | ANE utilization % at thermal steady state under ANE-first routing (baseline TBD from measurement) | Benchmarks 2–4 | **Done** — ~0% (MLX GPU path; documented) |
| Orchestration overhead | `energy_cpu_orchestration_joules` / total joules ratio documented | Benchmark 4 / `orchestration` | **Done** — ~4.3% CPU/total tight loop |

**Governing principle**: Thermal headroom and sustained IPJ take precedence over peak throughput.

## Documentation Audit Log (2026-06-30)

### Task 1 — Foundational physics grounding
- `Alalā_Physics_Corrected_Foundation.md` — §0 M4 silicon realities; thermal first-class; ANE-first default
- `IPJ_Measurement_Protocol_Alalā.md`, `Phase0_Microbenchmark_Suite_Plan.md`, memory docs, HCA — M4-specific phrasing

### Task 2 — Operational IPJ
- `IPJ_Measurement_Protocol_Alalā.md` §2.1–§2.3: executable IPJ\(_{phase0}\), SRAM cliff method, harness spec
- `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`: physical-M4-only execution guide
- `harness/README.md`: four harness modes

### Task 3 — Program Board + Risk Register
- Phase 0 success criteria table (above)
- `Risk_Register.md`: Phase 0 risks R02–R06 (SRAM cliff, thermal, 24 GB pressure, orchestration, dequant)

### Task 4 — Cross-consistency
- `Project_Index_Alalā.md`: full 19-doc navigation hub, operational IPJ, harness modes
- Terminology alignment: unified-memory spills, ≥30% SRAM cliff, execution constraint on planning docs

### Task 5 — Execution guidance
- `OSLab_Execution_Playbook.md` v2.1: physical M4 only, powermetrics required, thermal safety, ANE-first

## Human Review Flags (post-audit)
| Doc | Flag | Reason |
|-----|------|--------|
| `Compiler_Passes_Skeleton_Alalā.md` | Expected Benefit numbers unmeasured | Pass benefits are hypotheses until Phase 1 M4 IPJ validation |
| `Alalā_Improvement_Playbook.md` | Marginal IPJ not cross-linked to §2.1 | Self-improvement gating should cite operational IPJ after Phase 0 |
| `Meta_Controller_Skeleton_Alalā.md` | Threshold constants unspecified | Controller thresholds should be set from Phase 0 measured baselines |
| Phase 0 success criteria | Sustained ANE utilization % TBD | Target % intentionally deferred until thermal baseline on hardware |

## Active Tasks (as of today)
- W1-00: Docs audit (Tasks 1–5) — **Complete**
- W1-01: Implement `harness/m4_energy_harness.py` — **Complete** (`thermal_baseline` mode)
- W1-02: Thermal Baseline on physical M4 — **Complete** (2026-06-30)
- W1-03: ANE SRAM Cliff Characterization — **Complete** (2026-06-30)
- W1-04: FP16 vs int4 KV Comparison — **Complete** (2026-06-30)
- W1-05: Program Board & Risk Register — **Complete** (2026-06-30)

## Key Risks (Top 5 — see `Risk_Register.md`)
1. **R02** Thermal throttling under sustained ANE+CPU load
2. **R03** SRAM cliff impact on long-context decode
3. **R05** ANE utilization gaps due to orchestration
4. **R06** Dequantization energy eroding int4 gains
5. **R04** 24 GB working-set pressure

## Recent Decisions
- 2026-07-01: **Graph cleanup (scatter KV) + hybrid recommendation** — scatter int4 clean: **48.6 t/s** but **0.36% ANE** (mask int4: 27.7 t/s, 2.9% ANE). Scatter regresses ANE plan to 0%. **Recommend hybrid** (mask int4 for ANE energy, scatter for throughput). Run `6f90882a`.
- 2026-07-01: **int4 decode quant succeeds** — mask int4 @ ctx 512: **27.73 t/s**, **2.90% ANE proxy**. Run `1b69eca7`.
- 2026-07-01: **prefill-kv int4** — `qwen2.5-0.5b-prefill-kv-int4.mlpackage`; compute plan **29.1% ANE**.
- 2026-07-01: **torch.export decode succeeds** — `qwen2.5-0.5b-decode-kv-torch-export.mlpackage` achieves **44.8% ANE** in `MLComputePlan` (vs 0% MLState). Profile `bf783c54`.
- 2026-07-01: ANE placement diagnosis — decode MLState graph gets **0% ANE** in `MLComputePlan`; `CPU_AND_NE` fails ANE compile. Superseded for placement by torch.export path.
- 2026-07-01: Core ML decode export unblocked via MLState (`qwen2.5-0.5b-decode-kv.mlpackage`); first benchmark 7.45 t/s, 0.11% ANE @ ctx 512.
- 2026-07-01: Phase 1 seeding model — **Qwen2.5-0.5B-Instruct** selected for first ANE conversion attempt (see Seeding Model Decision).
- 2026-07-01: SRAM cliff detector updated for MLX GPU path (throughput + memory/power signals; ANE % optional).
- 2026-07-01: Phase 0 canonical results summary published (`Phase0_Results_Summary_Alalā.md`).
- 2026-06-30: Thermal headroom and sustained IPJ take precedence over peak throughput.
- 2026-06-30: No IPJ claim without raw powermetrics + thermal artifacts.
- 2026-06-30: ANE-first routing is default; measure CPU orchestration before minimizing.
- 2026-06-30: Adopted strict "no placeholder content" policy for all documentation.

## W1-02 Thermal Baseline Results (2026-06-30)

**Experiment**: `thermal_baseline_20260630T144128Z_8947f4d5`  
**Artifacts**: `logs/thermal_baseline_20260630T144128Z_8947f4d5.powermetrics.txt` (18.9 MB, 599 samples), JSONL + `results/thermal_baseline/.../summary.json`

| Metric | Value | Notes |
|--------|-------|-------|
| Idle power | **1.08 W** | 60 s pre-load |
| Sustained power | **11.85 W** | CPU spin load, thermal steady state |
| Temp start | **46.8°C** | macmon (smc sampler unavailable in powermetrics) |
| Temp steady-state | **82.7°C** | under sustained load |
| Peak temp | **83.8°C** | |
| Time-to-throttle proxy | **~5 s** | powermetrics thermal-pressure / power drop heuristic |
| Total energy | **6263 J** | CPU-dominated (`cpu_spin` interim load) |
| ANE utilization | **0%** | expected for CPU-only load |

**Proposed safe sustained `--temp-threshold`**: **85°C** (pending human confirmation; steady-state was 82.7°C on this run).

**Caveats**: Interim load was `cpu_spin`, not ANE-first decode; temperature via macmon supplement. Re-run with ANE decode workload when integrated for production baseline.

## W1-03 SRAM Cliff Results (2026-06-30)

**Experiment**: `sram_cliff_20260630T150641Z_f384fd3c`  
**Model**: `baseline` → `mlx-community/Qwen2.5-7B-Instruct-4bit`  
**Artifacts**: `logs/sram_cliff_20260630T150641Z_f384fd3c.jsonl` + per-context powermetrics in `results/sram_cliff/.../`

| Context | Sustained tok/s | Peak mem (GB) | Sustained power (W) |
|---------|-----------------|---------------|---------------------|
| 512 | **9.65** | 4.71 | 16.9 |
| 1024 | **6.40** | 4.92 | 17.5 |
| 2048 | **3.73** | 5.11 | 17.1 |
| 4096 | **2.13** | 5.25 | 17.2 |
| 8192 | **1.60** | 5.41 | 17.0 |

**\( L_{\text{cliff}} = 1024 \)** — first ≥30% sustained throughput drop vs prior step (9.65 → 6.40, −33.7%) with monotonic `peak_memory_gb` rise.

**Run parameters**: 90 s/step, 60 s steady window, 180 s inter-step cooldown, `--temp-threshold 88` (MLX decode exceeds cpu_spin 85°C baseline).

**Caveats**: MLX routes through GPU not ANE (`ane_utilization_pct` ≈ 0); cliff correlates with memory growth + throughput, not ANE utilization drop. Prior aborted run `...f4f68ae2` stopped at 85°C after ctx 512 only.

**Decode temp threshold for sweeps**: **88°C** (revised from 85°C cpu_spin baseline).

## W1-04 KV Comparison Results (2026-06-30)

**Experiment**: `kv_comparison_20260630T152942Z_54f06d2d` @ context **512** (below \( L_{\text{cliff}} \))

| Path | Sustained tok/s | IPJ\(_{phase0}\) | Energy (J) |
|------|-----------------|-----------------|------------|
| FP16 | 9.63 | 0.542 | 1022 |
| int4 KV | 9.87 | 0.539 | 1028 |

**ΔIPJ = −0.0028** (int4 **worse** despite higher tok/s). **energy_dequant_joules = +5.5 J** incremental vs FP16.

**Decision**: Reject int4 KV config at ctx 512 for this MLX GPU path — dequant energy erodes gains (R06 confirmed).

## W1-05 / Benchmark 4 Orchestration Results (2026-06-30)

**Experiment**: `orchestration_20260630T153406Z_6b6d5129` @ context **512**

| Profile | Sustained tok/s | CPU orchestration / total |
|---------|-----------------|---------------------------|
| Tight MLX loop | 10.10 | **4.34%** |
| Delayed agent-style (20 ms) | 10.07 | **3.67%** |

CPU orchestration is a small fraction of total joules on this workload; Python dispatch delay does not dominate GPU decode energy.

## Phase 0 Gate Summary

All five success criteria have **measured M4 numbers** with powermetrics artifacts in `logs/` and `results/`. MLX routes through **GPU not ANE** — ANE utilization baseline recorded as ~0%; Phase 1 should pursue Core ML / ANE-first path.

## Blockers

None for Phase 0 completion.

## Phase 1 — ANE-First Execution

**Status**: Active (2026-07-01)  
**Tools**: `phase1/coreml_convert.py`, `phase1/coreml_kv_convert.py`, `phase1/coreml_quantize.py`, `phase1/kv_decode.py`, `phase1/coreml_instrumentation.py`, `phase1/ane_residency_benchmark.py`, `phase1/ane_placement_profile.py`, `phase1/PROFILING.md`  
**Interface notes**: `phase1/NOTES.md`

### Entry Criteria

| Criterion | Status |
|-----------|--------|
| Phase 0 complete with measured baselines | **Done** |
| SRAM cliff detector fix | **Done** (2026-07-01) |
| Core ML conversion helper + ANE residency benchmark scaffold | **Done** (2026-07-01) |
| Measurable ANE utilization **> 60%** on seeding model | **38%** prefill proxy @ ctx 512; **44.8%** decode torch.export plan @ ctx 512; runtime proxy still ~0% |
| Safe operating region: context ≤ 1024 (or paged KV) + thermal duty cycle | Pending validation |
| First bounded self-improvement micro-scaffold with IPJ gating | After ANE residency gate |

### First Experiment (Week 1)

**Model**: `Qwen/Qwen2.5-0.5B-Instruct` (primary seeding candidate)

**Pipeline**:
1. Export KV models — `phase1/.venv/bin/python phase1/coreml_kv_convert.py --output-dir models --max-ctx 1024`
2. Run stateful decode benchmark — `phase1/.venv/bin/python phase1/ane_residency_benchmark.py --backend coreml --decode --context 512,1024`
3. Run MLX comparison — `python3 phase1/ane_residency_benchmark.py --backend mlx --decode --context 512,1024`
4. ANE placement profile (optional) — `PYTHONPATH=phase1 phase1/.venv/bin/python phase1/ane_placement_profile.py`
5. Log powermetrics + JSONL to `logs/` and `results/` (tracked in git)

**Controls**: batch=1, thermal threshold **85°C** steady-state (Phase 0 envelope), 60 s step / 30 s steady window.

### First Experiment Results (2026-07-01)

| Run | Backend | ctx | Sust. t/s | ANE proxy % | Temp steady | Notes |
|-----|---------|-----|-----------|-------------|-------------|-------|
| `ane_residency_20260701T001734Z_0bd0328f` | MLX 0.5B | 512 | **84.2** | ~0% | 81.9°C | Aborted before ctx 1024 (peak 92°C) |
| `ane_residency_20260701T002500Z_d1b410d0` | Core ML | 512 | 4.2 | **38.0%** | 62.9°C | Prefill proxy; **measurable ANE residency** |
| `ane_residency_20260701T002500Z_d1b410d0` | Core ML | 1024 | 3.9 | **11.7%** | 84.8°C | Aborted post-run (peak 87.7°C) |

**Decision**: Core ML path achieves real ANE energy attribution (38% @ ctx 512) vs ~0% MLX GPU. Throughput/IPJ not yet competitive — prefill-only proxy + no KV decode. Next: stateful Core ML decode + IPJ within 10% gate.

### KV Cache Decode Results (2026-07-01)

Stateful decode path landed in `phase1/kv_decode.py` + `--decode` on `ane_residency_benchmark.py`.

| Run | Backend | ctx | Sust. t/s | ANE proxy | Temp steady | Notes |
|-----|---------|-----|-----------|-----------|-------------|-------|
| `ane_residency_20260701T005044Z_d9ea7ae1` | MLX | 512 | **106.7** | ~0% | 79.8°C | Real mlx_lm KV decode |
| `ane_residency_20260701T005414Z_011715de` | MLX | 1024 | **57.6** | ~4% | 82.1°C | |
| `ane_residency_20260701T005247Z_b8d6539e` | Core ML | 512 | **35.0** | 0.3% | 83.4°C | Prefill Core ML + TorchScript decode (KV active) |
| `ane_residency_20260701T005431Z_c4529935` | Core ML | 1024 | — | — | — | GPU OOM on prefill-kv recycle |

**Decision:** KV hand-off works end-to-end; sustained Core ML decode throughput **35 t/s @ ctx 512** (~8× prefill proxy, ~3× slower than MLX — inside 3–5× band) with TorchScript fallback.

### Core ML Decode Conversion Status (2026-07-01)

**Export path:** MLState (`ct.StateType` for `keyCache` / `valueCache`) via `phase1/coreml_kv_convert.py --mode decode`. Follows HuggingFace Mistral7B stateful export; Qwen2-specific patches for RoPE `rotate_half`, `repeat_kv`, and SDPA dtype alignment.

**Artifact:** `models/qwen2.5-0.5b-decode-kv.mlpackage` ✅ (replaces blocked `No matching select or slice` / `int` op failures from stock `StaticCache` + dynamic shape ops).

| Run | Decode runtime | ctx | Sust. t/s | ANE proxy | Temp steady | Notes |
|-----|----------------|-----|-----------|-----------|-------------|-------|
| `ane_residency_20260701T005247Z_b8d6539e` | TorchScript `.pt` | 512 | **35.0** | 0.3% | 83.4°C | Prior fallback |
| `ane_residency_20260701T010929Z_830681e7` | **MLState `.mlpackage`** | 512 | **7.45** | **0.11%** | 83.5°C | First Core ML decode package benchmark |

**Decision:** Core ML decode export **unblocked**; autoregressive loop runs fully on Core ML with `MLState` I/O (`inputIds` + `causalMask`, in-place cache). ANE residency **not recovered** during decode (0.11% vs 38% prefill proxy; vs 0.3% TorchScript baseline). Throughput **regressed** vs TorchScript (7.45 vs 35 t/s) — investigate compute-unit placement, int4 quant, and graph size.

### ANE Placement Status (2026-07-01)

**torch.export decode experiment complete.** Re-export via `torch.export` + explicit KV I/O recovers **compile-time ANE placement** on decode. MLState path confirmed as root cause (TorchScript dialect + state ops).

| Artifact | Dialect | ANE plan % | GPU plan % | Sust. t/s | ANE proxy | `CPU_AND_NE` |
|----------|---------|------------|------------|-----------|-----------|--------------|
| `qwen2.5-0.5b-ane.mlpackage` | ATEN | 48.7% | 0.2% | — | — | ✅ |
| `qwen2.5-0.5b-prefill-kv.mlpackage` | ATEN | 31.0% | 8.3% | — | — | ✅ |
| `qwen2.5-0.5b-decode-kv.mlpackage` (MLState) | TorchScript | **0.0%** | **44.0%** | 7.45 | 0.11% | ❌ ANE compile fail |
| `qwen2.5-0.5b-decode-kv-torch-export.mlpackage` (fp16) | ATEN | 44.8% | 1.3% | 7.93 | 0.067% | ⚠️ |
| **`qwen2.5-0.5b-decode-kv-torch-export-int4.mlpackage`** | **ATEN** | **44.1%** | **2.0%** | **27.73** | **2.90%** | ⚠️ |

Benchmark: `ane_residency_20260701T022853Z_1b69eca7` (60 s, ctx 512). Tool: `phase1/coreml_quantize.py`.

**int4 mask impact:** ANE proxy **43×** fp16; throughput **3.5×**; compute-plan ANE ~44%; runtime proxy **2.9%**.

**Graph cleanup (scatter KV):** −468 ops; throughput **48.6 t/s** (exceeds TorchScript); ANE plan **0%** decode; proxy **0.36%**.

**prefill-kv int4:** 29.1% ANE plan.

**Decision:** ANE proxy **&lt;8%** after scatter cleanup → **hybrid architecture** (mask int4 for ANE energy; scatter int4 + int4 prefill for throughput). See `phase1/PROFILING.md`.

**Tracked artifacts (main @ 2026-07-01):**
- `results/ane_residency/ane_residency_20260701T024057Z_6f90882a/` — scatter int4 clean + prefill int4 benchmark
- `results/ane_residency/ane_residency_20260701T022853Z_1b69eca7/` — mask int4 decode benchmark
- `results/ane_placement_profile/ane_placement_profile_20260701T020740Z_bf783c54/` — fp16 torch.export profile
- `results/ane_residency/ane_residency_20260701T010929Z_830681e7/` — MLState baseline

**Gaps:** (1) Raise ANE proxy toward compute-plan 44%; (2) Match/exceed TorchScript 35 t/s (int4 at 27.7); (3) ctx 1024 + triple-model profile OOM; (4) Quantize prefill-kv.

### Success Metrics (First Experiment)

| Metric | Target | Status (2026-07-01) |
|--------|--------|---------------------|
| ANE utilization | **> 0%** first run; **> 60%** gate | mask int4 proxy **2.9%**; scatter clean **0.36%**; plan up to 44% |
| Sustained IPJ | Within **10%** of MLX baseline (or better) | scatter clean IPJ **2.86** @ 48.6 t/s; mask int4 IPJ 1.32 |
| Sustained throughput | Document tok/s at ctx 512 and 1024 | MLX 106.7; **scatter int4 clean 48.6**; mask int4 27.7; TorchScript 35.0 |
| Thermal compliance | Steady-state **≤ 85°C** | MLX aborted 92°C; Core ML completed with post-run abort 87.7°C |
| Energy attribution | ANE / CPU / GPU joules per step | **Done** — Core ML 200 J ANE @ ctx 512 |

**Reference baseline**: Phase 0 MLX GPU path (7B model) — ~9.65 t/s @ ctx 512, ~6.40 t/s @ ctx 1024, ~0% ANE (`sram_cliff_20260630T150641Z_f384fd3c`). Phase 1 compares 0.5B paths against this reference and records deltas.

### Seeding Model Decision

**Strategy doc**: `Phase1_ANE_First_Strategy.md`  
**Primary candidate for first conversion attempt**: **Qwen2.5-0.5B-Instruct**

Ranked by expected trade-off among model size, instruction capability, and ANE mapping quality on M4 (working set vs ~28–30 MB ANE SRAM budget; sustained IPJ under \( L_{\text{cliff}}=1024 \)):

| Rank | Model | Params | Physics rationale |
|------|-------|--------|-------------------|
| **1 (start here)** | **Qwen2.5-0.5B-Instruct** | ~0.5B | Smallest capable instruct model in set; weights + activations at ctx 512–1024 stay well below unified-memory pressure seen at 7B; transformer blocks map cleanly to Core ML fixed shapes; strong instruction following for seeding agent loops without 7B thermal/memory tax. |
| 2 | Phi-3.5-mini / Phi-3-mini-128k | ~3.8B / ~3.8B | Excellent reasoning per parameter, but larger than ideal for first ANE residency proof — higher KV footprint approaches \( L_{\text{cliff}} \) faster and may reduce ANE tile residency; reserve as fallback if Qwen 0.5B conversion under-delivers on capability. |
| 3 | Gemma-2-2B-IT (quantized) | ~2B | Mid-size; quantized weights help bandwidth but add dequant path (Phase 0: +5.5 J overhead, ΔIPJ −0.0028 on MLX GPU). Test after primary path establishes ANE utilization baseline. |
| 4 | Custom distilled ~350M | ~350M | Maximum ANE mapping surface area if off-the-shelf models fail residency targets; higher engineering cost — only if ranks 1–3 cannot reach >60% ANE utilization with acceptable IPJ. |

**Decision (2026-07-01)**: Begin Core ML / ANE-friendly conversion with **Qwen2.5-0.5B-Instruct**. Measure ANE utilization %, sustained tok/s at ctx 512 and 1024, energy per token, IPJ, and thermal behavior vs MLX GPU baseline (Phase 0: ~9.65 / ~6.40 t/s sustained, ~0% ANE).

## Next Milestone

Execute first ANE residency experiment on physical M4 (Qwen2.5-0.5B Core ML conversion + benchmark). Gate compiler pass prototyping on success metrics above per `Revised_Phase0_2_Systems_Plan_Alalā.md`.

## Human Review Flags
_See table above (post-audit)._

## Notes
This board must be updated by Grok Build after every significant task or discovery.
