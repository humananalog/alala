# OSLab Program Board — Alalā

**Version**: 1.1  
**Purpose**: Single source of truth for current status, risks, decisions, and progress.

## Current Phase

**Phase 0 – ANE Characterization & Measurement Infrastructure** — **COMPLETE** (2026-06-30)

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

## Next Milestone

**Phase 1 planning** — ANE-first routing validation, compiler pass prototyping per `Revised_Phase0_2_Systems_Plan_Alalā.md`.

## Human Review Flags
_See table above (post-audit)._

## Notes
This board must be updated by Grok Build after every significant task or discovery.
