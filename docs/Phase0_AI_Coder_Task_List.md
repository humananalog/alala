# Phase 0 — AI Coder Task List (Grok Build)

**Version**: 1.2  
**For**: Grok Build (local AI coding agent)  
**Rules**: You must strictly follow `AI_Coder_Rules_Guidelines_Alalā.md` at all times.

**Execution constraint**: All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold. No IPJ claim without raw `powermetrics` logs + thermal data (`IPJ_Measurement_Protocol_Alalā.md` §2.1).

This document contains explicit, numbered tasks for Phase 0. Complete them in order unless the Program Board instructs otherwise.

## Week 1 Tasks (ANE Characterization Focus)

**Task W1-01: Environment & Logging Setup**
1. Create the directory structure: `experiments/`, `logs/`, `harness/`, `results/`
2. Copy `m4_energy_harness.py` into the `harness/` folder.
3. Make the script executable if needed.
4. Run a basic test to verify `powermetrics` access and logging works.
5. Log the result in `logs/setup_log.jsonl`

**Success Criteria**: Harness can log power and temperature for at least 30 seconds without errors.

**Task W1-02: Thermal Baseline**
1. Ensure the machine is at a stable cool temperature (let it idle for 10+ minutes if warm).
2. Run the thermal baseline experiment using the harness (idle + sustained load).
3. Record results in `results/thermal_baseline/`
4. Update `OSLab_Program_Board.md` with key findings (peak temperature, time to stabilize, etc.)

**Success Criteria**: Clear thermal rise curve and steady-state temperature documented.

**Task W1-03: ANE SRAM Cliff Characterization**
1. Prepare workloads with increasing context lengths (start small, go up to where performance drops).
2. Run the SRAM cliff experiment using the harness.
3. Identify \( L_{\text{cliff}} \): context length where sustained throughput drops ≥30% (per IPJ protocol §2.2).
4. Document results clearly with graphs if possible.
5. Update the Program Board.

**Success Criteria**: Approximate SRAM cliff point identified and logged.

**Task W1-04: Fused int4 KV vs FP16 Decode Comparison (Short Context)**
1. Implement or use existing short decode workload in the harness.
2. Run comparison between FP16 and fused int4 KV path.
3. Measure energy and tokens/second.
4. Calculate IPJ\(_{phase0}\) delta per `IPJ_Measurement_Protocol_Alalā.md` §2.1 (with powermetrics artifacts).
5. Log results and update Program Board.

**Success Criteria**: Quantified IPJ difference between the two paths documented.

**Task W1-05: Update Program Board & Risk Register**
1. Summarize all findings from Week 1 tasks.
2. Update `OSLab_Program_Board.md` with progress and any new risks.
3. Update `Risk_Register.md` if needed.

**Success Criteria**: Program Board accurately reflects Week 1 results.

## Phase 0 Extended – Gap-Closing Experiments (Decision Gates)

**Posture**: These four experiments close material assumptions before model architecture or self-improvement cadence advances. Each extends Week 1 tasks, runs **only on physical Mac Mini M4 24 GB**, and uses `powermetrics` + package temperature logging. **Stop if temperature exceeds safe sustained threshold** (from W1-02). Status: **Defined – awaiting harness implementation**.

| ID | Extends | Harness mode (planned) | Decision if failed |
|----|---------|------------------------|-------------------|
| E1 | W1-04 | `ane_utilization` | Redesign routing/compilation before scaling model |
| E2 | W1-02 | `thermal_ipj_curve` | Redesign workload mix if sustained IPJ degrades ≥20% post-throttle |
| E3 | early Phase 1 | `meta_tax` | Do not scale self-improvement loop until net IPJ > 0 |
| E4 | W1-03 | `memory_spill` | Redesign memory hierarchy if spill joules/token > recompute |

### E1 – ANE Real Utilization Baseline

**Objective**: Measure what fraction of a minimal transformer forward pass (one MLX/Core ML block or baseline model) actually executes on ANE vs. CPU/GPU fallback under sustained ANE-first routing.

**Method**:
1. Run a fixed-shape minimal block end-to-end (batch=1, context below SRAM cliff).
2. Instrument: ANE vs CPU vs GPU time per forward pass; domain joules via `powermetrics`; orchestration gaps between invocations.
3. Sustain 5–10 min at thermal steady state.

**Instrumentation**: `powermetrics` (CPU/GPU/ANE power, 1 Hz); `temp_start_c`, `temp_steady_state_c`; optional Metal/Core ML ANE residency counters if available.

**Success**: `ane_compute_fraction_pct` ≥ 70% of forward-pass wall time **and** `energy_cpu_orchestration_joules` / `energy_joules` ≤ 25% at steady state.

**Failure (redesign gate)**: `ane_compute_fraction_pct` < 50% **or** orchestration energy fraction > 40% → halt architecture work; fix graph compilation/routing first.

### E2 – Sustained Thermal + IPJ Degradation Curve

**Objective**: Quantify how thermal headroom shrinkage under mixed ANE + CPU orchestration degrades sustained IPJ over 30–60+ minutes.

**Method**:
1. After W1-02 safe envelope is known, run representative mixed workload (decode + Python orchestration loop).
2. Log IPJ\(_{phase0}\), temperature, and throughput in 5-min windows for full duration.
3. Record `time_to_throttle_s`, post-throttle recovery after 10-min idle.

**Instrumentation**: Continuous `powermetrics`; per-window JSONL with `ipj`, `temp_steady_state_c`, `tokens_per_second_sustained`, `thermal_headroom_c` (margin to safe threshold).

**Success**: Sustained IPJ in final 10 min within **80%** of first steady-state window; clear `time_to_throttle_s` documented or null if none.

**Failure (redesign gate)**: Sustained IPJ drops **≥20%** from first steady-state window to post-throttle window → redesign batch size, precision, or orchestration duty cycle before Phase 1.

### E3 – Closed-Loop Meta-Tax Measurement (Early Phase 1 Gate)

**Objective**: Measure total joules consumed by one bounded self-improvement cycle (propose → evaluate → accept/reject a small automatically verifiable change) vs. joules saved in subsequent runs.

**Method**:
1. Define a micro-change with deterministic verifier (e.g. compiler flag, KV layout tweak).
2. Log `J_meta_propose`, `J_meta_evaluate`, `J_meta_apply` separately via `powermetrics`.
3. Run post-change workload N times; compute ΔIPJ vs. pre-change baseline at same thermal headroom.

**Instrumentation**: `powermetrics` per phase; JSONL fields: `energy_meta_total_joules`, `energy_saved_subsequent_joules`, `net_ipj_delta`.

**Success**: `net_ipj_delta` > 0 over amortization window (change cost recovered within defined N runs at sustained thermal conditions).

**Failure (redesign gate)**: `energy_meta_total_joules` ≥ `energy_saved_subsequent_joules` over amortization window → do not scale self-improvement cadence; simplify meta machinery.

### E4 – Memory Pressure & Spill Cost Quantification

**Objective**: Stress realistic working sets (weights + growing KV + activations + harness overhead) and quantify energy/throughput cost of ANE on-chip SRAM spills (~28–30 MB) vs. recompute or paging.

**Method**:
1. After W1-03 \( L_{\text{cliff}} \) is known, sweep context toward and past cliff with fixed model.
2. For each tier: measure joules/token, sustained tokens/s, unified-memory bandwidth proxy (if available).
3. Compare one recompute-at-tile vs. spill-to-unified-memory path at matched context.

**Instrumentation**: `powermetrics`; log `context_length`, `working_set_mb`, `spill_events` (or proxy), `energy_per_token`, `recompute_energy_delta_joules`.

**Success**: Documented spill cost curve; recompute or int4 KV path shows positive IPJ delta vs. FP16 spill at contexts above \( L_{\text{cliff}} \).

**Failure (redesign gate)**: Spill joules/token > recompute joules/token at target context → redesign hierarchical memory layout before scaling context or model size.

## Week 2 Tasks

**Task W2-01 to W2-06**: (To be expanded based on Week 1 results and Program Board updates)

You will receive updated tasks for Week 2 after completing and logging Week 1.

**Important Rules for Grok Build**:
- Never skip logging.
- Never make major architectural changes without updating the Program Board first.
- Always respect thermal limits and stop if temperature is too high.
- Follow `AI_Coder_Rules_Guidelines_Alalā.md` strictly.
- Ask for human input only when truly blocked or when a decision requires human judgment per the rules.
