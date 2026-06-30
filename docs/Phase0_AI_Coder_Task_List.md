# Phase 0 — AI Coder Task List (Grok Build)

**Version**: 1.3  
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

### Phase 0 Extended – Gap-Closing Experiments (Decision Gates)

Runs locally on **physical Mac Mini M4 24 GB** only. Harness modes: `ane_utilization`, `thermal_ipj_curve`, `meta_tax`, `memory_spill`. Status: **Harness ready — awaiting physical M4 runs**.

#### E1 – ANE Real Utilization Baseline

Instrument a minimal transformer block (or MLX baseline model) end-to-end on the physical M4. Measure what percentage of the forward pass actually executes on the ANE versus CPU/GPU fallback. Log ANE utilization %, orchestration overhead energy, power, and thermal under sustained load.

- **Objective**: Quantify real ANE forward-pass coverage and orchestration tax on unified memory under sustained ANE-first routing.
- **Decision gate**: Blocks model architecture commitment until ANE coverage and orchestration energy are measured — high CPU/GPU fallback or orchestration tax forces routing/compiler redesign before scaling.
- **Instrumentation**: `powermetrics` (CPU/GPU/ANE domain power); package temperature; `ane_utilization_pct`, `energy_cpu_orchestration_joules`, `temp_steady_state_c`.

#### E2 – Sustained Thermal + IPJ Degradation Curve

Extend the thermal baseline (W1-02) to longer-duration (30–60+ min) mixed ANE + orchestration workloads. Measure time-to-throttle, recovery behavior, and how IPJ degrades as thermal headroom shrinks. Thermal headroom is a first-class variable. Workloads must respect safe sustained temperature limits.

- **Objective**: Map sustained IPJ vs. thermal headroom under mixed ANE + CPU orchestration on the M4 thermal/DVFS envelope.
- **Decision gate**: Blocks workload and scheduling design if sustained IPJ degrades sharply as headroom shrinks — redesign batch size, precision, or duty cycle before Phase 1.
- **Instrumentation**: Continuous `powermetrics`; `time_to_throttle_s`, `ipj` per time window, `thermal_headroom_c`, `temp_steady_state_c`.

#### E3 – Closed-Loop Meta-Tax Measurement

Execute one bounded self-improvement cycle on the M4 (current model proposes + evaluates a small, automatically verifiable change). Measure the full energy cost of the improvement machinery itself versus any IPJ gain in subsequent runs. Net IPJ must be calculated and must be positive for the loop to scale.

- **Objective**: Measure total meta-overhead joules (propose, evaluate, accept/reject) vs. joules saved in subsequent runs at matched thermal headroom.
- **Decision gate**: Blocks self-improvement cadence scaling until `net_ipj_delta` > 0 — meta-tax exceeding marginal gains forces simpler machinery.
- **Instrumentation**: `powermetrics` per cycle phase; `energy_meta_total_joules`, `energy_saved_subsequent_joules`, `net_ipj_delta`, thermal logs.

#### E4 – Memory Pressure & Spill Cost Quantification

After the SRAM cliff test (W1-03), stress realistic working sets (model weights + growing KV cache + activations + harness overhead). Quantify energy and throughput cost of ANE on-chip SRAM spills (~28–30 MB) versus recompute strategies. This validates the hierarchical memory design against real 24 GB M4 physics.

- **Objective**: Quantify spill-to-unified-memory energy and throughput cost vs. recompute or tiling above \( L_{\text{cliff}} \).
- **Decision gate**: Blocks hierarchical memory and long-context design until spill joules/token are measured — expensive spills force memory-layout redesign before model scale-up.
- **Instrumentation**: `powermetrics`; `working_set_mb`, `context_length`, joules/token for spill vs. recompute paths, unified-memory bandwidth proxy if available.

## Week 2 Tasks

**Task W2-01: Close E1 (ANE Utilization)**
1. Run `sudo python harness/m4_energy_harness.py --mode ane_utilization --duration 300`
2. Validate artifacts with `validate_artifact.py --require-m4`
3. Record `ane_compute_fraction_pct`, `orchestration_tax_pct`; pass/fail per E1 gate

**Task W2-02: Close E2 (Thermal + IPJ Curve)**
1. Run `thermal_ipj_curve` for 30–60+ min with `--window-s 300`
2. Document `ipj_degradation_pct_final` and safe sustained envelope

**Task W2-03: Close E4 (Memory Spill)**
1. After W1-03, run `memory_spill` at contexts above and below \( L_{\text{cliff}} \)
2. Compare `energy_spill_joules_per_token` vs `energy_recompute_joules_per_token`

**Task W2-04: Week 1 Integration Report**
1. Update `measurement_status.json` with validated artifact paths
2. Update Program Board and Risk Register with quantified R-GAP likelihood/impact

**Task W2-05: E3 Meta-Tax (when meta machinery exists)**
1. Run `meta_tax` mode; require `net_ipj_delta` > 0 before scaling improvement cadence

**Task W2-06: Phase 0 Gate Review**
1. Confirm all Week 1 criteria + applicable E-gates pass or have redesign plans
2. Human sign-off before Phase 1 compiler work

**Important Rules for Grok Build**:
- Never skip logging.
- Never make major architectural changes without updating the Program Board first.
- Always respect thermal limits and stop if temperature is too high.
- Follow `AI_Coder_Rules_Guidelines_Alalā.md` strictly.
- Ask for human input only when truly blocked or when a decision requires human judgment per the rules.
