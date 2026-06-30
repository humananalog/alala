# OSLab Execution Playbook — How to Run Alalā

**Version**: 2.1  
**Date**: 2026-06-30  
**Purpose**: Single practical guide for executing the Alalā program day-to-day on **physical Mac Mini M4 24 GB**.  
**Primary Executor**: Grok Build (local AI coding agent on the M4).

**Execution constraint**: All workloads run locally on the target Mac Mini M4 24 GB using native tools (`powermetrics`, Metal/Core ML or MLX). Respect thermal limits — stop if temperature exceeds safe sustained threshold.

## 1. Daily Rhythm

1. Read `OSLab_Program_Board.md` for current status and blockers.
2. Check `Phase0_AI_Coder_Task_List.md` (or current phase task list).
3. Execute the next task **on the physical M4** — no simulation or remote hosts.
4. Log results (structured JSONL + raw `powermetrics` per `IPJ_Measurement_Protocol_Alalā.md`).
5. Update `OSLab_Program_Board.md` with progress and any new risks.
6. Commit only after passing `./verify.sh`.

## 2. Phase Execution Rules

- Complete all tasks in the current phase before moving to the next unless the Program Board explicitly authorizes otherwise.
- Every phase has clear entry and exit gates (see `Alalā_Physics_Corrected_Foundation.md`).
- **ANE-first routing** is default; measure CPU orchestration overhead before optimizing it away.
- If a task is blocked for more than 2 hours, update the Program Board and ask for human input.

## 3. Measurement & Logging (Physical M4 Only)

**Rule**: No IPJ claim without raw `powermetrics` logs + thermal data attached.

All experiments must log:
- Timestamp, task / experiment name
- Sustained IPJ (not peak burst before thermal steady state)
- Energy breakdown: total, ANE, CPU orchestration, dequantization (where applicable)
- ANE utilization at thermal steady state
- Thermal headroom: `temp_start_c`, `temp_steady_state_c`, `time_to_throttle_s`, `sustained_power_w`
- Before/after comparison when applicable

Use `harness/m4_energy_harness.py` modes per `How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md`.

**Priority**: Thermal headroom and sustained IPJ take precedence over peak throughput.

## 4. Rollback & Safety

- Keep the previous stable version easily recoverable.
- **Always respect thermal limits and stop if temperature is too high** (safe threshold from thermal baseline).
- Any change that degrades sustained IPJ by more than 10% or violates thermal/SRAM limits must be rolled back immediately.
- Self-improvement changes must pass HCA + marginal IPJ checks before being kept.

## 5. Communication with Human

Grok Build should proactively communicate when:
- A decision has significant IPJ or architectural impact (>10%).
- Thermal or SRAM limits are being approached.
- A task has been blocked for more than 2 hours.
- New risks are identified.

## 6. Success Criteria for This Playbook

Grok Build can execute phases with minimal human intervention while maintaining documentation quality, all measurements grounded in physical M4 `powermetrics` artifacts, and all changes gated by sustained IPJ under the thermal envelope.
