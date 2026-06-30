# OSLab Execution Playbook — How to Run Alalā

**Version**: 2.0  
**Date**: 2026-06-30  
**Purpose**: Single practical guide for executing the Alalā program day-to-day.  
**Primary Executor**: Grok Build (local AI coding agent).

## 1. Daily Rhythm

1. Read `OSLab_Program_Board.md` for current status and blockers.
2. Check `Phase0_AI_Coder_Task_List.md` (or current phase task list).
3. Execute the next task.
4. Log results (structured JSONL).
5. Update `OSLab_Program_Board.md` with progress and any new risks.
6. Commit only after passing the verification script.

## 2. Phase Execution Rules

- Complete all tasks in the current phase before moving to the next unless the Program Board explicitly authorizes otherwise.
- Every phase has clear entry and exit gates (see `Alalā_Physics_Corrected_Foundation.md`).
- If a task is blocked for more than 2 hours, update the Program Board and ask for human input.

## 3. Measurement & Logging

All experiments must be logged with:
- Timestamp
- Task / experiment name
- Key metrics (IPJ, utilization, energy, tokens/s, temperature, etc.)
- Before/after comparison when applicable
- Any anomalies or observations

Use the structured logging harness in `harness/`.

## 4. Rollback & Safety

- Keep the previous stable version easily recoverable.
- Any change that degrades IPJ by more than 10% or violates thermal/SRAM limits must be rolled back immediately.
- Self-improvement changes must pass HCA + marginal IPJ checks before being kept.

## 5. Communication with Human

Grok Build should proactively communicate when:
- A decision has significant IPJ or architectural impact.
- Thermal or SRAM limits are being approached.
- A task has been blocked for more than 2 hours.
- New risks are identified.

## 6. Success Criteria for This Playbook

This playbook is successful when Grok Build can execute phases with minimal human intervention while maintaining high documentation and code quality, and all changes are properly logged and gated by measurement.
