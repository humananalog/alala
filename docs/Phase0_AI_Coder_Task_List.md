# Phase 0 — AI Coder Task List (Grok Build)

**Version**: 1.0  
**For**: Grok Build (local AI coding agent)  
**Rules**: You must strictly follow `AI_Coder_Rules_Guidelines_Alalā.md` at all times.

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
3. Identify the approximate context length where throughput drops ~30%.
4. Document results clearly with graphs if possible.
5. Update the Program Board.

**Success Criteria**: Approximate SRAM cliff point identified and logged.

**Task W1-04: Fused int4 KV vs FP16 Decode Comparison (Short Context)**
1. Implement or use existing short decode workload in the harness.
2. Run comparison between FP16 and fused int4 KV path.
3. Measure energy and tokens/second.
4. Calculate rough IPJ difference.
5. Log results and update Program Board.

**Success Criteria**: Quantified IPJ difference between the two paths documented.

**Task W1-05: Update Program Board & Risk Register**
1. Summarize all findings from Week 1 tasks.
2. Update `OSLab_Program_Board.md` with progress and any new risks.
3. Update `Risk_Register.md` if needed.

**Success Criteria**: Program Board accurately reflects Week 1 results.

## Week 2 Tasks

**Task W2-01 to W2-06**: (To be expanded based on Week 1 results and Program Board updates)

You will receive updated tasks for Week 2 after completing and logging Week 1.

**Important Rules for Grok Build**:
- Never skip logging.
- Never make major architectural changes without updating the Program Board first.
- Always respect thermal limits and stop if temperature is too high.
- Follow `AI_Coder_Rules_Guidelines_Alalā.md` strictly.
- Ask for human input only when truly blocked or when a decision requires human judgment per the rules.
