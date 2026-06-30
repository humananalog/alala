# Meta-Controller Skeleton — Alalā

**Version**: 1.1  
**Goal**: Provide a bounded, auditable controller for self-improvement that respects HCA and IPJ constraints.

## Phase 0 Baseline Thresholds (set from M4 measurements)

Populate after W1-02 / E1 / E2 on physical hardware. Until then, controller uses conservative defaults and rejects aggressive changes.

| Threshold | Source experiment | Field | Default (pre-M4) |
|-----------|-------------------|-------|------------------|
| Safe sustained temp (°C) | W1-02 `thermal_baseline` | `max_temp_c` harness flag | unset — must measure |
| Max orchestration tax (%) | E1 `ane_utilization` | `orchestration_tax_pct` | 40 (gate from E1) |
| Max IPJ degradation (%) | E2 `thermal_ipj_curve` | `ipj_degradation_pct_final` | 20 (gate from E2) |
| Min net meta IPJ | E3 `meta_tax` | `net_ipj_delta` | must be > 0 |
| SRAM cliff context | W1-03 `sram_cliff` | `l_cliff` | measure on M4 |

See `results/measurement_status.json` for validation state.

## High-Level Design

The meta-controller sits above the main agent loop and decides:
- Whether a proposed improvement should be accepted
- How much compute/energy budget to allocate to self-improvement
- When to trigger rollback

It must remain lightweight and auditable.

## Core Loop (Pseudocode)

```python
def meta_controller_step(current_state, proposed_change):
    # 1. Evaluate HCA Impact
    hca_impact = evaluate_hca_impact(proposed_change)
    if hca_impact < 0:
        return reject_change("HCA violation")

    # 2. Estimate marginal IPJ
    marginal_ipj = estimate_marginal_ipj(current_state, proposed_change)
    if marginal_ipj <= 0:
        return reject_change("No positive marginal IPJ")

    # 3. Check resource budget
    if not within_energy_thermal_budget(proposed_change):
        return reject_change("Resource limit exceeded")

    # 4. Apply change with logging
    apply_change(proposed_change)
    log_improvement(proposed_change, marginal_ipj, hca_impact)

    return accept_change()
```

## Key Components

### 1. HCA Impact Evaluator
- Takes a proposed change and returns a simple score or statement.
- Changes that reduce human trust, autonomy, or flourishing are rejected.

### 2. Marginal IPJ Estimator
- Lightweight model or heuristic that predicts whether the change will produce net positive IPJ over its lifetime.
- Conservative by default (better to reject than accept harmful changes).

### 3. Resource & Thermal Guard
- Checks current temperature, power draw, and remaining energy budget.
- Rejects changes that would push the system into unsafe thermal or power regions.

### 4. Logging & Audit Trail
- Every decision (accept/reject) is logged with:
  - Proposed change
  - HCA impact
  - Estimated marginal IPJ
  - Resource check result
  - Final decision + reason

## Safety & Rollback

- All changes are applied behind a lightweight checkpoint/rollback mechanism.
- If post-change IPJ degrades significantly, the controller triggers automatic rollback.
- Human can force rollback at any time via the Program Board.

## Evolution Path

- **Phase 0–1**: Simple rule-based controller (as above).
- **Phase 2+**: Small learned controller trained on logged improvement outcomes, still heavily constrained by HCA and IPJ rules.

This skeleton ensures self-improvement remains bounded, measurable, and aligned with the project's core invariants.
