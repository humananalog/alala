# Risk Register — Alalā

**Version**: 1.2  
**Purpose**: Track and manage risks across the project, with Phase 0 M4 measurement focus.

## Risk Scoring

- **Likelihood**: Low / Medium / High
- **Impact**: Low / Medium / High
- **Owner**: Who is responsible for monitoring/mitigation

## Guiding Principle (Phase 0+)

**Thermal headroom and sustained IPJ take precedence over peak throughput.** Risks that inflate peak benchmarks while degrading sustained useful work per joule are High impact.

## Active Risks (as of 2026-06-30)

| ID | Risk | Likelihood | Impact | Owner | Mitigation / Notes | Status |
|----|------|------------|--------|-------|--------------------|--------|
| R-ANE-01 | Low real ANE coverage in forward pass → high orchestration & data-movement tax | To be quantified by experiment | To be quantified by experiment | Grok Build | **E1** ANE utilization baseline before architecture decisions; redesign if `ane_compute_fraction_pct` < 50% or orchestration tax > 40% | Open – experiment defined |
| R-THERM-02 | Rapid thermal throttling under sustained mixed load erodes IPJ | To be quantified by experiment | To be quantified by experiment | Grok Build | **E2** 30–60+ min thermal+IPJ curve; redesign workloads if sustained IPJ degrades ≥20% post-throttle | Open – experiment defined |
| R-META-03 | Self-improvement loop meta-tax exceeds marginal gains | To be quantified by experiment | To be quantified by experiment | Meta-Controller + Grok Build | **E3** closed-loop meta-tax measurement; do not scale loop until `net_ipj_delta` > 0 | Open – experiment defined |
| R-MEM-04 | Realistic working sets exceed 24 GB residency or cause expensive ANE SRAM spills | To be quantified by experiment | To be quantified by experiment | Grok Build | **E4** memory-pressure tests; validate hierarchical memory design against spill vs. recompute joules/token | Open – experiment defined |
| R01 | Noisy or unreliable `powermetrics` data | Medium | High | Grok Build | Multiple trials; attach raw logs to every result; cross-validate with external meter on calibration runs | Open |
| R02 | Thermal throttling under sustained ANE+CPU load | High | High | Grok Build | Thermal baseline first; log start/steady-state temp and time-to-throttle; stop if over safe sustained threshold; modulate batch/precision | Open |
| R03 | SRAM cliff impact on long-context decode | High | High | Grok Build | Benchmark 2: measure \( L_{\text{cliff}} \) (≥30% sustained throughput drop); redesign KV tiling/int4 if marginal IPJ negative past cliff | Open |
| R04 | 24 GB working-set pressure (KV + activations + harness) | Medium | High | Grok Build | Budget unified memory explicitly; measure harness overhead; avoid loading full FP16 KV at long context | Open |
| R05 | ANE utilization gaps due to orchestration | High | High | Grok Build | ANE-first default; Benchmark 4 measures `energy_cpu_orchestration_joules`; minimize only after measure | Open |
| R06 | Dequantization energy eroding theoretical int4 gains | Medium | High | Grok Build | Benchmark 3: log `energy_dequant_joules` vs FP16; reject config if IPJ delta ≤ 0 | Open |
| R07 | Grok Build underestimates low-level optimization difficulty | High | High | Human | Small gated tasks; frequent review in Phase 0 | Open |
| R08 | Documentation and code drift | Medium | Medium | Grok Build | Update docs after code changes; run `./verify.sh` before commit | Open |
| R09 | Over-optimism on ANE utilization gains | High | High | Team | Ground all claims in Phase 0 measured M4 numbers with powermetrics artifacts | Open |
| R10 | Self-improvement loop produces low-value or harmful changes | Medium | High | Meta-Controller + Human | HCA + marginal IPJ gates; automatic rollback | Open |

## Phase 0/1 Lab Uncertainties

**These risks are accepted lab uncertainties; we close them with physical M4 measurements before committing to model scale or self-improvement cadence.**

- **Measure first, redesign if marginal IPJ negative** — applies to R03, R05, R06, R-ANE-01, R-THERM-02, R-META-03, R-MEM-04
- No IPJ claim without raw `powermetrics` + thermal data and thermal headroom statement (`IPJ_Measurement_Protocol_Alalā.md` §2.5)
- All benchmarks on physical Mac Mini M4 24 GB only; stop if temperature exceeds safe sustained threshold

## Closed / Mitigated Risks

None yet.

## Risk Review Cadence

- Review all open risks every Sunday.
- Add new risks as soon as they are identified.
- Escalate High/High risks to human immediately.
