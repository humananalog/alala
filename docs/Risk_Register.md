# Risk Register — Alalā

**Version**: 1.1  
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
| R01 | Noisy or unreliable `powermetrics` data | Medium | High | Grok Build | Multiple trials; attach raw logs to every result; cross-validate with external meter on calibration runs | Open |
| R02 | Thermal throttling under sustained ANE+CPU load | High | High | Grok Build | **Measured** thermal baseline + decode sweeps; decode threshold **88°C**, 180 s inter-step cooldown | Mitigated (measured) |
| R03 | SRAM cliff impact on long-context decode | High | High | Grok Build | **Measured** \( L_{\text{cliff}}=1024 \) on Qwen2.5-7B-4bit MLX path (2026-06-30); throughput −33.7% at 1024 vs 512 | Mitigated (measured) |
| R04 | 24 GB working-set pressure (KV + activations + harness) | Medium | High | Grok Build | Budget unified memory explicitly; measure harness overhead; avoid loading full FP16 KV at long context | Open |
| R05 | ANE utilization gaps due to orchestration | High | High | Grok Build | **Measured** CPU orch ~4.3% of joules (tight loop); ANE util ~0% on MLX GPU — orchestration not dominant, ANE routing is | Open (ANE routing) |
| R06 | Dequantization energy eroding theoretical int4 gains | Medium | High | Grok Build | **Confirmed** at ctx 512: ΔIPJ −0.0028, `energy_dequant_joules` +5.5 J; reject int4 KV for this path | Mitigated (measured) |
| R07 | Grok Build underestimates low-level optimization difficulty | High | High | Human | Small gated tasks; frequent review in Phase 0 | Open |
| R08 | Documentation and code drift | Medium | Medium | Grok Build | Update docs after code changes; run `./verify.sh` before commit | Open |
| R09 | Over-optimism on ANE utilization gains | High | High | Team | Ground all claims in Phase 0 measured M4 numbers with powermetrics artifacts | Open |
| R10 | Self-improvement loop produces low-value or harmful changes | Medium | High | Meta-Controller + Human | HCA + marginal IPJ gates; automatic rollback | Open |

## Phase 0 Specific Notes

- **Measure first, redesign if marginal IPJ negative** — applies to R03, R05, R06
- No IPJ claim without raw `powermetrics` + thermal data (`IPJ_Measurement_Protocol_Alalā.md` §2.1)
- All benchmarks on physical Mac Mini M4 24 GB only

## Closed / Mitigated Risks

None yet.

## Risk Review Cadence

- Review all open risks every Sunday.
- Add new risks as soon as they are identified.
- Escalate High/High risks to human immediately.
