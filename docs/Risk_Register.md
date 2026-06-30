# Risk Register — Alalā

**Version**: 1.0  
**Purpose**: Track and manage risks across the project.

## Risk Scoring

- **Likelihood**: Low / Medium / High
- **Impact**: Low / Medium / High
- **Owner**: Who is responsible for monitoring/mitigation

## Active Risks (as of 2026-06-30)

| ID | Risk | Likelihood | Impact | Owner | Mitigation / Notes | Status |
|----|------|------------|--------|-------|--------------------|--------|
| R01 | Noisy or unreliable `powermetrics` data | Medium | High | Grok Build | Cross-validate with external measurement when possible; run multiple trials | Open |
| R02 | M4 thermal throttling during long experiments | Medium | Medium | Grok Build | Implement thermal-aware scheduling early; monitor temperature continuously | Open |
| R03 | SRAM cliff difficult to isolate cleanly | Medium | Medium | Grok Build | Use controlled workloads; accept approximate characterization | Open |
| R04 | Grok Build underestimates difficulty of low-level optimization | High | High | Human | Keep tasks small and gated; frequent human review in early phases | Open |
| R05 | Documentation and code drift | Medium | Medium | Grok Build | Update docs immediately after code changes; run verification script before commit | Open |
| R06 | Over-optimism on ANE utilization gains | High | High | Team | Ground all claims in measured data from Phase 0 | Open |
| R07 | Self-improvement loop produces low-value or harmful changes | Medium | High | Meta-Controller + Human | Strong HCA + marginal IPJ gates; automatic rollback capability | Open |

## Closed / Mitigated Risks

None yet.

## Risk Review Cadence

- Review all open risks every Sunday.
- Add new risks as soon as they are identified.
- Escalate High/High risks to human immediately.
