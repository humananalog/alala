# Experiments

## Phase 0 kickoff (physical M4)

```bash
./experiments/phase0_kickoff.sh
```

Runs `setup_check` + `thermal_baseline`. See `docs/OSLab_Program_Board.md` kickoff checklist.

## General

Each experiment should follow `docs/Alalā_Experimentation_Framework.md`:

1. Hypothesis
2. Method
3. Metrics (include IPJ when relevant)
4. Controls
5. Success criteria
6. Results
7. Decision

Output structured JSONL to `logs/` and artifacts to `results/`.
