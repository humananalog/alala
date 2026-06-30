# Alalā Experimentation Framework

**Version**: 1.0  
**Purpose**: Standard way to design, run, and evaluate experiments.

## Experiment Structure

Every experiment should contain:

1. **Hypothesis**  
   Clear statement of what you expect to happen and why.

2. **Method**  
   Detailed description of how the experiment will be run (workloads, models, parameters, logging).

3. **Metrics**  
   Primary and secondary metrics (must include IPJ when relevant).

4. **Controls**  
   What is being held constant.

5. **Success Criteria**  
   Pre-defined thresholds for considering the experiment successful.

6. **Results & Analysis**  
   Raw data + interpretation.

7. **Decision**  
   What will be done based on the results (keep change, rollback, modify, investigate further).

## Logging Standard

All experiments must produce structured logs (JSONL) with at minimum:
- experiment_id
- timestamp
- hypothesis
- key metrics (IPJ, utilization, energy, tokens/s, temperature, etc.)
- before/after comparison (when applicable)
- notes / anomalies

## Gating

- Small experiments (low risk, low resource use) can be run autonomously by Grok Build.
- Medium/large experiments require updating the Program Board before starting.
- Any experiment that could significantly affect IPJ or HCA must include those metrics.

## Documentation

After an experiment is complete, a short summary should be added to the relevant section of the Program Board or a dedicated experiment log.

This framework ensures experiments are rigorous, comparable, and contribute to long-term learning rather than one-off results.
