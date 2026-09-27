# Study boundaries

EP19 is currently a documented study design. The current files do not authorize
opening held-out signals or running the full model search.

## Decisions already made

- Predictions occur every 50 ms over a 900-ms horizon.
- The primary window is 300–600 ms; 0–300 ms is the near-event comparison.
- “Causal” means past-only information flow, not causal inference.
- WAY series 8–9 and eight whole self-paced participants are held out.
- The self-paced primary condition allows only four calibration parameters from
  the first 32 complete movements.

## Decisions still required before EEG scoring

- exact onset, stillness, synchronization, and ambiguity rules;
- five executable reference recipes and two source-specific non-neural models;
- the complete 32-event calibration procedure;
- finite ranges for every allowed model change;
- all numeric margins in `../SEARCH_POLICY.yaml`; and
- successful synthetic timing and future-information tests.

No scientific acceptance, external review decision, or permission to examine a
held-out result is implied by this summary.
