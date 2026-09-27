# Research memory

## Reusable lessons from the design

- A prediction timestamp before a recorded onset is not enough; filtering,
  normalization, state, and labels must all obey past-only information flow.
- A strong absolute EEG-model score is not neural evidence without a strong
  non-neural baseline and separately trained capacity-matched surrogates.
- The model-order question must compare the same fitted families across
  horizons rather than two independently selected winners.
- New-participant replication is stronger when calibration is small and
  explicit: one temperature and three offsets from 32 events.
- Historical event detection and strict forecasting are different scientific
  tasks.

## Open questions

- Can timing uncertainty be bounded below 300 ms in both sources?
- Does the 29-channel intersection behave consistently across sample rates?
- What practical EEG-increment, calibration, model-pair, and latency margins
  should be set before scoring?
- Can the full reference and surrogate panel fit within the resource limits?

These are design questions, not scientific results.
