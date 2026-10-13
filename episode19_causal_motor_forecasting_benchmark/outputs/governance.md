# Study boundaries

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md): Narrative clarification only. Horizons, baselines, surrogate controls, factorial construction, margins, model/search space, calibration and access permissions are unchanged.

EP19 is currently a documented study design. The current files do not authorize
opening held-out signals or running the full model search.

## Decisions already made

- Predictions occur every 50 ms over a 900-ms horizon.
- The primary window is 300–600 ms; 0–300 ms is the near-event comparison.
- “Causal” means past-only information flow, not causal inference.
- WAY series 8–9 and eight whole self-paced participants are held out.
- The self-paced primary condition allows only four calibration parameters from
  the first 32 complete movements.
- Component choices are now specified in `component_details.md`: earliest
  accepted sensor crossing, an online pending state, exact O/C voltage
  construction, source-specific baseline inputs and provisional calibration
  defaults. They are not implemented, verified or frozen.

## Decisions still required before EEG scoring

- implementation/development validation and freezing of the selected onset,
  stillness and ambiguity rules, with a supported full timing bound;
- five executable reference recipes and two source-specific non-neural models;
- development validation and freezing of the selected 32-event calibration
  procedure, including its history guard;
- finite ranges for every allowed model change;
- all numeric margins in `../SEARCH_POLICY.yaml`; and
- successful synthetic timing and future-information tests.

No scientific acceptance, external review decision, or permission to examine a
held-out result is implied by this summary.
