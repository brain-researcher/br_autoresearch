# Readiness check

## Document checks completed

- The primary 300–600 ms EEG-increment question appears before model search or
  historical compatibility.
- The text explicitly defines “causal” as past-only information flow, not causal
  inference.
- One EP12-style SVG shows synthetic past-only timing, the three visibly matched
  EEG specificity increments, all ten pairwise edge transitions, the frozen
  eight-person repetition test, and distinct bounded outcomes without implying
  an observed forecast result.
- `GOAL.md`, `DATASETS.md`, and `SEARCH_POLICY.yaml` agree on the 50-ms stride,
  900-ms horizon, WAY series roles, 15/8 self-paced split, and 32-event
  calibration.
- Search budgets, required controls, unset margins, and claim limits are stated
  explicitly.
- `paper_plan.md` gives each figure one scientific judgment and defines next
  steps for positive, negative, mixed, and uncertain results.

## Scientific readiness still pending

| Requirement | Status |
| --- | --- |
| Direct source timing, channel, and gap checks | Not run |
| Exact onset and stillness procedures | Not set |
| Development and held-out data views | Not prepared |
| Reference models and source-specific non-neural baselines | Not executable |
| Numeric margins and complete 32-event calibration | Not set |
| Synthetic timing and future-information tests | Not run |
| Real development analysis | Not started |
| Final held-out evaluation | Not started |

Passing document checks does not establish that EEG forecasts movement or that
one model family is better.
