# Readiness check

2026-10-02 — Viewed the actual [ep19_question_imagegen-v2.png](ep19_question_imagegen-v2.png) after generation and targeted corrections; a second reader checked its scientific operators and captions. Replaced the flat-peripheral assumption and pseudo winner bars; made older complement retention and all four input combinations explicit, with a hand-onset illustration. This verifies the illustration, not empirical results or pipeline readiness.

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md): Checked the new claim limits against the fixed forecast target and second-task procedure. No timing test, empirical analysis or implementation validation was performed.

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
- The 2026-09-30 component revision specifies pending-state detector behavior,
  exact shared-scale O/C reconstruction before features/state, the distinction
  between Q00 and zero-EEG, and whole-prefix calibration with identity fallback.
  Three focused internal reviews addressed the changed designs only.
- Scoped Git whitespace checks passed for edited documents and new files.
  A focused worklist review found all declared dependency producers present,
  no cycle, and only a combined final-evaluation output. It is not pinned
  ASTRA schema/semantic validation or empirical evidence.

## Scientific readiness still pending

| Requirement | Status |
| --- | --- |
| Direct source timing, channel, and gap checks | Not run |
| Onset, stillness and risk procedures | Starting choices selected; not implemented, validated or frozen |
| Development and held-out data views | Not prepared |
| Reference models and source-specific non-neural baselines | Not executable |
| Numeric scientific margins | Unset and unchanged |
| Complete 32-event calibration | Provisional recipe selected; not implemented, validated or frozen |
| Synthetic timing and future-information tests | Not run |
| Real development analysis | Not started |
| Final held-out evaluation | Not started |

ASTRA 0.0.14 schema/semantic validation remains open because pinned tooling
is unavailable on PATH. The design program is not execution or scientific
validation, and this tooling gap adds no execution gate.

Passing document checks does not establish that EEG forecasts movement or that
one model family is better.
