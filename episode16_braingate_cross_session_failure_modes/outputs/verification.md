# Readiness check

2026-10-02 — Viewed the actual [ep16_question_imagegen-v2.png](ep16_question_imagegen-v2.png) after generation and targeted corrections; a second reader checked its scientific operators and captions. Replaced intention/representational-cause wording with the direction proxy, target-local versus transport comparison, label-blind emulator and 32-label repair; no score bars. This verifies the illustration, not empirical results or pipeline readiness.

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md): Checked claim wording against the documented comparisons; no computation, empirical result or implementation validation is claimed.

## Document checks completed

- The question begins with the scientific comparison rather than process
  machinery.
- One EP12-style SVG shows synthetic near/long session codes, matched scores,
  three visibly distinct mechanism signatures, and bounded interpretations
  without implying observed success.
- `GOAL.md`, `DATASETS.md`, and `SEARCH_POLICY.yaml` agree on the 1–30 day and
  at-least-180 day gaps, six/three participant roles, 128-trial packets, and
  32-label primary recalibration.
- The proposed margins, search budgets, required controls, and claim limits are
  stated explicitly.
- `paper_plan.md` gives each planned figure one scientific judgment and defines
  what happens after positive, negative, mixed, and unresolved results.

## Scientific readiness still pending

| Requirement | Status |
| --- | --- |
| Comparable target schedules and sufficient near/long pairs | Not checked |
| Final six development and three held-out participants | Not chosen |
| Remaining numeric thresholds | Not set |
| Synthetic recovery and leakage tests | Not run |
| Real development analysis | Not started |
| Final held-out evaluation | Not started |

Passing document checks does not establish that the data support the study or
that any scientific conclusion is true.

## Component revision — 2026-09-30

- Focused parallel read-only review found no change to operative estimands,
  packet sizes, participant roles, numerical anchors or shared search budgets.
- Review refinements: zero-mean feature-independent noise; optional quality
  measurements respect time exposure; emulator parameters are reused unchanged
  on source evaluation trials.
- `component_details.md` leaves actual unresolved scientific choices explicit.
  SEARCH_POLICY was not edited; no numerical threshold was filled.
- The newly authored ASTRA program maps existing comparisons and separates
  recording-transform fitting from outcome-based reproduction evaluation.
  Pinned schema/semantic validation is still open: the current interpreter
  lacks YAML, ASTRA and Brain Researcher packages; no validator command was
  found. This does not block the completed documentation refinement.
- No payload access, neural score, synthetic experiment, held-out opening,
  software installation or Git mutation was performed.
- Focused Git whitespace checks reported no errors in tracked changes or the
  new component/worklist files. `git diff --exit-code` confirmed that
  SEARCH_POLICY remains unchanged. These are documentation checks, not ASTRA
  schema validation or scientific readiness.
