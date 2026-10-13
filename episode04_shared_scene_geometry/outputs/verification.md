# Verification

- Current Goal, dataset contract, and fixed-panel policy: present.
- Outcome computation: not run.
- One-shot audit: not opened.
- Task startup: allowed by an explicit scientist instruction naming EP04.
- Scientific qualification: candidate scoring and audit opening remain subject
  to `DATASETS.md`, `GOAL.md`, and the frozen fixed-panel policy.
- Primary development endpoint: image-disjoint RSA of native L2-normalized VLM
  cosine geometry versus fMRI response geometry, contrasted against one fixed
  native DINOv2 geometry and equally aggregated over the three registered
  high-level sectors and five participants. Development scores are explicitly
  selection-exposed.
- Final endpoint: one locked OOD audit using equal-weight within-category RSA;
  the mean VLM-over-DINOv2 gain must reach 0.02 Fisher-z units, at least four
  participant effects must be positive, all registered leave-one-out checks
  must remain positive, and positive unsubtracted VLM-to-brain RSA must pass
  the locked permutation null. The within-category low-level partial-RSA gain
  and its participant leave-one-out checks must also remain positive.
- Reproduction: not part of the current design; previously opened
  regular-shared responses are excluded.
- Controls and diagnostics: low-level geometry is a required nuisance control.
  Identical-fold encoding and raw EVC RSA with frozen reliability context are
  required diagnostics but cannot select or rescue the primary candidate;
  caption/object coverage is optional, and crossnobis is conditional on valid
  independent repeats.
- Inference: exact participant effects and leave-one-participant-out results
  are required; image/RDM-edge pseudoreplication and population significance
  claims from five participants are forbidden.
- Candidate-lock gate: the full three-VLM-plus-DINOv2 panel, every required
  falsifier, and both required diagnostics complete before the single candidate
  decision; one exact full-development rerun of the selected row then completes
  before configuration lock.

## 2026-09-30 — conceptual-figure check

The final generated image was visually inspected. Identical image IDs feed
all representation branches; model and neural RDMs use the same image order;
VLM and DINOv2 are compared to the same neural geometry without a learned
primary model-to-brain mapping. Development is image-disjoint, not participant-
held-out, and the OOD scope remains conditional stimulus-shift evidence in the
same five participants. Unmatched checkpoints, excluded exposed images and
the absence of observed data are labeled. This is a figure-only addition,
not a new scientific qualification or empirical result.

Focused checks passed: `git diff --check` reports no whitespace errors; the
new GOAL/README figure and prompt links resolve to readable episode-local
files; the saved image is a 1536 × 1024 RGB PNG. DATASETS, SEARCH_POLICY and
ASTRA are unchanged.

## 2026-10-02 local scope review

Checked narrative against native-RSA panel, equal-weight within-category OOD estimand, fixed comparator and 0.02 Fisher-z rule. No scientific validation or formal ASTRA validation run; scoped document checks only.

## 2026-10-02 figure review

Viewed the former 1536 × 1024 PNG and regenerated a sparse 1672 × 941 imagegen figure emphasizing direct native RSA and within-category OOD comparison. Accepted after correcting independent VLM/DINOv2-to-fMRI connectors, neural-RDM symmetry and abstract category exemplars; personally inspected and independently reviewed. Exact prompts saved, earlier assets retained, GOAL/output-index links updated. No source data, outcome, policy, runtime or Git change.
