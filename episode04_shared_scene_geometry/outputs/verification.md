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
