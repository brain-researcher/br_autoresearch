# Questions for future episodes

This is a backlog, not an automatic launch queue. One episode should take one
question or a tightly connected pair.

## Analysis decisions in NARPS

### What does smoothing actually change?

- **Why:** the strongest first-pass sensitivity may be concentrated in the
  transition from unsmoothed to smoothed maps, not in the difference between
  two smoothing kernels.
- **Data:** NARPS `ds001734` FitLins outputs and the frozen episode01/02 records.
- **Next experiment:** compare kernel-matched operators, edge/core masks, and
  amplitude-versus-shape effects under a fixed exclusion and confound rule.

### Are exclusion effects cohort-wide or subject-driven?

- **Why:** a median over excluded subjects can hide one influential participant
  or one group-specific numerical failure.
- **Data:** the same NARPS source with exact subject-level provenance.
- **Next experiment:** predeclare influence summaries, bootstrap calibration,
  and a second analysis path before inspecting the target comparison.

### Which decision family dominates after scale is normalized?

- **Why:** raw map sum-of-squares can mix scientific sensitivity with units,
  scaling floors, and near-zero variance estimates.
- **Data:** NARPS maps plus explicit numerical diagnostics.
- **Next experiment:** perform variance attribution on normalized, interpretable
  endpoints with matched negative controls.

## N-back measurement

### Is there an eligible discovery/retest/transport split?

- **Why:** LR/RL runs in one visit are not an independent session, and two
  datasets called `n-back` may have incompatible events, ages, or outcomes.
- **Data:** HCP-YA, AOMIC `ds002785`/`ds002790`, and a still-unverified NDA root.
- **Next experiment:** outcome-blind inventory of task semantics, sessions,
  behavior, access terms, derivatives, and non-overlapping data roles.

### Can reliability improve without losing construct validity?

- **Why:** optimizing ICC alone can produce a stable assay that measures the
  wrong thing.
- **Data:** one verified discovery source with held-out subjects and a genuine
  development retest if available.
- **Next experiment:** bounded Pareto search over assay representations, with
  locked ROI and whole-brain baselines.

### What survives independent transport without retuning?

- **Why:** a useful result should identify both invariant components and the
  task/site/population boundaries where they fail.
- **Data:** a sealed, task-compatible external cohort selected before the assay
  is locked.
- **Next experiment:** one-shot replay of a single frozen signature with motion,
  category, site, behavior, and difficulty controls.

## Propose another question

Add a short entry with four things: the question, why it is worth answering,
the data that may answer it, and the smallest discriminating experiment. It is
fine to start with a broad phenomenon. The runner can help formulate a bounded
test after the proposal is reviewed.
