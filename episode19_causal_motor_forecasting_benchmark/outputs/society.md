# External review status

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md): Local source-backed scope review, not an external review decision or canonical scientific acceptance.

No external scientific review has been requested or completed for EP19.

Three internal, read-only design reviews on 2026-09-30 examined onset/risk,
factorized EEG blocks, and baseline/calibration. Their implementation
clarifications are incorporated in `component_details.md`; they are not
external review, canonical Society evidence or scientific acceptance.

When review is eventually requested, reviewers should focus on:

- whether onset and all preprocessing are genuinely past-only;
- whether the non-neural and surrogate comparisons are strong and capacity
  matched;
- whether the same model fits support the near-to-strict ordering comparison;
- whether 32-event calibration leaves the encoder unchanged;
- whether the eight-person second-task rules support the proposed replication
  statement; and
- whether “causal” is consistently limited to information flow rather than
  biological inference.

This file contains no verdict and grants no permission to run or interpret the
study.
