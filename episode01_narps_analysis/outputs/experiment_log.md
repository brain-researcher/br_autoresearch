# Experiment log

> Client-maintained exploratory projection. This file grants no execution,
> review, scientific, memory, or Landscape authority. It is not a required
> Society review artifact and is not declared in the terminal bundle.

## Binding

- Terminal-owner Goal handoff: `goal_handoff_e3f79fc20a54776ebf304655`
- Non-owning diagnostic handoff: `goal_handoff_ed4e88575e29cb1a96acf016`
- Outer loop: `arl_8cbd25a7a8610a008c2830ba624052c4`
- Workspace: `/oak/stanford/groups/russpold/users/zijiao/NARPS`
- Handoff chronology: the fresh BR handoff was prepared after the existing
  client-native exploratory analysis completed; this replay is not a new
  analysis run.

## Entries

### Entry 1 — 2026-08-14T14:22:00Z

- Stage: local exploratory analysis setup, not an MCP stage assertion.
- Question or attempt: freeze the NARPS analysis-choice attribution contract
  before neural-outcome access.
- Action: recorded the 3 × 3 × 3 grid, estimand, decision rule, controls,
  stopping rule, and exploratory-only boundary.
- Outcome: the contract permits `closed_no_candidate` when the complete frozen
  rule is not met.
- Evidence refs: `outputs/frozen_analysis_contract.md`,
  `outputs/frozen_analysis_contract.json`.
- Failure or deviation: none recorded at contract freeze.

### Entry 2 — 2026-08-14T16:56:05Z through local analysis completion

- Stage: client-native exploratory analysis and repair, not canonical MCP
  execution.
- Question or attempt: complete the frozen grid and evaluate the frozen C/S
  and Q/S rule separately for `gain_demean` and `loss_demean`.
- Action: repaired one extra near-zero temporal-mean guard, retried the six
  affected subjects, and completed 54 contrast-by-grid cells from 108 subject
  checkpoints.
- Outcome: neither contrast met the complete frozen rule; the local result
  remained exploratory and produced no candidate for confirmation.
- Evidence refs: `outputs/engineering_repairs.json`,
  `outputs/aggregation_provenance.json`, `outputs/multiverse_manifest.tsv`,
  `outputs/RESULT.md`, `outputs/variance_attribution.tsv`.
- Failure or deviation: failed array tasks, a memory-limited diagnostic, and
  canceled pending aggregation jobs are recorded in
  `outputs/engineering_repairs.json`; the frozen scientific deviations list is
  empty.

### Entry 3 — 2026-08-16T12:58:52Z onward

- Stage: fresh terminal-handoff validation; no new scientific computation.
- Question or attempt: inspect the existing terminal evidence and prepare its
  exact `closed_no_candidate` replay.
- Action: verified all 34 declared output artifacts are present, non-symlink
  regular files inside the workspace; reran the two existing synthetic math
  audit entrypoints; prepared the fresh BR Goal handoff.
- Outcome: both synthetic audit entrypoints passed. Before submission, the
  authoritative outer loop remained `DISCOVERING` at revision 1 with no frozen
  review, execution, synthesis, or completion.
- Evidence refs: `outputs/candidate_bundle.json`, `outputs/RESULT.md`,
  `outputs/tests/test_math_review.py`,
  `outputs/tests/test_aggregate_results_math.py`.
- Failure or deviation: the declared environment lacks `pytest`; the two
  self-contained test entrypoints were therefore executed directly and both
  passed. No analysis artifact was rewritten.

### Entry 4 — 2026-08-16T13:08:23Z

- Stage: terminal-handoff recovery diagnosis; no scientific computation.
- Question or attempt: submit the existing terminal bundle through the fresh
  handoff.
- Action: called `autoresearch_goal_submit` once with the exact parsed
  `outputs/candidate_bundle.json`, then inspected both the fresh handoff and the
  previously persisted handoff after the server rejected ownership transfer.
- Outcome: the fresh handoff was rejected with
  `autoresearch_goal_submission_conflict`. The original handoff
  `goal_handoff_e3f79fc20a54776ebf304655` already owns the identical persisted
  `closed_no_candidate` submission, but its outer loop remains
  `DISCOVERING@1`; its observed action is `terminal_no_review`, not an allowed
  reconciliation action. No second submission, review, execution, or
  transition was created.
- Evidence refs: `outputs/candidate_bundle.json`, `outputs/governance.md`.
- Failure or deviation: this is a legacy bridge-reconciliation gap, not a
  scientific-analysis failure. No production record was edited manually.

### Entry 5 — 2026-08-16T14:16:59Z

- Stage: terminal-handoff reconciliation; no scientific computation.
- Question or attempt: close the already-persisted negative terminal through
  the original owning handoff without changing the bundle.
- Action: after production exposed `retry_terminal_reconciliation`, replayed
  the exact parsed `outputs/candidate_bundle.json` once through
  `goal_handoff_e3f79fc20a54776ebf304655`, then read both handoffs.
- Outcome: the replay was accepted with `idempotent_replay: true`. The outer
  loop is authoritative `COMPLETE@3` with `native_goal_terminal` and
  `closed_no_candidate`; the fresh sibling reports
  `closed_by_other_handoff`. Review, reward, approval, authorization,
  execution, synthesis, canonical episode, Society outcome, and Landscape
  transition remain absent.
- Evidence refs: `outputs/candidate_bundle.json`, `outputs/governance.md`.
- Failure or deviation: none in the reconciliation. This closes control-plane
  bookkeeping only; it does not change the exploratory scientific result.

## Remaining uncertainty

- This workspace result is not independent confirmation and establishes no
  scientific acceptance.
- The existing edge-scaling diagnostic is exploratory and does not resolve how
  much boundary behavior contributes to smoothing sensitivity.
- Outer-loop bookkeeping is now `COMPLETE`, but the result remains
  `exploratory_only`, `confirmation_eligible: false`, and not scientifically
  accepted; independent fresh confirmation is still required.
