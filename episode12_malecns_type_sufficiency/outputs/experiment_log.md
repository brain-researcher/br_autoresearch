# Experiment log

## 2026-09-26 — Executor cleanup, final access still closed

- Retired generic provenance/controller layers and obsolete launch wrappers
  from the live executor while preserving whole-type role separation, the
  18+9+9 development design, post-36 falsifiers, the procedure lock, and the
  99-replicate full-search null.
- Removed the runnable final-data surface. The remaining final transaction
  module is a library-only one-shot primitive with no MaleCNS reader, command,
  or Slurm launcher; a scientifically complete final adapter does not exist.
- Reconciled the current status summaries with the frozen run: the provisional
  18+9+9 development prefix contains 36 valid trials, but qualification,
  required post-36 falsifiers, procedure locking, null execution, and final
  access remain incomplete.
- All 126 lightweight executor tests pass in the current Sherlock scientific
  module environment. No final-role connectivity was opened or inspected.

## 2026-09-25 — Real adaptive coverage launched

- The user authorized approximately 2,750 core-hours for the minimum full
  observed-plus-null search and directed that the original 1,000-core-hour
  ceiling no longer stop this scope. The run-specific executable authorization
  is `outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml`.
- A frozen 18-row real development covering array now materially varies the
  partner vocabulary, representation, covariance estimator, K, pooling,
  ranks, pseudocount, nuisance complexity, and regularization within the
  declared policy grammar. Every row fits and scores a complete T/U/M triplet.
- Slurm jobs `45314943` (tests) and `45314976` (annotation-only hierarchy) were
  submitted. Real connectivity array `45314980` tasks `0-17` has dependency
  `afterok:45314943:45314976`, so a failed prerequisite prevents outcome work.
- Trial output is written under EP12 scratch and staged to
  `outputs/development_run002/observed/` only after each task succeeds. The 204
  final types remain unauthorized and unopened.
- The initial prerequisite jobs `45314943` and `45314976` failed before Python
  because the 16-GiB request caused a 3-CPU allocation inconsistent with the
  one-CPU task TRES. Coverage dependency `45314980` was cancelled without
  running. The scripts were corrected to one CPU and 6 GiB; replacement jobs
  `45315718`, `45315724`, and coverage array `45315734` were submitted with the
  same fail-closed dependency structure.

## 2026-09-25 — Real development execution started

- After the user twice explicitly directed that the actual execution be
  launched rather than additional placeholders, EP12 advanced from generated
  qualification to development-only MaleCNS access.
- Slurm job `45277224` entered `RUNNING` on `sh04-18n31`. It invokes the
  trusted role materializer over the real body-statistics and full minconf-0.5
  weight files.
- The source Feather files are not physically partitioned by EP12 role. The
  materializer scans their record batches but discards every non-development
  focal row before constructing any matrix, statistic, returned value, or log.
- The fixed annotation-only split remains 816 development and 204 final whole
  provider types. Final focal connectivity remains unauthorized and is not
  materialized or summarized.
- The first real complete T/U/M development trial is prepared to consume the
  role-filtered snapshot after materialization finishes. This is development
  execution, not final evaluation and not a MaleCNS conclusion.

### Materialization and first configuration completed

- The first 16-GiB attempt (`45277224`) failed closed from memory use. Integer
  partner indices replaced per-edge Python strings. A later attempt exposed
  that a dev allocation had launched three default tasks; its partial scratch
  NPZ was preserved as `development_snapshot_failed_45277879` and never used.
  The Slurm launchers now explicitly require `--ntasks=1`.
- Single-task job `45279289` completed in 6:11. It materialized 84,735 focal
  neurons, 40,891,895 development-source weight rows, 87,415,060 observed
  outgoing weight, and an 84,735 × 11,771 sparse raw-partner matrix. The
  durable exposure record states that final focal connectivity was not
  materialized or summarized.
- Job `45280169` executed development trial 001 over all 816 types using one
  declared grammar configuration. It accounted for all 816 types exactly once:
  807 complete T/U/M triplets and nine zero-positive-mass cases. The original
  runner mislabeled those nine cases as technical failures; reconciliation
  checked each against the frozen development snapshot, reclassified them to the
  prespecified `zero_observed_target_count` reason, and changed no score.
- For this one unselected development configuration, mean M-minus-reference
  point gain was 0.001554 nats/count left-to-right, 0.001949 right-to-left, and
  0.001752 across directions. Seven fitted types had both directional point
  estimates above 0.02. These values are not uncertainty-qualified, do not
  complete the adaptive search, and cannot open or predict the final bank.

## 2026-09-25 — EP12 launch, synthetic control plane only

This section records archived engineering history. Its implementation-specific
evidence is not a current scientific readiness gate.

- Read the repository `AGENTS.md` and EP12 `GOAL.md`, `DATASETS.md`,
  `SEARCH_POLICY.yaml`, and `outputs/paper_plan.md` before implementation.
- Implemented the standard-library episode executor under `outputs/executor/`.
  It has no real-data entry point and restricts run paths to EP12's output and
  scratch roots.
- Passed 21 synthetic tests in 87.662 seconds. Tests covered the frozen grammar,
  whole-type role boundary, covering-array contract, stop and budget behavior,
  replay, interruption recovery, exactly-once final opening, fail-closed
  malformed inputs, and all declared terminal classes.
- Passed all 16 checks in the durable six-scenario qualification at
  `outputs/executor_qualification/qualification_report.json`. The archived
  qualification also records the policy and executor identities used then;
  those identities do not gate the current workflow.
- No MaleCNS source or real connectivity outcome was opened. The 52-trial
  trajectories and 99 generated null-controller sets are fixtures, not
  scientific fits or executed full-search null replicates.

At this launch-only stage, no scientific experiment or outcome computation had
started.
