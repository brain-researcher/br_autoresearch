# Experiment log

## 2026-09-25 — EP12 launch, synthetic control plane only

- Read the repository `AGENTS.md` and EP12 `GOAL.md`, `DATASETS.md`,
  `SEARCH_POLICY.yaml`, and `outputs/paper_plan.md` before implementation.
- Implemented the standard-library episode executor under `outputs/executor/`.
  It has no real-data entry point and restricts run paths to EP12's output and
  scratch roots.
- Passed 21 synthetic tests in 87.662 seconds. Tests covered the frozen grammar,
  whole-type role boundary, covering-array contract, stop and budget behavior,
  hash-chained replay, interruption recovery, exactly-once final opening,
  fail-closed malformed inputs, all declared terminal classes, and artifact
  tamper detection.
- Passed all 16 checks in the durable six-scenario qualification at
  `outputs/executor_qualification/qualification_report.json`. Re-running the
  command returned the completed result only after re-verifying the durable
  journals, final vaults, and asserted null-receipt fixture hashes.
- Policy hash:
  `d64c5c9c56b70f3312472e056b9394f519107eda268902ceb586771d435d8b8d`.
  Executor code hash:
  `dad853859b17e6d426a07b85aa92723df330a79c492cd0c702c3ac7d083dbe38`.
- No MaleCNS source or real connectivity outcome was opened. The 52-trial
  trajectories and 99-receipt sets are generated controller fixtures, not
  scientific fits or executed full-search null replicates.

No scientific experiment or outcome computation has started.
