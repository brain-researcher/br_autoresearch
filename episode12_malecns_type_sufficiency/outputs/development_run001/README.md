# EP12 real development run 001

This directory contains the first actual MaleCNS development execution for
EP12. It is not a synthetic placeholder and it does not contain final-role
focal connectivity.

- `materialization/` records the real development role firewall and hashes of
  the development-only scratch snapshot.
- `trial001/` is the immutable raw first execution. Its nonzero process status
  reflects nine zero-count types initially mislabeled as technical failures.
- `trial001_reconciled/` verifies those same nine types against the hashed
  snapshot, records the prespecified `zero_observed_target_count` reason, and
  changes no fitted score.
- `real_cost_projection.json` preserves the measurement against the original
  1,000 CPU-core-hour policy ceiling. A later explicit user authorization for
  the approximately 2,750-core-hour minimum full search is recorded separately
  in `outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml`.

The reconciled first configuration has 807 complete T/U/M type fits, nine
unscorable types, zero remaining technical failures, and 204 final types still
locked. It is development evidence from one grammar configuration, not a
selected procedure, completed search, null calibration, or scientific
terminal.
