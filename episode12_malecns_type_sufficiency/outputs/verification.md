# Verification

## 2026-09-26 cleanup validation

- Full executor suite: 126 tests passed in 12.614 seconds with the Sherlock
  Python 3.12 scientific modules.
- Static validation: 64 Python files parsed, all 14 retained Slurm wrappers
  passed `bash -n`, and the EP12 diff passed `git diff --check`.
- Final-data surface: no MaleCNS final reader, command-line entry point, or
  Slurm launcher is present. Final-role connectivity accessed: no.
- ASTRA export remains open because the repository command reports that the
  Brain Researcher `astra` extra is unavailable; this is not a computation or
  final-access gate.

## Archived controller evidence

- The prior executor test run passed 29 tests in 143.484 seconds. Its launch
  chronology is retained in the experiment log; current validation is recorded
  above.
- The six-scenario synthetic controller qualification passed 16/16 checks
  across positive, adequate, unresolved, no-valid-comparison,
  budget-exhaustion, and input-failure fixtures.
- That evidence remains archived under `outputs/executor_qualification_v3/`.
  It verifies historical controller behavior only and is not a current
  scientific readiness gate.

## Current scientific and access status

- Development materialization: Slurm job `45279289` completed in 6:11; 816
  development types, 84,735 focal neurons, and 40,891,895 source-owned weight
  rows were materialized under the development firewall.
- First real T/U/M configuration: Slurm job `45280169` accounted for all 816
  development types. Reconciliation retained 807 valid fitted types and nine
  `zero_observed_target_count` unscorable types under the frozen development
  snapshot, with no score refit or change and no remaining technical failure.
- Provisional development prefix: the current readiness evaluator verifies 36
  valid development-only trials in the fixed 18+9+9 sequence, with the archive
  marked provisional pending falsifiers and null.
- Current readiness: `revise`. Whole-type roles, the 36-trial chain, post-36
  plan, and development resource authorization pass; full scientific
  qualification does not pass. Required post-36 falsifiers are incomplete, no
  procedure is locked, and the 99-search null has not run.
- MaleCNS source accessed: yes, development-only.
- Final-role focal connectivity accessed or summarized: no.
- Scientific-model qualification: incomplete.
- Executed full-search null completed: no.
- Final scientific evaluation: not opened. Synthetic one-shot final-access
  fixtures were exercised only to qualify controller mechanics.
- Concept figure: unchanged synthetic design mockup with no observed data.
