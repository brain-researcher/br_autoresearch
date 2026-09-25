# EP12 executor

This directory contains the episode-specific control plane for EP12. It is
currently **synthetic-only**: there is deliberately no command that can open
MaleCNS connectivity, body-statistic, synapse, or neurotransmitter values.

## Where this episode lives

| Role | Absolute location |
| --- | --- |
| Repository | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch` |
| EP12 | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency` |
| Read-only episode inputs | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/inputs` |
| Durable episode outputs | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs` |
| Temporary EP12 workspace | `/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency` |
| Search policy | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/SEARCH_POLICY.yaml` |
| Shared protocol | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/ADAPTIVE_SEARCH_PROTOCOL.md` |
| Controller interface | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/ADAPTIVE_CONTROLLER_INTERFACE.md` |

The dataset contract declares a read-only MaleCNS v1.0 source outside this
episode. Synthetic qualification neither opens nor lists its contents.

## Implemented boundary

The standard-library Python package in `ep12_executor/` implements:

- dependency-free loading and fail-closed validation of EP12's YAML subset;
- declarative grammar checks requiring the complete T/U/M triplet;
- whole-type role and prohibited-field guards;
- an atomic, idempotent, SHA-256 hash-chained JSONL journal;
- trial registration, execution, scoring, archive, coverage, promotion, stop,
  full-search-null receipt, lock, and final-evaluation operations;
- replay with frozen policy, code, environment, source, and split identities;
- distinct positive, adequate, unresolved, no-valid-comparison,
  incomplete-search, and technical-failure terminals; and
- durable final-opening reservation and reconciliation after a lost
  acknowledgement, with one effective evaluator access.

The synthetic covering array, scores, null trajectories, and final outcomes
are executor fixtures. They are not proposed real-analysis choices or
scientific evidence.

## Commands

Run tests with the ambient Python 3.13 standard library:

```bash
cd /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
PYTHONPATH=outputs/executor python3 -m unittest discover -s outputs/executor/tests -v
```

Run the durable synthetic qualification:

```bash
cd /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
PYTHONPATH=outputs/executor python3 -m ep12_executor qualify-synthetic
```

Inspect or replay a synthetic run journal:

```bash
PYTHONPATH=outputs/executor python3 -m ep12_executor status /absolute/path/to/run
PYTHONPATH=outputs/executor python3 -m ep12_executor replay /absolute/path/to/run
```

There is no `run-real`, development-data, or final-data command. Adding one
before the real-data contract is complete would be a policy violation.

## Qualification scope

The end-to-end synthetic program retains the production control-plane limits:
18 covering-array trials, at least 36 valid trials, a 16-trial qualified
patience tail, at least 40% post-coverage falsifiers, 99 asserted synthetic
full-search-null receipt fixtures, the fixed `0.02` margin, and one final
opening. The reported model-family fit count is protocol arithmetic—not
executed null reruns, measured runtime, or scientific-model fitting.

The five required data-shape tests—average, continuous, grouped, weak-signal,
and side-artifact—are listed as pending placeholders. They are **not** counted
as passed and do not qualify the future statistical estimators, sensitivity,
calibration, or power at MaleCNS sample sizes.

The 2026-09-25 test run passed 21/21 tests. The durable six-scenario
qualification passed 16/16 control-plane checks and is recorded in
`../executor_qualification/qualification_report.json` with its completion
manifest beside it. A repeat invocation re-verified every referenced ledger,
vault, and receipt artifact before returning the saved report.

## Real-data gate

`REAL_DATA_GATES.md` records the consequential decisions that remain unset.
The executor will stay fail-closed until those choices are frozen and their
scientific-model synthetic qualification and cost profile pass. The proposed
incoming/circuit follow-up is outside this executor and is not started.
