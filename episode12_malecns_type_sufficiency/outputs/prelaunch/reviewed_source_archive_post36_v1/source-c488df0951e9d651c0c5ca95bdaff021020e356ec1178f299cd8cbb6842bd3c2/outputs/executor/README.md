# EP12 executor

This directory contains the episode-specific control plane, bounded T/U/M
implementation, and development-only MaleCNS role firewall for EP12. Final
focal connectivity remains disabled.

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
episode. Synthetic qualification does not open its values. Development
commands use the frozen whole-type roles and cannot materialize final focal
connectivity.

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

The scientific modules additionally implement the common conditional-count
likelihood, T, the two frozen linear/curved U references, M2/M3, source-only
selection, reciprocal target scoring, simultaneous whole-type intervals,
generated biological counterexamples, and measured cost/stability checks.
`SCIENTIFIC_CONTRACT.yaml` and `SYNTHETIC_QUALIFICATION_SPEC.yaml` constrain
those computations; unsupported changes fail closed.

`development_data.py` materializes a development-source sparse matrix behind
the role firewall. `development_trial.py` executes a real complete T/U/M
configuration, retains whole-type accounting, and records prespecified
zero-count unscorable reasons.

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

Run annotation-only/source-schema preflight and the actual generated-data
fitter with Sherlock's pinned scientific modules:

```bash
module purge
module load python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
export PYTHONPATH="outputs/executor:${PYTHONPATH}"
python3 -m ep12_executor preflight-source
python3 -m ep12_executor qualify-methods --tier smoke
```

`slurm/qualify_methods.sbatch` packages the full declared qualification tier.

After explicit development authorization, materialize and run the first real
configuration with the packaged Slurm launchers:

```bash
sbatch outputs/executor/slurm/materialize_development.sbatch
sbatch outputs/executor/slurm/run_development_trial.sbatch
```

The second command requires the successful development snapshot. Interactive
`srun` calls must also specify `--ntasks=1` because Sherlock may allocate
multiple CPUs to satisfy a memory request.

Inspect or replay a synthetic run journal:

```bash
PYTHONPATH=outputs/executor python3 -m ep12_executor status /absolute/path/to/run
PYTHONPATH=outputs/executor python3 -m ep12_executor replay /absolute/path/to/run
```

There is no final-data command. `materialize-development`,
`materialize-partner-hierarchy`, and `run-development-trial` are restricted to
the development role and the EP12 scratch/output roots.

## Qualification scope

The end-to-end synthetic program retains the production control-plane limits:
18 covering-array trials, at least 36 valid trials, a 16-trial qualified
patience tail, at least 40% post-coverage falsifiers, 99 asserted synthetic
full-search-null receipt fixtures, the fixed `0.02` margin, and one final
opening. The reported model-family fit count is protocol arithmetic—not
executed null reruns, measured runtime, or scientific-model fitting.

The formerly placeholder data-shape tests now execute the actual fitter and
have been broadened to cover template, two bounded continuous forms, sparse
intermediates, grouped-with-continuum, weak groups, endpoint loss,
anatomy/quality confounding, and influential cells. A smoke or pilot output is
not promoted to a passing qualification merely because it completed.

The prior current-code test run passed 29/29 tests. Run 002 adds branch and real
covering-array tests under Slurm job `45314943`; its result is pending. The durable six-scenario
qualification passed 16/16 control-plane checks against the current code and
is recorded in `../executor_qualification_v3/qualification_report.json` with its completion
manifest beside it. A repeat invocation re-verified every referenced ledger,
vault, and receipt artifact before returning the saved report.

## Real-data gate

`REAL_DATA_GATES.md` records both the choices now executable and the remaining
closed gates. Development execution has started and real coverage array
`45314980` is dependency-submitted, but final access stays fail-closed until
the full adaptive grammar, mode-support controls, adequacy
sensitivity, scientific-model qualification, full-search null, and
grammar-wide cost profile pass. The proposed incoming/circuit follow-up is
outside this executor and is not started.
