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

The Python package in `ep12_executor/` implements the bounded scientific
workflow: whole-type development/final separation, the complete T/U/M grammar,
the fixed 18+9+9 prefix, the registered post-36 falsifiers and stop rule, the
procedure lock, and the 99-replicate full-search null. The live journal is a
plain append-only event log with event-ID idempotency. It does not build a hash
chain or an environment/provenance attestation layer.

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

Run tests in an authorized compute session with the scientific dependencies
available:

```bash
cd /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
PYTHONPATH=outputs/executor python3 -m unittest discover -s outputs/executor/tests -v
```

Run generated-data method qualification after loading the scientific Python
environment:

```bash
cd /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
export PYTHONPATH="outputs/executor:${PYTHONPATH}"
python3 -m ep12_executor qualify-methods --tier smoke \
  --contract outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml \
  --qualification-spec outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml
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

There is no final-data command or Slurm wrapper. `final_evaluation.py` contains
only a library-level transaction primitive that tests one immutable opening,
same-operation retrieval-only reconciliation, and refusal of a second
operation. It has no MaleCNS outcome reader or command-line entry point. Before
it can be wired to final data, EP12 still needs the frozen simultaneous-interval
calculation, bounded result allowlist, detailed terminal-decision mapping, and
canonical prerequisite bindings. No final-access authorization is stored here.

`materialize-development`, `materialize-partner-hierarchy`, and
`run-development-trial` remain restricted to the development role and the
EP12 scratch/output roots.

The active continuation wrappers are `check_production_readiness.sbatch`,
`run_post36_tail.sbatch`, `run_procedure_lock.sbatch`, and the two
`run_full_search_null*.sbatch` wrappers. Readiness produces one pass/revise
decision. The obsolete prefix-only `pre_null_gate.sbatch` shortcut is removed;
procedure locking uses the complete selected comparison, mode-support
inference, and registered falsifiers. Planning, finalization, and diagnostic
commands write their scientific payloads directly without generic sidecar file
inventories or source/hash receipts. Source archives, generic outer checksum
manifests, contract-transition receipts, environment attestations, and nested
handoff controllers are retired from the live path; their historical output
records remain unchanged.

## Qualification scope

The scientific program retains its frozen limits: 18 covering-array trials,
at least 36 valid trials, a 16-trial patience tail, at least 40%
post-coverage falsifiers, 99 full-search-null replicates, the fixed `0.02`
margin, and one immutable final opening.

The formerly placeholder data-shape tests now execute the actual fitter and
have been broadened to cover template, two bounded continuous forms, sparse
intermediates, grouped-with-continuum, weak groups, endpoint loss,
anatomy/quality confounding, and influential cells. A smoke or pilot output is
not promoted to a passing qualification merely because it completed.

The current lightweight executor suite passes 126/126 tests. Run 002 completed
and provisionally archived the 18+9+9 prefix as 36 valid development-only
trials. The durable six-scenario qualification passed 16/16 control-plane
checks and is archived in `../executor_qualification_v3/`; it is historical
controller evidence, not a current scientific readiness gate. Current
production readiness remains `revise` because the full scientific
qualification has not passed.

## Real-data gate

`REAL_DATA_GATES.md` records both the completed provisional 36-trial
development prefix and the remaining closed gates. Final access stays
fail-closed until scientific qualification, required post-36 falsifiers,
mode-support and adequacy gates, procedure locking, and the 99-search null are
complete. The proposed incoming/circuit follow-up is outside this executor and
is not started.
