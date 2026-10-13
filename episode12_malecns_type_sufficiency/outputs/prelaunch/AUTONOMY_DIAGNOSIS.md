# EP12 autonomous-launch diagnosis

Status date: 2026-09-26.

## Current correction

The reviewed-v6 source-archive DAG described below is retained as failed
historical engineering evidence, not as the active execution path. Its probe
job `45410522` failed on a staged-path root bug and its descendants were
cancelled. Commit `58d989c` intentionally replaced that machinery with the
live scientific path. Jobs `45446080 -> 45446082 -> 45447999` now implement
qualification, readiness, and the development-only post-36 tail. The one
missing normal-stop handoff from tail to procedure lock was added and tested;
the null dispatcher remains bounded by the existing development resource
authorization. Final connectivity remains unauthorized and unopened.
The preceding live chain failed before Python because its sanitized launch
environment lacked the `module` function; its dependents were cancelled.
Every retained wrapper now initializes Sherlock Lmod explicitly, and the
replacement qualification requests one task, one CPU, and 6 GiB.
The first bootstrap-corrected attempt also exposed a no-op module entrypoint;
its children were cancelled before running, the wrapper now calls the
executable package entrypoint, and a subprocess regression test prevents
another empty success.
The next real-fitter attempt was cancelled after 5:33 because its loaded
resource check still misread the 2,750-core-hour planning estimate as a hard
cap. The partial synthetic artifact was preserved. The current code enforces
the explicit absence of a new hard CPU ceiling, authenticates full qualification completeness,
and makes the downstream tail/null dispatch restart-safe.

- Episode: `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency`
- Temporary computation root: `/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency`
- Durable artifacts: `outputs/`
- Read-only source inputs: `inputs/`
- Final connectivity: locked.

## What was wrong

Earlier work treated three different states as though they were equivalent:

1. executor code and synthetic tests existed;
2. an individual Slurm job had been submitted; and
3. every successful stage durably authenticated and submitted its successor.

Only the third state is an autonomous launch.  EP12 initially had gaps between
those states: downstream launchers accepted human-supplied job identities,
some handoffs were not exact-once transactions, source/configuration bytes were
not fully frozen, and several wrappers inherited or destroyed important shell
state.  The first three legacy contract probes exposed concrete environment
defects before any generated fit: missing `MODULEPATH`, an `srun` export
boundary, a package initializer outside the minimal archive, and loss of the
module-generated scientific `PYTHONPATH`.

The earlier launch failures had a narrower scheduler cause.  The controller
passed completed finalizer `45323481` back to Slurm as a new live dependency
after Slurm no longer accepted that terminal job ID for dependency creation.
Sherlock also rendered a user hold as `(JobHeldUser)` in contexts where the
executor accepted only `JobHeldUser`.  The reviewed v5 launcher now persists
the exact successful finalizer accounting row as a satisfied logical
predecessor, omits only that stale scheduler edge, and canonicalizes exactly
the two valid user-hold spellings.  All active predecessor edges remain
`afterok` dependencies.

The resulting v5 chain did start, but replacement job `45396786` failed closed
on a producer/consumer contract mismatch: real adaptive successor
configurations use `stage: adaptive`, while the attester incorrectly required
the logical cycle labels `adaptive_cycle_1` or `adaptive_cycle_2`. Synthetic
fixtures had mirrored the consumer assumption instead of exercising the real
producer output. The corrected test now feeds both cycles from the real
successor planner into the attester. Failed-v5 artifacts were authenticated and
rotated without overwriting them before the v6 transaction was created.

The planning estimate was also materially wrong.  Approximately 2,750
core-hours described an earlier observed-search planning scale, not the full
99-search null program.  The minimum full program must be measured from the
frozen procedure before an exact additional resource authorization can be
issued.  A low estimate cannot silently authorize a much larger launch.

## Historical reviewed-v6 evidence

Full prelaunch qualification job `45405244` passed 358 of 358 tests, and its
dependent relaunch job `45408036` completed successfully. The actual
reviewed-v6 dependency graph is visible in Slurm:

- replacement observed attestation: `45410406` (`COMPLETED`, `0:0`);
- post-36 contract probe: `45410522`;
- comparison: `45410527`;
- scientific-transition publisher: `45410528`; and
- downstream coordinator: `45410530`.

The durable controller receipt is
`outputs/prelaunch/post36_orchestration/receipt.json`; the replacement launch
receipt is `outputs/prelaunch/replacement_attestation_launch/receipt.json`.
Obsolete job `45332084` is terminally cancelled.  The source archive tree is
`6caca9e53657c6490113dac5eb85685e46fbf2ee7334f44f88c156f372342361`.
Final connectivity remains locked and has not been read by this relaunch.

## Current autonomous boundary

After qualification and readiness succeed, the current live tail submits and
recovers the remaining development-only chain exactly once:

`synthetic qualification -> production readiness -> post-36 tail -> procedure lock -> runtime bundle -> 99 full-null searches -> null aggregate`

Safe technical or scientific gate failures remain allowed to stop the chain.
The approximately 2,750 core-hours is a measured planning value, not a hard
cap. The explicit minimum-full-search authorization has
`new_hard_cpu_ceiling: null`; the frozen maximum trial count, per-trial CPU,
memory and wall limits, the overall 120-hour wall limit, and zero-GPU limit
remain enforced. No further CPU-ceiling decision is expected within that
authorized scope.

No full-null array and no final-connectivity evaluation is claimed launched by
this document.

## Recorded operator limitation

An operational resource search at `2026-09-26T06:12:01-0700` was scoped too
broadly and displayed real development result rows. The values were not
analyzed or used for any decision, but the exposure is real and is recorded in
`outputs/prelaunch/exposure_incidents/development_outcome_terminal_exposure_20260926T061201-0700.json`.
Future transition evidence must preserve this limitation; it may claim an
outcome-blind programmatic transition extractor, not a globally outcome-blind
operator. Final connectivity remains locked.

## Scientific-gate termination is autonomous behavior, not an autonomy gap

The v6 chain did continue without human intervention through all 2,520
generated qualification replicates. It stopped because the scientific report
was false, whereupon the CLI returned 1 and Slurm correctly cancelled both
`afterok` descendants. Automatically retrying the same method, editing the
report, or releasing readiness would have defeated the gate.

The postmortem identified method defects rather than an FD/socket leak: the
old “max-t” code did not studentize bootstrap resamples, qualification omitted
aggregate/joint coverage, false adequacy was not margin-relative, and 256 draws
failed the existing stability check. The repair path is therefore a new
generated-only qualification version. Pilot `45494117` is an actual scheduled
T/U/M execution for the numerical component; it does not broaden access or
weaken any scientific threshold.
