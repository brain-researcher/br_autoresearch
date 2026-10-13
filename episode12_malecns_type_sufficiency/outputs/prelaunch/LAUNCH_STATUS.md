# EP12 execution launch status

Status date: 2026-09-26.

## Current live continuation (19:20 local)

The reviewed-v6 archive chain below is now historical: probe job `45410522`
failed before any generated fit because its staged source location was
incorrectly treated as the episode root, and its dependent jobs were cancelled.
Current commit `58d989c` intentionally retired that archive/nested-controller
path. The active live-executor continuation is:

- full scientific qualification `45446080`;
- production readiness `45446082`, `afterok:45446080`; and
- development-only post-36 tail `45447999`, `afterok:45446082`.

Pending predecessor `45446084` was cancelled before execution and replaced
exactly once so the scheduled tail contains the validated hardened wrapper.
Qualification `45446080` entered `RUNNING` at 19:24:24 PDT and continues to
produce generated-data replicates without log errors; its children remain
dependency-blocked.

The immediately preceding chain (`45440965 -> 45440968 -> 45440971`) failed
closed when qualification reached a node without an inherited `module` shell
function; it exited before Python and its dependents never ran. All retained
wrappers now bootstrap Sherlock Lmod explicitly under the sanitized launch
environment. The replacement qualification requests one task, one CPU, and
6 GiB, retaining ample headroom over the historical memory peak.

The first post-bootstrap qualification `45444151` then exposed a separate
empty-success path: invoking `ep12_executor.cli` as a module returned zero
without calling `main()` and produced no artifact. Its two children were
cancelled before running. Qualification now uses the executable package
entrypoint; both package and direct CLI module entrypoints are regression
tested.

Qualification `45444664` subsequently ran the real synthetic fitter for 5:33
but was deliberately cancelled when audit showed its loaded gate still treated
the approximately 2,750-core-hour planning value as a hard ceiling. Its partial
JSONL is preserved under `outputs/scientific_qualification/` and its children
did not run. The current chain enforces `new_hard_cpu_ceiling: null`, strict
qualification completeness, verified tail recovery/handoff, safe null requeue,
scratch-to-durable aggregation, and phased idempotent null dispatch.

The tail self-submits bounded successor slots. On a normal frozen scientific
stop it now hands off to procedure locking, whose existing dispatcher submits
the authorized 99-null array and aggregate when eligible. A wall-deadline stop
does not make that handoff. Final connectivity remains locked.

- Episode: `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency`
- Temporary root: `/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency`
- Durable output root: `outputs/`
- Source inputs: `inputs/` (read-only)

## Formal scientific qualification

Full frozen qualification job `45275957` was submitted but remained pending.
It was cancelled without running when the user prioritized actual development
execution and the pending request prevented an interactive development
allocation under the Sherlock per-user QOS.

The reduced pending pilot job `45275184` was cancelled after the full run was
submitted.

## Fixed-margin compatibility diagnostic

Generated-data diagnostic job `45275899` was likewise cancelled while still
pending to free QOS capacity for actual development execution. No result is
claimed for either cancelled job.

## Real development execution

The user subsequently explicitly directed that EP12 launch the actual
execution. After two fail-closed engineering attempts (`45277224`, OOM; and
`45277879`, unintended multi-task contention) and one immediate empty-directory
check (`45278884`), single-task job `45279289` successfully materialized the
role-filtered real MaleCNS development view. It retained 84,735 focal neurons
from 816 development types and left all 204 final types locked.

Real development T/U/M job `45280169` then accounted for all 816 development
types under one complete grammar configuration. The reconciled durable result
at `outputs/development_run001/trial001_reconciled/` contains 807 fitted types,
nine snapshot-verified zero-count unscorable types, and no remaining technical
failures. Reconciliation did not refit or change any score.

## Adaptive search launch

The user subsequently authorized the measured approximately 2,750 core-hour
planning scale for the minimum full search and removed the original 1,000
core-hour ceiling as a stop condition for that scope. The executable override
is recorded in `outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml`.

Run 002 froze an 18-row real covering array with complete pairwise coverage of
partner vocabulary, representation, covariance, and component count. Slurm
jobs `45314943` (tests) and `45314976` (annotation-only hierarchy preparation)
were submitted, followed by real development coverage array `45314980` with
an `afterok` dependency on both. These are actual development connectivity
trials, not synthetic or placeholder tasks. The remaining trials needed to
reach the 36-trial minimum will be outcome-adaptive successors after coverage.

## Access boundary

The two qualification/diagnostic jobs do not read real connectivity outcomes.
The completed real development jobs do. The prelaunch exposure record at
`outputs/prelaunch/source_preflight/exposure_record.json` records
the state before this first access. `outputs/executor/SCIENTIFIC_CONTRACT.yaml`
now authorizes `development_only`; final connectivity remains explicitly
unauthorized. The development materializer writes the post-access exposure
record under `outputs/development_run001/materialization/`.

## Run 002 relaunch correction

The initial prerequisite jobs `45314943` and `45314976` failed before Python
because their memory request made Slurm allocate three CPUs while task TRES
remained one CPU. Dependent coverage array `45314980` was cancelled without
executing. The corrected one-CPU, 6-GiB replacement jobs are tests `45315718`,
annotation hierarchy `45315724`, and real coverage array `45315734`, with
dependency `afterok:45315718:45315724`.

## Current scheduler chain (2026-09-26)

The failed v5 attempt was not a placeholder: replacement job `45396786`
started and failed closed because the consumer required the logical label
`adaptive_cycle_1` where the real successor producer writes the frozen stage
value `adaptive`. Its dependent jobs `45397274`--`45397277` were therefore
cancelled by `afterok`. The old transaction trees and scheduler evidence are
preserved under names ending in
`failed_v5_adaptive_stage_contract_45396786`.

The corrected executor passed all 358 prelaunch tests in Slurm job `45405244`
(`COMPLETED`, `0:0`). Dependent relaunch job `45408036` then completed and
published the immutable reviewed source at
`outputs/prelaunch/reviewed_source_archive_post36_v6/source-6caca9e53657c6490113dac5eb85685e46fbf2ee7334f44f88c156f372342361/`.
It rotated the failed v5 transaction crash-safely and submitted this real,
receipt-bound chain:

- replacement observed attestation `45410406`;
- post-36 contract probe `45410522`, `afterok:45410406`;
- comparison `45410527`, `afterok:45410522`;
- scientific-transition publisher `45410528`, `afterok:45410527`; and
- downstream coordinator `45410530`, `afterok:45410528`.

Replacement `45410406` completed successfully in 22 seconds. Probe
`45410522` is released and waiting for normal-partition scheduling; the three
later stages remain dependency-gated. These are actual scheduled jobs, not
launch placeholders. The controller receipt records that all child submission
and release receipts were durable before the replacement was released last.

The downstream coordinator will autonomously submit and recover the remaining
authenticated
`tests -> synthetic qualification -> production readiness -> post-36 tail -> procedure lock -> measured resource plan`
chain after its active predecessors succeed. The measured plan will stop at an
authorization-pending receipt if its exact request exceeds the user's
authorized approximately 2,750 core-hour planning scale. Final connectivity
remains locked and has not been authorized or accessed by this launch.

## Operator exposure limitation

At `2026-09-26T06:12:01-0700`, an overly broad resource-keyword grep
accidentally included development result files and displayed development model
scores, directional deltas, provider-type identifiers, and source-fit metadata
in terminal output. No displayed value was analyzed or used to change a
scientific contract, threshold, model, or selection, and final connectivity
was not accessed. Nevertheless, global operator outcome blindness can no
longer be claimed. The self-hashed incident is recorded at
`outputs/prelaunch/exposure_incidents/development_outcome_terminal_exposure_20260926T061201-0700.json`.
The post-36 transition must bind this limitation and distinguish its own
allowlisted programmatic extraction from the operator exposure.

## 2026-09-27 scientific qualification terminal and repair launch

The live v6 generated-data qualifier `45446080` completed every planned fit but
failed its scientific gates and intentionally exited nonzero. Readiness
`45446082` and post-36 tail `45447999` were cancelled without execution. This
is the correct autonomous fail-closed behavior, not a missing human handoff.

The failed result is preserved under a unique immutable archive with content
hashes. No unchanged replacement chain was submitted. A real generated-only
integration-refinement job, `45494117`, now tests 512 versus 1024 draws with
the original stability and identity criteria. Method repair remains upstream
of a new qualification; no full-null or final-connectivity job is launched.
