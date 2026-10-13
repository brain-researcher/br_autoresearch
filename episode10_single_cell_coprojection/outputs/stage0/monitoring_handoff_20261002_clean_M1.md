# Clean M1 refit launched; command tool unavailable

The user approved the twelve-hour M1 production limit with `ok`. The approved
amendment, qualification, numerical evidence, focused regressions and resource
reservation are recorded in the execution ledger. ASTRA schema and semantic
validation passed in `outputs/astra/v0.0.14/validation_20261002_m1_12h.json`.

Actual submitted M1 job: **46332190**. Last verified scheduler observation:
2026-10-02 19:19:39 UTC, RUNNING, start19:17:18UTC, elapsed141s,3allocatedCPUs.
The job requests12h/1CPU/16GiB/AMD7502 and uses one numerical thread. Its actual
checkpoint manifest passed qualification, resource and backend gates and entered
`membership_loaded_outer_tuning` on194cells/12groups. Historical M0 opening
45760290 is retained; new-process M0/final/unassigned/SWC access remains false.
The thirteen launcher-owned mocked orchestration tests passed in0.306s; their
fold events are not actual scientific completion. No completed M1 result was
present at the last verified observation.

Other existing jobs at the same observation: optional46237243 RUNNING75586s
with9allocatedCPUs; permutation diagnostic46325527 RUNNING3955s with2CPUs.
The latter's last progress was `permutation_M1_started`,zero completed folds.
Optional recovery last had3/12completed historical low-rank cases and no final
report. Preserve their48h/4hlimits; no duplicates or successors were submitted.

After that observation, `exec_command` twice failed before creating a shell:
`No file descriptors available (os error 24)`. The retry was separated by one
minute. No scheduler command ran in either failed attempt. This is an agent
command-tool failure; it does not establish a Slurm outage or scientific failure.
No job was cancelled or modified in response. Further active monitoring is
unavailable until command execution is restored. Do not infer job completion
or continue a dependent stage from this handoff.

Next action when command execution is available: inspect these exact three job
IDs and their authorized reports, preserve actual terminal artifacts under Oak,
and evaluate the existing frozen gates. Do not resubmit them merely because
monitoring was interrupted. New M1 runtime root:
`$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m1_real_fit_single_thread_v3_job_46332190`.

Current resource record:
`outputs/stage0/resource_accounting_20261002_m1_clean_refit_launch.json`.
All reservations leave77.04strict wall-hours and994.202944allocated core-hours
unreserved. The same episode and Goal remain preserved; no cumulative budget,
data boundary, scientific threshold, or stopping criterion changed.
