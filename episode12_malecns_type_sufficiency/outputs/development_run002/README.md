# EP12 adaptive development run 002

Episode root:
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency`

Temporary root:
`/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency`

This is the first real multi-configuration adaptive-search run. It uses only
the 816 assigned development types; all 204 final types remain locked.

## Launch receipt

- executor test job: `45314943`
- annotation-only partner-hierarchy job: `45314976`
- real observed coverage array: `45314980`, tasks `0-17`
- coverage dependency: `afterok:45314943:45314976`
- submitted: 2026-09-25

Those first submissions are retained as a failed launch attempt. Jobs
`45314943` and `45314976` exited before Python because the memory request on
the 6-GiB-per-core normal partition produced a 3-CPU allocation inconsistent
with the one-CPU task TRES. Dependency array `45314980` was automatically
cancelled without running a real trial.

The corrected one-CPU, 6-GiB relaunch is:

- executor test job: `45315718`
- annotation-only partner-hierarchy job: `45315724`
- real observed coverage array: `45315734`, tasks `0-17`
- coverage dependency: `afterok:45315718:45315724`

See `relaunch_identity.json` for replacement script hashes and resource
requests.

The 18 coverage configurations and their hashes are frozen in `search_plan/`.
They cover every pair among partner vocabulary, fitting representation,
covariance branch, and component count. Each task computes a complete T/U/M
comparison over the real development snapshot. Job I/O is written first under
scratch and staged to `observed/` only after successful completion.

The user authorized the approximately 2,750 core-hour measured planning scale
for the minimum full observed-plus-null search and explicitly removed the
original 1,000-core-hour stop for that scope. The durable authorization is
`outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml`.

This launch does not authorize final connectivity, does not by itself complete
the 36-trial minimum, and does not imply a scientific conclusion.
