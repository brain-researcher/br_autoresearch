# One bounded retry after the diagnosed allocation-time guard failure

Status: prepared; not approved or submitted.

Job **46330408** failed with `parent_allocation_time_approval_mismatch` before
the public check, converter conformance, protocol lock, or real image access.
Slurm records confirm the allocation was **1 GPU, 4 CPUs, 16 GiB, 8 hours**,
with elapsed time **17 seconds**, exit `1:0`, and batch peak RSS 27,064 KiB.
No model weights, images, feature arrays, or neural arrays were processed.
The actual start/end environment values were not logged, so their exact
missing-value or timestamp difference is unknown.

The incorrect guard treated the difference between environment timestamps
as the requested allocation limit. It now queries the current job's actual
`TimeLimit` once, validates that it does not exceed the approval, and passes
that job-bound fact explicitly to the bounded public step. The original
projected end remains the shared deadline; the public step retains its
separate 30-minute cap. No resource or scientific limit was relaxed.

All **18 focused allocation-guard regressions passed**, including bounded
time records, unchanged cache reuse, child resource limits and failure
stopping. Four additional focused entrypoint/tighter-limit regressions
passed. The real feature-only context's prior 12 focused regressions remain
applicable. These are engineering checks, not live conformance or scientific
results. No model/data conformance was performed on the login node.

## Exact retry requested

Approve **one replacement job**, with **1 GPU, 4 CPUs, 16 GiB RAM and a
7-hour-59-minute limit**, keeping **≤1.6 TB durable output and ≤340 GB
temporary storage as shared caps across both attempts**. Together with the
first attempt's 17 seconds, its maximum elapsed allocation time is
7 hours 59 minutes 17 seconds, below the originally approved 8 hours.

Scientific scope and order stay unchanged: existing public check → required
converter conformance and prospective lock → real feature-only context →
all 24,681 regular-image VGG19 features. Neural fitting/scoring,
reconstruction, external-site execution and all OOD access remain closed.
Stop on failed prerequisites or a cap; no further automatic replay,
extension or successor. The new approval changes only the one-job replay
permission and tighter wall-time request.

Pending authorization:
`outputs/regular_vgg19_retry_authorization_2026-10-02.json`.
After explicit approval, the reviewable submission is:

```bash
sbatch --parsable --time=07:59:00 \
  --output=outputs/steward_provisioning/regular_vgg19_attempt002-%j.out \
  --error=outputs/steward_provisioning/regular_vgg19_attempt002-%j.err \
  outputs/steward_provisioning/tools/export_regular_vgg19.sbatch \
  --authorization outputs/regular_vgg19_retry_authorization_2026-10-02.json
```

Run from the EP21 directory. Existing public/converter attempt IDs007/008
have no completed artifacts; their first actual execution remains pending.
No existing data export, passing fixture, or sealed artifact is overwritten.

## Preserved evidence and export

- Failed job stdout/stderr: `regular_vgg19_attempt001-46330408.out/.err`.
- Scheduler evidence: `regular_vgg19_failed_job_46330408.txt`.
- Repair regressions: `outputs/pipeline_conformance/allocation_time_guard_regression_2026-10-02.log`
  and `feature_retry_entrypoint_regression_2026-10-02.log`.
- Official timestamp definitions: https://slurm.schedmd.com/sbatch.html .
- The complete 17-analysis ASTRA program is unchanged. Export status is
  recorded separately from the technical failure and absent scientific
  outcomes in the execution ledger.

The original approved proposal explicitly permits one job and forbids
automatic resubmission. That job has been consumed; the agent has not
submitted or requeued a replacement.
