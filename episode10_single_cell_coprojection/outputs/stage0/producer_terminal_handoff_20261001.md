# EP10 terminal producer handoff — 2026-10-01

Observed at 13:14:08 UTC. Active ASTRA experiment:
`v3_search.adaptive_model_search`; episode run `ep10-stage0-20260926T063634Z`.
The existing native Goal was resumed, not replaced. Both recorded producer
attempts are terminal. No fresh execution is authorized by this handoff.

## Actual result and terminal evidence

| Attempt | Terminal state | Actual retained evidence | Missing prerequisite |
| --- | --- | --- | --- |
| Optional recovery 46115632 | TIMEOUT, 12:00:21 | 34 focused tests pass; low-rank population ranks 1, 2, 3 pass with bit-exact repeats | Nine cases incomplete; no final qualification report |
| Falsifier timing 45881554 | TIMEOUT, 48:00:29 | Eight focused tests pass; time-limit cancellation and accounting retained | No timing supplement or measured-unit timing artifacts |

Parent exit code `0:0` does not make a TIMEOUT successful. Batch cancellation
`0:15` and explicit Slurm stderr support the time-limit diagnoses. No numerical
exception was reported in the preserved logs. The timing directory is empty;
its total elapsed duration includes an unmeasured scaffold and cannot be used
as the missing measured-unit projection or as proof that its statistical
resource criterion failed.

Durable actual optional evidence is in
`outputs/stage0/v3_m2_optional_family_qualification_job_46115632/`, including
three producer case JSONs, qualification log, Slurm logs, exact accounting and a
separate root terminal diagnostic. Timing evidence is in
`outputs/stage0/v3_m2_falsifier_timing_v1_job_45881554/`, including Slurm logs,
exact UTC accounting and a separate root terminal diagnostic. Root diagnostics
are not substitute producer reports. Original artifacts remain in Scratch.

The retained QC figure is
`outputs/figures/v3_m2_required_recovery_partial_job_46115632_cases3.png`
(with SVG and numeric source JSON). It was rendered and inspected when three
cases completed, before termination; its nine pending cases are now incomplete.
This prescribed synthetic QC supports numerical recovery only at those three
ranks. It does not establish the whole qualification or a biological finding.
Neurofig plotting was unavailable; existing Matplotlib was used without installation.

## Frozen resource gate and stop

Final accounting is
`outputs/stage0/resource_accounting_20261001_producers_terminal.json`.
The prior conservative floor is preserved, including both Slurm overruns.
Used strict wall: 129.832667/144 hours; remaining: **14.167333 hours**.
Used allocated core time: 302.025667/1800 hours; remaining: 1497.974333 hours.
There are no remaining live reservations for these jobs. The unchanged scale
request is 24 wall-hours, exceeding the remaining ceiling by 9.832667 hours.
The combined all-99-replay gate is unresolved because complete optional, scale
and timing reports are absent. No first empirical M2 search or full-controller
null replay was launched; real M2 membership and closed outcomes remain closed.

Under the current plan, the search is **INCOMPLETE at the preaccess gate**.
This is neither success nor a negative biological result. The frozen model
families, thresholds, eligibility, data roles, budget and stopping criteria
remain unchanged. A focused timeout diagnosis found substantial prescribed
solver work but no demonstrated incorrect solver or sufficient repair; no new
code, profiling cycle, synthetic cycle or automatic resubmission followed.

No experiment is currently runnable with satisfied dependencies and execution
authority. The already-asked bundled question remains pending: close EP10 as
INCOMPLETE SEARCH (recommended), or retain the human gate for a separately
specified continuation. Neither option by itself grants fresh execution.
The next action is that human disposition. Do not resubmit cancelled scale
45924664 or observed 45933922, rebind their stale producer paths, open real M2,
launch a successor, or infer permission from silence.

## Scientific interpretation and BR evidence impact

The existing M1 result and its adequacy checks remain retained; they were not
rerun or replaced by these QC attempts. Source-backed BR retrieval and focused
scientific review were completed within permitted summary boundaries. Review
verdict `ext_review_verdict_7c31bcf5fcb9` supports the restricted interpretation:
a small M1–M0 improvement is incremental to an already informative M0, and a
future calibrated M2 gain would challenge the particular fitted M1 observation
generator rather than establish biology, mechanism, terminal anatomy or animal
replication. Nonlinear source effects, latent populations and correlated
recording remain competing explanations. The returned evidence **supports the
current claim boundary and changes no scientific or execution decision**.

Actual review, verdict and source retrieval artifacts are retained under
`outputs/stage0/scientific_review_20261001/`; the existing exploratory M1 figure
is retained in `outputs/figures/v3_m1_reference_gains_exploratory_20261001.png`.
A standalone hosted focused-critique endpoint was unavailable; a native focused
critic used the exposed BR review directive and its verdict was submitted to BR.
Unusable retrieval hits were rejected and source-access gaps were recorded.
This assistance did not invoke formal Society or alter its frozen packet or
coordinator workflow. No restricted artifact was implicitly uploaded.

## ASTRA and closeout boundary

The preserved experiment program/export path is
`outputs/astra/v0.0.14/astra.yaml`. It remains unchanged; its passing pinned
schema and semantic validation are reused. Runtime states and attempts are
recorded in `outputs/experiment_log.md` under the stable ASTRA analysis ID.
No scientific finding has been fabricated for missing M2 results.

Export issue, separate from scientific outcome: the legacy
`bin/astra-milestone`/`br-export-astra` adapter is a lossy single-analysis
projection. It was not used to overwrite the multi-analysis program; no new
complete export is claimed. The durable program plus execution ledger and this
handoff preserve the actual result. No canonical acceptance, reward,
confirmation, Landscape transition, replacement Goal or successor was created.
The pending human gate is not a claim that the scientific episode succeeded.

BR closing snapshot persisted at 13:17:16 UTC in the existing run/session;
its initial missing-start error and transparent late-start repair are recorded
in `outputs/stage0/br_session_closeout_20261001.json`. Logging success denotes
snapshot persistence, not scientific completion or authority. No protected
artifact, transcript or external trace was uploaded.
