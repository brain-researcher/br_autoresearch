---
name: run-episode
description: >-
  Implement, launch, resume, repair, monitor, or analyze a named research
  episode under its existing scientific contract. Use for experiment execution
  and results, not designing a new study or managing a formal BR campaign.
---

# Run an Episode

Turn an existing research design into interpretable evidence. Work in the
named standalone episode, not the private BR source checkout that distributes
these instructions. Repository `AGENTS.md` and the episode contract govern.

## Match the requested action

- Implementation or execution: advance the next authorized experiment.
- Analysis or plotting: use permitted existing evidence; do not launch new
  experiments merely to strengthen a result or complete a figure.
- Status or diagnosis: inspect and explain; do not mutate files, repair,
  resubmit, or start a campaign unless that action is also requested.
- A new question or substantive redesign belongs to `design-episode`.
  Outcome-adaptive choices already delegated by the active search policy stay
  here. A redesign request does not silently amend a running frozen contract.
- Explicit formal BR campaign work belongs to `manage-research-campaign`.
  A Codex goal, local launch, or request for scientific critique does not
  establish that binding.

## Resume from actual evidence

Read the named episode's `GOAL.md`, `DATASETS.md`, and single search policy.
Determine the current data-access stage before opening artifacts. Follow
`outputs/README.md` to the relevant ASTRA program, execution-log entries and
permitted evidence; use known safe paths if navigation is missing or stale.
Read the state needed for this decision, not every prior attempt.

Identify the selected experiment, its actual dependencies, and whether it is
planned, running, completed, or blocked. Reuse the designated monitor's
observation. Never infer completion from a missing job or resubmit a running
attempt. Preserve failed attempts and start from completed prerequisites.

Input payloads remain read-only. Positive read boundaries also cover indexes,
terminal output and `outputs/`: no root-wide searches with exclusion globs.
Instructions alone are not filesystem isolation. Preserve biological-group
independence, overlap controls, data roles, support/atlas compatibility and
unknown versus verified absence. Do not open closed outcomes to debug or plot.

## Execute and repair

- Use the runner's existing entrypoint and recorded validations. Add checks
  only for an explicit contract requirement or a concrete failure that changes
  the next action. No generic preflight, schema, SHA, provenance, receipt or
  synthetic-qualification chain. Preserve requirements of frozen contracts.
- Record the selected analysis and attempt in `outputs/experiment_log.md`
  before a process or scheduler side effect, then the returned ID and observed
  outcome. A plan, ASTRA export, dry run or synthetic success is not an
  empirical launch or scientific finding.
- Keep exploration as broad as the active question and policy allow. Do not
  import another episode's candidate counts, model menu, pilot size or search
  program. Respect the shared budget and the declared stopping rule.
- Make contract-preserving engineering choices directly. For a diagnosed
  defect, implement the focused fix, test it on compute, and retry the smallest
  affected stage within existing authority. Honor explicit retry/job caps.
- When a consequential code, input, runtime or protocol change invalidates a
  result, mark only that result and its affected dependents stale in the
  ledger. Reuse unaffected checks and evidence.
- Missing required inputs, authority, or a consequential undelegated scientific
  choice needs one early question with evidence and a recommendation. Optional
  branches do not block ready work. Do not replace blocked empirical work with
  unrequested synthetic demonstrations or hardening.
- A required synthetic or calibration stage needs a scientific purpose, cost
  and stop condition. Run short required checks inside the substantive
  allocation. If a requirement dominates the budget or times out, report the
  trade-off and propose a sufficient amendment; do not reset consumed budget,
  enlarge a sweep, silently waive the requirement or cancel unrelated work.

Preserve the requested fidelity, including an adapted replication with stated
deviations. Do not demand perfect upstream provenance or silently substitute a
different dataset. Changes to estimand, model family, eligibility, frozen
thresholds, data roles, claim or resources need authority unless already
delegated by the active contract.

## Compute, parallel work and waiting

On Sherlock, follow `/etc/agents/AGENTS.md` and the relevant topic files.
Run tests, scientific code, builds and substantial I/O in Slurm, not on login
nodes. For interactive testing, reuse a suitable task-owned compute session
or request a bounded one after resource discovery. Verify `hostname` and
`SLURM_JOB_ID` inside the execution shell; a new shell-tool call or subagent
does not automatically inherit the compute PTY. Stop if the allocation expires;
never fall back to login-node testing. GPU work requires an actual GPU request.

Delegate useful independent implementation, interpretation, plotting or
monitoring with explicit permitted paths and a shared resource/thread budget.
Give each submission and writable path one owner; the lead integrates shared
records. One monitor owns each dependency chain. Workers coordinate compute
with the lead rather than starting tests on a login node or duplicating polls.

Use scheduler dependencies or the host-permitted wait/monitor mechanism.
The site's minimum interval is a floor, not a polling target. Keep unchanged
waits quiet unless periodic updates were requested. Report meaningful results,
failures and decisions, and advance independent authorized work when useful.
Do not fill waits with audits, status rewrites or repeated plots. No unattended
login-node polling, keepalives, automatic renewals or supervisors.

## Interpret and show the result

Report what the comparison establishes, uncertainty at the independent-unit
level, the strongest surviving alternative, and what remains unresolved.
Separate exploratory evidence, held-out evaluation, replication and formal
confirmation. A performance win alone does not identify its cause. Negative
results and informative limitations are valid outcomes, not failed storytelling.

At consequential scientific decisions, use exposed read-only BR literature/KG
retrieval or focused plan/result critique within the disclosure boundary.
Reuse prior evidence and inspect tool schemas only as needed. Record the
question, evidence reference and decision impact in the existing log. Optional
tool absence is not an installation or audit gate; never assume an external
server can read an OAK path. Advisory critique grants no canonical authority.

Plot new permitted evidence when it clarifies the comparison, uncertainty or
a diagnosed problem. Use a source-backed plotting tool, with
`neurofig-plotting` if available; identify data and evidence status. Image-gen
is for conceptual illustration, not invented measurements, brain maps or
result curves. If image-gen is explicitly requested for quantitative material,
clarify whether the desired artifact is an illustrative schematic or an exact
data plot; do not silently replace the requested tool or present generated
numbers as evidence. Keep useful figures in `outputs/figures/` and share them
during work. Do not add experiments, unlock outcomes or install a renderer
merely to make a plot. Unchanged evidence needs no new figure.

## Keep one clear handoff

Maintain the scientific program in `outputs/astra/v0.0.14/astra.yaml`, attempts
and job state in `outputs/experiment_log.md`, and concise navigation in
`outputs/README.md`. Read [ASTRA and record ownership](references/astra.md)
when authoring or changing these records; status-only requests remain read-only.
Missing navigation or ASTRA tooling does not block authorized science.

Hand off the observed result or stage, direct evidence, blocker if any, and
next scientific action. Do not manufacture a successor run or canonical
acceptance. Record changes once and link them; do not create dated audit,
review or status documents for routine progress.

For skill maintenance only, use the relevant
[behavioral scenarios](references/behavioral_evals.md).
