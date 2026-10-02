# Autoresearch campaign instructions

This OAK repository contains independent Brain Researcher episodes. It is the
durable workspace for episode files and human-readable campaign summaries. It
is not the canonical Brain Researcher runtime database.

## Quick boundaries

- Work on one explicitly named direct-child episode directory, such as
  `episodeNN_topic`, at a time. Never write to a sibling episode or use live
  sibling outputs as run inputs.
- This repository root is the only authorized persistent OAK workspace for a
  task launched here. It does not grant write access elsewhere on OAK.
- Provisioned scientific payloads under `inputs/` are read-only.
- Plain documentation under `inputs/`, such as `inputs/README.md`, may be
  updated to keep source roles, locations, access limits, and availability
  accurate. Documentation edits must not change, replace, rename, or delete a
  payload or expose protected outcomes.
- Put generated scientific artifacts under the active episode's `outputs/`.
  Put transient work under
  `$SCRATCH/br_autoresearch/<episode-name>/`.
- Make only deliberate, reviewable Git changes. Do not run `git add`,
  `git commit`, `git push`, or another Git mutation unless the user explicitly
  requests it.
- Standalone episode work is the default. It is not canonical campaign action
  or scientific acceptance.

These are instruction-level boundaries, not OS-level isolation. Each episode
must enforce its data roles and record any limitation.

## Episode layout and local records

Every active episode must contain:

- `GOAL.md`
- `DATASETS.md`
- one search policy
- `inputs/`
- `outputs/`

Record execution events once in `outputs/experiment_log.md`. Keep
`outputs/README.md` a short, dated entrypoint linking the current log entry,
actual evidence stage, active run, useful results, blocker, and next action.
Runner/scheduler observations and their artifacts establish execution facts;
the log records them and the README provides navigation. Resolve conflicting
summaries from the relevant recent evidence or one targeted check, not a scan
of the whole episode. Refresh navigation at material transitions, not every
resume or scheduler poll; missing navigation does not block execution.

Preserve existing `memory.md`, `governance.md`, `society.md`, `loop.md`,
`landscape.md`, and `verification.md` as domain-specific decisions and links.
Create or update them only when their subject changes or a contract requires
them. Do not copy the same live status into all six. They are not canonical
MCP evidence or state-transition gates. Keep scientific goals and data roles
in the contracts, and job history in the execution log.

Reuse the existing artifact layout. Group new run configuration, scheduler
logs, metrics, and results together, such as
`outputs/runs/<analysis-id>/<attempt>/`; direct new Slurm logs there instead
of the episode root. Link current artifacts separately from historical attempts.
Mark obsolete status pages as historical and link the active entrypoint.
Preserve frozen paths and active writers; moving or deleting old artifacts is
a separate scoped change. No new manifests or status files are needed for
navigation. See the episode skill for output organization.

Use `$br-autoresearch-episode` for each new or resumed standalone episode.
Maintain `outputs/astra/v0.0.14/astra.yaml` as the episode's ASTRA experiment
program: one root analysis with one `analyses` entry per independently runnable
or evaluable experiment already required by the episode contracts. Express
experiment dependencies through ASTRA inputs and outputs. Keep mutable runtime
state—attempts, Slurm or process IDs, failures, artifacts, blocker, and exact
next action—in `outputs/experiment_log.md`, keyed by the ASTRA analysis IDs;
ASTRA 0.0.14 does not define those runtime fields.

The current `bin/astra-milestone` adapter emits only a lossy single-analysis
plan or terminal snapshot. It is a compatibility projection, not the ASTRA
experiment program and not a complete handoff. ASTRA and the execution log are
episode-local records, not canonical evidence, scientific acceptance, reward,
execution authority, or a Landscape transition. Authoring or validation
failure leaves ASTRA compliance open; it does not block otherwise-authorized
computation, invalidate completed work, or authorize a different outcome.

The six auxiliary domain records must not be listed in a `candidate_bundle.json` as
`output_artifacts`. `experiment_log.md` may be included only as explicitly
optional supplemental context under the project-local
`$brain-autoresearch-loop` contract.

## Execution first, checks when needed

Default to the next authorized action toward the requested task. There is no
mandatory audit, preflight, conformance, smoke, or qualification sequence.
Check only an explicit contract condition, concrete access/validity risk, or
observed blocker whose result can change whether to execute, repair, stop, or
ask. Reuse passing checks unless relevant code, inputs, or protocol changed.
Validate changed schemas or concrete compatibility failures, not every resume.

Synthetic work is for requested or contract-required experiments, or a bounded
reproducer of a diagnosed defect. Prefer a small real-input run when authorized.
Name synthetic stages plainly and use the existing contract to identify their
decision, cost, and stop condition. If qualification dominates the budget or
repeatedly times out, propose a concrete smaller sufficient option or amendment
within the scientist's delegated authority. Do not reset budgets, increase the
sweep, or relax frozen requirements without existing authority for that change.
Already authorized work may continue within its bounds.
Fix the selected experiment's blocker, run its focused regression, then resume;
do not start another broad audit to find hypothetical defects.

Preserve the requested scientific fidelity. For an adapted replication, explain
and disclose inherited limitations; do not silently require perfect upstream
provenance or an unlocated replacement dataset. Stricter frozen requirements
remain in force until an authorized amendment changes them.

Report missing required inputs, permission, or scientific choices early in one
bundled question with evidence and a recommendation; do not substitute synthetic
or hardening work. Optional tools or branches do not block authorized execution.

Tests, audits, or documents count as progress only if they advance the task or
remove its actual blocker. Babysit recorded jobs using dependencies or a
host-permitted wait/monitor mechanism. Check around meaningful milestones;
the scheduler's minimum interval is a floor, not a polling target. Keep
unchanged waits quiet unless periodic updates were requested. Waiting does not
require repeated reviews, log reads, document refreshes, or plots. Report new
results, failures, actionable stalls, and decisions with the actual evidence
stage; synthetic launch is not empirical launch.

Use native multiagent delegation proactively to advance independent authorized
work in parallel. The lead keeps the scientific task moving while workers
monitor status, interpret completed results, plot new evidence, or implement
disjoint changes. A waiting dependency stalls only its dependent branch.
Assign one read-only monitor per job or dependency chain; share timestamped
changes and source links instead of repeating scheduler queries. Give every
writable path and submission one owner, and integrate shared README/ledger
updates once. Workers share the episode's access limits and total approved
budget. Use disjoint permitted paths or isolated workspaces for edits. Do not
create duplicate audits, status files, or tasks to occupy agents; use the direct
path when no useful parallel work or native delegation is available.

Visualize useful new evidence while working, not only at final reporting. Use
available Neurofig layouts or existing plotting tools to explain progress,
comparisons, uncertainty, and diagnosed problems. Share the figure with a brief
interpretation at the relevant milestone; retain it under the episode's
`outputs/figures/`. Read only authorized artifacts and label schematic, QC,
exploratory, or confirmatory evidence. Do not invent data, open sealed outcomes,
add experiments or synthetic work, or make figure quotas or renderer installation
new execution gates. Do not replot unchanged evidence.

Do not add generic hashes, receipt chains, source copies, interpreter bindings,
schema gates, or provenance inventories. Retain provider identities and checks
required by the frozen contract or a concrete integrity failure.

Preserve biological-group identity/independence, overlap/leakage,
development/final separation, group support, source/atlas compatibility, and
unknown versus verified absence. Failures require affected-stage repair,
scientific amendment, or stop.

Apply modeling, lock, audit, and confirmation rules only at their required
stage. Retain frozen identity, timing, and access guarantees unless the scientist
amends the contract. Lighter implementations must preserve those guarantees
before outcome access; never normalize prior records.

## BR scientific assistance in standalone work

Standalone does not mean BR-free. At consequential hypothesis, design, or result
decisions, actively use relevant exposed read-only BR retrieval or critique
within the episode's access and resource boundaries. Reuse prior evidence and
record its decision impact in the existing experiment log; do not require a
fixed tool checklist for every episode or resume.

Check actual tool exposure when needed, not merely whether MCP is configured.
Report missing optional capabilities without blocking otherwise-authorized
work or adding installation, retry, or audit chains. Scientific assistance is
not formal Society, canonical binding, approval, or scientific acceptance; the
canonical boundary below remains unchanged.

## Canonical Brain Researcher boundary

The configured Brain Researcher MCP service is authoritative for Society
packets, reward, accepted claims, confirmation, and Landscape transitions.
Local episode files describe standalone work only.

- Do not set `BR_AUTORESEARCH_DATA_ROOT` to this OAK repository. The live MCP
  service uses its own canonical storage under `/app/jobstore/autoresearch`.
- Do not infer or replay a canonical action from local notes or an old
  terminal. Read canonical state again immediately before a canonical
  mutation.
- A successful exploration is not scientific acceptance. Do not record reward,
  approve or execute confirmation, or apply a Landscape transition unless the
  current canonical MCP state explicitly authorizes it.
- Use `$brain-autoresearch-loop` and its canonical preflight only when the
  scientist explicitly requests the Brain Researcher loop, Society review,
  reward, registered handoff, canonical confirmation, or another canonical
  operation.

Canonical state is not a prerequisite for standalone acquisition,
qualification, analysis, or continuation.

## Sherlock storage, compute, and secrets

- Keep durable inputs and final artifacts inside the active episode on OAK.
- Use `$SCRATCH/br_autoresearch/<episode-name>/` for new transient
  intermediates. Scratch is temporary and subject to Sherlock purge policy.
- Existing `$SCRATCH/autoresearch/` trees are legacy runtime state. Do not use
  them as canonical inputs, move them, or delete them without an explicit
  migration decision.
- Run expensive or sustained computation through Slurm, not on a login node.
- Never store credentials, tokens, or private keys in this repository. The MCP
  configuration reads `BR_MCP_TOKEN` from the environment.
- All cluster-wide rules in `/etc/agents/AGENTS.md` still apply.

## Protected history and campaign files

The following are immutable prior records:

- `_examples/historical_prior_records/episode01_narps_analysis`
- `_examples/historical_prior_records/episode02_narps_analysis`

They were moved from the repository root by explicit scientist instruction on
2026-09-22. Do not rewrite, rename, delete, normalize, reward, or otherwise
mutate them. Their old display aliases do not make them current episodes. Any
canonical action involving them requires a separate scientist-authorized turn
and a fresh canonical-state read.

The current formal EP01 is `episode01_narps_deep_search`. It may use those
records only through its verified episode-local, content-addressed prior
packet. This is a grandfathered EP01 boundary, not a default requirement for
new episodes. Never reuse a legacy loop or Goal handoff as the new episode
identity.

Update root `CAMPAIGN.md`, `LANDSCAPE.md`, and `QUESTIONS.md` only from
observed canonical state or an explicit human decision. Never promote a local
result into a shared campaign claim automatically.

## Starting or resuming an episode

1. Require an explicit Codex request naming one episode.
2. Use `$br-autoresearch-episode` and read this file plus that episode's
   `GOAL.md`, `DATASETS.md`, and search policy.
3. Confirm the required episode files and directories exist.
4. Work only in that episode. Parallel workers may own disjoint permitted paths;
   each writable path, shared status record, and job submission has one owner.
   Continue the same coordinated task when practical.
5. Preserve provisioned input payloads; limit input documentation edits to the
   exception above.
6. Write durable results under `outputs/` and transient work under the
   episode-specific scratch path.
7. Run and record validation, qualification, falsification, held-out
   evaluation, and audit work inside the episode at the stage where each is
   scientifically required.

Starting a Codex task does not automatically snapshot contracts, evaluate
scientific policy, inspect Git state, call MCP, submit a Slurm job, mutate Git,
approve a claim, or advance Landscape state. Missing optional hashes, schemas,
receipts, snapshots, or projection refreshes do not block standalone work.
