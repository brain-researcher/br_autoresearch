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

Maintain these seven local status files under `outputs/`:

1. `experiment_log.md`
2. `memory.md`
3. `governance.md`
4. `society.md`
5. `loop.md`
6. `landscape.md`
7. `verification.md`

They are local, human-readable records—not canonical MCP evidence or
state-transition gates. Keep them current at material milestones, but do not
let projection maintenance delay the next scientific action.

Use `$br-autoresearch-episode` for each new or resumed standalone episode. It
exports a validated ASTRA plan after the contracts are read and a terminal
projection only after real closeout artifacts exist. Both milestones go
through `bin/astra-milestone`, which selects the deployed Sherlock launcher
when available. The ASTRA file is written to
`outputs/astra/v0.0.14/astra.yaml`. It is an interoperability projection, not
canonical evidence, scientific acceptance, reward, execution authority, or a
Landscape transition. ASTRA is expected at plan and terminal milestones, but
it is not a launch or computation gate. An export failure leaves ASTRA
compliance open and does not retroactively invalidate completed computation or
authorize a different scientific outcome.

The six non-log files must not be listed in a `candidate_bundle.json` as
`output_artifacts`. `experiment_log.md` may be included only as explicitly
optional supplemental context under the project-local
`$brain-autoresearch-loop` contract.

## Keep checks scientifically necessary

Use the smallest check that answers the scientific or operational question.
A check is warranted only when it:

- enforces an explicit eligibility or validity condition in the frozen
  `GOAL.md`, `DATASETS.md`, or search policy;
- is needed for access control, safety, or correct use of real data or code; or
- diagnoses a concrete inconsistency or failure that has actually occurred.

Do not add SHA/checksum inventories, strict schema validators,
content-addressed copies, duplicate-key policing, attestation or receipt
chains, exact-interpreter bindings, repeated qualification reruns, or extra
provenance files as generic readiness work. Provider-supplied identifiers and
checksums may be recorded, but generating new machinery is not a launch gate
unless exact identity is scientifically essential or a real integrity problem
requires it.

Preserve checks that directly support the claim: biological-group identity
and independence, duplicate or cross-episode overlap, leakage,
development/final separation, adequate group support, source and atlas
compatibility, and the distinction between detected, verified absent, and
unknown outcomes. Summarize them with the lightest useful pass/revise/stop
note. If one fails, revise the resource, narrow the claim, or stop instead of
building a larger validation framework.

Stage-specific modeling, lock, audit, and confirmation rules apply only at
their scientific stage. A lighter representation may replace an old
engineering mechanism only before candidate-discriminating outcome access and
only when identity, timing, and access guarantees remain equivalent. If the
mechanism itself enforces a frozen identity or access boundary, retain it
unless the scientist explicitly amends that contract. Never rewrite prior
records merely to normalize them.

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
4. Work only in that episode and do not run two writing tasks against it at
   once. Continue the same task when practical.
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
