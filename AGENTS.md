# Autoresearch campaign instructions

This directory is the Sherlock project root for Brain Researcher autoresearch.
It contains several independent research episodes.  The root is a lightweight
Git repository for durable campaign instructions and human-readable campaign
projections, not the canonical runtime database.

For tasks explicitly launched from this project, this exact OAK root is the
narrow authorized persistent workspace.  It grants no
write authority to any other OAK location.

## Scope and authority

- The active scientific workspace is one direct child directory such as
  `episodeNN_topic`.  Never write to a sibling episode.
- Each episode must contain `GOAL.md`, `DATASETS.md`, `inputs/`, and `outputs/`.
  Treat `inputs/` as read-only.  Put all new scientific artifacts, including
  the seven required Markdown projections, under that episode's `outputs/`.
- The configured Brain Researcher MCP service remains the authority for
  Society packets, reward, accepted claims, and Landscape transitions. It is
  not a prerequisite for starting an explicitly requested episode-managed
  exploration run. Local episode state and artifacts describe that run only.
- Do not set `BR_AUTORESEARCH_DATA_ROOT` in a Sherlock client shell to point
  at this OAK directory.  The live MCP service currently resolves its own
  canonical storage under `/app/jobstore/autoresearch`; OAK is the workspace
  and projection plane, not a way to migrate or impersonate that authority.
- A successful exploration run is not a scientific acceptance.  Do not record
  reward, approve confirmation, execute confirmation, or apply a Landscape
  transition unless the canonical MCP state explicitly authorizes it.
- Never replay an action from local notes or an old terminal.  Query canonical
  state again, especially when resuming an episode.

## Required episode documents (project-local policy)

The project-local `$brain-autoresearch-loop` skill is authoritative when an
episode opts into the canonical Brain Researcher review/reward workflow. For a
standalone episode-managed run, use the same seven projections as the local
format and update convention. Initialize and maintain them under
`<episode>/outputs/`:

1. `experiment_log.md`
2. `memory.md`
3. `governance.md`
4. `society.md`
5. `loop.md`
6. `landscape.md`
7. `verification.md`

This seven-file projection requirement is a local reproducibility policy, not
an MCP state-transition gate.

The six non-log projections are not `candidate_bundle.json` artifacts and must
never be declared in the bundle's `output_artifacts`.  They must not be treated
as canonical evidence.  `experiment_log.md` may be declared only as explicit
optional supplemental context under the project-local skill contract.

## Sherlock storage and compute

- Keep durable inputs and final artifacts in the episode directory on OAK.
- Put new transient intermediates in
  `$SCRATCH/br_autoresearch/<episode-name>/`. Existing
  `$SCRATCH/autoresearch/` trees are legacy runtime state: do not treat them as
  canonical inputs, move them, or delete them without an explicit migration
  decision.
- Run expensive computation through Slurm; do not run long or resource-heavy
  analyses on the login node.
- Do not put credentials, tokens, or private keys in this repository.  The
  project MCP configuration reads `BR_MCP_TOKEN` from the environment.

## Versioning and frozen prior history

- `_examples/historical_prior_records/episode01_narps_analysis` and
  `_examples/historical_prior_records/episode02_narps_analysis` predate this
  root bootstrap. They were relocated once from the repository root by an
  explicit scientist instruction on 2026-09-22. Their contents and canonical
  identities remain immutable prior records: do not rewrite, rename, delete,
  or retroactively normalize them. Do not reward or otherwise mutate a legacy
  canonical record as part of work on the new EP01; any such action requires
  its own explicit scientist-authorized turn and a fresh canonical-state read.
  Their former EP01/EP02 display aliases do not make either directory a current
  episode.
- The current formal EP01 is `episode01_narps_deep_search`. It may consume the
  legacy records only through its verified, episode-local, content-addressed
  prior packet. Never read live sibling outputs during a run, and never reuse a
  legacy loop or Goal handoff as the new episode identity.
- Starting episode-managed work requires an explicit user instruction in a
  Codex task. The instruction must name one episode, whose directory must
  contain `GOAL.md`, `DATASETS.md`, one search policy, `inputs/`, and
  `outputs/`. A Codex task does not automatically snapshot contracts, inspect
  Git state, or evaluate scientific policy. Data validation, runtime
  qualification, falsification, held-out evaluation, and any audit logic are
  executed and recorded inside the episode.
- Commit only deliberate, reviewable changes.  Do not run `git add`, `git
  commit`, `git push`, or any other Git mutation unless the user explicitly
  requests it.
- Update the root `CAMPAIGN.md`, `LANDSCAPE.md`, and `QUESTIONS.md` only from
  observed canonical state or explicit human decisions.  Do not turn a local
  exploration result into a shared campaign claim automatically.

## Starting episode work in Codex

Start a new Codex task from the repository project and explicitly name one
episode. The initial instruction must tell Codex to read this file plus that
episode's `GOAL.md`, `DATASETS.md`, and search policy; work only on that
episode; treat `inputs/` as read-only; write durable artifacts under
`outputs/`; and use `$SCRATCH/br_autoresearch/<episode-name>/` for transient
work. Continue the same task when resuming work. Do not run two writing tasks
against the same episode at once.

These are instruction-level boundaries, not OS-level input isolation. The
episode is responsible for enforcing its data roles and recording any
limitation. Direct Codex task startup grants bounded episode-managed
exploration only; it does not itself call MCP, submit Slurm jobs, change Git,
approve a scientific claim, or advance Landscape state. Brain Researcher
review and reward can be requested afterward without becoming a prerequisite
for doing the episode work.
