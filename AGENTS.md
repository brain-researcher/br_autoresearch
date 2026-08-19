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
- The canonical authority for episode state, actions, Society packets, reward,
  launch approval, and Landscape transitions is the configured Brain
  Researcher MCP service.  Local Markdown is a readable projection only.
- Do not set `BR_AUTORESEARCH_DATA_ROOT` in a Sherlock client shell to point
  at this OAK directory.  The live MCP service currently resolves its own
  canonical storage under `/app/jobstore/autoresearch`; OAK is the workspace
  and projection plane, not a way to migrate or impersonate that authority.
- A successful exploration run is not a scientific acceptance.  Do not record
  reward, approve confirmation, execute confirmation, or apply a Landscape
  transition unless the canonical MCP state explicitly authorizes it.
- Never replay an action from local notes or an old terminal.  Query canonical
  state again, especially when resuming an episode.

## Required episode documents

The project-local `$brain-autoresearch-loop` skill is authoritative for their
format and update rules.  Initialize and maintain these under
`<episode>/outputs/`:

1. `experiment_log.md`
2. `memory.md`
3. `governance.md`
4. `society.md`
5. `loop.md`
6. `landscape.md`
7. `verification.md`

The six non-log projections are not `candidate_bundle.json` artifacts and must
never be declared in the bundle's `output_artifacts`.  They must not be treated
as canonical evidence.  `experiment_log.md` may be declared only as explicit
optional supplemental context under the project-local skill contract.

## Sherlock storage and compute

- Keep durable inputs and final artifacts in the episode directory on OAK.
- Put transient intermediates in `$SCRATCH/autoresearch/<episode-name>/`.
- Run expensive computation through Slurm; do not run long or resource-heavy
  analyses on the login node.
- Do not put credentials, tokens, or private keys in this repository.  The
  project MCP configuration reads `BR_MCP_TOKEN` from the environment.

## Versioning and frozen history

- `episode01_narps_analysis` and `episode02_narps_analysis` predate this root
  bootstrap.  Their existing artifacts are frozen historical material: do not
  rewrite, rename, or retroactively normalize them.
- Before `bin/codex-episode launch`, `GOAL.md` and `DATASETS.md` must be
  tracked and clean at this root repository's current `HEAD`.  This creates a
  reproducibility pin for the local contract; it does not replace the
  canonical MCP freeze or any scientific gate.
- Commit only deliberate, reviewable changes.  Do not run `git add`, `git
  commit`, `git push`, or any other Git mutation unless the user explicitly
  requests it.
- Update the root `CAMPAIGN.md`, `LANDSCAPE.md`, and `QUESTIONS.md` only from
  observed canonical state or explicit human decisions.  Do not turn a local
  exploration result into a shared campaign claim automatically.

## Launching Codex

Use `bin/codex-episode` from this root.  It starts Codex with the episode as
its writable working directory and adds only that episode's scratch directory.
`inputs/` remains an instruction-level read-only boundary inside the writable
episode workspace; the launcher does not claim OS-level input isolation:

```bash
bin/codex-episode launch episodeNN_topic
bin/codex-episode resume episodeNN_topic <codex-session-id>
```

The launcher grants bounded native exploration only.  It does not itself call
MCP, submit Slurm jobs, change Git, approve an experiment, or advance science.
