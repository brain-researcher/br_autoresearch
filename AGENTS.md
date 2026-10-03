# Research episode workspace

This repository is the durable workspace for independent Brain Researcher
episodes. Local files describe the work; the configured Brain Researcher MCP
service owns canonical campaign state.

## Choose the workflow

| Requested work | Guidance |
| --- | --- |
| Propose a new EP, assess novelty, revise a scientific question/design, or create a concept figure | Use [`design-episode`](.agents/skills/design-episode/SKILL.md). Start from the question and available evidence; existing episode files are not a prerequisite for ideation. |
| Implement, launch, resume, repair, monitor, analyze, or report on an existing standalone episode | Use [`run-episode`](.agents/skills/run-episode/SKILL.md). Read its `GOAL.md`, `DATASETS.md`, and single search policy; status/diagnosis alone remains read-only. |
| Explicit formal BR campaign, Society, reward, registered handoff, confirmation, or Landscape action | Use [`manage-research-campaign`](.agents/skills/manage-research-campaign/SKILL.md) and the current canonical state. |
| Repository maintenance or ordinary document edits with no scientific-design change | Stay within the requested files and action; no research workflow needs to start. |

Design and standalone execution are the everyday workflows. Use only the skill
needed for the current action, not all three in sequence. A design discussion
does not authorize compute; an analysis request does not authorize another
experiment. A persistent Codex goal does not imply a canonical Brain Researcher
Goal. Do not run Git mutations, including staging, committing, or pushing,
unless the user explicitly requests them.

## Workspace and data boundaries

- Each execution task owns one explicitly named direct-child episode, such as
  `episodeNN_topic`. Do not write to siblings or use live sibling outputs as
  run inputs. A request covering several episodes needs separate task scopes.
- This repository is the only persistent OAK workspace authorized by a task
  launched here; access to it does not grant write access elsewhere on OAK.
- Each active episode contains `GOAL.md`, `DATASETS.md`, one
  `SEARCH_POLICY.yaml` or `SEARCH_POLICY.json`, `inputs/`, and `outputs/`.
- Scientific payloads under `inputs/` are read-only. Plain input documentation
  may be corrected without changing payloads, data roles, or exposing outcomes.
- Determine the current data-access stage before reading episode artifacts.
  Read or search only explicitly permitted paths, including under `outputs/`
  and in indexes. Root-wide searches with exclusion globs do not protect closed
  outcomes; terminal output is access too.
- Keep durable scientific artifacts under the active episode's `outputs/`
  and transient work under `$SCRATCH/br_autoresearch/<episode-name>/`.
  Preserve frozen paths, historical attempts, and active writers. Moving or
  deleting artifacts requires a separately scoped change.
- Instructions are not OS-level isolation. Enforce the episode's declared data
  roles and disclose any weaker boundary.

## Execution and scientific decisions

Advance the next authorized work. Check an explicit contract condition, a
concrete access/validity risk, or an observed defect that changes the next
action; reuse passing runner checks. There is no universal preflight,
conformance, smoke, or qualification sequence. Do not add generic hashes,
receipt chains, source copies, interpreter bindings, schemas, or provenance
inventories. Retain mechanisms required by a frozen contract or a concrete
integrity failure.

Keep exploration as open as the scientific question and active policy allow.
Do not import candidate caps, modeling choices, or an adaptive-search program
from a different episode. [Adaptive search](ADAPTIVE_SEARCH_PROTOCOL.md) and its
[controller interface](ADAPTIVE_CONTROLLER_INTERFACE.md) apply only at their
declared stages in episodes that adopt them.

Preserve the requested scientific fidelity, biological-group independence,
overlap/leakage controls, development/final separation, group support,
source/atlas compatibility, and unknown versus verified absence. Frozen
identity, timing, access, stopping, budget, and claim limits remain in force
until an authorized amendment. A cleanup or mechanical repair is not such an
amendment.

Make contract-preserving engineering choices directly. Report missing required
data, authority, or consequential scientific choices early, with a recommendation.
Do not replace blocked empirical work with unrequested synthetic or hardening
work. Optional tools do not block otherwise-authorized execution.

Use useful parallel work with one owner per writable path or submission, one
monitor per dependency chain, and a shared resource budget. Share meaningful
new evidence and sourced figures when they clarify the work; unchanged waits
do not need repeated checks, reports, or plots. Detailed execution, repair,
monitoring, and result-interpretation guidance lives in `run-episode`;
novelty, study design and concept-figure guidance lives in `design-episode`.

## Records and artifacts

| Record | Purpose |
| --- | --- |
| `GOAL.md`, `DATASETS.md`, search policy | Scientific question, data roles, and operative contract; not live job status |
| `outputs/astra/v0.0.14/astra.yaml` | Experiment program with stable analysis IDs and artifact dependencies |
| `outputs/experiment_log.md` | Execution events, attempts, job IDs, observed evidence, failures, blocker, and next action |
| `outputs/README.md` | Short, dated navigation to current log entries, runs, and useful results |
| Existing auxiliary Markdown records | Domain-specific decisions and links, updated when their subject changes |

Maintain the ASTRA program for standalone episodes; use the
[ASTRA reference](.agents/skills/run-episode/references/astra.md)
when authoring it or organizing outputs. The legacy snapshot exporter is not
the full program. Missing navigation or ASTRA tooling does not block
otherwise-authorized computation.

Record execution facts once in the log and link to their evidence. Resolve
conflicting summaries from the relevant permitted runner/scheduler observations
or artifacts. Refresh navigation at material transitions, not on every resume.
Preserve `memory.md`, `governance.md`, `society.md`, `loop.md`,
`landscape.md`, and `verification.md`; do not copy live status into all six or
create them unless their workflow or contract requires them. Canonical
projection and CandidateBundle rules are in
[workspace projections](.agents/skills/manage-research-campaign/references/workspace_projections.md).

## Canonical Brain Researcher authority

- Do not set `BR_AUTORESEARCH_DATA_ROOT` to this repository. The live MCP
  service uses canonical storage under `/app/jobstore/autoresearch`.
- Re-read canonical state immediately before a canonical mutation or replay;
  local notes and old terminal output cannot authorize it.
- Scientific retrieval or advisory critique is not formal Society, reward,
  launch approval, confirmation, scientific acceptance, or a Landscape
  transition. Use exposed read-only scientific assistance where it informs a
  decision, within the current read/disclosure boundary.
- A successful local run cannot approve its own claim. Apply canonical actions
  only when both the scientist's request and current canonical state authorize
  them. Canonical state is not a prerequisite for standalone execution.
- Update root `CAMPAIGN.md`, `LANDSCAPE.md`, and `QUESTIONS.md` only from
  observed canonical state or an explicit human decision.

## Protected history

These prior records are immutable:

- `_examples/historical_prior_records/episode01_narps_analysis`
- `_examples/historical_prior_records/episode02_narps_analysis`

Do not rewrite, rename, delete, normalize, reward, or otherwise mutate them.
Canonical action involving them requires a separate scientist-authorized turn
and fresh canonical state. Their old display aliases are not current episodes.

The current EP01 is `episode01_narps_deep_search`. It may use those records only
through its verified episode-local, content-addressed prior packet. This is an
EP01-specific boundary, not a default requirement for other episodes. Do not
reuse a legacy loop or Goal handoff as the new episode identity.

## Sherlock and credentials

Follow `/etc/agents/AGENTS.md` and its relevant topic files. Tests, scientific
code, builds and substantial I/O belong in Slurm, not on a login node. Verify
the execution shell's hostname and `SLURM_JOB_ID`; an allocated session does not
move other shell calls or subagents onto compute. Coordinate testing through
the task-owned compute session and stop if it expires. Scratch is purgeable; legacy
`$SCRATCH/autoresearch/` trees are not canonical inputs and must not be moved or
deleted without an explicit migration decision.

Never store credentials, tokens, or private keys in this repository. The MCP
configuration reads `BR_MCP_TOKEN` from the environment.
