---
name: br-autoresearch-episode
description: >-
  Run, launch, resume, or babysit one named standalone br_autoresearch episode
  on OAK/Sherlock. Enforce staged data access and repair-first execution;
  exclude canonical Society, reward, confirmation, and Landscape actions.
---

# BR Autoresearch Episode

Use this skill for one explicitly named, direct-child episode. The repository
`AGENTS.md` and the episode contracts remain authoritative. For canonical Brain
Researcher actions—Society, reward, confirmation, registered handoff, or
Landscape—use `$brain-autoresearch-loop` instead.

## Start

1. Read `AGENTS.md`, then the episode's `GOAL.md`, `DATASETS.md`, and single
   `SEARCH_POLICY.yaml` or `SEARCH_POLICY.json`.
2. Confirm `inputs/` and `outputs/` exist. Keep `inputs/` read-only, durable
   work in `outputs/`, and temporary work in the episode-specific scratch path.
3. Determine the current data-access stage and name the files and directories
   that are safe to read.
4. Choose the next executable action. For a build, launch, resume, continue, or
   babysit request, do not stop at planning while authorized work remains.

## Protect closed outcomes

Data roles apply to `outputs/` as well as `inputs/`. While an outcome is closed,
read and search only explicitly allowed paths. Never search the episode root or
all of `outputs/` and rely on exclusions; terminal output counts as access.
Use filesystem or runner isolation when available; instructions alone are not
equivalent isolation.

## Move the episode forward

Default to the next authorized experiment. There is no compulsory audit,
preflight, conformance, smoke, or qualification chain.

- If dependencies are satisfied and execution is authorized, execute in this
  run. Add a prerequisite only when the contract requires it or a concrete
  access, validity, or implementation problem blocks this experiment. Reuse
  passing checks unless relevant code, inputs, or protocol changed.
- Use synthetic work for a requested or contract-required experiment, or a
  bounded reproducer of a diagnosed defect. Prefer a small real-input run when
  access and execution are authorized; synthetic fixtures do not establish
  empirical readiness.
- Repair only a diagnosed defect blocking the selected experiment. Make the
  focused fix, run its regression, then resume. Do not start another broad or
  adversarial audit to search for additional hypothetical blockers.
- If required data, permission, or a scientific choice is unavailable, report
  the exact blocker early and ask once with a recommendation. Do not substitute
  synthetic or hardening work. Optional branches do not block unrelated work.
- Record the process or scheduler ID and which experiment actually launched.
  Qualification or synthetic launch is not empirical launch; ASTRA, plans,
  placeholders, and dry runs are not execution. Prefer scheduler dependencies.
- Babysit existing authorized jobs to completion or the next real decision;
  repair technical failures at the smallest affected stage. Never duplicate a
  pending job or manufacture prerequisites to keep a goal active.

## Visualize while working

When new authorized artifacts materially explain progress, a comparison,
uncertainty, or a diagnosed problem, proactively plot and share them in the
current update; do not wait for the final report. Prefer one clear primary view:
learning/progress curves, coverage, paired effects with intervals, prediction
and residual diagnostics, or modality-appropriate brain/QC views as relevant.

Reuse available `neurofig-plotting` guidance and source-backed layouts/renderers;
gallery examples supply style, not values or scientific claims. If unavailable,
use an existing plotting tool rather than installing a stack or calling an
unverified MCP renderer. Keep figures under `outputs/figures/`, identify the
source artifacts and evidence status in the caption or ledger, and display the
figure when the client supports it; otherwise provide its usable artifact path.

Stay within the current read boundary. Do not invent observations, statistics,
anatomy, or transforms; create new experiments or synthetic data just for a
figure; or unlock held-out outcomes to plot. Label design diagrams as schematics
and QC/exploratory evidence accordingly. No figure quota, dashboard, mandatory
plotting gate, repeated unchanged render, or implicit remote upload.

## Decide or ask

Decide contract-preserving engineering details yourself, including seeds,
numerical methods, Slurm sizing, chunking, checkpointing, logging, and
mechanical fixes whose behavior is already determined.

Ask only when proceeding would change the estimand, model family, eligibility,
frozen threshold or meaningful margin, data role, permitted claim, or
authorized resources—or when the requirements cannot be met with the available
data or resources. Honor explicit delegation. Ask once with evidence, concrete
options, trade-offs, and a recommendation.

## ASTRA and handoff

At a plan or genuine terminal milestone, read
[references/astra.md](references/astra.md). ASTRA is non-authoritative and must
never grant execution authority, prove scientific validity, change canonical
state, delay execution, or bypass the read boundary.

Lead every handoff with the real execution state, job IDs, current blocker, and
next action. Mention ASTRA status second; documentation is evidence, not a
state transition.

Maintainers changing this skill should use
[references/behavioral_evals.md](references/behavioral_evals.md).
