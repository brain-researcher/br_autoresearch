---
name: br-autoresearch-episode
description: >-
  Run, launch, resume, or babysit one named standalone br_autoresearch episode
  on OAK/Sherlock from an ASTRA experiment program. Enforce staged data access
  and advance the next authorized experiment; exclude canonical Society,
  reward, confirmation, and Landscape actions.
---

# BR Autoresearch Episode

Use this skill for one explicitly named, direct-child episode in its standalone
workspace, not in the private BR source checkout that distributes this skill.
The standalone workspace's `AGENTS.md` and episode contracts remain authoritative.
For canonical Brain Researcher actions—Society, reward, confirmation, registered
handoff, or Landscape—use `$brain-autoresearch-loop` instead.

## Start

1. Read the standalone workspace's `AGENTS.md`, then the episode's `GOAL.md`,
   `DATASETS.md`, and single `SEARCH_POLICY.yaml` or `SEARCH_POLICY.json`.
2. Confirm `inputs/` and `outputs/` exist. Keep `inputs/` read-only, durable
   work in `outputs/`, and temporary work in the episode-specific scratch path.
3. Determine the current data-access stage and name the files and directories
   that are safe to read.
4. Read the existing ASTRA program and execution ledger within that boundary.
   Use [references/astra.md](references/astra.md) when creating or changing the
   program; update only what is missing or changed in the contract-required work.
5. Select the next authorized experiment from current dependencies and job state.
   Resume from completed prerequisites rather than restarting readiness work.
   An unchanged resume does not require ASTRA revalidation, export, projection,
   or documentation refresh.

## Protect closed outcomes

Data roles apply to `outputs/` as well as `inputs/`. While an outcome is closed,
read and search only explicitly allowed paths. Never search the episode root or
all of `outputs/` and rely on exclusions; terminal output counts as access.
Use filesystem or runner isolation when available; instructions alone are not
equivalent isolation.

## Use BR at scientific decision points

Actively use relevant exposed, read-only Brain Researcher capabilities when
evidence or critique could change the next scientific decision. Workflow state
and ASTRA export alone are not scientific use of BR.

- When choosing hypotheses, competing explanations, or controls, use
  source-backed KG or literature retrieval to identify prior evidence,
  contradictions, and discriminating tests. Reuse already retrieved evidence.
- For a consequential unresolved design choice, request focused plan critique
  rather than a general preflight or audit.
- After a substantive authorized result, use scientific critique to challenge
  its interpretation and strongest surviving alternative before closeout.
  Mock success, scheduler status, and mechanical repairs do not require a
  scientific review.

Inspect the actual exposed tools and their schemas when needed; reuse known
capability information. A configured MCP or installed skill is not proof that
its scientific tools are available. If a relevant capability is unavailable or
degraded, report the gap and continue otherwise-authorized work; do not start an
installation, retry, or qualification chain. There is no tool-call quota.

Stay inside the current read and disclosure boundaries. Use only permitted
evidence summaries or an exposed external-review handoff; never assume the
hosted server can read an OAK path or upload restricted artifacts implicitly.
Record the question, tool/evidence reference, and decision impact, including
no change, in the existing `outputs/experiment_log.md`. Do not rewrite frozen
selection records after observing outcomes.

Early scientific critique is not formal canonical Society. It does not adopt
this episode into a canonical loop, grant approval, open closed outcomes, or
advance reward or Landscape. Formal Society still requires its frozen packet
and the explicitly requested canonical workflow.

## Move the episode forward

Default to the next authorized experiment. There is no compulsory audit,
preflight, conformance, smoke, or qualification chain.

- If dependencies are satisfied and execution is authorized, execute in this
  run. Add a prerequisite only when the contract requires it or a concrete
  access, validity, or implementation problem blocks this experiment. Reuse
  passing checks and results. When a known consequential change to relevant
  code, runtime, inputs, or protocol invalidates a result, mark that result stale
  in the ledger and rerun only the affected required stage.
- Use the episode runner's normal entrypoint for required validations it already
  owns. Consume its recorded outcomes rather than duplicating contract, schema,
  input, or prerequisite checks in agent-side scripts. Validate a changed object
  or a concrete compatibility failure only where the runner does not cover it.
  Ordinary standalone work needs no new hashes, fingerprints, receipt manifests,
  or artifact inventories; preserve explicit frozen contract requirements.
- Run short contract-required conformance as a bounded stage within the already
  authorized substantive allocation, before its dependent scale step. Do not
  submit a separate job just for checks or pad resource use to justify one.
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

## ASTRA experiment program and handoff

ASTRA is the episode's scientific experiment program, not a plan/terminal
attachment. Use one ASTRA `analyses` entry per independently runnable or
evaluable experiment, with stable IDs and explicit artifact dependencies.
Keep scheduler state, attempts, job IDs, failures, and the exact next action in
`outputs/experiment_log.md`, keyed by those ASTRA IDs; these are not valid ASTRA
fields.

Keep the handoff sufficient to identify running work, completed artifacts, the
actual blocker, and the next authorized action without duplicating execution.

ASTRA and the ledger remain episode-local and non-authoritative. They cannot
grant execution authority, prove scientific validity, change canonical Brain
Researcher state, delay otherwise-authorized execution, or bypass the read
boundary.

Focused maintainer scenarios: [references/behavioral_evals.md](references/behavioral_evals.md).
