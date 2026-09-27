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

```text
PLAN -> CONFORMANCE -> SMOKE -> QUALIFICATION -> NEXT_AUTHORIZED_EXECUTION
                    failure -> REPAIR -> CONFORMANCE
consequential scientific change -> ASK_ONCE
```

- Before scaling synthetic work, verify that the active fitter and scorer match
  the frozen contract. Also verify each acceptance gate's estimand, reference,
  sampling unit, dependence assumptions, uncertainty method, and decision rule.
  A descriptive diagnostic is not an acceptance gate.
- Run the smallest useful smoke test, then one bounded qualification. Add
  another pilot only when new failure evidence requires it.
- When a test or audit finds a concrete defect, fix it next, add the focused
  regression test, and rerun the smallest affected stage. Do not re-audit
  unchanged code, rerun a failed qualification without a relevant change, or
  replace the repair with more documentation.
- Once required gates pass and execution is already authorized, submit the
  actual job or dependency chain in the same run. A launch requires a process
  or scheduler job ID; record it with dependencies and the next action. Prefer
  scheduler dependencies to login-node watchers. ASTRA, plans, placeholders,
  and dry runs do not count.
- For babysitting, follow authorized jobs to completion or the next real
  decision. Diagnose technical failures, repair them, and resume at the
  smallest affected stage. Never duplicate a pending job.

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
