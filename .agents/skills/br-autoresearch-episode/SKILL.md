---
name: br-autoresearch-episode
description: Run or resume one direct-child standalone br_autoresearch episode, including non-authoritative ASTRA plan and terminal milestone exports. Use for named OAK/Sherlock episode work; exclude canonical Society, reward, confirmation, and Landscape actions.
---

# BR Autoresearch Episode

Use this skill only for one explicitly named standalone episode. The repository
`AGENTS.md` and the episode's own contracts remain authoritative for workspace,
data-role, scientific, and execution boundaries. This skill adds the two ASTRA
interoperability milestones; it does not create a parallel lifecycle.

If the scientist explicitly requests canonical Brain Researcher binding,
Society, reward, confirmation, or a Landscape action, route that work through
`$brain-autoresearch-loop` instead. Do not infer canonical authority from a
standalone episode or its ASTRA file.

## Start or resume

1. Resolve the named episode as a direct child of the repository root.
2. Read `AGENTS.md`, `GOAL.md`, `DATASETS.md`, and the single
   `SEARCH_POLICY.yaml` or `SEARCH_POLICY.json` before doing episode work.
3. Confirm `inputs/` and `outputs/` exist and treat `inputs/` as read-only.
4. Export the plan milestone:

   ```bash
   episode_dir=$(cd -P -- "<absolute-episode-dir>" && pwd)
   repo_root=$(dirname -- "$episode_dir")
   "$repo_root/bin/astra-milestone" plan "$episode_dir"
   ```

The adapter selects the deployed `br-export-astra` launcher when available and
otherwise uses the repository entrypoint. Do not initialize modules, activate
the release venv, or rewrite `PYTHONPATH` yourself on Sherlock.

An ASTRA export failure leaves `ASTRA compliance: open`. Report the observed
error and continue work allowed by the episode contract. It is not a launch,
computation, or scientific-validity gate.

## Terminal closeout

Export terminal ASTRA only for a genuine episode terminal, after the final
scientific artifacts exist and before writing a terminal state marker or final
handoff.

1. Select one or more existing regular files below this episode's `outputs/`
   that actually support the terminal result. Prefer primary results; use a
   verification or report file only when it contains the relevant evidence.
2. State the observed finding, refutation, or limitation without strengthening
   the episode's scientific status.
3. Use the actual UTC closeout time in ISO 8601 form.
4. Run:

   ```bash
   episode_dir=$(cd -P -- "<absolute-episode-dir>" && pwd)
   repo_root=$(dirname -- "$episode_dir")
   "$repo_root/bin/astra-milestone" terminal "$episode_dir" \
     --output-ref outputs/<actual-artifact> \
     --finding "<observed finding, refutation, or limitation>" \
     --created-at <actual-UTC-timestamp>
   ```

Repeat `--output-ref` when several selected artifacts jointly support the
finding. Never invent a path, finding, or timestamp. Do not export a terminal
projection merely because one Slurm job ended; a job may be only an
intermediate episode step.

If terminal export fails, preserve the scientific terminal and original job
outcome. Report `ASTRA compliance: open` with the error; do not change reward,
Landscape, canonical state, or scientific acceptance to make the projection
pass.

## Handoff

Report the ASTRA stage, `outputs/astra/v0.0.14/astra.yaml` path, validation
warnings, or the exact open compliance error. Always label the projection as
non-authoritative.
