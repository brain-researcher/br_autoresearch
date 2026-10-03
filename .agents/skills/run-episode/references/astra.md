# ASTRA experiment program

Use ASTRA for the episode's experiments, scientific decisions and artifact
dependencies. Execution and resume behavior lives in [SKILL.md](../SKILL.md).

## Record responsibilities

| Record | Contains |
| --- | --- |
| `outputs/astra/v0.0.14/astra.yaml` | Stable experiment IDs, purpose, inputs, outputs, decisions, recipes, resources, dependencies, and findings |
| `outputs/experiment_log.md` | State, attempts, Slurm or process IDs, timestamps, observed failures, durable artifacts, blocker, and next action |
| `outputs/README.md` | Short human entrypoint with observation time, actual evidence stage, and links to the current log entry, run, results, and next action |
| `GOAL.md`, `DATASETS.md`, search policy | Scientific question, data roles and execution contract; no live job status |

Key every execution-log entry by its ASTRA analysis ID. ASTRA 0.0.14 does not
define runtime status or job fields, so never add `status`, `job_id`, retries,
or `next_action` to `astra.yaml`.

Runner/scheduler observations and their artifacts establish execution facts;
the log records them and the README provides navigation. If summaries disagree,
use relevant recent evidence or one targeted check, then correct the affected
summary. A filename such as `LATEST` or `STATUS` grants no authority and does
not justify scanning the output tree. Preserve failed and superseded attempts.

Update the log when execution state, evidence or the next action changes and
the README at material transitions. Link current artifacts separately from
historical attempts; label obsolete status sections as historical and point
to the active entrypoint. Reuse these records for handoff instead of creating
dated reports, per-agent status files or empty placeholders.

Preserve `memory.md`, `governance.md`, `society.md`, `loop.md`, `landscape.md` and
`verification.md` for their domain-specific decisions and links; update only
when their subject changes or a contract requires it. Do not duplicate live
status across them or list them as `candidate_bundle.json` `output_artifacts`.
`experiment_log.md` is optional supplemental bundle context only under the
project-local `$manage-research-campaign` contract. These are local records,
not canonical MCP evidence or transition gates.

## Make outputs easy to find

Reuse the existing layout. Group new configuration, scheduler logs, metrics and
results under, for example, `outputs/runs/<analysis-id>/<attempt>/`; stable IDs
describe purpose and attempts preserve history. Direct new Slurm logs there.

Keep figures under `outputs/figures/` and approvals/protocol amendments together.
Create directories only for real artifacts. Link reusable execution code in its
existing code/tools location instead of copying it per attempt. Navigation needs
no new manifests, receipts, duplicate archives or `STATUS` files.

Preserve paths referenced by frozen contracts, ASTRA, scripts or active writers;
moving or deleting old artifacts requires a separate scoped change. The skill's
positive read boundary covers indexes and directory inspection too. Missing
navigation or an optional projection does not delay authorized execution.

## Build the program

- Make the root ASTRA `Analysis` describe the episode.
- Put each independently runnable or evaluable experiment under `analyses:`.
  Use a stable snake_case ID. Do not split by script or invent work that the
  episode contracts do not require.
- Include a contract-required check or gate as an analysis when it is itself
  independently runnable or evaluable, even if engineering or synthetic.
  Runner-internal checks stay inside their owning experiment; do not duplicate
  them as agent tasks or separate checking-only scheduler jobs.
- Give each experiment a short purpose, declared inputs and outputs, and its
  real methodological decisions. Add a recipe and resource request when the
  experiment is executable from the record.
- Express dependencies through artifacts. A downstream experiment consumes a
  sibling result with an input such as
  `from: ../planned_baseline.baseline_report`; ASTRA has no `depends_on`
  field.
- Use universes for defensible decision combinations, not attempts, statuses,
  bug fixes, or versions.
- Add an ASTRA `finding` only when a declared output supports a scientific
  claim. A run completing or a gate passing is execution state by itself.

A compact shape for a baseline and scientific comparison already required by
the episode contract is below. Replace these entries with the actual authorized
experiments; the example adds no baseline, qualification, or other gate.

```yaml
version: "0.0.14"
name: Episode experiment program
inputs:
  - {id: goal, type: data, source: GOAL.md}
outputs:
  - id: final_report
    from: planned_comparison.comparison_report
analyses:
  planned_baseline:
    name: Run the declared baseline analysis
    inputs:
      - {id: goal, from: ../goal}
    outputs:
      - id: baseline_report
        type: report
        format: json
        inputs: [goal]
        recipe:
          command: python baseline.py --out {output}
          resources: {cpus: 1, memory: 4Gi, time_limit: 30m}
  planned_comparison:
    name: Run the declared scientific comparison
    inputs:
      - {id: goal, from: ../goal}
      - id: baseline_report
        from: ../planned_baseline.baseline_report
    outputs:
      - id: comparison_report
        type: report
        format: json
        inputs: [goal, baseline_report]
        recipe:
          command: python compare.py --baseline {inputs.baseline_report} --out {output}
```

Large experiments may use `path:` entries that point to directories containing
their own `astra.yaml`. Do not combine `path:` with inline content.

When creating or changing the program, validate the changed ASTRA objects and
affected cross-analysis references with the pinned ASTRA 0.0.14 schema and
semantic validator. Also check an object when a concrete compatibility error
requires it. Reuse validation already performed by the runner; do not revalidate
unchanged objects or every frozen contract. ASTRA validation is not an execution
gate; unavailable tooling leaves compliance open.

## Compatibility boundary

The current `bin/astra-milestone` / `br-export-astra` adapter emits one lossy
plan or terminal snapshot. It does not export ASTRA `analyses`, recipes,
resources, universes, or the execution ledger. It may be retained as a legacy
interchange projection, but it does not create this experiment program and
must not be reported as a complete ASTRA handoff.

If program authoring or validation tooling is unavailable, report that gap and
continue otherwise-authorized episode work. ASTRA never opens sealed outcomes,
authorizes execution, or establishes scientific acceptance.
