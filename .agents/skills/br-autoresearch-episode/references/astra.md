# ASTRA experiment program

Use ASTRA as the episode's organized scientific worklist. The YAML is only its
serialization; the useful object is the set of experiments, their decisions,
their artifact dependencies, and the evidence they produce.

## Record responsibilities

| Record | Contains |
| --- | --- |
| `outputs/astra/v0.0.14/astra.yaml` | Stable experiment IDs, purpose, inputs, outputs, decisions, recipes, resources, dependencies, and findings |
| `outputs/experiment_log.md` | State, attempts, Slurm or process IDs, timestamps, observed failures, durable artifacts, blocker, and next action |
| `outputs/README.md` | Short human entrypoint with observation time, actual evidence stage, and links to the current log entry, run, results, and next action |

Key every execution-log entry by its ASTRA analysis ID. ASTRA 0.0.14 does not
define runtime status or job fields, so never add `status`, `job_id`, retries,
or `next_action` to `astra.yaml`.

This mirrors the original Brain Researcher pattern: a frozen experiment
worklist is separate from mutable, reconnect-safe execution state. These local
records are not canonical MCP objects.

The README is navigation, not another state store. Runner/scheduler observations
and the corresponding artifacts establish execution facts; the log records
them. If summaries disagree, use the relevant recent evidence or one targeted
state check, then correct the affected current summary. Do not infer authority
from a filename such as `LATEST` or `STATUS`, or reread the whole output tree.

Update the entrypoint at material transitions, not every poll or resume. Label
obsolete current-status sections as historical and link the active entrypoint;
preserve the underlying records. Keep job state out of `GOAL.md` and `DATASETS.md`.
Existing `memory.md`, `governance.md`, `society.md`, `loop.md`, `landscape.md`, and
`verification.md` retain their domain-specific decisions and links. Do not copy
one status paragraph into all six or create empty records unless a contract
requires them. An unavailable index or optional projection cannot delay a run.

## Make outputs easy to find

Reuse an intelligible existing layout. When a new run needs a directory, group
its configuration, scheduler logs, metrics, and results together, for example
`outputs/runs/<analysis-id>/<attempt>/`. Use the stable analysis ID for purpose
and the attempt for history; a job number alone does not describe the experiment.
Direct new Slurm logs into that run directory rather than the episode root.

Keep useful figures under `outputs/figures/`, reviews under an existing review
directory, and approvals/protocol amendments together. Create directories only
when they hold real artifacts. Keep reusable execution code in the existing
code/tools location and link it from a run instead of copying a source tree per
attempt. List current artifacts and historical attempts separately in the
README. Do not create manifests, receipts, duplicate archives, or new `STATUS`
files to organize the files.

Improve navigation first in an established episode. Preserve paths referenced
by frozen contracts, ASTRA, scripts, or active writers; moving or deleting old
artifacts is a separate scoped change. Apply the positive read boundary to
indexes and directory inspection too: no discovery scan of closed outcomes.

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
- Keep failed and superseded attempts in the execution log. Do not erase them
  by rewriting the scientific program as if they never happened.
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

## Execute and resume

Before a scheduler or process side effect, record the selected experiment and
attempt in `experiment_log.md`. After submission, record the returned job or
process ID. On completion or failure, append the observed outcome and artifact
refs; never infer completion from a missing job.

At resume:

1. Follow the existing index to the relevant ASTRA experiments and recent log
   entries within the current positive read boundary; use known paths directly
   if the index is missing or stale.
2. Reconcile active job state when needed for an execution decision. Reuse the
   current monitor observation; never submit a duplicate or repoll just for
   this checklist.
3. Select dependency-satisfied, authorized work. Advance independent experiments
   or supporting tasks in parallel where the contract and combined resource
   budget permit; a waiting branch need not stall the whole episode.
4. Give each submission and mutable artifact one owner. Consume the designated
   monitor's observations and integrate worker results in the shared log and
   entrypoint; do not duplicate jobs, polling, or per-agent status records.

Use existing ledger entries, completed artifacts, and passing runner checks on
an unchanged resume. Do not rerun ASTRA validation or export, refresh a legacy
projection, rewrite documentation, or create hashes or receipt inventories to
establish that nothing changed. A known consequential change invalidates only
the affected result: state that explicitly in the ledger, then continue from
the required affected stage. Update the ledger when execution state, evidence,
or the next action changes, rather than rewriting it merely because of a resume.

Short contract-required conformance runs before its dependent scale step within
the already authorized substantive allocation. An ASTRA analysis entry does not
require its own allocation or justify a checking-only job or padded resource use.

The handoff should make the active experiment, actual evidence stage, completed
artifacts, blocker, and exact next action easy to locate through stable IDs and
direct links. Reuse the current entrypoint and log instead of creating a dated
handoff document on every resume. A linked run can be synthetic or engineering
work; its completion alone is not a scientific finding.

## Compatibility boundary

The current `bin/astra-milestone` / `br-export-astra` adapter emits one lossy
plan or terminal snapshot. It does not export ASTRA `analyses`, recipes,
resources, universes, or the execution ledger. It may be retained as a legacy
interchange projection, but it does not create this experiment program and
must not be reported as a complete ASTRA handoff.

If program authoring or validation tooling is unavailable, report that gap and
continue otherwise-authorized episode work. ASTRA never opens sealed outcomes,
authorizes execution, or establishes scientific acceptance.
