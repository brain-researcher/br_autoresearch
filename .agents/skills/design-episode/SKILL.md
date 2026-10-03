---
name: design-episode
description: >-
  Create or revise a research episode from a question, dataset, or permitted
  prior evidence. Use for ideation, literature and novelty assessment, study
  design, paper framing, or conceptual figures; not experiment execution.
---

# Design an Episode

Develop a question worth answering and a study the available data can actually
support. This is the entrypoint for generating new EPs and revising existing
ones. It does not require a pre-existing `GOAL.md` or a formal BR campaign.

Apply only the parts needed for this request. Ideation, novelty assessment and
substantive redesign need the scientific reasoning below. A wording or figure
revision can reuse the agreed design and its current evidence; it does not
restart literature review, add a paper plan or change the research question.

## Establish the task, not a new bureaucracy

Read repository `AGENTS.md` and the user's scientific intent and prior decisions.
For an existing episode, read its current `GOAL.md`, `DATASETS.md`, search policy,
and specifically permitted design material. Old audits and conversation
summaries are leads to verify, not proof of current literature, data or state.
Do not scan outputs or open held-out outcomes to choose a more attractive story.

Distinguish proposing an idea, reviewing a design, implementing a revision, and
authorizing an experiment. A review remains read-only; a requested revision can
change design documents within scope. A new EP can start from an informal
question. Before creating a numbered directory, check the relevant registry
entry for collisions or reserved identities; do not renumber existing work or
register a formal campaign implicitly. If identity or scope is genuinely
undecided, propose it without creating an arbitrary numbered episode.

Read [study figures](references/study_figures.md) when creating or revising a
conceptual figure. Do not run execution checks merely to begin design work.

## Find the scientific contribution

Start with the phenomenon, what is unknown, and why a changed answer would
matter. Preserve the user's intended purpose: discovery, replication, methods,
boundary testing, or explanation. Do not replace a broad question with the
easiest available benchmark, or turn a replication into a discovery claim.

Read the closest primary papers, including the dataset's own paper and recent
work that could already answer the question. Use current retrieval rather
than memory for novelty claims. Distinguish published evidence from preprints
and uncertainty. Identify what the nearest work actually tested, what is
already known, and the specific remaining distinction our data could resolve.
A new model checkpoint, stricter split, or positive familiar comparison is not
automatically a new scientific finding; say when the value is replication or
methodological rigor instead. Do not promise publication from a positive result.

Use exposed, read-only BR literature/KG or scientific critique when it can
change the question, controls or interpretation. Reuse applicable evidence;
ordinary novelty review is not formal Society and does not require a loop.
Unavailable optional BR tools do not block ordinary primary-source research.
Send only permitted summaries, never restricted data or closed outcomes.

In a saturated field, look for an unresolved comparison, prediction, boundary
condition or explanation that matters. If the data cannot address one, state
that and recommend reframing, merging related work, or stopping. Do not add
methods just to make a narrow question look larger.

## Connect observations to claims

Identify the actual measured quantities, independent sampling unit, repeated
measures, relevant missing information, confounds and cross-episode exposure.
Check feasibility from permitted metadata and existing evidence, not from an
unrequested pilot or opened final set. Keep uncertainty about availability
distinct from verified absence. Metadata checks are not license for a data scan.

Make the intended claim match the measurements: representation similarity is
not automatically encoding or causality; prediction need not establish a
mechanism; simulation need not establish biological or hardware performance.
Use such distinctions where they change this study's interpretation, not as a
generic list of caveats. Consider detectability at the independent-unit level
and what a null result could and could not resolve. A power simulation or new
pilot is an explicit next experiment when needed, not a compulsory design gate.

Develop competing accounts and comparisons that can distinguish them. Specify
the relevant target/estimand, controls, evaluation support and uncertainty.
For component-based methods, explain each component's inputs, outputs, fitted
quantities and fitting data. Match information, capacity and tuning allowance
where the claim depends on a fair comparison. Add detail until the scientific
comparison is clear, not until every implementation choice is frozen.

## Keep the search open where it should be open

Use supported starting families and parameter ranges as a launch point, not
an arbitrary closed menu. Do not impose universal candidate counts, two-metric
limits, a fixed pilot-trial count, calibration prerequisites, or every available
research pattern. Use a pattern only when it helps this question.

Separate exploratory choices from data roles, resource limits, stopping rules
and final evaluation commitments. Existing frozen policies remain binding;
request or document an authorized amendment before changing them. Already-seen
outcomes cannot become unobserved confirmation after a redesign. Share paper
ideas across EPs only within scope; sharing a manuscript does not merge their
data permissions or make overlapping samples independent replications.

## Write one coherent study

Make the title, opening question, nearest-prior comparison, proposed test,
expected decision and concept figure tell the same story. Explain the science
first; put concrete limitations beside the claim they limit. Avoid defensive
administrative language dominating the narrative or artwork.

For a requested new EP with an agreed identity, create its core documents and
useful `inputs/README.md` and `outputs/README.md` navigation. State proposed
policy choices as proposed; file creation is not a freeze or launch approval.
For existing work, update living documents instead of adding review reports:

- `GOAL.md`: question, rationale/novelty, decisive comparisons and claim scope.
- `DATASETS.md`: observations, source roles, units, availability and limitations.
- The single search policy: operational choices and commitments the runner
  needs; distinguish proposed changes from frozen ones.
- Existing design/paper-plan and figure material under `outputs/`, when within
  the permitted design scope. Create a separate paper plan only if useful or
  requested; an EP need not support an independent paper by itself.

Use English text in repository artifacts. Keep filenames and entrypoints
stable where possible, and update their relevant links/captions after a
revision. A concept picture, paper outline or well-written contract is not a
result. Do not generate schema, SHA, provenance, preflight, review or status
files as default design deliverables.

End with the revised scientific question, evidence-backed contribution,
data-supported claim, remaining consequential choice, and next useful action.
Once execution is requested, hand the agreed design to `run-episode` without
restarting ideation. Design approval alone does not launch compute, open
outcomes, authorize Git mutations or invoke `manage-research-campaign`.
