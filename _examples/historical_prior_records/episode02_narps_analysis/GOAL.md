# NARPS Analysis-Decision Landscape v2

## Mission

This is a bounded, exploratory successor to the completed NARPS v1 Goal exploration.
Using the supplied public `ds001734` sources, determine which technically and
scientifically defensible explanation best accounts for v1's unexpectedly
large smoothing contribution to continuous gain- and loss-related group-map
variation. A useful terminal result may either show that smoothing sensitivity
persists under a mechanism-focused analysis or show that the v1 pattern is
substantially explained by a defined numerical, masking, or sampling mechanism.

This workspace must not claim independent confirmation, a biological neural
mechanism, or a causal effect of quality control. All new artifacts belong
under `outputs/`; `inputs/` is read-only.

## Scientific lineage and non-repetition rule

Parent loop: `arl_8cbd25a7a8610a008c2830ba624052c4`  
Parent Goal handoff: `goal_handoff_e3f79fc20a54776ebf304655`  
Parent terminal state: `closed_no_candidate`, `exploratory_only`

The exact v1 question was:

> Using public NARPS `ds001734` data, are gain- and loss-related task-fMRI
> group effect maps more sensitive to confound handling and outcome-blind
> subject exclusion than to spatial smoothing?

Its frozen support rule required, for at least one contrast, both `C/S > 1`
and `Q/S > 1`, with both simultaneous interval lower bounds above 1. The v1
directional hypothesis was falsified under its frozen raw map-space estimand
because no contrast met that complete rule. For `gain_demean`, `C/S=0.0758674`
and `Q/S=0.0427427`, with simultaneous lower bounds 0.0608696 and 0.00244018;
for `loss_demean`, `C/S=0.0863375` and `Q/S=0.0304137`, with lower bounds
0.0661793 and 0.00173066. Smoothing's attributed share was about 0.894 and
0.8955, respectively.

Do not rerun the v1 3 x 3 x 3 grid merely to restate this result. The curated
parent packet is prior exploratory evidence that may generate successor
mechanisms. It is not v2 validation, independent confirmation, or a substitute
for new source-derived analysis. Because v1 used the full dataset, any
within-`ds001734` split or resampling analysis remains internal robustness,
not an untouched confirmation set.

## Competing mechanism candidates

Before reading new source BOLD values, effect-map values, or group-map values,
write a short `outputs/mechanism_selection.md` that ranks these candidates by
mechanism specificity, falsifiability, available inputs, and computational
cost. It may retain at most two candidates for frozen testing.

1. **Substantive spatial sensitivity:** Gaussian smoothing genuinely changes
   group-map geometry enough to dominate the other v1 decision families.
2. **Amplitude-versus-shape dependence:** the v1 functional-ANOVA result is
   dominated by map amplitude/scale differences even where spatial pattern
   similarity is relatively preserved.
3. **Mask-edge and estimator mechanism:** low-intensity boundary voxels,
   mean-scaling behavior, or inverse-variance fixed effects contribute
   disproportionately to smoothing attribution.
4. **Task-version or sample-composition mechanism:** EI/ER design differences
   and the ER-only frozen exclusions alter the apparent ranking without a
   general smoothing effect.
5. **Continuous-versus-thresholded inference disconnect:** a smoothing effect
   on continuous geometry is not commensurate with a conclusion-stability
   effect after a fixed thresholding procedure.

The selection note may use source metadata and the curated parent packet, but
not new neural outcome values. It must state why unselected candidates are not
tested in this episode.

## Stage 0: source and resource inventory

First produce `outputs/source_inventory.md` and a machine-readable inventory
that verify, without changing sources:

- subject, run, task-version, events, confound, mask, space, and contrast
  availability;
- readability and semantic compatibility of the supplied raw, fMRIPrep, and
  FitLins trees;
- which existing maps or designs can support each candidate and whether a
  bounded recomputation is actually necessary;
- available Sherlock compute, storage, and an execution plan that keeps
  durable artifacts in this workspace and high-frequency intermediates on
  scratch when needed.

Missing optional FitLins material is not automatically a technical failure if
the needed evidence can be obtained from the other supplied sources. Do not
download a replacement dataset or introduce an external analysis-team bundle.

## Stage 1: freeze a successor test contract

After Stage 0 and candidate selection, but before reading new source neural
outcomes for the selected tests, create both
`outputs/frozen_successor_contract.md` and
`outputs/frozen_successor_contract.json`. The contract must name no more than
two mechanism tests and, for each, specify:

- the target contrast and estimand;
- a directional prediction, SESOI or practical decision threshold, and
  falsifier;
- the exact comparison layers, masks, weighting/normalization choices,
  subject handling, and fixed random seeds where applicable;
- a matched negative control and the strongest alternative explanation it
  discriminates against;
- which quantities are confirmatory within this exploratory episode and which
  are post-hoc descriptive;
- stop conditions, resource cap, and the required terminal artifacts.

Analyze `gain_demean` and `loss_demean` separately. Do not use a pooled EI/ER
gain-minus-loss contrast as the primary result. Any EI-versus-ER result is
task-version-associated and exploratory.

## Stage 2: bounded mechanism tests

Use existing FitLins statistics or fMRIPrep data first. A BOLD-level
recomputation is allowed only if the frozen selected test requires it to
adjudicate an identified mechanism. Do not expand into a new unrestricted
Cartesian multiverse: there may be at most two selected mechanism tests and no
replacement candidate after their neural outcomes are inspected.

Each selected test must include a directly comparable control, preserve
subject-level provenance, and distinguish continuous-map geometry from
thresholded conclusion stability. Interpret any QC-set difference as
analysis-set/sample-composition sensitivity, not as a causal QC effect.

## Stage 3: internal robustness and scientific critique

For a result to become the sole `candidate_ready` finding, it must survive its
frozen internal robustness check and the specified negative control. The
episode must also document:

- the strongest remaining alternative explanation;
- whether evidence is driven by mask periphery, scale, task version, or a
  small set of subjects;
- an outcome-blind follow-up or held-out-style internal check when feasible;
- why the conclusion is methodologically informative beyond the v1 null; and
- the minimal independent confirmation design that would be required next.

If no selected mechanism passes its predeclared test, preserve the ranking and
report a meaningful `closed_no_candidate` result rather than widening the
search after outcome access.

## Resource and terminal rules

Keep the Goal exploration bounded to the Stage 0 inventory plus at most two frozen
mechanism tests. Stop when those tests and their controls are complete, or
when a stated technical blocker prevents completion. Do not modify the parent
workspace or its frozen contract.

The terminal `CandidateBundleV1` must declare
`schema_version: br.autoresearch_goal_candidate_bundle.v1`,
`isolation_assurance: client_unattested`, and exactly one of:

- `candidate_ready`: one reproducible exploratory methods candidate with its
  prediction, falsifier, primary test, controls, limitations, and independent
  confirmation requirement;
- `closed_no_candidate`: completed bounded tests with the ranked, rejected,
  or inconclusive mechanisms and their evidence; or
- `technical_failure`: a source, estimability, runtime, or storage blocker
  that prevents the frozen selected test from completing.

Every terminal bundle remains `scientific_status: exploratory_only`,
`confirmation_eligible: false`, and `scientific_acceptance: false`, with
`confirmation_requirement: independent_fresh_confirmation_required`.
Only a valid `candidate_ready` handoff may enter governed Society review.
`closed_no_candidate` and `technical_failure` must reconcile the outer loop to
`COMPLETE` with Society not eligible and not called; they must not create a
reward, launch approval, canonical execution, or Landscape transition.

## Required outputs

At minimum, write `source_inventory`, `mechanism_selection`, the frozen
successor contract, analysis manifests, result tables/figures, a concise
`RESULT.md`, and a schema-valid terminal candidate bundle under `outputs/`.
During work, maintain `outputs/experiment_log.md`, `outputs/memory.md`, and
`outputs/governance.md` as human-readable projections. Do not create a separate
`outputs/society.md`; a frozen Society review is an external governed step only
for a valid `candidate_ready` handoff.
