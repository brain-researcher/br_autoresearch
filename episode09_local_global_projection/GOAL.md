# EP09 — adaptive dendrite-to-distal-arbor prediction

This episode is governed by [the common adaptive protocol](../ADAPTIVE_SEARCH_PROTOCOL.md)
and is the morphology-search flagship.

## Authority and history boundary

This is the current local contract. It creates no canonical loop,
authorizes no download or compute, opens no morphology outcomes, and records
no finding. Prior EP09/10/11 work is exposure, not initialization evidence.
Before launch, bind canonical identities and create one shared
cell/group/duplicate/exposure ledger for all three morphology episodes.

EP09, EP10, and EP11 reuse the same SEU neurons and axonal outcomes. Their
audits are correlated, not independent replications. All three role manifests
and claims must be frozen before any shared audit group is opened; otherwise
only the first opening retains a sealed audit and later episodes become
outcome-exposed development studies.

## Adaptive scientific question

Which bounded representation of a neuron's **local dendritic morphology** adds
reproducible predictive information about qualifying **distal axonal-arbor
detection** in unseen biological groups, beyond soma location, source anatomy,
layer, labels, quality, and technical covariates?

The agent searches mechanisms—scale, topology, orientation, geometry, model
class, and calibration—not arbitrary feature code. The terminal comparison is
a capacity-controlled context model `M0` versus a paired `M1` that adds only
dendritic information.

## Outcome, unit, and data roles

For each eligible cell-target pair, the frozen observation rule yields
`detected`, `not_detected`, or `uncertain`; uncertain is missing, never zero.
The most conservative verified animal/brain/specimen group is the evidence and
split unit. Random-neuron splits are forbidden.

Before outcome inspection, assign whole groups to:

- **adaptive development:** at least 12 independent groups, used for feature,
  model, calibration, and hyperparameter search through nested grouped folds;
- **locked audit:** at least 8 independent groups, outcome-inaccessible until
  lock and opened once; and
- **technical-only:** groups failing outcome-blind completeness/support, never
  moved into development or audit after scores are seen.

If verified biological provenance cannot support these minima, the episode is
blocked or explicitly downgraded to exploratory grouped cross-validation. It
may not substitute cells for biological groups.

## Bounded scientific grammar

Every candidate is a declarative paired `M0/M1` pipeline composed from:

1. **Context block (`M0` and `M1`):** flexible soma coordinates, source parcel,
   cortical depth/layer, independently measured labels, acquisition/QC fields,
   and missingness indicators available prospectively. Axon-derived
   `Projection class` is prohibited.
2. **Dendrite blocks (`M1` only):** topology/branch order, Sholl or multiscale
   radial profiles, path-length and tortuosity geometry, orientation/extent,
   hemispheric symmetry, persistent-homology summaries, and explicitly
   normalized combinations. Native dendrites are primary; CCF-derived
   dendrites are a sensitivity, not a selectable replacement.
3. **Transforms:** robust scaling, signed log, block-wise normalization, PCA
   ranks in `{4, 8, 16, 32, 64}` where supported, or supervised PLS ranks in
   `{2, 4, 8, 16}` fit inside grouped training folds.
4. **Learners:** elastic-net logistic, group-sparse logistic, low-rank
   multi-target logistic, calibrated shallow gradient boosting, or a bounded
   additive spline model. Target-specific intercepts and context capacity are
   matched between `M0` and `M1`.
5. **Calibration:** none, Platt, or isotonic when grouped development support
   meets a frozen minimum.
6. **Ensembling:** at most two diversity-qualified candidates, with weights
   learned from out-of-group development predictions only.

The grammar forbids axonal features, outcome-derived subtype labels, evaluation
prevalence, pretrained representations with unresolved SEU lineage, arbitrary
neural embeddings, free-form feature generation, and target- or group-specific
winner selection.

## Objective and capacity control

For independent group `g`, average pairwise Brier loss within target and cell,
then define `D_g = Brier(M0)_g - Brier(M1)_g`. The primary objective is the
equal-group mean `Delta_Brier`; secondary objectives are worst-source-family
gain, log-loss, calibration error, sparsity, and stability.

`M0` and `M1` must share training groups, observable pairs, target intercepts,
context basis, calibration opportunity, tuning folds, and an explicit capacity
budget. A dendrite candidate is feasible only if it improves over both `M0`
and a capacity-matched nuisance/noise block, has no prespecified source family
worse by more than `0.002` Brier, and does not depend on one group. The smallest
useful effect `delta_morph` and interval procedure are frozen before audit.

## Search stages

1. **Measurement preflight:** authenticate cells/groups, native/CCF transforms,
   target readability, distal-arbor rules, support, and split sealing.
2. **Baseline stage:** establish context-only `M0`, prevalence, location-only,
   and capacity-matched nuisance controls under grouped folds.
3. **Coverage stage:** evaluate at least 20 diverse candidates covering every
   dendrite block, transform family, and learner family.
4. **Adaptive stage:** propose successors from the append-only evidence ledger;
   each trial states its mechanistic hypothesis and parent, and the trusted
   evaluator returns only approved grouped diagnostics.
5. **Falsification stage:** run block ablations, label and feature controls,
   batch/location matching, and leave-one-group influence on every promotion
   candidate.
6. **Lock/audit:** select one paired `M0/M1` pipeline, refit from immutable
   development inputs, freeze all measurement and inference code, hash the
   bundle, and open the group-sealed audit once.

## Required falsifiers and ablations

- dendritic-feature permutation within prespecified context strata and groups;
- group-preserving outcome permutation under target prevalence;
- capacity-matched random and smooth nuisance feature blocks;
- exact soma-location/source/layer matching or weighting;
- topology, Sholl, geometry, orientation, and multiscale block ablations;
- native-versus-CCF dendrite sensitivity without winner swapping;
- leave-one-group, leave-one-source-family, batch, and missingness influence;
- negative controls for local-shape scale and registration quality;
- synthetic recovery under null, context-confounded, sparse, distributed, and
  miscalibrated morphology effects; and
- lineage audit against MouseLight, ION, integrated releases, and any
  pretrained representation.

## Incumbent, budget, and stopping

The incumbent is nonterminal and development-only. It grants no claim and no
audit access. Run at least **40** and at most **96** valid paired-pipeline
trials. After the minimum, stop only after **18** consecutive valid trials
without a feasible Pareto improvement of at least `0.001` equal-group Brier
and after every feature/learner family plus all falsifiers have coverage. At
most 16 engineering failures may be retried outside scientific patience.

Resource envelope: CPU only; at most 1,500 aggregate CPU-hours, 120 wall-clock
hours, 64 concurrent cores, and 1 TB scratch. Resource exhaustion is
`incomplete_search`, not a negative biological finding.

## Lock, one-shot audit, and terminal boundary

The lock bundle includes source hashes, parsers, atlas/coordinate contract,
outcome rule, cell/group/duplicate/exposure ledger, group roles, grammar,
append-only trial ledger, all out-of-fold predictions, Pareto archive, chosen
`M0/M1`, capacity proof, calibration, `delta_morph`, intervals, multiplicity,
falsifiers, environment, seeds, and artifact hashes. Audit groups open exactly
once; the resulting score cannot alter representation, model, threshold, or
target set.

`candidate_ready` requires the frozen group-level interval to exceed
`delta_morph`, concordant prespecified source-family signs, no single-group or
batch dependence, capacity-matched control success, acceptable calibration,
and all decisive falsifiers passed. Otherwise report `closed_no_candidate`,
`unresolved`, or `technical_failure` under the lock.

A positive result supports only incremental prediction of the frozen distal
arbor observation in unseen groups from this source and atlas pipeline. It
does not establish synapses, causal growth rules, cell type, functional
connectivity, whole-axon fate, cross-species universality, or independence from
EP10/EP11. If biological group identity or negative-target observability is
unresolved, no neuron-level sample size can rescue the claim.
