# EP04 — adaptive shared scene geometry

## Status, protocol, and exposure boundary

This is the current local episode contract. It follows
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md) and does not
authorize outcome access, compute, or any canonical transition.

Previously opened regular-shared responses and model comparisons are exposed
development evidence and cannot become an audit. The current contract inherits
no live action, approval, reward, or scientific acceptance from prior work.

## Scientific question

Can an adaptive but bounded search learn a capacity-matched representation of
scene relations and context that predicts held-out high-level visual-cortex
geometry better than object-only, vision-only, or caption-only alternatives,
and does the locked representation retain that advantage under a genuinely
unseen image-distribution boundary?

The target is representational/predictive alignment. It is not identity of
coordinates, computation, syntax, mechanism, or evidence for a Platonic
representation. Five participants are biological observations; images,
voxels, RDM edges, sessions, and repetitions are not independent people.

## Evidence roles

- **Adaptive development and selection:** each participant's 4,712 regular
  subject-unique images, using image-disjoint, repetition-grouped folds. These
  outcomes may be searched repeatedly and remain development evidence.
- **Historical prior only:** 1,121 regular images shared by all five
  participants. Their responses were opened in the earlier round. They may be
  cited to motivate operators or used in a fixed reproduction report, but
  never score, rank, prune, or promote adaptive trials.
- **One-shot boundary audit:** the 371 shared OOD images only if a complete
  exposure ledger proves their neural responses and candidate-comparison
  summaries remained sealed through configuration lock. If that test fails,
  the OOD pool is development/prior evidence and audit requires a new,
  independently sealed compatible dataset or acquisition.

An OOD audit on the same five LAION-fMRI participants tests stimulus-shift
robustness, not participant-population generalization. A new independent
participant/data source is required for that stronger claim.

## Bounded scientific operator grammar

Each trial is a declarative pipeline. It may choose only registered values from
these operator families:

1. **Representation blocks:** human-caption MPNet; object/category inventory;
   OpenCLIP; DINOv2; PEcore; SigLIP2; or registered low-level image covariates.
   Use released, hash-pinned embeddings and metadata; training or fine-tuning a
   foundation model is out of scope.
2. **Caption construction:** mean of human-caption embeddings, token/word
   average, signed relation/action/object sub-blocks from one pinned parser, or
   a registered caption-minus-object residual. AI-generated captions are
   forbidden.
3. **Feature transform:** normalization only, train-fold PCA, train-fold PLS,
   whitened kernel, or covariance/spectrum matching. Dimension comes from a
   finite rank schedule and is fit without evaluation responses.
4. **Modality combination:** no fusion, residualized incremental block,
   banded/stacked ridge, or a convex blend of at most two already evaluated
   blocks. The same effective-capacity rule applies to every comparator.
5. **Neural target:** session-standardized, repetition-averaged GLMsingle beta
   patterns in a registered ROI; either cross-validated encoding or
   cross-validated representational-distance prediction. Preprocessing,
   repeat selection, and voxel reliability are training-fold operations.
6. **Readout:** ridge, fractional ridge, reduced-rank ridge, linear kernel
   alignment, or cross-validated RSA with correlation/crossnobis distance.
   Neural performance cannot select a new ROI definition.
7. **ROI family:** retinotopic EVC as a negative/specificity control and the
   frozen LAION ventral, lateral, and dorsal high-level sectors. A trial may
   not search arbitrary voxel masks.

Raw stimulus access, new detector execution, or new embeddings are outside
this grammar unless their DUA, provenance, frozen model, and complete coverage
are separately approved before search. Arbitrary code mutation, outcome-based
caption editing, per-audit-category model selection, and nearest-neighbour
access across folds are forbidden.

## Objective and constraints

The primary development objective is the participant-balanced mean gain in
held-out-image predictive correlation in the frozen high-level ROI family for
the candidate shared representation relative to the strongest
capacity-matched single-block comparator evaluated on the identical folds.
Correlations are Fisher transformed before participant/ROI aggregation; the
exact aggregation weights and practical minimum are frozen before search.

A promotable trial must also:

- improve in a frozen majority of participants and not depend on one ROI;
- retain a positive caption-context increment over object inventory;
- retain a positive shared-component increment over the best text-only or
  vision-only component;
- pass image-label permutation and capacity-matched random-feature nulls;
- remain directionally consistent under cluster/tau-like image separation and
  near-neighbour exclusion; and
- show the prespecified high-level-over-EVC specificity pattern as a hard
  development feasibility constraint. Because this constraint can reject a
  candidate, EVC is selection evidence even though it is never the primary
  objective or an adaptively searched ROI.

RSA agreement, retrieval, individual OOD categories, and ceiling-normalized
scores are secondary diagnostics. They cannot rescue a failed predictive
objective. The Pareto archive tracks prediction, robustness, complexity, and
participant consistency, but produces exactly one locked candidate.

## Adaptive loop

1. **Lineage and method gate:** bind release, beta/trial/image joins, exposure
   ledger, ROI masks, caption and embedding manifests, licenses, and synthetic
   reproductions of standardization, repetition averaging, RSA, and encoding.
   Before any neural score, verify image-ID-aligned caption, object/category,
   every retained visual embedding, and low-level covariate coverage on the
   **subject-unique development images**. Object segmentations reported only
   for shared images do not satisfy this gate. If the object comparator is
   absent, stop at readiness or approve and hash a frozen DUA-compatible
   extractor before outcome access; do not drop the object comparison later.
2. **Fold freeze:** export subject-unique development/selection image IDs,
   group all repetitions, block near neighbours, and reserve identical folds
   for every pipeline.
3. **Coverage stage:** evaluate fixed anchors spanning caption, object, each
   visual embedding, RSA, encoding, and one registered shared-component model.
4. **Adaptive stage:** one hypothesis-led operator change per trial; append the
   parent, mechanism, full DAG/hash, fold predictions, scores, constraints,
   runtime, and failure reason to the ledger. Update a nonterminal incumbent
   and Pareto archive.
5. **Successive fidelity:** prune only on frozen partial folds. Every promoted
   finalist is rerun across all five participants, eligible subject-unique
   images, registered ROIs, seeds, and mandatory falsifiers.
6. **Stress stage:** compare the incumbent and at most three challengers under
   source-stratum balance, nearest-neighbour exclusion, ROI specificity,
   caption-count matching, feature-capacity matching, and participant
   influence.
7. **Configuration lock:** choose one executable pipeline and freeze the code,
   environment, feature hashes, ROI/split manifests, rank/regularization,
   aggregation, nulls, thresholds, and expected output schema.
8. **One-shot audit:** an independent custodian verifies the OOD/new-data
   firewall, then runs the locked pipeline once. No OOD-category-specific
   tuning, candidate swap, threshold change, or second look is allowed.

The incumbent is never terminal during search. A promising score does not
permit stopping before minimum trials, coverage, falsifiers, stress tests, and
patience are complete.
At least 40% of valid post-coverage trials must be falsifiers, ablations,
negative controls, influence guards, synthetic recovery, or direct
replications. At least two outcome-adaptive successor cycles and two recorded
incumbent/challenger decisions are required before lock.

## Mandatory falsifiers and ablations

- whole-image label permutation preserving participant/session structure;
- caption-to-image shuffling and object-word-only caption construction;
- object block, relation/action block, text block, vision block, and fusion
  ablations;
- capacity-, covariance-, and spectrum-matched random features;
- near-neighbour exclusion and LAION/THINGS/THINGSplus source-stratified
  results;
- training-data-lineage/exposure sensitivity for each pretrained model;
- EVC negative/specificity control and leave-one-high-level-ROI influence;
- leave-one-participant influence and exact per-participant results;
- caption-count matching and repeat-selection sensitivity; and
- identical-fold comparison of RSA versus encoding conclusions.

Any mandatory falsifier failure blocks promotion regardless of aggregate
score.

## Budget and stopping

- minimum valid scientific trials: **24**;
- maximum valid scientific trials: **64**;
- improvement patience after the minimum: **10** valid trials;
- CPU ceiling: **2,500 core-hours**;
- GPU ceiling: **0 GPU-hours** because released embeddings are used;
- wall-clock ceiling: **120 hours** from the first scientific trial;
- finalists: at most **4** including the incumbent;
- audit openings: **1**;
- maximum parallel CPU cores: **48**;
- per-trial memory ceiling: **256 GB**;
- scratch-storage ceiling: **1,500 GB**.

Exact reruns after proven infrastructure failure are ledgered but are not new
hypotheses. They still consume resources. The search can stop early only for a
hard resource ceiling, technical impossibility, or policy violation.

## Terminal classes

- `candidate_ready`: one locked pipeline satisfies every development
  constraint and passes the one-shot sealed boundary audit.
- `closed_no_candidate`: the bounded search is valid but no feasible pipeline
  remains, or the locked candidate fails audit.
- `search_exhausted_no_audit`: deep development finishes but neither sealed OOD
  responses nor new independent compatible data are available.
- `technical_failure`: source identity, legal access, stimulus-response joins,
  folds, ROI/beta validity, or executable evaluation cannot be established.
- `policy_violation`: prohibited outcome access, leakage, undeclared operators,
  or post-audit adaptation invalidates the round.

For canonical outer status, `candidate_ready` maps to `candidate_ready`;
`closed_no_candidate` and `search_exhausted_no_audit` map to
`closed_no_candidate`; and `technical_failure` or `policy_violation` map to
`technical_failure`.

## Claim boundary

A successful sealed LAION OOD audit supports a shared representation's
predictive robustness across the prespecified image shift in these five
participants and ROIs. It does not establish a universal modality-general
space, mechanism, participant-population generalization, foundation-model
scaling law, or independent replication. Only an independently sealed new
dataset can extend that boundary.
