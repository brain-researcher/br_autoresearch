# Does native VLM–brain alignment survive image-distribution shift?

## Status, protocol, and exposure boundary

This is the current local episode contract. An explicit scientist instruction
naming EP04 may start bounded episode work. It uses the evidence-separation,
configuration-lock, and audit rules from
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md); it does not
run an adaptive primary-model search and alone does not authorize
protected-outcome access or a Brain Researcher transition.

Previously opened regular-shared responses and model comparisons are excluded
exposed evidence and cannot become development data or an audit. The current
contract inherits no live action, approval, reward, or scientific acceptance
from prior work.

## Scientific question

Can a frozen vision-language-model (VLM) image embedding retain an advantage
over DINOv2 in its direct alignment with high-level visual-cortex geometry
across an unseen image-distribution boundary, including differences **within**
the prespecified OOD categories? The complete fixed panel first selects one
native VLM geometry on image-disjoint regular-image responses; the locked OOD
test then asks whether its advantage survives. A registered
low-level geometry is a required nuisance control. Caption and object
geometries are optional secondary semantic anchors, not alternative primary
endpoints.

### What this could add beyond another model ranking

VLM–brain correspondence is already established. For example, a
[2026 Nature Human Behaviour study](https://www.nature.com/articles/s41562-025-02357-5)
compares CLIP with vision models across multiple fMRI datasets and adds
brain-lesion evidence; [BrainSAIL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/73af055566f5514b9863315133b84eda-Paper-Conference.pdf)
compares CLIP, DINO, and SigLIP encoding representations. A newer checkpoint or
one more positive RSA is not sufficient novelty.

EP04's candidate contribution is a **native-geometry boundary test**: the same
locked VLM must exceed the same DINOv2 comparator on within-category OOD
geometry, where broad separation between OOD categories cannot produce the
primary score. This is the existing OOD endpoint, not a new architecture,
reproduction requirement, or learned brain mapping. It does not identify why
multimodal training helps, and within-category prediction is not automatically
semantic or independent of low-level visual structure.

Development and OOD use different image supports and aggregations; the audit
tests whether the registered advantage criterion is met again, not a causal
shift effect, a percentage of alignment retained, or a formal ID–OOD interaction.
A precise model-ranking result alone remains a bounded benchmark. A broader
paper needs informative evidence about this boundary.

The primary target is direct geometry-to-geometry alignment between each
release-supplied native VLM image embedding and fMRI activity patterns, not
reproduction of an earlier analysis and not prediction for its own sake.
Alignment does not imply identity of coordinates, computation, syntax,
mechanism, or a shared Platonic representation. Five participants are the
biological observations; images, voxels, RDM edges, sessions, and repetitions
are not independent people.

### Conceptual figure

![EP04: direct native RSA and within-category image-shift test](outputs/ep04_conceptual_question-v2.png)

The same image identities generate release-supplied VLM and DINOv2 embedding
RDMs and participant fMRI response RDMs. Native cosine model geometry is
compared directly with correlation-distance neural geometry by Spearman RSA,
not by learning a model-to-brain coordinate mapping. The fixed-panel contrast
is Fisher-transformed VLM-to-brain RSA minus DINOv2-to-brain RSA, with the
registered low-level control and five equally weighted participant effects.
Development holds out images within participants, not people. The conditional
371-image OOD audit tests within-category geometry in the same five participants;
1,121 previously exposed regular-shared images are excluded. Panel D's A/B/C
cards are abstract category/exemplar placeholders, not actual OOD images or
classes. The audit repeats an advantage criterion, not an ID–OOD interaction.
[Exact imagegen prompts](outputs/ep04_conceptual_question-v2-prompt.md).

This is an image-generated scientific schematic: thumbnails, colored feature
strips, RDM entries and brain shading are illustrative, not source data or
results. No winning model is shown. Unmatched checkpoints cannot identify a
causal language/multimodal-training effect, and similarity does not establish
shared mechanism, neural specificity or participant-population generalization.

## Evidence roles

- **Fixed-panel development and selection:** each participant's 4,712 regular
  subject-unique images, using image-disjoint, repetition-grouped folds. These
  outcomes compare the complete prespecified panel and remain selection-exposed
  development evidence.
- **Excluded exposed slice:** 1,121 regular images shared by all five
  participants. Their responses were opened in the earlier round. No
  reproduction analysis is required or planned; these responses never score,
  rank, or select a panel candidate.
- **One-shot boundary audit:** the 371 shared OOD images only if a complete
  exposure record establishes that their neural responses and
  candidate-comparison summaries remained sealed through configuration lock.
  If that test fails,
  the OOD pool is development/prior evidence and audit requires a new,
  independently sealed compatible dataset or acquisition.

An OOD audit on the same five LAION-fMRI participants tests stimulus-shift
robustness, not participant-population generalization. A new independent
participant/data source is required for that stronger claim.

## Bounded scientific analysis grammar

Each fixed panel row or diagnostic may use only the following registered
definitions:

1. **Primary image geometries:** one release-supplied, version-identified final
   pooled image embedding from each of OpenCLIP, PEcore, and SigLIP2. L2
   normalization followed by cosine distance defines each native model RDM.
   Layer search, neural fitting, foundation-model training, and fine-tuning are
   outside the primary grammar.
2. **Primary comparator:** one release-supplied final pooled DINOv2 image
   embedding, frozen before neural scoring and evaluated with the identical L2
   normalization and cosine-distance rule. It is a strong fixed vision-only
   comparator, not an architecture-, parameter-, or training-data-matched
   causal control.
3. **Controls and anchors:** registered low-level image geometry is a required
   nuisance control. Mean human-caption MPNet and object/category geometries
   are optional secondary semantic anchors; missing caption or object coverage
   cannot block the primary VLM-versus-DINOv2 test. AI-generated captions are
   forbidden.
4. **Geometry sensitivities:** train-image-only PCA, whitening, and
   covariance/spectrum matching are labeled sensitivity analyses. They may not
   replace the native cosine RDM, select the winner, or rescue a failed primary
   endpoint. No neural-outcome-supervised transform or learned fusion may
   define a model geometry.
5. **Secondary predictive analysis:** train-fold PLS, ridge, fractional ridge,
   reduced-rank ridge, or banded ridge may be used for the identical-fold
   encoding diagnostic only. Encoding never chooses or rescues the primary VLM
   candidate.
6. **Neural geometry:** session-standardized, repetition-averaged GLMsingle
   beta patterns in a registered ROI, requiring at least two valid
   presentations per image. The primary neural RDM uses correlation distance
   on image-disjoint evaluation images. Correlation-versus-cosine neural RDM is
   an always-evaluable sensitivity; crossnobis is an optional sensitivity only
   where independent repeat partitions make it valid. Preprocessing, repeat
   selection, voxel reliability, and nuisance fitting are training-fold
   operations.
7. **Readout:** the primary readout is image-split cross-validated RSA between
   the upper triangles of the native model RDM and fMRI response RDM, using a
   frozen Spearman correlation and Fisher-z aggregation. Linear kernel
   alignment and cross-validated ridge/reduced-rank encoding are secondary
   convergence diagnostics. Neural performance cannot select a new ROI.
8. **ROI family:** the frozen LAION ventral, lateral, and dorsal high-level
   sectors define the primary ROI family. Retinotopic EVC is a descriptive
   control whose raw RSA is reported alongside frozen ROI reliability/noise-
   ceiling information; it is not a hard specificity or candidate-lock gate.
   An evaluation may not search arbitrary voxel masks.

Raw stimulus access, new detector execution, or new embeddings are outside
this grammar unless their DUA, source/model identity, frozen settings, and
complete coverage are separately approved before fixed-panel scoring.
Arbitrary code mutation, outcome-based caption editing, per-audit-category
model selection, and nearest-neighbour access across folds are forbidden.

## Objective and constraints

For candidate model `m`, participant `p`, image fold `f`, and frozen high-level
sector `r` (ventral, lateral, or dorsal), let `A_dev(m,p,f,r)` be the
Fisher-transformed Spearman correlation between the upper triangles of the
native model RDM and fMRI response RDM on the image-disjoint evaluation fold.
Define the participant-level VLM gain over DINOv2 as
`G_dev(m,p) = mean_f,r[A_dev(m,p,f,r) - A_dev(DINO,p,f,r)]`, with equal fold
and sector weights. The development selection score is the unweighted mean of
the five participant effects.

All development fold scores are used once in the fixed-panel candidate
decision. They are out-of-fold within a model evaluation but selection-exposed
because the same development results choose the winner, so they are not
described as an unbiased final test. The decision order is fixed: compute all
three `G_dev` scores; construct each family-wise null by taking the
unconditional `argmax G_dev` across all three VLM rows in every null draw;
screen every observed row against those null thresholds and all other gates;
then select the highest-`G_dev` passing row. If no row passes, close without a
candidate. No gate is applied inside the null-draw argmax, so the rule is not
circular.

No layer, distance, transform, rank, caption, object, or encoding result may
replace that primary candidate definition. A frozen tie is broken by lower
native embedding dimension and then lexicographic released model identifier;
a tie means an absolute score difference no larger than `1e-6` Fisher-z units.

The practical VLM-over-DINOv2 margin is `delta_min = 0.02` Fisher-z units. It
is the smallest scientifically worthwhile increment—approximately a two-
percentage-point correlation difference near zero—so dependent image pairs
cannot make a negligible advantage look important; it is not a population-
significance threshold. A development row is eligible only if its mean
gain reaches `delta_min`, at least four of five participant gains are positive,
every leave-one-participant-out mean is positive, and every
leave-one-high-level-sector-out mean is positive. Its positive unsubtracted
VLM-to-brain RSA must
exceed the 95th percentile of 10,000 frozen whole-image label permutations,
and its alignment must exceed the 95th percentile of 1,000 dimension- and
pre-L2-covariance/spectrum-matched random geometries. The low-level partial-RSA,
source-stratum,
near-neighbour, repeat/missingness, and neural-distance sensitivity rules below
must also pass.

Images are split before RDM construction; RDM edges sharing an image are never
split across folds or treated as independent inferential units. Primary native
model geometry has no neural fit. Every fMRI preprocessing, reliability, or
nuisance operation and every secondary transform/readout is fit on training
images only. Label permutations act jointly on model-RDM rows and columns at
the whole-image level within the registered source stratum; RDM edges are never
permuted independently.

The five participant effects are the complete biological evidence: each has
weight `1/5`, all five values and leave-one-participant-out results are
reported, and conventional participant-population significance is not claimed.
The participants' development images differ, so development heterogeneity
mixes participant and stimulus-sample variation. The shared OOD audit holds
stimuli fixed across participants. Resampling whole images or frozen
near-neighbour clusters quantifies stimulus uncertainty only; it does not turn
images or RDM edges into additional participants.

Encoding accuracy, linear kernel alignment, retrieval, caption/object RSA,
EVC comparisons, transformed-geometry RSA, optional crossnobis, and any valid
ceiling-normalized scores are secondary diagnostics. They cannot rescue a
failed native-RSA objective. RSA/encoding disagreement is reported as readout
dependence rather than silently averaged away. Raw high-level-versus-EVC
differences are descriptive and are interpreted alongside the frozen
reliability context, not as neural specificity. The fixed decision table
reports alignment, robustness, and participant consistency, and produces at
most one locked VLM candidate.

## Locked OOD estimand and pass rule

Before neural outcomes are opened, the locked candidate, DINOv2, registered
low-level geometry, category labels, and common valid-image coverage must be
complete for all 371 OOD images. The VLM and DINOv2 are then applied without
fitting. For each participant `p`, high-level sector `r`, and prespecified OOD
category `c`, `A_ood(m,p,r,c)` is the Fisher-transformed Spearman RSA computed
from only the within-category RDM edges. Categories and sectors receive equal
weight, so
`G_ood(p) = mean_r,c[A_ood(VLM,p,r,c) - A_ood(DINO,p,r,c)]` and the audit
summary is the unweighted mean of the five participant effects. The all-image,
all-pair RDM is reported only as a secondary diagnostic because between-
category separation could dominate it.

The OOD audit passes only if the mean `G_ood` reaches `delta_min = 0.02`, at
least four of five participant effects are positive, and the aggregate remains
positive after leaving out each participant, high-level sector, or OOD category
in turn. Positive unsubtracted VLM-to-brain RSA must also exceed the 95th
percentile of 10,000 whole-image label permutations performed within OOD
category, with each permutation shared across participants and ROIs. After
partial Spearman control for the locked low-level RDM within category, the
participant-balanced
VLM-minus-DINOv2 gain and every leave-one-participant-out gain must remain
positive. All categories are evaluated in one release; no category-specific
candidate choice, threshold, exclusion, or second look is allowed. Missing
pre-open feature/category coverage or a structurally unusable prespecified
category makes the audit technically unevaluable rather than silently changing
the estimand.

## Fixed-panel comparison and lock procedure

1. **Readiness gate:** identify the release, verify beta/trial/image joins,
   evidence-slice exposure status, ROI masks, feature availability, licenses,
   and focused checks of standardization, repetition averaging, RSA, and
   encoding. Bind exactly one released final pooled embedding and model
   identifier for each named VLM family and for DINOv2; if multiple arrays are
   released, use only the documented default final pooled array or stop for an
   outcome-blind scientist choice. No layer grid remains open after this gate.
   Before any primary neural score, verify image-ID-aligned VLM, DINOv2, and
   low-level-covariate coverage on the **subject-unique development images**.
   Record caption/object coverage separately; it affects only the labeled
   secondary anchors and does not block the primary analysis.
2. **Fold freeze:** export subject-unique development/selection image IDs,
   group all repetitions, block near neighbours, and reserve identical folds
   for every model and diagnostic.
3. **Complete primary panel:** evaluate all three native VLM geometries and the
   fixed DINOv2 geometry on every frozen fold, participant, and high-level ROI.
   No partial-fold pruning or outcome-created primary candidate is allowed.
4. **Locked controls and diagnostics:** run the low-level control, nulls,
   influence checks, distance/repeat sensitivities, identical-fold encoding,
   EVC context, and every available caption/object anchor.
5. **Single candidate decision:** construct the family-wise null thresholds,
   screen all three observed rows against every frozen constraint, and use the
   tie-breaker to choose the highest-scoring passing row. If no VLM passes,
   close without a candidate; do not repair it with a transformed geometry or
   secondary endpoint.
6. **Full rerun:** recompute the chosen row once on all eligible development
   images, participants, high-level ROIs, fixed seeds, and mandatory
   falsifiers. Any mismatch is a technical failure, not a new primary row.
7. **Configuration lock:** choose one executable pipeline and record a
   write-once lock identifying its code/model versions, native embedding,
   normalization/distance, ROI/split/category IDs, aggregation, nulls, and
   thresholds.
8. **One-shot audit:** an independent custodian first verifies complete VLM,
   DINOv2, low-level, category, and valid-image coverage without opening neural
   outcomes, then runs the locked within-category estimator and diagnostics
   once. No OOD-category-specific tuning, candidate swap, threshold change, or
   second look is allowed.

Panel completeness, every mandatory falsifier, both required diagnostics, the
full rerun, and the single candidate decision are required before lock. Nulls,
sensitivities, and diagnostics are not counted as extra primary candidates.

## Mandatory falsifiers and ablations

- **Image-label null:** each of 10,000 deterministic permutations relabels
  model-RDM rows and columns relative to the fixed neural RDM, applies the same
  source-stratified whole-image permutation to all three VLMs and DINOv2, and
  takes the unconditional `argmax G_dev` with the frozen tie-breaker, without
  applying any gate. The resulting family-wise 95th-percentile threshold then
  screens every observed VLM row's positive, unsubtracted RSA—not its absolute
  value.
- **Random-geometry null:** each of 1,000 deterministic draws replaces every
  VLM embedding matrix with its seeded SVD/Haar random-geometry analogue,
  takes the same unconditional `argmax G_dev`, and records that row's positive,
  unsubtracted RSA. Its family-wise 95th-percentile threshold screens every
  observed row.
- **Vision-only comparator:** the VLM-minus-DINOv2 mean must reach
  `delta_min = 0.02`, with at least four positive participant effects and all
  participant/sector leave-one-out means positive.
- **Low-level nuisance control:** after fixed partial Spearman control for the
  registered low-level RDM, the VLM-minus-DINOv2 aggregate and every
  leave-one-participant-out aggregate must remain positive.
- **Stimulus structure:** the gain must remain positive after frozen
  near-neighbour-cluster exclusion and under equal weighting of the registered
  LAION/THINGS/THINGSplus development strata.
- **Repeat and distance sensitivity:** the gain must remain positive on the
  frozen complete-repeat subset and when cosine replaces correlation distance
  for the neural RDM; missing repeats follow the pre-score fixed rule.
- **Influence reporting:** exact effects for all five participants and all
  three high-level sectors are required; every registered leave-one-out gate
  above must pass.

An identical-fold encoding analysis and raw EVC RSA with the frozen ROI
reliability/noise-ceiling context remain required diagnostics. Their results
cannot select or rescue the primary RSA candidate. Readout disagreement and
EVC reliability dependence are reported rather than silently averaged away.
Crossnobis is additionally reported only when valid independent repeat
partitions exist; structural
ineligibility does not fail candidate lock because the always-evaluable neural-
distance sensitivity above is the mandatory estimator check.

Any mandatory falsifier failure blocks candidate lock regardless of aggregate
score.

## Budget and stopping

- fixed primary panel: **3 VLM rows plus 1 DINOv2 comparator row**;
- candidate decisions: **1** after complete panel and falsifier evaluation;
- eligible locked VLM candidates: **at most 1**;
- exact full-development reruns: **1**;
- CPU ceiling: **2,500 core-hours**;
- GPU ceiling: **0 GPU-hours** because released embeddings are used;
- wall-clock ceiling: **120 hours** from the first panel evaluation;
- audit openings: **1**;
- maximum parallel CPU cores: **48**;
- per-evaluation memory ceiling: **256 GB**;
- scratch-storage ceiling: **1,500 GB**.

Exact reruns after proven infrastructure failure are ledgered but are not new
model evaluations. They still consume resources. The fixed comparison ends
after the complete panel, controls, diagnostics, decision, and full rerun, or
earlier only for a hard resource ceiling, technical impossibility, or policy
violation.

## Terminal classes

- `candidate_ready`: one locked pipeline satisfies every development
  constraint and passes the one-shot sealed boundary audit.
- `closed_no_candidate`: the fixed panel is valid but no VLM passes every
  constraint, or the locked candidate fails audit.
- `panel_complete_no_audit`: development finishes but neither sealed OOD
  responses nor new independent compatible data are available.
- `technical_failure`: source identity, legal access, stimulus-response joins,
  folds, ROI/beta validity, or executable evaluation cannot be established.
- `policy_violation`: prohibited outcome access, leakage, undeclared operators,
  or post-audit adaptation invalidates the round.

For canonical outer status, `candidate_ready` maps to `candidate_ready`;
`closed_no_candidate` and `panel_complete_no_audit` map to
`closed_no_candidate`; and `technical_failure` or `policy_violation` map to
`technical_failure`.

## Claim boundary

A successful sealed LAION OOD audit supports robustness of the locked VLM
image-embedding-to-fMRI response-geometry alignment across the prespecified
image shift in these five participants and ROIs. It does not establish a shared
neural/model code, mechanism, universal modality-general space,
participant-population generalization, foundation-model scaling law, or
independent replication. Comparing these unmatched VLM and DINOv2 checkpoints
also cannot identify a causal benefit of multimodal training. Only an
independently sealed new dataset can extend that boundary.
