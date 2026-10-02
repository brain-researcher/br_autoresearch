# EP17 paper plan: which visual-model advantages survive changes in the measured cortex?

Status: claim-specific design amendment, 2026-10-02. No neural score, sealed-concept
result, checkpoint qualification or execution lock is reported here. The
existing three-pair/five-edge search and audit contract remains operative.

## The intended contribution

The lead comparison is **DINO versus category-supervised ResNet-50 in
anatomical bilateral fusiform**. The lead operation is **D-stage anatomical
versus reliable voxel membership**, with matched counts/spatial quotas and
identical voxel predictions. The question is whether an apparent advantage
belongs consistently to the measured anatomy or changes when reliable voxels
are emphasized—and whether that change can be predicted on new concepts.

This is not a new claim that SSL fits visual cortex, a new voxel-selector, or
a causal experiment on learning objectives. It is a proposed conditional
model-comparison result about the sampled neural population. A named operator
and a reproducible, consequential pattern of cortical sampling dependence,
not the number of model pairs, must carry the paper. For fixed support or
ceiling weights the forecast is numerically the development edge itself;
passing it on new concepts is a repeatability test, not a new mechanistic law.

For a controlled model pair, map where model A or model B has the larger
voxelwise held-out advantage. Then compare that map with an independently
defined map of what the measurement operation changes:

- reliability gained by moving from B→C or C→D beta estimates;
- voxel membership gained by moving from an anatomical to a
  reliability-selected support; or
- positive weight gained under noise-ceiling normalization.

The hypothesis is conditional: selection favors whichever model's development
advantage aligns with the change in inclusion weights. DINO need not benefit,
and the relation need not reverse. A stable relation, an accurately forecast
shift, a failed forecast, and imprecise evidence are different possible results.
The forecast is frozen in development and tested on new concepts in the same
four people; no audit advantage enters it.

All operator and advantage maps are defined within participant and visual
ROI. Voxels are averaged inside each participant/ROI before equal-participant
aggregation; native voxels from different people are never pooled as
independent observations.

## Why the three operations must remain distinct

| Operation | Mathematical mechanism | Explanatory test |
| --- | --- | --- |
| Beta construction | Can change the response and therefore the A–B advantage at each voxel | Does a reliability-associated high/low pattern repeat? The intercept-containing aggregate forecast alone cannot establish incremental gain-map explanation. |
| Voxel support | Leaves each voxel's fixed-prediction advantage unchanged but changes which voxels are averaged | Does the development inclusion decomposition forecast the audit change? |
| Ceiling normalization | Leaves every raw prediction, voxel, and within-voxel winner unchanged; positive weights alone change | Does the development weight decomposition forecast the audit change? |

Calling all three “pipeline sensitivity” would discard the most informative
part of the experiment.

## Closest primary work and the remaining claim

| Source reviewed | What is already established | Boundary for EP17 |
| --- | --- | --- |
| [Konkle and Alvarez, Nature Communications 2022](https://pubmed.ncbi.nlm.nih.gov/35078981/) | Self-supervised representations can fit ventral-stream geometry alongside supervised models. | SSL-versus-supervised brain fit is not new. |
| [Conwell et al., Nature Communications 2024](https://www.nature.com/articles/s41467-024-53147-y) | Controlled visual-model comparisons; linking methods and training diets materially affect conclusions. | Neither controlled comparison nor method-dependent brain alignment is a first claim. |
| [Prince et al., GLMsingle, eLife 2022](https://elifesciences.org/articles/77599) | Better response estimates, reliability and downstream analyses. | Better denoising or reliability is not this paper's discovery. |
| [Schütt et al., eLife 2023](https://elifesciences.org/articles/82566) | Representational model inference includes measurement and selection assumptions. | Sensitivity to measurement assumptions is known; retain subject/stimulus uncertainty. |
| [CNeuroMod-THINGS data paper](https://arxiv.org/abs/2507.09024) | Dense repeated visual measurements and supplied GLMsingle derivatives. | Dataset availability or another use of the release is not novelty. |

The candidate distinction is narrower: forecast the isolated membership effect
for one named comparison in a fixed anatomical scope, then test its sign and
magnitude on untouched concept roles without refitting. This distinction is an
inference from the reviewed sources, not an exhaustive global-priority claim.
The algebra below is elementary; only a supported empirical conditionality or
useful forecast adds substance. A generic ranking grid or an imprecise null
would be a pilot/methods comparison, not a major visual-neuroscience paper.

## Named choices and the exact support forecast

[GOAL.md](../GOAL.md) nominates P1 DINO versus supervised, P2 standard-width
SwAV-800 versus supervised, and P3 DINO versus SwAV. All are nominal
ResNet-50/ImageNet-1K contrasts; they share models and are not independent
replications. Backbone, corpus, recipe and exposure qualification remains open.
Training duration, augmentation and optimization differences are not erased
by choosing a common backbone. The October 2 amendment permits these pairs
for measurement-conditional representation comparisons, while prohibiting
causal training-property attribution. The measurement operation, not the
training recipe, is isolated. Exposure and checkpoint qualification remain
open. No checkpoint was fetched.

The nominated anatomical scope is DK fusiform labels 1007/2007, with equal
hemispheric-ROI weight inside each of the four people. It is not a functional
FFA or all high-level visual cortex. Local atlas compatibility, finite common
support and quota feasibility are still unverified; no outcome-driven ROI
substitution is permitted.

For each participant and hemispheric ROI, let `d_dev[v]` be DINO-minus-
supervised raw R² from development concepts. Let `w_R[v]` be reliable-support
inclusion divided by N, and `w_A[v]` the average normalized inclusion over
the frozen anatomical subset bank. Both weight vectors sum to one:

```text
forecast_support_shift = sum_v (w_R[v] - w_A[v]) * d_dev[v]
observed_audit_shift   = sum_v (w_R[v] - w_A[v]) * d_audit[v]
```

Compute each expression inside a participant/ROI, then equal-weight ROIs and
people. Calibration concepts alone define support weights; development
concepts define the forecast; audit concepts alone supply the observed target.
No weights or fits change when the support is applied. The development formula
exactly attributes a development shift, but cannot prove future agreement.

The existing forecast test requires the observed audit shift to share the
frozen direction and stay inside the presigned absolute-error margin. A
reversal additionally requires both endpoint bounds to cross opposite
practical margins. A correctly predicted shift that does not reverse the
relation is not a reversal. A near-zero forecast, unsupported margins, or wide
bounds remains unresolved rather than a test of universal SSL superiority.

This paper focus does not preselect a candidate-ready result or shrink the
operative panel. The other two nominated pairs and all five edges still need
the current coverage. If a different relation survives the active search,
report it separately; it does not retroactively validate the named DINO
support-selection story.

## Study sequence

### 1. Build isolated comparisons

Register fixed model pairs with disclosed training differences and matched
neural fitting opportunities; isolate one measurement change. Use concepts,
not images, as the leakage boundary. Reserve 480 concepts for development, 120
for calibration of reliability/support/ceilings, and 120 for one sealed audit.

Build the exact four-person image intersection and align every image, trial,
voxel, and response unit across B/C/D. Freeze the anatomical visual ROIs,
support size, subset bank, spatial quotas, ceiling estimator and floor before
candidate scores.

The initial five-edge family is fixed in advance: B→C and C→D on raw R² and
the same `A_all` voxels; D-stage `A_all→A_N` as a size control; D-stage
`A_N→R_N` with equal counts and quotas; and raw→normalized D-stage R² on
identical `V_NC` voxels. Every endpoint receives a named audit homolog. Only a
prespecified structural or calibration failure can make one inapplicable.

For beta edges, use one common layer/PCA/ridge selection across stages. For
support edges, fit one voxel-prediction bank and apply the two support masks
afterward. For ceiling edges, change only the positive voxel weights. A
comparison that violates these rules is composite and cannot establish a
measurement-specific reversal.

### 2. Establish a stable relation or a genuine reversal

Use raw held-out voxelwise R² as the primary score. A stable relation requires
the same signed practical-margin result across every eligible primary
contract. A reversal requires both ends of one isolated edge to cross opposite
margins under simultaneous uncertainty.

Raw and ceiling-normalized R² have separate practical margins. A
raw-to-normalized reversal controls their joint uncertainty through
studentized max-t statistics while retaining each effect in its own units; the
two score values are never averaged.

Show all four participants, all concept blocks, and all eligible contracts.
Do not treat a point sign flip, a selected ROI, or a nonsignificant endpoint as
evidence of reversal or equivalence.

### 3. Predict the measurement effect

For every edge, define the operator map without candidate-model scores. On
development concepts, estimate the voxelwise pair advantage.

For support and ceiling, first compute the development edge change directly
from the frozen development voxel advantages and membership or ceiling
weights. This exact reaggregation attributes the development result; it does
not predict new data. Then freeze a separate audit forecast calculated from
development advantages only. The evaluator compares it with an observed
reaggregation of audit advantages. Audit values cannot enter the forecast.

For beta edges, use two deterministic repeat halves to estimate calibration
reliability gain. Within each participant and ROI, cross-fit one ordinary
least-squares model of the voxelwise advantage change from standardized gain
and fixed spatial-bin intercepts. Fit on 12 development blocks and score the
other 12, then reverse. Score participant-by-block prediction errors, not
voxel rows. Refit the same equation on all 24 development blocks and freeze
its direction, predicted magnitude, absolute-error margin, and median-split
high- versus low-gain comparison.

The bin intercepts force fitted residuals to sum to zero within each bin.
Thus the fitted aggregate equals the corresponding development target mean
on the same voxel support. Aggregate prediction error measures repeatability,
not the gain map's explanatory increment over intercepts. The high-minus-low
test adds a descriptive spatial association; it does not compare against a
bin-only predictor. An incremental spatial-prediction test remains a prospective
option. Mean blockwise R² and pooled-image R² are not silently equated.

The beta audit forecast is evaluated against the equal mean of the twelve
audit-block raw-R² stage changes, using the frozen training baseline and
prediction bank with the registered within-block aggregation. The primary
beta relation and high/low contrast remain pooled-image endpoints. This
blockwise repeatability companion cannot substitute for them. Support and
ceiling forecasts keep their corresponding aggregate-edge targets.

The preferred explanation predicts both sign and magnitude on sealed
concepts. A real reversal with a failed prediction is reported as unexplained
response-geometry or readout sensitivity rather than as recovered signal.
Successful aggregate and stratum tests likewise do not identify a causal
recovered-signal mechanism.

### 4. Test new concepts without refitting

Fit the final voxel coefficients on all 480 development concepts. The trusted
evaluator applies them directly to the 120 audit concepts. No audit concept can
choose a layer, penalty, support, ceiling, margin, or model pair.

The audit is concept-disjoint but uses the same four people. Its uncertainty
crosses participants and twelve audit concept blocks; it is not a population
estimate.

A robust relation must repeat every frozen endpoint. A reversal must repeat
both opposite endpoint effects and the development-only forecast. The same
intersection of at least three participants must satisfy every signed
component, and all participant and audit-block leave-one-out means must retain
the required direction. Score-specific adequacy-width thresholds distinguish
a precise nonreplication from an unresolved audit.

### 5. Keep secondary questions secondary

Held-out correlation checks whether the result depends on R². Fixed-rank
linear CKA tests one alternative linking criterion. Preregistered ROI or
semantic interactions can support a conditional relation. None may rescue a
failed primary comparison or become an independent candidate-ready terminal,
and their numerical scales are never pooled.

## Result branches

| Result | Interpretation | Next action |
| --- | --- | --- |
| Stable relation repeats on sealed concepts | The controlled pair relation is robust within the declared measurement family | Report its scope and the contracts actually covered |
| Reversal and operator prediction both repeat | A named measurement operation reproducibly favors one fixed representation | Explain the measured population dependence; do not claim a training or neural mechanism from reaggregation |
| Reversal repeats but prediction fails | The ranking change is real, but the proposed reliability/reweighting explanation is wrong | Retain a narrower measurement-sensitivity result or stop the independent-paper direction |
| Relation differs by preregistered ROI or semantic group | A scientific stratum, rather than a generic pipeline choice, conditions the relation | Report only if the interaction and both within-stratum effects replicate |
| Development relation fails on sealed concepts with adequate precision | The relation does not transfer to new concepts in the same participants | Report nonreplication without reopening search |
| Bounds are wide or the edge is composite | The comparison does not distinguish the explanations | Report underidentification; do not call models tied |

## Figures, contingent on observed evidence

| Figure | Scientific judgment | Required content |
| --- | --- | --- |
| **1. Named pair, fixed predictions, changed membership** | What does reliable selection actually change? | DINO/supervised backbones, anatomical fusiform, D-stage common bank, matched quotas and independent forecast inputs; broader three-operation schematic is supporting context |
| **2. Is the relation stable or reversed?** | Do both edge endpoints clear practical margins? | Every contract, participant, concept block, and simultaneous interval |
| **3. Does operator–advantage alignment predict the change?** | Is the direction and magnitude explained before audit? | Reliability-gain strata for beta; exact support and ceiling decompositions; strongest falsifier |
| **4. Does the prediction survive new concepts?** | Which relation and explanation repeat without refitting? | All 120 audit concepts in twelve blocks, all four people, prediction error and uncertainty |
| **5. Where does the rule fail?** | Is the boundary measurement-related, regional, semantic, or linking-dependent? | Prespecified negative, conditional, and CKA results without post-hoc rescue |

The first figure is a synthetic question schematic. Later figures must display
complete participant and concept-block evidence rather than only an ROI mean.

## Current conceptual figure

![Three measurement operations and held-out repetition; no observed results](ep17_conceptual_question-v3.png)

The current schematic shows fixed named checkpoints, matched response/readout
comparisons, equal-size support changes and positive ceiling reweighting of
the same voxels. Ring sizes denote schematic weights, not scores. The
held-out relation is tested in the same four people. This is not a causal
training-objective test or a newly explanatory weighted-mean forecast;
contract-robust stability still needs every eligible five-edge endpoint.
[Exact generation and correction prompts](ep17_conceptual_question-v3-prompt.md)
are saved. Earlier images remain historical assets.

## When to deepen, narrow, or stop

- Deepen the paper only when one controlled pair has an isolated relation, a
  useful operator prediction, and a valid homologous sealed-concept test.
- Narrow the claim when the relation repeats but its explanation fails, or
  when it holds only in a preregistered ROI or stimulus stratum.
- Stop the isolated measurement claim when checkpoints or neural fitting change across an edge,
  support or beta edges require endpoint-specific retuning, the audit cannot
  be isolated, or precision cannot distinguish a margin from equivalence.
- Do not compensate for a failed explanatory prediction by expanding the
  model zoo or selecting another contract after audit.

## Claim language

A successful abstract should name the controlled pair, the measurement
operation, its measured spatial association or repeatable weighting effect,
and the same-participant
held-out-concept scope.

It should not claim that one model is universally more brain-like, that four
participants establish population prevalence, that the result transfers to a
new dataset, or that a training objective causally produced the neural
relation.

## Current next action

Complete the existing outcome-independent checkpoint/pair and anatomical
support decisions, exposure eligibility, roles and numerical inference choices.
The design now gives those decisions named targets, not fabricated answers.
If the nominations cannot qualify under the amended claim-specific rule,
resolve remaining checkpoint/exposure issues before scores rather than
silently replacing a model or ROI. No extraction, neural
access, model fitting, audit opening or canonical action was authorized by
this writing milestone.

The [October 2 scope review](scope_novelty_review_20261002.md) identifies
open explanatory follow-ups without adding them to the executable program.
