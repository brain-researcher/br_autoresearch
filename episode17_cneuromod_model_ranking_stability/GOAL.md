# Which visual-model advantages survive changes in the measured cortex?

Suppose model A predicts visual-cortex responses better than model B for the
same images and people. If B wins after denoising the beta estimates, selecting
more reliable voxels, or dividing by a noise ceiling, what changed?

Those operations are scientifically different:

- beta construction can change the response estimated at each voxel;
- voxel selection keeps a voxel's score but changes which voxels enter the
  regional average; and
- positive noise-ceiling normalization keeps the voxels and predictions fixed
  and only changes their weights. It cannot reverse A versus B within one
  voxel, although it can reverse the regional average.

EP17 asks whether a model relation is stable across those operations and,
when it changes, whether the change can be predicted from where each model's
voxelwise advantage lies. The intended paper is not a catalogue of fragile
rankings. It should explain which measurable property of the response makes a
particular model benefit from a particular measurement choice.

## A concrete starting comparison

The lead design is now concrete: compare the official **DINO ResNet-50**
backbone with **Torchvision ResNet-50 `IMAGENET1K_V1`** in anatomical left and
right fusiform cortex. Both nominally use ImageNet-1K and the same backbone
family; their pretraining recipes are not matched interventions. They are
named representation comparisons, not interventions on self-supervision. Ask whether
selecting reliable voxels changes that relation, and whether a forecast made
from development model advantages predicts the change on new concepts.

For the headline operation, keep the D-stage responses, ridge prediction bank,
voxelwise raw R², voxel count and spatial quotas fixed. Change only anatomical
subset membership to calibration-reliability membership, `A_N → R_N`.
If DINO's advantage concentrates in the newly emphasized voxels, the frozen
forecast predicts a shift toward DINO; if the supervised advantage concentrates
there, it predicts the opposite. There is no unconditional prediction that
self-supervision wins, and no new preferred sign chosen from audit results.

This distinguishes **a stable representation comparison** from **a changed
sample of measured cortex**. Reliable selection need not recover a uniquely
correct biological population. Anatomical fusiform is not a functionally
localized FFA, and repeat reliability is not selectivity or mechanistic truth.

The [paper plan](outputs/paper_plan.md) names the checkpoint proposals, exact
forecast, closest literature and failure cases. This is a pre-score paper
focus, **not an execution lock or a reduction of the operative search**:
three eligible pairs, all five initial edges, the 32–72-trial budget and the
one-shot audit remain required. No model-ranking result is claimed.

## What the literature already closes off

[Konkle and Alvarez (2022)](https://pubmed.ncbi.nlm.nih.gov/35078981/) already
compare self-supervised and supervised ventral-stream representations.
[Conwell et al. (2024)](https://www.nature.com/articles/s41467-024-53147-y)
already conduct controlled visual-model comparisons and show that linking
methods matter. [GLMsingle](https://elifesciences.org/articles/77599) already
improves response reliability and downstream analyses. Thus neither
self-supervision, controlled model comparison, reliability selection, nor a
pipeline-dependent winner is a first claim.

The candidate contribution is a reproducible account of **which measured
cortical population supports a model advantage**, separating response
estimation, voxel membership, and score weighting. A named support-selection
forecast tests transfer to new concepts with fixed predictions and spatially
matched supports. For fixed support weights that forecast equals the observed
development edge carried forward: audit agreement establishes repeatability,
not a new predictive mechanism. The exact decomposition is elementary.
The reviewed sources motivate a methods study; a standalone paper still needs
a consequential empirical pattern beyond known selection sensitivity.
[The scope review](outputs/scope_novelty_review_20261002.md) separates that
contribution from stronger, still-open scientific directions.

## At a glance

| Question | EP17 design |
| --- | --- |
| What is compared? | Lead proposal: DINO versus category-supervised ResNet-50; the existing three-pair controlled panel must still qualify before scores. |
| What stays the same? | The four CNeuroMod participants, exact images, concept folds, linear readout family, and pairwise scoring rule. |
| What changes one at a time? | Lead paper operation: D-stage anatomical versus reliable voxel support; beta and ceiling edges remain separate required comparisons. |
| What is the primary score? | Held-out voxelwise explained variance, aggregated with equal participant weight. |
| What is the explanatory prediction? | A measurement operation should favor the model whose voxelwise advantage aligns with the operation's independently defined reliability, inclusion, or weighting map. |
| What is held out? | Concepts, not random images: 480 development, 120 support/calibration, and 120 sealed audit concepts. |
| What does the audit establish? | Whether one frozen relation and its explanation repeat on new concepts in the same four people, without refitting the readout. |
| What can the study conclude? | Stable relation, predicted measurement reversal, unexplained reversal, scientific conditionality, or an unresolved comparison. |

## Why a robustness grid is not enough

Model rankings are known to depend on analysis choices. Showing another set of
rank flips would therefore be descriptive rather than explanatory. EP17 must
add two things:

1. isolate one measurement operation while holding the model pair, images,
   predictions, voxels, or weights fixed as required; and
2. predict the direction of the ranking change from an independently defined
   property of the measured response.

The source data paper and GLMsingle processing family establish the dataset
and the B/C/D response stages. Before a paper claim, a focused novelty review
must compare the selected model pair and operator–advantage prediction with
prior work on fMRI encoding-model reliability, voxel selection, noise-ceiling
normalization, and analysis-dependent model rankings. “We tested more
contracts” is not a sufficient contribution.

## Competing explanations

| Explanation | Prediction |
| --- | --- |
| **Stable model relation** | The same model clears the practical margin under every eligible measurement contract, and its advantage is not confined to one reliability or spatial stratum. |
| **Reliability-associated stage change** | The signed beta-stage change differs between calibration-defined high- and low-reliability-gain voxels and repeats on sealed concepts; the aggregate forecast alone does not establish incremental gain-map explanation. |
| **Support reweighting** | Reliable-voxel selection changes the regional relation by selecting voxels where one model already has a larger fixed-prediction advantage. Development gives an exact membership decomposition; its frozen value becomes a forecast for audit concepts. |
| **Ceiling reweighting** | Raw and normalized scores differ because inverse-ceiling weights align with one model's voxelwise advantage. Development gives an exact weight decomposition; agreement on audit concepts is a separate empirical test. |
| **Scientific specialization** | A relation differs reproducibly by a preregistered visual ROI or external semantic stratum, not merely by a measurement operation. |
| **Readout dependence** | Linear encoding and a fixed-dimensional representational comparison favor opposite models under the same response and support. |
| **Unexplained measurement sensitivity** | A genuine reversal survives isolation controls, but the frozen operator–advantage prediction has the wrong sign or magnitude. |
| **Underidentification or nonreplication** | Bounds are too wide, the comparison is not isolated, or the development relation does not survive on sealed concepts. |

A point-estimate sign change is not a reversal. Nonsignificance is not
equivalence, and wide intervals are not evidence that two models are tied.

## Data roles and scope

EP17 uses CNeuroMod-THINGS 1.0.1 and the same four intensively sampled
participants throughout. Whole concepts receive one role across every image,
repetition, participant, and beta stage:

| Role | Concepts | Use |
| --- | ---: | --- |
| Development | 480 | Fit and compare controlled model pairs; choose one final relation and explanation |
| Support/calibration | 120 | Define repeat reliability, reliable-voxel support, and B/C/D noise ceilings; never fit or score a candidate model |
| Sealed audit | 120 | Apply the frozen development predictor once, with no readout refit |

The audit uses new concepts but the same four participants. It is not a new-
participant, cross-dataset, or population replication. Participants and
concept blocks are the uncertainty dimensions; voxels and images are not
independent biological replicates.

The exact four-person image intersection and 720 eligible concepts must be
reconstructed from events before neural values are opened. Candidate model
features cannot define or repair the split.

## Model pairs and the claim they can support

**Pre-score design amendment, 2026-10-02:** eligibility now distinguishes a
controlled measurement comparison from a controlled training intervention.
The former is the primary EP17 question and does not require two checkpoints
to differ in exactly one training property. Training-recipe differences are
held fixed across a measurement edge and disclosed. A causal attribution to a
training objective, architecture, or corpus remains outside the permitted
claim. This replaces the original single-property eligibility requirement;
three registered pairs, exposure rules, matching of the neural readout, five
edges, budgets, roles, margins, and the one-shot audit are unchanged.

### Concrete nominations, not a qualified registry

The outcome-independent design nominates these three pairs:

| Pair | Named backbone proposals | Declared contrast |
| --- | --- | --- |
| P1, paper lead | DINO `dino_resnet50_pretrain.pth` versus Torchvision `ResNet50_Weights.IMAGENET1K_V1` | Two fixed representations learned with different training recipes |
| P2 | Official SwAV ResNet-50, 800-epoch standard-width release versus the same supervised backbone | A second fixed-representation contrast |
| P3 | The same DINO versus SwAV backbones | Two fixed self-supervised representations; not an isolated objective effect |

Checkpoint sources are the official [DINO](https://github.com/facebookresearch/dino),
[SwAV](https://github.com/facebookresearch/swav) and
[Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
records. Use backbones, not category heads or SSL projection heads. Do not
silently substitute DINOv2, DINOv3, widened SwAV, or a moving `DEFAULT` weight.
These pairs share models and are dependent contrasts, not three replications.

The nominal common architecture/corpus does not certify identical training
images, augmentation, optimization, epochs or checkpoint eligibility.
Those differences must be disclosed; they prevent a causal objective claim.
Under the amended criterion, these pairs may support measurement-conditional
comparisons if the checkpoints, exposure status and matched evaluation qualify;
they cannot identify why their training produced different representations.
Known THINGS-image overlap remains control-only and unknown exposure remains
unknown. If fewer than three pairs qualify, the operative panel is incomplete;
no post-score replacement or relaxed exposure rule is authorized.

The nominated anatomical scope is the Desikan–Killiany cortical fusiform
labels `ctx-lh-fusiform` (1007) and `ctx-rh-fusiform` (2007), as defined in
the [FreeSurfer label table](https://github.com/freesurfer/freesurfer/blob/dev/distribution/FreeSurferColorLUT.txt).
Treat them as two hemispheric ROIs with equal ROI weight within each person,
then equal weight over four people; hemispheres are not extra participants.
This nomination does not establish that the local derivative is that atlas,
has sufficient common finite voxels, or supports the required quotas. Those
outcome-independent mappings remain open. Do not select another anatomical
region because the nominated relation fails.

Freeze three required promotion-eligible model pairs and at most one optional
pair before candidate-discriminating neural scores are opened. Each pair must
name two fixed representations, disclose architecture/corpus/recipe differences,
and keep both checkpoints unchanged across each measurement edge. Shared
backbone/corpus is useful matching, not proof of a single-property intervention.
The one-factor isolation applies to the measurement operation. A future
training-causal question would need its own matched intervention design.

A trained-versus-deterministically-initialized-random pair is a mandatory
falsifier and cannot become the paper's winner. Every checkpoint must have a
named version, license, parameter count, training lineage, image
preprocessing, eligible layers, and THINGS exposure status. Known exact or
near-duplicate exposure makes a checkpoint a control rather than a clean
promotion candidate; unknown exposure is not evidence of no exposure.

Within a pair, both models receive the same folds, feature-dimensionality
rule, layer budget, PCA-rank grid, ridge grid, images, voxels, and score
weights.

## The three measurement operations

For model pair A versus B, define the held-out voxelwise advantage separately
for participant `s`, visual ROI `r`, and that participant's native voxel `v`:

`d_s,r,v = R²_A,s,r,v - R²_B,s,r,v`.

All maps, support choices, and reweighting calculations are made within a
participant and ROI. Effects are averaged over voxels within `(s,r)`, then over
a predeclared ROI set with equal ROI weight if a relation contains more than
one ROI, and finally over the four participants with equal weight. Native
voxels are never pooled across people as though they were independent cases.

The primary paper distinguishes three ways that a regional average can change.

| Operation | What changes? | What must remain fixed? | What a reversal would mean |
| --- | --- | --- | --- |
| **Beta B→C or C→D** | The response target at each voxel | Images, voxel identity, folds, model features, layer/PCA/ridge choice, and score | The measurement stage changed `d_s,r,v` itself |
| **Support A_N→R_N** | Which voxels enter the mean | Stage D, one common out-of-fold voxel-prediction bank, `N`, spatial quotas, and per-voxel scores | The selected population carried a different mix of fixed voxel advantages |
| **Raw→ceiling-normalized R²** | The positive weight applied to each voxel | Images, response, predictions, voxel set, raw R², and ceiling floor | Reweighting fixed voxel advantages changed the regional mean |

`A_N` is the mean over a fixed bank of anatomical subsets of size `N`.
Both are built separately within participant and ROI. `R_N` contains the top
D-stage calibration-reliability voxels with exactly the same subparcel and
spatial-bin quotas. `A_all→A_N` is reported as a size-control, not a headline
biological result.

The complete initial contract family is fixed before model scores:

| Edge | Endpoint 1 | Endpoint 2 | Score and fixed anchor |
| --- | --- | --- | --- |
| Beta B→C | B on `A_all` | C on the same `A_all` | Raw R²; common anatomical voxels |
| Beta C→D | C on `A_all` | D on the same `A_all` | Raw R²; common anatomical voxels |
| Size control | D on `A_all` | D on `A_N` | Raw R²; one D-stage prediction bank |
| Support A_N→R_N | D on `A_N` | D on `R_N` | Raw R²; equal `N` and spatial quotas |
| Ceiling treatment | Raw D score on `V_NC` | Normalized D score on the same `V_NC` | Identical predictions and voxels |

`V_NC` is the D-stage `A_all` subset that passes the outcome-independent
response-variance rule and has a finite calibration ceiling before flooring.
Its ceiling-quality threshold and floor are fixed before scores. An endpoint
may be removed only for a prespecified structural or calibration failure; its
reason remains visible. The audit uses the homologous stage, support, weights,
and endpoint definition on the 120 audit concepts. Contracts cannot be added,
dropped, or relabeled after a model score is seen.

For beta edges, each model uses one layer, PCA rank, and ridge setting selected
jointly for B/C/D inside development training concepts. The selection
objective is mean inner-validation raw R² across the three stages after
voxel-within-ROI, equal-ROI, and equal-participant aggregation. Ties choose the
shallower registered layer, then the smaller PCA rank, then the larger ridge
penalty. Voxel coefficients are fitted separately to B, C, and D because the
response target differs, and all endpoint-specific coefficient banks are
locked for audit. Stage-specific retuning is a sensitivity and cannot support
a pure beta reversal.

For support edges, generate one out-of-fold prediction bank for all eligible
voxels before applying either support mask. If `A_N` and `R_N` use different
layers, ranks, penalties, prediction fits, or spatial quotas, the comparison is
support × fitting and cannot be called a support-only reversal.

For ceiling edges, use the identical finite voxel set and predictions. Divide
each voxel's raw held-out R² by `max(NC_s,r,v, tau)` and average the voxelwise
ratios. Do not divide one ROI mean by another, clip negative R², or clip values
above one. A positive denominator cannot change the winner within a voxel; it
can only change the aggregate by giving voxels different weights.

## The explanatory prediction: operator–advantage alignment

Support/calibration concepts define an outcome-independent map within each
participant and ROI for every operation:

| Edge | Calibration-defined operator map |
| --- | --- |
| B→C | Per-voxel change in repeat reliability from B to C |
| C→D | Per-voxel change in repeat reliability from C to D |
| A_N→R_N | Difference between reliable-support inclusion weight and the mean anatomical-bank inclusion weight |
| Raw→normalized | The fixed positive weight `1 / max(NC_v, tau)` |

Development concepts estimate the voxelwise model advantage. The analysis
then asks whether that advantage aligns with the relevant operator map.

There are two different claims, and they must not be confused:

1. **Development attribution.** For support and ceiling edges, reaggregating
   development `d_s,r,v` under the two frozen weight vectors exactly reproduces
   the development edge change. This is an algebraic identity within the
   development data, not held-out evidence.
2. **Audit forecast.** Before audit, use development `d_s,r,v` to calculate and
   freeze a signed forecast for the audit edge. The evaluator then computes the
   observed edge from audit `d_s,r,v`. Audit advantages never enter the
   forecast. Agreement in sign and absolute error is therefore empirical, not
   tautological.

For beta edges, the response and voxelwise advantage can change, unlike pure
support reaggregation. Calibration repeat reliability is the Pearson
correlation between two deterministic repeat halves
across calibration images; its Fisher-z stage difference is the operator map.
Within each participant and ROI, fit

`change in d = spatial-bin intercept + beta * reliability gain`

by ordinary least squares, with no tuned penalty. The gain map is standardized
within `(s,r)` using calibration values only. Fit on the mean voxelwise change
from 12 development concept blocks and predict each of the other 12 blocks,
then reverse the halves. Participant-by-held-out-block errors, not voxel rows,
score this cross-fit. A rank-deficient design or inadequate residual variation
in reliability gain makes the beta explanation inapplicable rather than
inviting another model.

The spatial-bin intercepts imply `sum_bin(y - y_hat) = 0`. On the same
voxel support the aggregate fitted prediction therefore equals the training
target's mean change. Aggregate cross-fit/audit agreement tests repeatability;
it does not demonstrate incremental explanatory value of reliability gain.
This identity concerns the fitted block-mean target, not necessarily a score
recomputed from pooled images. The registered high-minus-low comparison tests
a descriptive reliability-associated spatial pattern, not superiority over a
spatial-bin-only predictor. A direct incremental spatial-prediction comparison
is an open follow-up, not an additional criterion in the current protocol.

After cross-fit qualification, refit that same equation to all 24 development
blocks and freeze its audit forecast. Its beta-specific observed target is
the equal mean of the twelve audit-block raw-R² stage changes: compute each
block's scores with the frozen development training mean and predictor, then
aggregate voxels, ROIs and participants as specified above. This is a
blockwise repeatability companion. The primary beta-edge endpoint and its
high-minus-low contrast retain their pooled-image R² definitions; the
companion cannot replace either. Support and ceiling forecasts retain their
corresponding aggregate-edge targets. High- and low-gain voxels are the upper
and lower halves of the calibration gain map within each `(s,r)`; the median
and tie rule are fixed without development or audit scores. The beta
explanation predicts both the aggregate stage change and a larger signed
change in the high-gain half.

After development, freeze for the selected relation:

- the operator map and voxel set;
- the development-only signed audit forecast and an edge-specific absolute
  prediction-error margin;
- the high- versus low-operator strata;
- the exact aggregation and uncertainty rule; and
- the participant and audit-block units used to judge the forecast.

On sealed concepts, the forecast must have the same direction as its declared
observed target and absolute error no larger than its frozen margin. For beta
this is the audit-block-mean companion, not the pooled primary edge; for
support/ceiling it is the corresponding aggregate edge. For a beta edge, the
preregistered high-minus-low gain contrast must also cross its own signed
margin. If those tests fail, the reversal may be real but the proposed
reliability-associated pattern or repeatable reweighting account is unsupported.
It is reported as
unexplained response-geometry or readout sensitivity.

Passing them supports the specified association and repeatability, not a
causal recovery-of-signal mechanism or an independently validated gain-map
explanation beyond spatial intercepts.

This explanatory test cannot rescue a primary relation that fails its own
margin or audit requirement.

## Response, fitting, and score

Within participant, beta stage, image, and voxel, average all retained
trialwise betas with equal repetition weight. Each unique image then receives
equal weight in fitting and scoring.

The primary linking model is voxelwise ridge regression. Every learned
operation—feature centering, PCA, layer choice, rank, and ridge penalty—is
nested inside development training concepts. No outer-held-out concept may
choose a model setting.

The primary score is held-out explained variance:

`R² = 1 - sum(y - y_hat)² / sum(y - training_mean)²`.

The baseline mean always comes from the corresponding development training
data, including when sealed audit concepts are scored. Negative held-out R²
values are retained. Before model outputs, a voxel must have finite response
values and nondegenerate response variance under every endpoint in its edge.
The numerical variance threshold is fixed in qualification. A model-specific
missing prediction or score invalidates the whole paired trial; pairwise or
model-specific deletion is prohibited.

Pairwise effects are calculated within participant and ROI, then aggregated
with equal ROI and participant weights as declared by the relation. The common
prediction bank and the finite voxel set are locked before applying support or
ceiling weights.

Held-out correlation is a sensitivity, not a substitute for R². A
fixed-dimensional linear CKA comparison is a secondary linking falsifier. R²,
correlation, normalized R², and CKA keep their own margins and are never pooled
on one numerical scale.

## Uncertainty and scientific decisions

The primary uncertainty procedure crosses equal-weight resampling of the four
participants with resampling of 24 development or 12 audit concept blocks.
All exemplars and repetitions of a concept stay together. Voxelwise rows never
create the inferential sample size.

Raw R², normalized R², correlation, CKA, and beta high-minus-low effects each
have a separately frozen practical margin. For a raw-to-normalized reversal,
the two endpoints retain their own margins and units. Familywise uncertainty
is controlled with one max-t bound over their studentized statistics; the raw
and normalized effect values are never averaged or put on one numerical
scale.

For the component-specific margin `epsilon_j`:

- **stable positive:** every eligible-contract lower bound is above
  `epsilon`;
- **stable negative:** every upper bound is below `-epsilon`;
- **genuine reversal:** both ends of one isolated edge cross opposite margins;
- **practical equivalence:** every complete simultaneous interval lies inside
  its own `[-epsilon_j, epsilon_j]`; and
- **underidentified:** the bounds overlap both a decision region and the
  equivalence or null region.

The bootstrap describes these four participants and sampled concept blocks.
It does not establish population prevalence.

## Development search and one-shot audit

Development covers all three required controlled pairs on the isolated beta,
support, and ceiling edges before adaptive successors. Successors change one
scientific operator at a time and must state a prediction, competing
explanation, falsifier, cost, and retirement condition.

The first round permits 32–72 valid trials, requires at least two adaptive
successor cycles and two incumbent/challenger decisions, and devotes at least
40% of post-coverage trials to falsification. Limits are 6,000 CPU-core-hours,
256 GPU-hours, 240 wall-clock hours, and 4 TiB of transient scratch.

Development selects one primary relation and at most two disclosed secondary
checks. ROI/semantic conditionality and CKA remain secondary and cannot become
independent candidate-ready outcomes. Before audit, freeze the model pair,
all endpoints and their exact audit mappings, prediction bank, measurement
edge, operator map, development-only forecast, response and score, every
component margin and adequacy-width threshold, concept blocks, uncertainty
procedure, and evaluator.

The evaluator applies the frozen development voxel coefficients directly to
all 120 sealed concepts. No layer selection, hyperparameter tuning, readout
refit, voxel replacement, subset rescue, margin change, or second audit run is
allowed.

Audit gates depend on the selected relation. A robust relation must clear all
frozen contract endpoints. A reversal must clear both edge endpoints in
opposite directions and pass the frozen forecast; a beta explanation also
requires the high-minus-low gain contrast. Secondary conditionality requires
its interaction and both stratum effects, while linking dependence requires
both encoding and CKA, but neither can rescue a failed primary relation.

The same intersection of at least three of the four participants must clear
every signed component of the selected relation. It is not enough for
different groups of three to clear different components. Every
leave-one-participant and leave-one-audit-block mean must retain the required
direction. An adequately precise nonreplication additionally requires the
pre-audit width threshold for that score scale; wider bounds are
underidentified, not negative evidence.

## Planned question figure

The figure below is a synthetic design illustration. It contains no
CNeuroMod image or neural result.

![EP17: fixed visual models compared under response, support and weighting changes](outputs/ep17_conceptual_question-v3.png)

The current figure shows all three measurement operations. Changing response
estimates requires matched readout refitting; support selection holds
predictions fixed while matching counts and spatial quotas; ceiling weighting
holds both voxels and predictions fixed and changes positive weights. The
small grids, traces and object drawings are illustrations, not source images
or results. Held-out concepts test repetition in the same four people, not
new-participant generalization or a causal effect of training. Carrying a
fixed-weight development edge to audit remains a repeatability test, not an
independent explanation. Contract-robust stability still requires the full
eligible five-edge family. [Exact generation and correction prompts](outputs/ep17_conceptual_question-v3-prompt.md)
are saved; v1 and the support-specific v2 remain historical illustrations.

## Possible conclusions

| Outcome | Evidence required | Interpretation |
| --- | --- | --- |
| **Contract-robust relation** | One model clears the same signed margin under every eligible primary contract and repeats on sealed concepts. | The controlled pair relation is stable within the declared response, support, and weighting family. |
| **Predicted measurement reversal** | Both isolated edge endpoints cross opposite margins; the frozen operator–advantage prediction has the correct direction and acceptable error on sealed concepts. | A named measurement operation predictably changed which model was favored. |
| **Unexplained measurement reversal** | The reversal itself passes, but the operator–advantage prediction fails. | The ranking change is real within the design, but the proposed reliability or reweighting explanation is unsupported. |
| **Secondary scientific conditionality** | A preregistered ROI or semantic interaction and both within-stratum effects clear their margins and repeat on sealed concepts. | The relation may be conditional on a named neural or stimulus stratum; this cannot rescue or replace the primary result. |
| **Secondary readout dependence** | Encoding and fixed-dimensional CKA cross opposite margins under the same data contract. | The conclusion may depend on the linking criterion; this is disclosed as a falsifier, not an independent positive terminal. |
| **Held-out-concept nonreplication** | Development is precise, but an adequately powered sealed-concept test supports equivalence, the opposite direction, or stable heterogeneity. | The development relation did not transfer to new concepts in the same people. |
| **Underidentified** | Intervals are too wide, the edge is composite, or required calibration/support is inadequate. | The data do not distinguish the competing explanations. |
| **Technical failure** | Role separation, identity alignment, fixed prediction banks, or audit execution fails. | No scientific interpretation is permitted. |

## Data readiness and access boundary

The restricted CNeuroMod source and stimulus archive are available, but EP17
is not analysis-ready. The stimulus archive remains encrypted and unextracted;
the exact event-image join, four-person common universe, 480/120/120 roles,
role-filtered handoffs, B/C/D alignment, ROI crosswalk, support sizes and
quotas, ceiling estimator, controlled model pairs, feature grids, margins, and
operator-prediction tolerance remain to be fixed or verified.

The 2026-09-30 nominations make the paper question specific; they do not
complete those scientific decisions. No checkpoint was fetched, stimulus
extracted, anatomical array opened, neural model fitted, or audit accessed
for this revision.

The mixed neural source must not be mounted to the search worker. A trusted
builder may create metadata, calibration, and development artifacts; the
sealed audit handoff remains evaluator-only until the final relation and
prediction are frozen.

EP17 and EP18 share the THINGS stimulus ecosystem. Before either sealed
outcome is opened, record exact concept and image overlap, feature or
checkpoint reuse, and first-access history. The two episodes cannot be called
independent stimulus-family replications.

## Claim boundary

A successful EP17 would show that a controlled visual-model relation is stable
or changes for a predictable measurement reason across specified contracts
and new concepts in the same four intensively sampled participants.

It would not establish cross-dataset transport, generalization to new people,
population prevalence, a universal model ranking, causal superiority of an
architecture or objective, or correctness outside the registered response and
linear-readout family.
