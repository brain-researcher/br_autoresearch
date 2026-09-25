# Which Visual-Model Relations Survive the Measurement Contract?

## Episode 17 — CNeuroMod-only model-ranking stability

## Decision snapshot

EP17 asks whether a **controlled pairwise relation between visual models** is
stable, reverses for an identifiable measurement reason, is scientifically
conditional, or is too weak to identify.

The experiment uses one pinned source, CNeuroMod-THINGS 1.0.1, but gives its
stimuli three irreversible roles before any neural outcome is inspected:

| Concept role | Target count | What it may do |
| --- | ---: | --- |
| Development | 480 | adaptive model comparison, nested tuning, successors, and falsifiers |
| Support/calibration | 120 | voxel reliability, support selection, and stage-specific noise ceilings |
| Sealed audit | 120 | one application of the fully frozen development-trained predictor |

All exemplars and repetitions of a concept, in all four participants, inherit
the same role. The primary analysis uses only exact image IDs shared by all
four participants and meeting a frozen repeat-completeness rule.

The audit is therefore a **same-participant, concept-disjoint, held-out-stimulus
replication**. It is not an independent-dataset replication, a new-participant
replication, or a population estimate.

## Authority and scope

This file is a local drafting contract. It does not register a canonical Brain
Researcher loop, authorize scientific compute, expose sealed neural outcomes,
or assert that one model is generally more brain-like than another.

All claims are conditional on:

- the registered controlled model pairs;
- a frozen linear voxel-encoding readout;
- the CNeuroMod-THINGS acquisition and GLMsingle products;
- the declared anatomical ROIs and stimulus strata; and
- the finite B/C/D, voxel-support, and ceiling-treatment contract graph.

The object of inference is a **pairwise model relation under a declared
contract**, not an absolute model leaderboard.

## The scientific question

For controlled model pairs, which conclusions remain practically invariant
when we change:

1. beta construction;
2. independently defined voxel support; and
3. reliability normalization?

When a conclusion changes, can a single measurement axis explain the change?
When it does not change, does the same signed relation survive once on sealed
concepts that never influenced fitting?

## What would count as an answer?

| Explanation | Decisive pattern |
| --- | --- |
| Contract-robust relation | One pairwise advantage clears its practical margin under every eligible homologous contract. |
| Beta-stage sensitivity | B→C or C→D changes the relation while images, voxels, score, and readout protocol remain fixed. |
| Support sensitivity | Anatomical and reliability-selected supports reverse the relation at fixed size and spatial composition. |
| Ceiling sensitivity | Raw and ceiling-normalized held-out variance reverse the relation on the identical voxel set. |
| Scientific conditionality | A preregistered ROI or semantic-stratum interaction and both within-stratum effects clear their margins. |
| Linking dependence | Encoding and fixed-dimensional linear CKA cross opposite signed margins on the same data. |
| Underidentification | Bounds cannot distinguish the practical pass region from the null/equivalence region. |
| Held-out-concept nonreplication | A development relation fails on sealed concepts with bounds narrow enough to distinguish a real negative pattern from low information. |

A point-estimate sign flip is not a reversal. Nonsignificance is not
equivalence. Failure to pass with wide uncertainty is not a scientific
nonreplication.

## What this episode is not

EP17 is not:

- an unrestricted model-zoo search;
- a contest between unrelated architectures with uncontrolled training data;
- evidence that linear encoding is the uniquely correct linking hypothesis;
- direct transfer to new people or a new dataset;
- a claim that four intensively sampled participants represent a population;
- a causal account of why a preprocessing stage changes a relation; or
- permission to tune on the sealed concepts.

## Dataset partition and audit firewall

### Eligibility universe

Before neural values are read, the trusted data builder derives an
outcome-blind eligibility table from events and derivative metadata.

A primary image must:

1. have the same immutable image ID and concept ID in all four participants;
2. have the prespecified number of usable presentations in every participant;
3. map identically across B, C, and D trial axes; and
4. pass only structural checks fixed before model features or neural values.

The expected public design has 720 shared concepts, but neither 720 concepts
nor 3,840 common images is hard-coded as an observed local fact. If exactly 720
eligible concepts are not recovered, neural-outcome access remains closed
unless the scientist freezes a revised split before any neural outcome is
opened.

### Role assignment

Using metadata only, concepts are deterministically assigned by a versioned
hash-and-balance procedure to:

- 480 development concepts;
- 120 support/calibration concepts; and
- 120 sealed-audit concepts.

The procedure balances frozen THINGS/THINGSplus taxonomy strata, the
five-versus-six-exemplar pattern, participant acquisition order, session
coverage, and repetition lag. Before role assignment, the taxonomy input is
pinned by release, file hash, concept mapping, unmapped and multilabel rules,
group-merging rule, and minimum cell size. Balance tolerances and tie-breaking
are fixed in the role-manifest generator. Candidate-model embeddings cannot
define or repair the split.

Every image, repetition, participant row, beta stage, and derived neural value
inherits its concept's role. No concept can cross roles. Development concepts
form six fixed outer folds of 80 concepts and 24 independent uncertainty
blocks of 20 concepts. Sealed audit concepts form twelve fixed influence and
uncertainty blocks of 10 concepts. These counts become operative only after
the eligibility gate passes.

### Permission boundary

The source B/C/D files physically mix all roles. Acquisition may copy and hash
those bytes into a restricted raw area, but the adaptive search worker never
receives that area. A trusted splitter creates:

1. a metadata-only handoff;
2. a builder-only calibration-neural handoff;
3. a search-worker handoff containing development neural values plus only the
   frozen calibration-derived reliability/support/ceiling artifacts; and
4. an evaluator-only sealed-audit handoff.

The metadata handoff exposes only an allowlist of image/concept IDs,
presentation indices, session/run/trial positions, acquisition order, and
repetition lag, plus a provider-defined schedule-exception flag. Recognition
responses, correctness, reaction times, and other behavioral outcomes are
excluded from role assignment.

The role manifest and handoff hashes are immutable. File permissions are a
required implementation control, while the no-outcome-access rule is the
scientific audit boundary. Raw calibration values cannot be used for candidate
fitting, selection, or scoring.

## Controlled model-pair panel

Before any candidate-discriminating neural score is opened, freeze three
required promotion-eligible controlled pairs and at most one optional fourth.
Each pair isolates exactly one declared contrast:

- training objective at fixed architecture and image corpus;
- architecture at fixed objective and image corpus; or
- training diet at fixed architecture and objective.

At least one trained-versus-deterministically-initialized random pair is a
required falsifier and is never promotion-eligible. Across all pairs, use six
to eight unique checkpoints, including random checkpoints. Reusing a
checkpoint across controlled pairs is allowed only when each pair still
isolates one contrast; the resulting comparisons are explicitly dependent.

Every checkpoint needs a version, weight hash, license, parameter count,
training-corpus lineage, preprocessing recipe, eligible layers, feature shape,
and target-stimulus exposure status. The admissible exposure labels are:

- `lineage_audited_no_known_target_exposure`;
- `known_related_corpus_exposure`;
- `known_exact_or_near_duplicate_exposure`; and
- `exposure_unknown`.

Known target exposure makes a checkpoint a control, not a clean promotion
candidate. Unknown exposure is not evidence of absence.

All models receive the same dimensionality rule, layer-selection budget, PCA
rank grid, ridge grid, outer concept folds, and score weights within a pair.

## Measurement-contract graph

### Beta stages

CNeuroMod exposes one aligned GLMsingle family:

| Stage | Frozen interpretation |
| --- | --- |
| B | fitted HRF, without GLMdenoise or ridge regularization |
| C | fitted HRF plus GLMdenoise, without ridge regularization |
| D | fitted HRF plus GLMdenoise plus ridge regularization |

The pure beta edges are B→C and C→D. On either edge, image IDs, voxel identity,
folds, score family, feature representation, and readout procedure remain
fixed. Any stage-specific voxel replacement turns the comparison into a
`beta_x_support_composite` and removes isolated-axis credit.

### Voxel support

Within each preregistered anatomical ROI:

- $A_{all}$: all common finite atlas voxels;
- $A_N$: the mean estimand over a bank of $K$ deterministic hashed anatomical
  subsets of size $N$, with frozen subparcel and spatial-bin quotas; and
- $R_N$: the top reliability voxels selected only from calibration concepts,
  under the identical $N$, subparcel, and spatial-bin quotas.

$A_{all}\to A_N$ tests support size while representing subset variability.
$A_N\to R_N$ tests reliability selection at fixed size and spatial
composition. Without exact quota matching, the edge is a composite support
policy and cannot support a pure support reversal.

### Noise ceiling

Stage-specific B, C, and D ceilings are recomputed from calibration concepts
and reserved repetitions using one frozen estimator. The public D ceiling is
a post-lock positive-control diagnostic only. Because its source computation
may mix stimulus roles, it is never visible to the search worker and cannot
select support, normalize a terminal score, or alter a candidate.

For a pure ceiling-treatment edge, define one common finite-NC voxel set before
model scores. Raw and normalized held-out $R^2$ use identical betas, images,
voxels, and predictions. Only the reported value changes:

$$
R^2_{norm,v}=\frac{R^2_v}{\max(NC_{b,v},\tau_b)}.
$$

Negative raw $R^2$ and normalized values outside $[0,1]$ are retained.
Changing voxel identity because of NC eligibility creates a
`support_x_ceiling_composite`, not a pure ceiling effect.

### Required isolated edges

Every required promotion pair must cover:

1. B→C on fixed voxels;
2. C→D on fixed voxels;
3. $A_{all}\to A_N$ at fixed beta stage;
4. $A_N\to R_N$ at fixed $N$ and spatial quotas; and
5. raw $R^2\to R^2/NC$ on the identical $V_{NC}$.

With three required pairs, initial coverage is therefore 15 valid trial atoms.

Initial isolation uses fixed anchors: beta edges use outcome-independent
$A_{all}$; support edges use stage D; and the required ceiling-treatment edge
uses stage D. Every mandated score is emitted, but a relation ID binds exactly
one score scale and its own margin. Other cross-contract combinations are
successors, not substitutes for these five anchors.

### Frozen terminal contract universe

Before the first candidate-discriminating neural score, freeze one hashed
`eligible_terminal_contract_manifest` for every controlled pair and score
family. It enumerates the exact contract vertices and directed edges that may
drive a stable-relation or reversal terminal, the outcome-blind applicability
rule for each entry, and its exact development-to-audit homologous mapping.

At minimum, wherever the score is mathematically defined, the manifest
contains both endpoints of all five required isolated edges above. A vertex
may be declared inapplicable only by a prespecified structural or
calibration-feasibility rule evaluated before candidate scores. Its exclusion
and reason are retained; it cannot be replaced by a more favorable vertex.
Once any candidate score is visible, the terminal-driving set cannot be
shrunk, expanded, or relabeled. Later contracts are diagnostics or successors
and cannot retroactively make a relation `contract_robust`.

## Linking and response contract

### Primary response

For participant $s$, beta stage $b$, unique image $i$, and voxel $v$, the
primary response is the equal mean of all retained trialwise betas:

$$
\bar y_{s,b,i,v}
=
\frac{1}{R_{s,i}}\sum_{r=1}^{R_{s,i}}y_{s,b,i,r,v}.
$$

Each unique image then has equal weight in fitting and scoring. Trialwise,
inverse-variance-weighted, or presentation-weighted endpoints are
sensitivities, not substitutes for the primary estimand.

### Nested development fitting

The primary linking model is voxelwise ridge regression. In each development
outer fold, every learned operation is fit inside its training concepts:

1. deterministic image preprocessing and frozen feature extraction;
2. feature centering and PCA;
3. inner-fold layer, PCA-rank, and ridge selection; and
4. prediction on the outer-held-out concepts.

No voxel, ROI, image, layer, rank, or regularization choice may use an
outer-fold score. The final audit predictor is refit once on all development
concepts using a predeclared outer-fold-to-final aggregation and tie-break
rule. Its feature transform, hyperparameters, and voxel coefficients are hashed
before the audit handoff is opened.

### Correlation score

For model $m$, participant $s$, contract $c$, and scientific stratum $g$:

$$
S^r_{m,s,c,g}
=
\frac{1}{|V_{s,c,g}|}
\sum_{v\in V_{s,c,g}}
\operatorname{atanh}\!\left(r_{m,s,v,c,g}\right).
$$

The pairwise effect is:

$$
\Delta^r_{ab,c,g}
=
\frac{1}{4}\sum_{s=1}^{4}
\left(S^r_{a,s,c,g}-S^r_{b,s,c,g}\right).
$$

The atanh boundary tolerance is frozen before held-out outcome access.
Degenerate correlations are invalid rather than silently replaced.

### Held-out explained variance

For an evaluation set $I$, the voxelwise held-out quantity is:

$$
R^2_{m,s,v,c,I}
=1-
\frac{\sum_{i\in I}(y_i-\hat y_{m,i})^2}
     {\sum_{i\in I}(y_i-\bar y^{train}_{s,v,c})^2}.
$$

The baseline mean always comes from the corresponding development training
data, including on sealed audit concepts. Raw and ceiling-normalized pairwise
effects are means of voxelwise differences or ratios as declared; a ratio of
ROI means is prohibited.

Correlation and explained variance are different score families. Their
numerical margins and bounds are never pooled. Disagreement between them is
metric-target dependence, not a ceiling-only reversal.

### Fixed linking falsifier

Fixed-dimensional linear CKA is mandatory for locked relations. For a
column-centered held-out feature matrix $X^{(q)}$ and neural matrix $Y$:

$$
G(X^{(q)},Y)
=
\frac{\|{X^{(q)}}^T Y\|_F^2}
{\|{X^{(q)}}^T X^{(q)}\|_F\,\|Y^T Y\|_F}.
$$

Use the locked model layer, training-only centering and PCA, one common
prespecified rank $q$, the same held-out images, response, beta/support
contract, and unique-image fold weights. A zero denominator invalidates the
fold. CKA has its own signed practical margin. Linking dependence requires
encoding and CKA to cross opposite margins under the same contract and
stratum; their scales are never combined.

## Scientific strata are not measurement axes

Preregistered anatomical ROIs and one externally defined THINGS/THINGSplus
semantic grouping may support conditionality claims. The taxonomy release,
file hash, concept mapping, unmapped and multilabel handling, group-merging
rule, minimum cell size, pooling rule, and interaction margin are frozen before
neural scores. Candidate-model embeddings cannot define a scientific stratum.

A relation that differs by ROI or semantic group is scientific
specialization. It is not pipeline fragility and does not count as an
isolated-axis reversal.

## Uncertainty and identification

Participants and concepts are crossed, not nested. The primary uncertainty
procedure uses 10,000 fixed-seed crossed bootstrap draws:

1. resample the four participants with equal weight;
2. resample the 24 development or 12 audit concept blocks with replacement;
3. keep every exemplar and repetition of a concept together; and
4. recompute all signed components of one relation jointly.

Simultaneous max-$t$ bounds control the components of one relation within one
score family. Frozen out-of-fold predictions and selected hyperparameters are
treated as fixed in the primary bootstrap; a full selection-refit bootstrap is
a nonterminal sensitivity.

The bootstrap quantifies uncertainty over these participants and stimulus
blocks. It does not license population inference from four people.

For a relation on one score scale, let
$[L_{ab,c,g},U_{ab,c,g}]$ be the simultaneous bound under contract $c$.
Its descriptive contract uncertainty envelope is:

$$
E_{ab,g}
=
\left[\min_c L_{ab,c,g},\;\max_c U_{ab,c,g}\right].
$$

This envelope is not a new confidence interval and never combines correlation,
variance, normalized variance, or CKA scales.

For one score family and practical margin $\epsilon$:

- stable positive: every eligible-contract lower bound is $>\epsilon$;
- stable negative: every upper bound is $<-\epsilon$;
- genuine reversal: the two ends of one isolated edge cross opposite margins;
- practical equivalence: all simultaneous bounds lie inside
  $[-\epsilon,\epsilon]$; and
- resolution-limited: neither a margin nor the equivalence region is
  identified.

Specialization additionally requires the interaction
$\Gamma=\Delta_{g_1}-\Delta_{g_2}$ and both within-stratum effects to cross
their prespecified signed margins.

## Adaptive-search contract

### Trial atom

One valid trial is:

> one registered model pair × one directed isolated contract edge × all four
> development participants.

It includes every frozen ROI and stimulus stratum, paired concept folds, all
required score outputs, uncertainty outputs, and native falsifiers. A trial
cannot be split into extra counts by participant, ROI, concept block,
repetition, fold, seed, subset-bank member, score scale, or scheduler job.

The unique key is a scientific-configuration hash. An engineering retry keeps
the same ID only if the scientific configuration is byte-identical.

### Depth and budgets

- 32 minimum and 72 maximum valid trials;
- 15 mandatory initial coverage trials for three required pairs;
- at least two outcome-adaptive successor cycles;
- at least two incumbent/challenger decisions;
- at least 40% falsification trials after initial coverage;
- patience of 12 qualified valid trials, starting only after trial 32 and
  complete branch coverage;
- 6,000 CPU-core-hours, 256 GPU-hours, 240 wall-clock hours, and 4 TiB
  transient scratch ceilings.

Qualification failures and engineering failures do not count as valid trials.
Budget exhaustion is not scientific success.

### Outcome-linked successors

Every successor must name its scored parent, directional prediction,
competing explanation, native falsifier, expected information gain and cost,
retirement condition, visible-ledger prefix hash, and exactly one changed
scientific operator.

| Observed pattern | Required next attack |
| --- | --- |
| Reliability-localized effect | Fixed-identity reliability strata, $A_N$ versus $R_N$, and repeat split |
| B/C or C/D reversal | Fixed support and score scale, stage-specific NC, layer stability, and repeat split |
| Ceiling-only reversal | Alternate frozen NC estimator/floor and no-clipping check |
| Apparent universal stability | CKA, participant influence, concept-block influence, and semantic-stratum attacks |
| ROI/semantic flip | Fixed-contract interaction and block stability |

A branch retires only after at least two valid configurations and a direct
falsifier or replication. A failed qualification check or one nonsignificant
trial cannot retire a scientific branch.

## One-shot sealed audit

After development stops, lock exactly one terminal-driving primary relation
and at most two secondary checks. Secondary checks are disclosed but cannot
rescue the primary.

The lock binds:

- model and checkpoint hashes;
- image preprocessing, features, layer, PCA rank, and ridge choices;
- final voxel coefficients fitted only on development concepts;
- B/C/D, ROI, support, NC, and score contracts;
- concept roles, audit blocks, response aggregation, and exclusions;
- all practical margins, adequacy widths, and uncertainty seeds;
- evaluator code, environment, permissions, and expected output schema; and
- the EP17/EP18 exposure decision.

The trusted evaluator opens the sealed handoff once and applies the frozen
predictor directly to the 120 audit concepts. It may not refit a voxel
coefficient, choose a layer, change support, repair a subset, alter a margin,
or rerun after seeing a partial result.

### Positive audit gate

The primary relation passes only if:

1. every registered signed component clears its simultaneous bound and
   practical margin;
2. the same intersection of at least three of four participants exceeds the
   corresponding participant-level margin for every signed component;
3. every leave-one-participant mean retains the required direction;
4. every leave-one-audit-block mean retains the required direction;
5. integrity, exposure, and exact-score-contract checks pass; and
6. no audit neural outcome influenced fitting or selection.

### Negative versus uninformative audit

`closed_heldout_concept_nonreplication` is allowed only when prespecified,
score-specific adequacy widths are narrow enough to identify practical
equivalence, an opposite effect, or stable participant/concept heterogeneity.

If the simultaneous bounds still overlap both the pass region and the
null/equivalence region, the outcome is `closed_audit_underidentified`. It is
non-evidential, not a negative scientific result.

If metadata or structural compatibility fails before any audit outcome is
opened, audit access remains closed. If the frozen audit pool becomes
incommensurate after audit access opens but before valid scoring, it closes as
`closed_audit_pool_incommensurate`.

## Terminal outcomes

Positive candidate terminals:

- `candidate_ready_contract_robust_pair`;
- `candidate_ready_heldout_concept_measurement_reversal`;
- `candidate_ready_stable_conditional_specialization`; and
- `candidate_ready_linking_criterion_dependence`.

Scientific or non-evidential closures:

- `closed_underidentified`;
- `closed_no_isolated_relation`;
- `closed_unresolved`;
- `closed_heldout_concept_nonreplication`;
- `closed_audit_underidentified`; and
- `closed_audit_pool_incommensurate`.

Integrity failures are `technical_failure` or `policy_violation`, never a
scientific null.

## Requirements before neural-outcome access

Downloading the source does not grant neural-outcome access. Candidate scoring
and audit access remain closed until:

- the THINGS/THINGSplus taxonomy release, hash, concept mapping, label-handling
  rules, group merges, and minimum cell size are frozen;
- the 720-concept eligibility and exact four-person image intersection are
  verified from events;
- the 480/120/120 role manifest and role-filtered handoffs are frozen;
- B/C/D trial axes, response units, geometry, and voxel indices align;
- B/C/D stage-specific NC estimators are frozen and calibration-feasible;
- all ROIs, $N$, $K$, spatial quotas, folds, grids, margins, tolerances,
  audit-adequacy widths, simultaneous-bound estimator, and seeds are frozen,
  with every numeric scientific threshold signed by the scientist;
- the hashed eligible terminal contract manifest and every exact
  development-to-audit homologous mapping are frozen before candidate scores;
- six to eight exact checkpoints form at least three defensible controlled
  pairs and one trained/random falsifier;
- model exposure and EP17/EP18 split-exposure ledgers are signed;
- the permission-separated evaluator passes a synthetic dry run; and
- no mixed raw neural source is mounted to the search worker.

## Claim boundary

A successful EP17 would show that a controlled visual-model relation survives
specified measurement choices and one concept-disjoint test in the same four
densely sampled people.

It would not establish:

- cross-dataset transport;
- generalization to new participants;
- population prevalence;
- universal model ranking;
- causal superiority of an architecture, objective, or training diet; or
- correctness outside the registered linear-readout and contract family.
