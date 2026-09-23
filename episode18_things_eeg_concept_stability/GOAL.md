# EP18 — cross-exemplar concept-template stability in THINGS-EEG

This episode follows the
[common adaptive-search protocol](../ADAPTIVE_SEARCH_PROTOCOL.md). Data facts,
event fields, and access requirements are specified in
[DATASETS.md](DATASETS.md).

## Scientific question

For the fixed THINGS concept and image inventory, does a regularized
categorical concept basis predict held-out continuous EEG better than a
label-aware basis with matched fitting capacity, when both include the same
image-derived feature panel? Does the true THINGS grouping also outperform
alternative 12-image groupings matched to the same fold-specific visual and
temporal structure?

For a concept such as `dog`, ten images train the concept-specific response
and two different dog images test it. The question is whether that shared
response improves prediction after pixel-derived and label-derived models have
had comparable fitting capacity.

Concept-name and human concept features already identify the named concepts.
A categorical term therefore does not introduce nominal information that was
absent from those features. It introduces a different predictive basis and
inductive bias. EP18 tests whether that categorical basis has out-of-sample
predictive value under a frozen feature and readout family.

A positive result would support a reproducible categorical concept-template
increment conditional on that family. It would not establish abstract,
amodal, purely semantic, or vision-independent coding.

The EEG release and the authorized THINGS image archives have been acquired
and verified, so the sources can support a future outcome-blind setup phase
that reconstructs events, builds the exact image-event manifest, and assigns
participant roles. EP18 remains an incomplete local draft: no episode-managed
run may launch until its input/output guards and seven workspace projections
exist. After that scaffold is complete, EEG-informed candidate comparison may
begin only after the manifest is validated, participant roles are frozen, the
development view is provisioned, and audit EEG is evaluator-only as specified
in `DATASETS.md`.

## Why this episode is needed

Cross-exemplar concept structure is not a new result by itself. The original
THINGS-EEG1 paper reported it, and later work showed that low-level visual
statistics can produce apparently semantic decoding. Other studies have
compared perceptual, conceptual, vision-model, and language-model structure in
THINGS EEG and MEG:

- [Grootswagers et al. (2022)](https://doi.org/10.1038/s41597-021-01102-7)
  introduced THINGS-EEG1 and its cross-exemplar analyses;
- [Holm et al. (2024)](https://doi.org/10.1016/j.neuroimage.2024.120626)
  demonstrated low-level visual confounding in THINGS-EEG; and
- [Kim et al. (2026)](https://doi.org/10.1167/jov.26.8.2),
  [Watson et al. (2026)](https://doi.org/10.1016/j.neuroimage.2026.122012),
  and [Rong et al. (2025)](https://doi.org/10.7554/eLife.108915) examined
  perceptual, conceptual, behavioral, vision-model, and language-model
  contributions in related datasets.

EP18's novelty threshold is therefore stricter: single-exemplar prediction in
continuous 10 Hz EEG; explicit modeling of overlapping responses; a strong
feature-and-readout sufficiency baseline; true concepts tested against visually and
temporally matched alternatives; and replication in whole participants not
used for development.

## Competing explanations

1. **Categorical concept-template increment.** Different images with the same
   concept share EEG structure that remains predictive after the frozen image
   features and label-aware, capacity-matched readout.
2. **Feature-and-readout sufficiency.** Visual, vision-language,
   label-semantic, or human-perceptual features explain the predictable
   structure once their readout has comparable capacity.
3. **Partition nonspecificity.** True concepts help, but matched alternative
   image groupings help just as much.
4. **RSVP sequence artifact.** The apparent effect comes from neighboring
   images, sequence position, filtering, targets, responses, or drift.
5. **New-participant nonreplication.** A development effect does not recur
   under the fixed procedure in audit participants.
6. **Measurement limitation.** Reliability, design rank, or precision is
   insufficient to distinguish these explanations.

## Evidence structure

The provider marks four of the 50 participants for exclusion, leaving 46
potentially eligible participants. The planned split is:

- 30 whole participants for development; and
- 16 whole participants for audit.

The deterministic assignment is generated and recorded during episode setup
from provider metadata and fixed integrity checks. It is an episode output,
not a missing external dataset.

Exact eligibility and assignment rules are in `DATASETS.md`. Participant IDs
must be fixed before any EEG-derived quantity is used to choose among analysis
candidates, and no audit participant may later be replaced. The participant is
the inferential and replication unit; folds, sequences, images, electrodes,
and EEG samples are repeated measurements.

All participants saw the same main image collection. A scored exemplar is new
to that participant's fold-specific fit, but the image may already have been
seen in a development participant. The primary claim is therefore
cross-exemplar replication in new participants, not generalization to globally
unseen images.

The audit interval treats participants as the sampling unit and conditions on
the observed 1,854 concepts and 22,248 images. It does not estimate
generalization to unseen concepts or to a new image population.

## Within-participant folds and continuous model

The main session has 12 blocks. Each block presents one exemplar of every
concept and is divided into six physical sequences. Use six fixed outer folds:
fold \(j\) scores blocks \(j\) and \(j+6\), for \(j=0,\ldots,5\), and fits on
the other ten blocks. Every image is assigned to a held-out fold exactly once;
the fixed boundary mask determines which EEG samples contribute to its score.

Within each fold:

- adaptive preprocessing, feature reduction, covariance estimation,
  regularization, and coefficient fitting use only the ten training blocks;
- physical-sequence boundaries are preserved;
- samples affected by filter or FIR boundary support are excluded from
  scoring; and
- no raw or transformed EEG sample contributes to both fitting and scoring.

The five remaining block pairs form five inner folds. Each inner fit uses
eight blocks and validates on two. Every permitted training-derived transform,
feature reduction, whitening covariance, temporal rank, model rank, and
penalty is re-estimated inside every inner-training split. For a participant
and outer fold, the inner one-standard-error rule uses the mean and standard
error across those five block-pair losses. After one rule is selected, it is
refitted on all ten outer-training blocks; the two outer-held-out blocks are
scored once. Lower-complexity and lexicographic tie rules are fixed in
`SEARCH_POLICY.yaml`.

At 10 Hz, an epoch around one image contains responses to several neighboring
images. The primary analysis therefore predicts continuous multichannel EEG
with one time-expanded model over every event in a physical sequence:

\[
y_p(t)
=
\sum_{\tau\in\mathcal T} W_{p,\tau}x_p(t-\tau)
+
\sum_{\tau\in\mathcal T} U_{p,\tau}z_p(t-\tau)
+\epsilon_p(t).
\]

Here, \(x\) contains the fixed image and nuisance features, \(z\) contains true
or alternative group membership, and \(\mathcal T\) is a fixed causal FIR or
low-dimensional temporal basis. Every image enters at its actual event time,
so overlapping responses are modeled jointly rather than treated as
independent epochs.

The primary pipeline may not use ordinary prestimulus baseline correction that
folds responses to earlier images into the baseline. A two-sided filter, if
retained, must operate within each physical sequence and discard its full edge
influence. No filter, drift estimate, bad-channel rule, normalization,
covariance estimate, or artifact model may learn from held-out blocks.

The primary sensor space begins with the fixed set of 62 recorded channel
labels shared by the standard montage and the `sub-49`/`sub-50` montage. Any
participant- or fold-specific bad-channel removal and interpolation is
prohibited in the primary analysis. Remove the across-channel mean and express
the data in one frozen 61-dimensional orthonormal contrast basis for that
common-average subspace.
Estimate and regularize the whitening covariance on training blocks in this
full-rank basis; do not invert a 62-channel common-average covariance. Channels
outside the shared set are sensitivity analyses only and cannot select or
rescue the primary result.

## Models being compared

### Image-derived model \(M_{\mathrm{image}}\)

The image-derived model contains:

- luminance, contrast, color, spatial frequency, texture, image geometry,
  object extent, foreground/background structure, and composition;
- intermediate and final representations from supervised and self-supervised
  vision models;
- image-encoder and automatically generated caption representations from the
  frozen VLM panel; and
- target onsets, validated button responses, physical-sequence position,
  sequence boundaries, and slow drift.

Every image-derived feature job receives only pixels and an opaque image ID.
It may not read a path, filename, folder, concept name, human category, or a
prompt constructed from those fields. Preceding and following images enter as
their own rows at their actual event times in the joint temporal design; they
are not copied into extra current-image columns.

### Label-aware models

\(M_{\mathrm{label,lin}}\) adds frozen concept-name embeddings and fixed human
perceptual or conceptual dimensions to \(M_{\mathrm{image}}\), using the
declared linear readout.

\(M_{\mathrm{label,cap}}\) adds one nonlinear kernel slot, built only from
those same continuous label features, to \(M_{\mathrm{label,lin}}\). It never
receives a one-hot concept ID. On every outer-training fold, that slot's
conditional effective degrees of freedom must match or slightly exceed those
of the categorical slot within the frozen tolerance. It also receives the same
temporal basis and frozen 61-dimensional output space. A candidate that cannot
make this match is invalid; it cannot fall back to the weaker linear label
baseline.

Encoder weights, raw outputs, label features, kernels, and the admissible
feature-family menu are fixed without EP18 EEG. Declared downstream reductions
and penalties may use only nested training-block development EEG. A mandatory
family cannot be removed because its removal increases the categorical effect.
At least one independently trained expanded image panel is required, and the
terminal decision is made on the additive expanded panel; the core-panel
result is explanatory and cannot substitute for it.

The common encoding fit is also fixed. Training-only PCA produces whitened
component scores; a requested rank above a family's numerical rank is replaced
by that numerical rank and recorded. After temporal expansion, each penalized
predictor block is scaled from its training rows to unit root-mean-square
energy. Nuisance terms are unpenalized; image blocks share one ridge penalty,
linear label blocks share a second, and the added replacement slot has its own
ridge penalty. All penalties use the frozen grid and scaling convention in
`SEARCH_POLICY.yaml`. Continuous label features are normalized without EEG,
with equal total weight for each declared feature family, before either their
linear readout or the nonlinear label kernel is constructed.

The image and label-readout recipe is selected only by its own held-out
prediction loss. The concept and pseudo-group contrasts cannot choose a weaker
baseline, alter branch retirement, or break a tie. The expanded panel keeps
all core feature blocks and adds its extra encoder; a shared rank cap cannot
silently displace core features.

### Categorical and alternative-template models

\(M_{\mathrm{image,true}}\) adds the regularized true-concept template to
\(M_{\mathrm{image}}\). This comparison describes how much the template adds
to image-derived features alone; it does not drive the terminal claim.

\(M_{\mathrm{true}}\) adds one true-concept categorical slot to
\(M_{\mathrm{label,lin}}\), while \(M_{\mathrm{pseudo},k}\) places one
alternative grouping in that same slot. Thus \(M_{\mathrm{label,cap}}\),
\(M_{\mathrm{true}}\), and each \(M_{\mathrm{pseudo},k}\) have matched
capacity within the declared 5% conditional-effective-degrees-of-freedom
tolerance: they share \(M_{\mathrm{label,lin}}\) and differ only in the added
slot's basis. These are competing replacements for one added slot:
\(M_{\mathrm{true}}\) does not also contain the nonlinear label-kernel slot.
The added slots use the same temporal basis, sensor output space, penalty menu,
tuning budget, and effective-capacity rule. An unconstrained
`1,854 × channels × lags` lookup table is not allowed.

All models use identical folds, samples, transformations, temporal support,
nuisance terms, covariance, and fitting opportunities. Each is fitted directly
to EEG with all of its covariates included. Fitting a baseline once and then
searching exported EEG residuals is not allowed. Because unique label
embeddings and a categorical basis can span overlapping predictive spaces, any
surviving gain is interpreted as an advantage of the frozen categorical basis
and regularization, not as new nominal information absent from the label
features.

## Matched alternative partitions

Random labels test capacity but do not distinguish concepts from unmodeled
visual clusters. The primary comparator bank therefore contains alternatives
built without EEG.

The primary bank contains exactly eight alternatives. Every alternative has
1,854 groups of 12 images, covers all 22,248 main images exactly once, and
places one image from every main block and 12 different true concepts in each
group. It uses the same group term, temporal basis, regularization, and tuning
budget as \(M_{\mathrm{true}}\).

Each pseudo-group is treated exactly like a concept: its ten training-block
images predict its two held-out images. Matching design spectrum, leverage,
and effective degrees of freedom prevents a pseudo-grouping from being easier
or harder to fit merely because of its numerical geometry.

For every outer fold separately, the generator must match true groups on:

- held-out-to-training nearest-neighbor and centroid distances in low-level,
  vision-model, and image-derived VLM or caption spaces;
- training and held-out within-group scatter;
- the post-constraint design spectrum, leverage, and conditional effective
  degrees of freedom; and
- physical-sequence position and all image, target, and validated-response
  context inside the complete FIR, filter, and scoring-support horizon.

A partition that matches only full 12-image dispersion but fails a
held-out-versus-training diagnostic in any outer fold is not admissible.

Concept-name and concept-level human features cannot be matched: their
within-group dispersion is zero for a true concept but not for a group of 12
different concepts. Those continuous features remain in the shared
\(M_{\mathrm{label,lin}}\), and the remaining semantic mismatch must be
reported.

Image order differs across participants, so a fixed generator may create a
different literal partition for each participant. Index \(k\) denotes the same
predeclared generator seed, objective, fold-specific tolerances, diversity
rule, acceptance order, and retry ceiling across participants. These choices
are frozen before any concept comparison. If eight acceptable partitions
cannot be generated within that ceiling, the episode is underidentified; the
tolerances may not be relaxed after EEG inspection. Achieved matching quality
is reported for every participant, fold, and partition.

Random partitions remain separate capacity diagnostics. The matched bank is a
finite set of competing models, not an exchangeable permutation sample, and
must not be reported as a permutation-test null.

## Primary quantities and decision rule

Participant \(p\)'s held-out data in outer fold \(f\) comprise 12 physical
sequences with model-independent scoring masks \(S_{pfq}\), set by timing,
boundary, missingness, and artifact rules. Define the whitened error in
sequence \(q\) as

\[
E_{pfq}(M)=\frac{1}{61|S_{pfq}|}\sum_{t\in S_{pfq}}
\left\|\Sigma_{pf}^{-1/2}
\left[y_p(t)-\widehat y_{pfM}(t)\right]\right\|_2^2,
\]

and the primary loss as

\[
L_{pf}(M)=\frac{1}{12}\sum_{q=1}^{12}
\frac{E_{pfq}(M)}{E_{pfq}(M_{\mathrm{nuisance}})}.
\]

Each retained continuous sample is scored exactly once. Sequences are equally
weighted, samples are equally weighted within a sequence, and the 61 frozen
sensor contrasts are whitened with one covariance \(\Sigma_{pf}\) estimated
from a fixed nuisance-only model on the outer-training blocks. The same mask
and covariance are used for every model in the fold. Division by the held-out
nuisance-model error makes the loss a dimensionless fraction of nuisance-only
error and is identical across candidate comparisons. Here,
\(M_{\mathrm{nuisance}}\) contains only target, response, sequence-position,
boundary, and drift terms. Lower loss is better; a value of 1 means no
improvement over that nuisance-only prediction. No sample may be removed
because of a candidate residual, concept label, or observed effect. Concept
and category concentration are reported as influence diagnostics rather than
silently changing the loss weights.

For each fold, define

\[
d^{\mathrm{image}}_{pf}
= L_{pf}(M_{\mathrm{image}})-L_{pf}(M_{\mathrm{image,true}}),
\]

\[
d^{\mathrm{cap}}_{pf}
= L_{pf}(M_{\mathrm{label,cap}})-L_{pf}(M_{\mathrm{true}}),
\]

\[
c^{(k)}_{pf}
= L_{pf}(M_{\mathrm{pseudo},k})-L_{pf}(M_{\mathrm{true}}).
\]

Positive \(d^{\mathrm{image}}_{pf}\) is the image-panel increment. Positive
\(d^{\mathrm{cap}}_{pf}\) means the categorical basis improves on the
capacity-matched label readout. Positive \(c^{(k)}_{pf}\) means the true
grouping predicts better than matched alternative \(k\).

Average the six folds before inference:

\[
d^{\mathrm{image}}_p=\frac{1}{6}\sum_f d^{\mathrm{image}}_{pf},
\qquad
d^{\mathrm{cap}}_p=\frac{1}{6}\sum_f d^{\mathrm{cap}}_{pf},
\qquad
c^{(k)}_p=\frac{1}{6}\sum_f c^{(k)}_{pf}.
\]

Let \(P_B\) be the fixed set of 16 audit participants. The audit estimands are
equal-participant means:

\[
\Delta_{\mathrm{image}}=\frac{1}{|P_B|}\sum_{p\in P_B}d^{\mathrm{image}}_p,
\qquad
\Delta_{\mathrm{cap}}=\frac{1}{|P_B|}\sum_{p\in P_B}d^{\mathrm{cap}}_p,
\qquad
C_k=\frac{1}{|P_B|}\sum_{p\in P_B}c^{(k)}_p.
\]

A positive primary result requires all of the following:

1. the cohort-level image/task positive control exceeds its fixed floor;
2. on the frozen additive expanded image panel, the simultaneous lower audit bound
   for \(\Delta_{\mathrm{cap}}\) exceeds practical margin \(\delta_C\);
3. on that panel, simultaneous lower bounds for all eight \(C_k\) exceed
   partition margin \(\delta_P\);
4. temporal, task, sequence, and label-shift controls pass their fixed
   margins; and
5. leave-one-participant, held-out-block-pair, physical-sequence, and broad
   object-category analyses remain within a fixed influence tolerance.

\(\Delta_{\mathrm{image}}\) and the linear-label comparison are explanatory;
neither can rescue failure against \(M_{\mathrm{label,cap}}\). The rules for
\(\delta_C\) and \(\delta_P\) are fixed by the label-free image positive
control rule in `SEARCH_POLICY.yaml`. Development reliability, injections, and
null simulations must qualify those margins but cannot lower them, and the
observed true-versus-pseudo effect cannot enter their calculation. The interval
and multiplicity procedure, positive-control floors, influence tolerance,
equivalence bounds, image panels, and matched bank are set before audit.

Failure to establish \(C_k>\delta_P\) is not by itself evidence of partition
nonspecificity. That conclusion requires adequate precision and a predeclared
equivalence test showing that a matched alternative is comparable to, or
better than, the true grouping; otherwise the result is underidentified.

## Development and audit boundary

Development may compare bounded choices in:

- causal FIR versus low-rank temporal bases;
- training-only feature reductions;
- image and linear-label ridge penalties;
- nonlinear label-kernel bandwidth; and
- downstream choices within the admissible expanded-panel menu. Every panel
  designated as required is carried into audit.

The true and pseudo categorical slots always use the same frozen full penalty
grid and nested rule. Their scores, selected penalties, and capacity-match
status cannot change the adaptive search grammar or choose the baseline.

Every candidate evaluates \(M_{\mathrm{image}}\),
\(M_{\mathrm{label,lin}}\), \(M_{\mathrm{label,cap}}\),
\(M_{\mathrm{true}}\), and the primary matched bank on identical folds and
samples. Development may strengthen the image or label baseline. It may not
choose a favorable weak baseline, create EEG-informed partitions, select a
different winner for each participant, or inspect audit EEG to decide what to
try next.

During adaptive baseline search, the controller sees only
\(M_{\mathrm{label,cap}}\) prediction, positive controls, conditioning,
complexity, and resource use. Concept, pseudo-group, and concept-dependent
control scores—and even pass/fail proxies for them—remain sealed. The baseline
is chosen across the 30 participant-level outer-CV losses and locked
irreversibly. Only then are the already computed scientific contrasts for that
one locked configuration released. No further development candidate may be
proposed after that release; a failed control closes or classifies the episode
rather than sending the search back to a more favorable baseline.

One procedure advances to audit. For each audit participant, the evaluator
fits that procedure on ten blocks and scores the two held-out blocks in each
outer fold. Feature families, admissible basis and rank grids,
penalty-selection rules, masks, partitions, controls, margins, and decisions
may not change. Coefficients and the realized rank or penalty may vary by
participant and outer fold only through that locked nested-selection rule.

This primary audit tests whether the participant-refit procedure reproduces
the cross-exemplar increment in new participants. It is not zero-shot
application of a development-participant EEG template. Shared-template
transport is outside the terminal contract for this episode; if performed
later, it is exploratory and cannot qualify, rescue, or overturn EP18.

Before audit access, the locked procedure must pass a precision gate using
development participant vectors and blinded injections under the same
simultaneous interval rule planned for 16 participants. If the projected
simultaneous half-width cannot distinguish \(\delta_C\) or \(\delta_P\), the
episode closes as underidentified without opening audit EEG.

Audit reliability and positive controls are evaluated only at the cohort
level after lock. No audit participant may be excluded or replaced because of
signal quality, reliability, or effect direction. A cohort-level failure makes
the result underidentified rather than creating a smaller favorable sample.

## Qualification before candidate comparison

Before model comparison, EP18 must establish that:

1. every retained participant's 12 blocks and 72 physical sequences must be
   reconstructed unambiguously from events and raw markers;
2. filtering, FIR construction, and boundary trimming must prevent any EEG
   sample from entering both fitting and scoring;
3. the time-expanded feature and group-membership design must have adequate
   rank and conditioning under the fixed 10/2 folds;
4. outcome-blind simulations must recover declared concept increments without
   confusing concept, neighbor, target, response, drift, or latency terms;
5. the matched-partition generator must meet its visual and sequence targets
   without EEG;
6. development-participant repeated-image reliability and image/task positive
   controls must support a meaningful participant-level test, and the frozen
   16-participant precision projection must resolve both practical margins;
7. all eight fold-matched alternatives must be generated within the frozen
   retry budget; and
8. the 30/16 participant IDs must be frozen, the development runtime must have
   no access to audit EEG values, and the shared EP17/EP18 exposure ledger must
   be initialized.

Failure of rank, injection recovery, reliability, matching, or precision makes
the question underidentified. It cannot be repaired by treating folds, trials,
or samples as independent participants.

## Checks carried into audit

The fixed finalist includes:

- the primary matched partition bank and same-size random capacity controls;
- every required expanded image panel;
- negative-lag or prestimulus checks under the same preprocessing;
- concept-label shifts longer than the complete response and filter horizon,
  with no circular wrap across physical sequences;
- separate target-onset and validated-response terms, sequence position, and
  drift sensitivities;
- filter-support and physical-sequence boundary checks;
- the fixed influence analyses in the primary decision rule;
- null, visual-only, concept-increment, neighbor-effect, and latency-jitter
  simulations using the real event design; and
- positive controls for recoverable image- and task-related EEG signal.

The 200 repeated validation images are used only for signal-recovery and
reliability qualification, positive controls, and injection calibration.
Their repetition, adaptation, and late-session structure prevents treating
them as a primary noise ceiling, and they cannot rescue the main result.
Time-resolved plots are secondary. Their uncertainty and multiplicity rules
must be set in advance, and an isolated latency peak cannot replace the global
continuous-prediction result.

## Possible conclusions

| Outcome | Interpretation | Terminal status |
| --- | --- | --- |
| Categorical concept-template increment | The full conjunctive decision rule passes | `candidate_ready` |
| Feature-and-readout sufficiency | The increment is absent against the strongest required feature and capacity-matched readout, with adequate precision | `closed_no_candidate` |
| Readout insufficiency | An increment against a linear label readout disappears against the capacity-matched label readout | `closed_no_candidate` |
| Partition nonspecificity | With adequate precision, a matched alternative is equivalent within the fixed margin or predicts better | `closed_no_candidate` |
| Sequence or filtering artifact | Negative lags, shifts, task terms, drift, or support checks explain the effect | `closed_no_candidate` |
| New-participant nonreplication | A precise development effect fails in the participant-refit audit | `closed_no_candidate` |
| Underidentified | Reliability, rank, matching, injection recovery, or participant-level precision is inadequate | `closed_no_candidate` |
| Technical or access failure | Event reconstruction, sample separation, preprocessing, audit isolation, or the fixed procedure is violated | `technical_failure` |

## Claim boundary

The strongest permitted conclusion is:

> Within the fixed THINGS-EEG1 concept and image inventory, a categorical
> true-concept template supplied a reproducible cross-exemplar predictive
> advantage over the frozen label-aware capacity-matched basis, with the same
> image-derived panel, and outperformed the fixed bank of fold-matched visual and
> temporal alternatives in participants not used for development.

This is a participant-generalization claim conditional on the observed 1,854
concepts and 22,248 images. It is not an interval over unseen concepts or
unseen image populations.

This does not establish abstract semantics, amodal concepts, lexical coding,
independence from every possible visual feature, causal computation, or a
universal object ontology.

EP17 and EP18 overlap in the THINGS stimulus ecosystem. EP18 may provide new
EEG evidence, but it is not an independent stimulus-family confirmation of
EP17. The two episodes therefore share a stimulus-exposure ledger recording
exact image and concept overlap, feature or checkpoint reuse, and first-access
history.
