# EP18 — residual cross-exemplar concept information in THINGS-EEG

This episode follows the
[common adaptive-search protocol](../ADAPTIVE_SEARCH_PROTOCOL.md). Data facts,
event fields, and access requirements are specified in
[DATASETS.md](DATASETS.md).

## Scientific question

When a participant sees a new image of a concept represented in the training
blocks, does true THINGS concept membership improve continuous-EEG prediction
after a strong visual, vision-language, perceptual, task, and sequence model
has already been fitted?

The decisive comparison is not concept labels versus no labels. It is true
concept membership versus equally flexible alternative groupings that preserve
as much visual and sequence structure as possible.

A positive result would support a residual concept-indexed component relative
to the fixed feature panel. It would not establish abstract, amodal, purely
semantic, or vision-independent coding.

The EEG release and the authorized THINGS image archives have been acquired
and verified. EP18 can therefore begin with an outcome-blind setup phase that
reconstructs events, builds the exact image-event manifest, and assigns
participant roles. EEG-informed candidate comparison begins only after that
manifest is validated, the participant roles are frozen, the development view
is provisioned, and audit EEG is evaluator-only as specified in `DATASETS.md`.

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
feature-sufficiency baseline; true concepts tested against visually and
temporally matched alternatives; and replication in whole participants not
used for development.

## Competing explanations

1. **Residual concept component.** Different images with the same concept
   share EEG structure that remains predictive after the fixed feature panel.
2. **Feature-panel sufficiency.** Visual, vision-language, label-semantic, or
   human-perceptual features explain the predictable structure without a
   categorical concept term.
3. **Partition nonspecificity.** True concepts help, but matched alternative
   image groupings help just as much.
4. **RSVP sequence artifact.** The apparent effect comes from neighboring
   images, sequence position, filtering, targets, responses, or drift.
5. **New-participant nonreplication.** A development effect does not recur
   under the fixed procedure in audit participants.
6. **Participant-specific organization.** Participant-refit replication
   succeeds, but a common EEG template does not transport across people.
7. **Measurement limitation.** Reliability, design rank, or precision is
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
adaptive bad-channel or interpolation transform is learned on training blocks
and applied unchanged to held-out blocks before the final reference transform.
Then remove the across-channel mean and express the data in one frozen
61-dimensional orthonormal contrast basis for that common-average subspace.
Estimate and regularize the whitening covariance on training blocks in this
full-rank basis; do not invert a 62-channel common-average covariance. Channels
outside the shared set are sensitivity analyses only and cannot select or
rescue the primary result.

## Models being compared

### Feature-sufficiency model \(M_0\)

The mandatory baseline contains:

- luminance, contrast, color, spatial frequency, texture, image geometry,
  object extent, foreground/background structure, and composition;
- intermediate and final representations from supervised and self-supervised
  vision models;
- image-derived VLM and caption representations;
- embeddings of the true concept name and fixed human perceptual or conceptual
  dimensions; and
- target onsets, validated button responses, physical-sequence position,
  sequence boundaries, and slow drift.

Preceding and following images are represented by their own feature rows at
their own event times in the joint FIR design; they are not duplicated as
extra current-image columns.

Image-derived VLM features and embeddings of the true concept name are
different predictor families and remain separately identifiable. Encoder
weights, raw outputs, and the admissible feature-family menu are fixed without
EP18 EEG. Declared downstream reductions and model choices may use only
training-block development EEG. A mandatory family cannot be removed because
its removal increases the concept effect.

At least one independently trained stronger visual panel must be declared in
advance. Each stronger baseline \(M_0^{(s)}\) either adds features to \(M_0\),
or makes a predeclared substitution shown by EEG-independent criteria to be no
weaker than the visual features it replaces. All other baseline families stay
unchanged.

### True and alternative membership models

\(M_{\mathrm{true}}\) adds a regularized true-concept term to \(M_0\).
\(M_{\mathrm{pseudo},k}\) places the same term in the same model slot but uses
matched alternative partition \(k\). The group term must use a bounded-rank or
otherwise capacity-controlled representation; an unconstrained
`1,854 × channels × lags` lookup table is not allowed.

Concept-name and human concept features in \(M_0\) already occupy part of the
true one-hot concept space. For each true or alternative membership matrix,
estimate its projection on the overlapping \(M_0\) columns using training rows
only, then apply those training-estimated coefficients to the held-out rows of
that same matrix. Use the same algorithm and hyperparameters for every
partition. An exactly equivalent joint constraint is also allowed. This
prevents a gain caused only by duplicating predictors under regularization.

All models use identical folds, samples, transformations, temporal support,
nuisance terms, covariance, tuning budget, and fitting opportunities. Each
candidate is fitted directly to EEG with all \(M_0\) covariates included.
Fitting \(M_0\) once and searching exported EEG residuals is not equivalent
and is not allowed.

## Matched alternative partitions

Random labels test capacity but do not distinguish concepts from unmodeled
visual clusters. The primary comparator bank therefore contains alternatives
built without EEG.

Every alternative has 1,854 groups of 12 images. Each group contains one image
from every main block and 12 different true concepts. It uses the same group
term, temporal basis, regularization, and tuning budget as
\(M_{\mathrm{true}}\). Its post-constraint effective rank and leverage must
meet fixed matching tolerances, which are reported for every accepted
partition.

The generator matches true groups as closely as possible on:

- dispersion in low-level, vision-model, and image-derived VLM or caption
  spaces;
- physical-sequence and within-sequence position;
- preceding and following image features; and
- target and validated-response proximity.

Concept-name and concept-level human features cannot be matched: their
within-group dispersion is zero for a true concept but not for a group of 12
different concepts. Those continuous features remain in \(M_0\), and the
remaining semantic mismatch must be reported.

Image order differs across participants, so a fixed generator may create a
different literal partition for each participant. Index \(k\) denotes the same
predeclared generator seed, objective, tolerances, and acceptance rule across
participants. The primary bank and its size are fixed before audit EEG is
opened, and achieved matching quality is reported for every partition.

Random partitions remain separate capacity diagnostics. The matched bank is a
finite set of competing models, not an exchangeable permutation sample, and
must not be reported as a permutation-test null.

## Primary quantities and decision rule

Let \(D_{pf}\) be participant \(p\)'s held-out continuous data in outer fold
\(f\). Let \(L\) be a concept-balanced multichannel prediction loss whitened by
a covariance estimate learned from that fold's training blocks. The scoring
rule must specify how the fixed boundary mask preserves or restores equal
concept weighting.

For each fold, define

\[
d^{\mathrm{true}}_{pf}
= L(M_0,D_{pf})-L(M_{\mathrm{true}},D_{pf}),
\]

\[
c^{(k)}_{pf}
= L(M_{\mathrm{pseudo},k},D_{pf})-L(M_{\mathrm{true}},D_{pf}).
\]

Positive \(d^{\mathrm{true}}_{pf}\) means that true membership improves on
\(M_0\). Positive \(c^{(k)}_{pf}\) means that true membership predicts better
than matched alternative \(k\).

Average the six folds before inference:

\[
d^{\mathrm{true}}_p=\frac{1}{6}\sum_f d^{\mathrm{true}}_{pf},
\qquad
c^{(k)}_p=\frac{1}{6}\sum_f c^{(k)}_{pf}.
\]

Let \(P_B\) be the fixed set of 16 audit participants. The audit estimands are
equal-participant means:

\[
\Delta_C=\frac{1}{|P_B|}\sum_{p\in P_B}d^{\mathrm{true}}_p,
\qquad
C_k=\frac{1}{|P_B|}\sum_{p\in P_B}c^{(k)}_p.
\]

For every stronger baseline \(M_0^{(s)}\), fit the corresponding
\(M_{\mathrm{true}}^{(s)}\) and \(M_{\mathrm{pseudo},k}^{(s)}\), then define
\(\Delta_C^{(s)}\) and \(C_k^{(s)}\) by the same equations above.

A positive primary result requires all of the following:

1. the lower audit confidence bound for \(\Delta_C\) exceeds a practical
   margin \(\delta_C\);
2. simultaneous lower bounds for every \(C_k\) in the primary matched bank
   exceed zero;
3. simultaneous lower bounds for every required \(\Delta_C^{(s)}\) and every
   required \(C_k^{(s)}\) exceed zero;
4. temporal, task, and sequence controls pass their fixed margins; and
5. leave-one-participant, held-out-block-pair, physical-sequence, and broad
   object-category analyses remain within a fixed influence tolerance.

The rule for \(\delta_C\) is set from development reliability and injection
tests and may not depend on the observed true-concept effect. The interval and
multiplicity procedures, control margins, stronger panels, matched bank, broad
object categories, equivalence margin, and influence tolerance are all fixed
before audit.

Failure to establish \(C_k>0\) is not by itself evidence of partition
nonspecificity. That conclusion requires adequate precision and a predeclared
equivalence test showing that a matched alternative is comparable to, or
better than, the true grouping; otherwise the result is underidentified.

## Development and audit boundary

Development may compare bounded choices in:

- causal FIR versus low-rank temporal bases;
- training-only feature reductions;
- ridge, group-regularized, and reduced-rank linear encoding models;
- concept-term rank and penalty;
- training-side covariance shrinkage; and
- downstream choices within the admissible stronger-panel menu. Every panel
  designated as required is carried into audit.

Every candidate evaluates \(M_0\), \(M_{\mathrm{true}}\), and the primary
matched bank on identical folds and samples. Development may strengthen the
baseline. It may not choose a favorable weak baseline, create EEG-informed
partitions, select a different winner for each participant, or inspect audit
EEG to decide what to try next.

One procedure advances to audit. For each audit participant, the evaluator
fits that procedure on ten blocks and scores the two held-out blocks in each
outer fold. Participant-specific coefficients may be refitted; feature
families, temporal bases, ranks, penalty-selection rules, masks, partitions,
controls, margins, and decisions may not change.

This primary audit tests whether the procedure reproduces the cross-exemplar
increment in new participants. It is not zero-shot application of a
development-participant EEG template. A separately fixed shared-template
transport analysis is a stricter secondary endpoint; it can qualify, but
cannot rescue or overturn, the primary result.

Audit-participant reliability and positive controls are computed only by the
evaluator after lock. Failure makes the result underidentified; it never
permits excluding or replacing that participant.

## Qualification before candidate comparison

These are setup and qualification tasks performed within EP18, not
prerequisites for creating the episode. The coding agent may reconstruct
events, build the image-event join, assign participants, generate folds and
manifests, and run outcome-blind checks before comparing biological
candidates.

Before comparing biological candidates:

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
   controls must support a meaningful participant-level test; and
7. the 30/16 participant IDs must be frozen, the development runtime must have
   no access to audit EEG values, and the shared EP17/EP18 exposure ledger must
   be initialized.

Failure of rank, injection recovery, reliability, matching, or precision makes
the question underidentified. It cannot be repaired by treating folds, trials,
or samples as independent participants.

## Checks carried into audit

The fixed finalist includes:

- the primary matched partition bank and same-size random capacity controls;
- every required stronger visual panel;
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

The 200 repeated validation images are used only for reliability, positive
controls, and injection calibration. Time-resolved plots are secondary; their
uncertainty and multiplicity rules must be fixed, and an isolated latency peak
cannot replace the global continuous-prediction result.

## Possible conclusions

| Outcome | Interpretation | Terminal status |
| --- | --- | --- |
| Residual concept component | The full conjunctive decision rule passes | `candidate_ready` |
| Feature-panel sufficiency | The increment is absent after the fixed primary or required stronger panel, with adequate precision | `closed_no_candidate` |
| Partition nonspecificity | With adequate precision, a matched alternative is equivalent within the fixed margin or predicts better | `closed_no_candidate` |
| Sequence or filtering artifact | Negative lags, shifts, task terms, drift, or support checks explain the effect | `closed_no_candidate` |
| New-participant nonreplication | A precise development effect fails in the participant-refit audit | `closed_no_candidate` |
| Precise conditional null | Positive controls pass and the audit interval excludes the smallest useful increment | `closed_no_candidate` |
| Underidentified | Reliability, rank, matching, injection recovery, or participant-level precision is inadequate | `closed_no_candidate` |
| Technical or access failure | Event reconstruction, sample separation, preprocessing, audit isolation, or the fixed procedure is violated | `technical_failure` |

A positive result receives one secondary label: **shared-template transport
supported**, **participant refit required**, or **shared-template
unresolved**. That label does not change the primary terminal status.

## Claim boundary

The strongest permitted conclusion is:

> Within THINGS-EEG1 and the fixed feature panel, true concept membership
> supplied a reproducible cross-exemplar predictive increment beyond matched
> visual and sequence alternatives in participants not used for development.

This does not establish abstract semantics, amodal concepts, lexical coding,
independence from every possible visual feature, causal computation, or a
universal object ontology.

EP17 and EP18 overlap in the THINGS stimulus ecosystem. EP18 may provide new
EEG evidence, but it is not an independent stimulus-family confirmation of
EP17. The two episodes therefore share a stimulus-exposure ledger recording
exact image and concept overlap, feature or checkpoint reuse, and first-access
history.
