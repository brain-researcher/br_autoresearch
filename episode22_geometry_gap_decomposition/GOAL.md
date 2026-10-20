# Spectrum, eigenvectors or stimulus selection? Decomposing the VLM–DINOv2 brain-geometry gap

## Status and decisions

**Current status, reconciled October 15, 2026: `confirmation_complete`.**
D0–D3 and the single OOD evaluation are complete. The
[write-once lock](outputs/ood_confirmation_lock.json) preceded the
[recorded October 13 opening approval](outputs/experiment_log.md#2026-10-13-0032-utc--ood_confirmation--scientist-approved-the-single-ood-opening)
and [completed run](outputs/experiment_log.md#2026-10-13-0129-utc--ood_confirmation--single-opening-complete-terminal-class-confirmation_complete).
This status means the evaluation finished, not that every prediction passed:
the [result summary](outputs/summary.md) retains the failed original C2,
the adopted revised C2a, descriptive C2b, indeterminate C1 and failed PEcore
secondary prediction. The same five participants contributed development and
OOD evidence; this is not independent participant replication. The OOD pool
has been consumed, and this reconciliation authorizes no new opening.

### Initial design decisions and their subsequent adoption

EP22 is a standalone diagnostic follow-up to the closed
[EP04](../episode04_shared_scene_geometry/GOAL.md). The design was drafted on
2026-10-12, with the following initial scientist decisions:

1. Run this diagnostic before any theory-level episode, such as one on
   co-occurrence operators or human-curated training data.
2. Reassign the 371 shared LAION-fMRI OOD images from EP04 to EP22 as its
   one-shot confirmation set. EP04 had reserved them for auditing a locked VLM
   candidate. That audit never ran, because EP04 closed as
   `closed_no_candidate` on 2026-10-06 with the OOD outcomes unopened.
3. Later the same day, the scientist adopted the default procedural OOD
   separation and authorized D0.

At that initial stage, the remaining design was proposed. Subsequent recorded
decisions adopted D1–D3, C1–C3 and the pre-lock C2 revision; the completed
exposure review, corrected embeddings and write-once lock supported the single
approved OOD opening. The scientific specifications below preserve that design
and its history; they are not an outstanding request to rerun it.

EP22 does not adopt the adaptive-search protocol.

## Scientific question

EP04 compared released image embeddings with fMRI response geometry in the
LAION ventral, lateral and dorsal high-level sectors of five LAION-fMRI
participants. Two readouts disagreed:

- **Native RSA:** DINOv2 and PEcore aligned better than CLIP and SigLIP2.
- **Identical-fold linear encoding:** all four models were essentially tied.

EP22 asks which of three accounts explains each VLM's native-RSA gap:

- **Spectrum (S).** The models share brain-relevant principal axes but weight
  them differently.
- **Eigenvectors (V).** The principal axes themselves differ.
- **Selection (C).** The LAION images were selected to maximize effective
  dimensionality in CLIP space. On exactly these images, that flattens CLIP's
  own spectrum, and those of CLIP-like models.

It also asks whether the explanation found on development images holds once,
on sealed OOD images whose documented construction does not involve CLIP-space
sampling.

## Why the answer matters

Work on representational convergence increasingly separates relational
structure from metric geometry. After calibration, models agree much more on
which samples are neighbours than on how far apart they are
([Gröger et al. 2026](https://arxiv.org/abs/2602.14486);
[You et al. 2026](https://arxiv.org/abs/2609.27252)).

A spectral account makes this concrete. Suppose a representation is roughly
its leading eigenvectors times a spectral weighting specific to the learner.
Then readouts that refit feature weights, such as encoding, largely ignore the
weighting, while fixed-geometry RSA depends on it. This is the
representational-model view of encoding versus RSA
([Diedrichsen & Kriegeskorte 2017](https://doi.org/10.1371/journal.pcbi.1005508)).
EP04's pattern, equal encoding with unequal RSA, is the brain-side signature
of such a split. A planned theory episode would build on that split, so it
must first be shown to belong to the representations (S or V) rather than to
how the stimuli were sampled (C).

The selection account also matters for method. The provider's
[stimulus-selection documentation](https://laion-fmri.hebartlab.com/stimulus_selection.html)
says that a large LAION-natural sample "was embedded with CLIP, reduced with
PCA, and clustered", and that candidates "were then chosen to maximize the
effective dimensionality of the selected set". Suppose selecting in a model's
own embedding space lowers that model's fixed-geometry RSA. Then RSA-based
model comparisons are biased against the selecting model family, both on
LAION-fMRI and on any dataset sampled the same way. Encoding comparisons would
be affected much less.

## Motivating EP04 observations (exposed development evidence)

All values are participant-balanced EP04 development results, taken from the
[saved decision](../episode04_shared_scene_geometry/outputs/runs/panel_decision/registered_panel_decision.json)
and the
[native sensitivities](../episode04_shared_scene_geometry/outputs/runs/native_sensitivities/20261005-all01/native_sensitivities.json).

| | DINOv2 | PEcore | CLIP | SigLIP2 |
| --- | ---: | ---: | ---: | ---: |
| Native RSA, Fisher z | 0.129 | 0.130 | 0.094 | 0.090 |
| Identical-fold encoding R² | 0.176 | 0.173 | 0.170 | 0.177 |

On 2026-10-12 the `equal_source_strata` rows were re-summarized by source.
Each entry below is the mean Fisher z over 75 cells (participant × fold ×
sector). The median number of evaluation images per fold was 849 for LAION,
29 for THINGS and 64 for THINGSplus.

| Within-source native RSA | LAION | THINGS | THINGSplus |
| --- | ---: | ---: | ---: |
| DINOv2 | 0.142 | 0.105 | 0.119 |
| PEcore | 0.126 | 0.118 | 0.132 |
| CLIP | 0.088 | 0.114 | 0.114 |
| SigLIP2 | 0.093 | 0.070 | 0.066 |

What the per-participant gaps to DINOv2 show:

- **CLIP and PEcore.** On LAION images their gaps are negative in all five
  participants (means −0.054 and −0.016). On THINGS and THINGSplus the gaps are
  small and change sign (CLIP +0.010 and −0.005; PEcore +0.013 and +0.013).
  This makes them candidates for account C.
- **SigLIP2.** Its gap is negative in all five participants on both LAION
  (−0.049) and THINGSplus (−0.053). This makes it a candidate for S or V.

These summaries are post hoc and come from evidence that EP04 had already
exposed. They motivate the design but cannot confirm it. The THINGS cells are
small, and the strata also differ in content.

## Competing accounts and what would distinguish them

| Account | Outcome-blind signature | Development signature | OOD signature |
| --- | --- | --- | --- |
| S, spectrum | The models' spectra differ in shape. | Imposing DINOv2's eigenvalues on a VLM's eigenvectors closes most of the gap. A one-parameter spectral reshaping, fitted on training images, also closes it on held-out images. | The same closure appears on sealed OOD images, and the development-fitted reshaping transfers. |
| V, eigenvectors | Cross-model subspace overlap is low. | Swapping eigenvectors closes the gap; swapping eigenvalues does not. Only readouts that can rotate close it. | The same pattern holds on OOD. |
| C, selection | CLIP's ED relative to DINOv2 is elevated on LAION images, but not on THINGS or OOD images. | Maximizing ED by subsampling in a model's own space lowers that model's RSA more than other models'. | No CLIP deficit on photographic OOD images, which were not selected in CLIP space. |

The accounts can co-occur, and C is a sampling-induced case of S.

## Development analyses

Development uses the 23,560 subject-unique images (4,712 per participant).
EP04 already exposed them, so all results here are exploratory development
evidence. Each family below covers:

- all four models;
- all five participants;
- the three primary sectors, with EVC descriptive;
- EP04's frozen image-disjoint folds.

Every family is reported in full.

**Common geometry.** For model m on an image set:

1. L2-normalize each released embedding row, as EP04 did.
2. Centre the columns over the set.
3. Take the SVD `X_m = U_m Σ_m V_mᵀ`.

The native model RDM is the Euclidean distance between rows of `U_m Σ_m`. It
ranks pairs exactly as EP04's cosine RDM did. Neural RDMs, Spearman RSA,
Fisher z and the aggregation follow EP04: folds, then sectors, then
participants, each weighted equally.

- **D0. Outcome-blind spectra (no neural data).** Per participant and source
  stratum, on random subsets of matched size, compute:
  - participation-ratio effective dimensionality (ED);
  - spectral decay;
  - cross-model subspace overlap, as linear CKA and as principal angles between
    the leading image-space eigenvectors.

  The same measures are computed on the corrected OOD embeddings before the
  lock, since they involve no responses. C predicts that CLIP's ED relative to
  DINOv2's is higher on LAION images than on THINGS, THINGSplus or OOD images.

- **D1. Eigenvalue × eigenvector factorial (no neural fitting).** Take each VLM
  m and DINOv2 r on the same evaluation images. Compute RSA for the four
  geometries `U_a Σ_b` with a, b ∈ {m, r}, where `Σ_b` is imposed in rank
  order. The native gap `G = RSA(U_m Σ_m) − RSA(U_r Σ_r)` splits exactly into
  two parts:
  - `E_val = ½[(RSA(U_m Σ_m) − RSA(U_m Σ_r)) + (RSA(U_r Σ_m) − RSA(U_r Σ_r))]`
  - `E_vec = G − E_val`

  S predicts that `E_val` dominates; V predicts that `E_vec` dominates.

- **D2. Flexibility ladder at matched flexibility.** Every rung is applied
  identically to the VLM and to DINOv2. Fitting uses training images only, and
  RSA is scored on held-out images.

  | Rung | Geometry |
  | --- | --- |
  | R0 | native |
  | R1 | one spectral exponent per model and fold, `U Σ^α` with α ∈ [0, 4], in the leading-256 training-PCA basis |
  | R2 | non-negative per-axis weights on the leading 64 axes, in the style of feature-reweighted RSA ([Kaniuth & Hebart 2022](https://doi.org/10.1016/j.neuroimage.2022.119294)) |
  | R3 | the geometry of responses predicted by ridge encoding |

  Closure at rung R is `1 − |gap_R| / |gap_R0|`. How to read it:
  - Closure by R1 or R2 means the difference is spectral.
  - Closure only at R3 means the difference lies along axes other than the
    model's own principal axes.
  - No closure even at R3 would mean the difference is informational, which
    contradicts EP04's encoding parity.

- **D3. Selection intervention.** Within each evaluation fold's LAION images:
  1. For each selector model s, greedily choose half the images to maximize ED
     in s's 64-dimensional PCA space. This mirrors the provider's PCA-plus-ED
     step.
  2. Draw at least 20 random half-subsets with fixed seeds as baselines.
  3. For every evaluated model e, let `Δ_e(s)` be its native RSA on s's subset
     minus its mean over the random subsets.

  The own-selection effect is `O(s) = Δ_s(s) − mean_{e≠s} Δ_e(s)`, and the
  selection mechanism predicts `O(s) < 0`. The most informative selector is
  DINOv2, which the provider did not use. This tests the direction of the
  mechanism. It does not test magnitude, because the provider selected from a
  pool of 120M images.

Stimulus uncertainty comes from whole-image bootstrap within each participant.
All five participant values are always reported, and no population inference
is made.

## Sealed OOD confirmation (one opening)

All five participants saw all 371 OOD images, four times each. The images fall
into nine provider types:

- **Confirmatory categories (adopted 2026-10-12):** the photographic types
  with at least 20 images, weighted equally:
  - `cropped` (60)
  - `relations` (72)
  - `selfmade` (30)
  - `unusual` (64)
- **Descriptive only:**
  - `illusion-natural` (8 images, too few for a stable within-category RSA)
  - the synthetic types `gabor`, `gaudy`, `illusion-classic` and `shape`

Provider documentation describes no CLIP-space sampling for any OOD type. If
such sampling is found before the lock, that category leaves the confirmatory
set, and the change is recorded.

The primary OOD estimand follows the estimator EP04 had planned:

1. Compute Spearman RSA on within-category edges only.
2. Average Fisher z equally over categories and sectors, giving one effect per
   participant.
3. Average equally over the five participants.

All-pair RDMs are secondary.

D0–D3 changed the expected outcome. The native-RSA gaps turned out spectral,
and DINOv2's spectrum collapses on OOD images, so a spectral account predicts
a CLIP deficit on OOD images. A selection-only account predicts none. The
scientist adopted the following predictions on 2026-10-12. They are frozen at
the lock:

- **C1, a discriminating test.** This is the native CLIP − DINOv2 gap on the
  confirmatory categories:
  - a gap ≤ −0.02 Fisher z supports the spectral account;
  - a gap > −0.02 supports the selection-only account.

  Each classification needs the participant-balanced mean and at least four of
  five participants to agree; otherwise C1 is recorded as indeterminate.
  Leave-one-category-out means are reported. PEcore receives the same
  classification as a secondary result.
- **C2, spectrum.** This was rewritten post hoc on 2026-10-12, after the
  pre-lock dev-test and before any OOD access. On development pseudo
  categories, the within-category eigenvalue swap closed only 22–30% of the
  gap. D1's dominance of the eigenvalue component had been measured on whole
  folds, not within categories.
  - **C2a, within-category sign.** For CLIP and SigLIP2, the within-category
    E_val is below 0 for the mean and in at least four of five participants.
  - **C2b, all-pairs scale.** This mirrors D1. Take all pairs among each
    participant's eligible confirmatory images. For CLIP and SigLIP2, imposing
    DINOv2's eigenvalues on the VLM's eigenvectors closes at least half of the
    native all-pairs gap, for the mean and in at least four of five
    participants. It can be evaluated only if the mean native all-pairs gap
    is ≤ −0.02. C2b is confirmatory only if the rerun dev-test met this rule
    for both models; otherwise it is reported descriptively, and the lock
    records which case holds.
  - **Secondary, for PEcore:** the within-category E_val < 0 and E_vec > 0,
    for the mean and in at least four of five participants.
- **C3, transfer (secondary).** The R1 exponent fitted on development data,
  applied without refitting, raises each VLM's OOD within-category RSA above
  its native value.
- **Informativeness gate.** DINOv2's participant-balanced within-category RSA
  on the confirmatory categories must exceed the 95th percentile of 10,000
  within-category whole-image permutations. The same permutations are used
  across participants and sectors. If the gate fails, the opening is recorded
  as uninformative, and no prediction is scored as passing or failing.

After the opening, no prediction, category, margin, model or aggregation may
change.

## Decision this episode informs

- **S or C explains the gaps, and the OOD predictions hold.** The brain–model
  differences in EP04 are then metric or sampling effects on shared structure.
  - The next theory episode can target what sets each learner's spectral
    weighting, using structure-level and selection-aware comparisons.
  - LAION-fMRI RSA model rankings should carry a selection caveat.
- **V dominates.** The VLMs and DINOv2 differ in brain-relevant axes. The
  theory episode must then explain which training statistics or representation
  stages produce which axes, instead of assuming shared structure.
- **Confirmation fails or is uninformative.** The development explanation
  remains exploratory, and EP04's model differences should not be built on.

## Nearest prior work and contribution

- **Reweighting.** Fitting feature weights removes most native-RSA differences
  among DNNs as models of human IT
  ([Storrs et al. 2021](https://direct.mit.edu/jocn/article-abstract/33/10/2044/103001/Diverse-Deep-Neural-Networks-All-Predict-Human)).
  Feature reweighting also changes which model wins
  ([Kaniuth & Hebart 2022](https://doi.org/10.1016/j.neuroimage.2022.119294)).
  EP22 asks how much of that flexibility is actually needed: a parameter-free
  eigenvalue swap, one exponent, per-axis weights, or a full linear readout.
- **Spectral theory.** A spectral theory already splits regression-based
  neural prediction into eigenspectrum and eigenvector alignment
  ([Canatar et al. 2023](https://arxiv.org/abs/2309.12821)). D1 is the
  analogous factorial for fixed-geometry RSA.
- **Stimulus-set dependence.**
  - Model similarity depends on the stimulus set
    ([Ciernik et al. 2025](https://arxiv.org/abs/2411.05561)).
  - Controversial stimuli separate models, but they can favour the model
    trained to invert the generator that produced them
    ([Golan et al. 2022](https://arxiv.org/abs/2211.15053)).
  - Higher latent ED predicts better encoding
    ([Elmoznino & Bonner 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011792)).

  A brief search on 2026-10-12 found no direct test of whether ED-maximizing
  selection in a model's own space lowers that model's RSA; D3 and C1 test
  this. A fuller search should come before any novelty claim.

The contribution is a diagnosis of EP04's result and a test for selection bias
in datasets sampled in an embedding space. It is not a new model ranking.

## Claim boundary

- **Participants.** These are EP04's five participants, and the development
  images are the ones EP04 already exposed. The OOD set is a stimulus shift
  within the same participants, not a participant or population replication.
- **Representation stage.** The released embeddings fix each model's stage.
  DINOv2 is a mean of layer-23 patch tokens; the VLMs are final projected
  embeddings. So "spectrum" covers the training objective, the projection head
  and the stage together. EP22 makes no causal claim about language
  supervision or training objectives.
- **Selection.** D3 emulates selection in a model's own space and tests its
  direction. The provider's CLIP variant, cluster count and pool are not fully
  documented, so the provider's effect size cannot be reproduced.
- **Decomposition.** The spectral decomposition is algebraically exact for
  these RDMs. It attributes RSA differences between models; it does not
  identify mechanisms inside the networks.
- **Sealing.** The OOD seal is procedural, not enforced by the operating
  system; see [DATASETS.md](DATASETS.md#sealed-confirmation-requirements).

## Conceptual figure

This is still pending. The repository's image-generation route was not
available in the 2026-10-12 design session, so no schematic was generated. The
intended panels are:

- **A.** EP04's dissociation: equal encoding, unequal native RSA.
- **B.** The eigenvector × eigenvalue factorial.
- **C.** Selection in one model's space flattens that model's spectrum and
  lowers its RSA.
- **D.** The sequence from development, to a write-once lock, to a single
  sealed OOD opening on the photographic categories.
