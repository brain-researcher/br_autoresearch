# EP23: A conditional spectral-operator account of representational convergence

## Status

status: adopted_initial_analyses_complete_A4_inactive (reconciled 2026-10-15)

The adopted initial analyses A0–A3 and P1–P5 theory checks are complete; see the
[released results](outputs/README.md) and [execution record](outputs/experiment_log.md).
All analyses of exposed LAION development data remain exploratory, and A3 is
synthetic. A4 independent confirmation remains inactive. This status correction
does not authorize additional execution or change the scientific rules below.

## Scientific question

Under what conditions do learners that experience similar co-occurrence
operators share coarse representational subspaces while differing in
representational metric? Do those conditions leave testable signatures in
models and brains?

The primary empirical test (A0) asks a direct question. Do the leading
eigenspaces of an externally measured concept co-occurrence operator predict
the leading subspaces of vision models and of individual brains? And does that
prediction survive controls for taxonomy, frequency and visual similarity?

The restricted account under test:

- a stable coarse operator subspace survives learner-specific perturbations;
- objectives and measurement change how that subspace is weighted;
- fine structure can differ when learner-specific perturbations dominate
  weakly separated modes.

Whether natural experience, real learning objectives and brain measurement
satisfy these conditions is the empirical question. The scientist's broader
interest is how the brain learns its world model. This episode can constrain
that question but cannot answer it.

## Why it matters

Representation similarity across models and brains is well documented. An
explanation has to connect three things: the statistics a learner
experiences, its objective, and how its representation is measured.
Similarity between fitted representations does not establish that
connection.

EP22's sealed OOD confirmation (the same five participants, procedurally
sealed) found:

- **A VLM spectral penalty within categories.** E_val was −0.020 for CLIP and
  −0.023 for SigLIP2, negative in every participant.
- **Transferable steepening.** Development-fitted spectral powers
  (α ≈ 1.6–2.3) raised OOD VLM RSA by +0.017 (PEcore) to +0.035 (SigLIP2).
  DINOv2 also improved (+0.045), which was not a locked prediction.

These findings motivate separating shared subspaces from their spectral
weighting. They do not identify an experienced operator, or the source of the
brain's preference for steep spectra.

EP22 also constrains the account:

- Reweighting each model's own axes did not close the development gap, but a
  full linear readout did. An explanation may therefore need changes of axes,
  not only weights.
- C1 was indeterminate, and PEcore's within-category eigenvector advantage did
  not replicate.
- The observation that eigenvector agreement depends on scale was formed after
  the outcomes were seen. It is a hypothesis here, not evidence. OOD E_vec
  moves from within-category to all-pairs values as follows:

  | Model | Within category | All pairs |
  | --- | ---: | ---: |
  | CLIP | −0.003 | 0.087 |
  | SigLIP2 | −0.029 | 0.041 |
  | PEcore | −0.010 | 0.068 |

## Theory component

### Definitions

- Use a finite common event set and an explicitly fixed sampling measure.
- Start with symmetric stochastic operators T_S, and remove the stationary
  mode explicitly. A reversible extension needs the correct
  stationary-measure similarity transform. Directed, nonnormal, partially
  observed and policy-dependent cases need separate arguments.
- Write T_S = U_S Λ_S U_Sᵀ and Φ_S = U_S g_S(Λ_S) R_S, where R_S is an
  orthogonal change of feature coordinates and g_S weights the retained modes.
  This form is a hypothesis unless it is derived for a stated objective.
- Empirically, write the centered features as X = U D Vᵀ. EP22's power
  reweighting is Φ_α = U_k D_k^α in the leading-k training-PCA basis:
  - α = 1 is the centered, rank-k truncation of the native feature geometry;
  - Gram eigenvalues are d^{2α};
  - α = 2 squares the feature Gram matrix.

  A brain preference for α ≈ 2 is numerically what a geometry tracking K²
  rather than K would show, e.g. a two-step or shared-context kernel. P1 and
  P5 explain why α alone does not identify that reading.
- Shared subspaces, shared neighbourhood identities and shared distances are
  distinct claims. Nonuniform weighting can change neighbourhoods as well as
  distances.

### Propositions to prove

**P1 — Operator relations share an eigenbasis under stated conditions.**

- For symmetric T, the operators T, TTᵀ = T² and the successor operator
  (I − γT)⁻¹ = Σ_{k≥0} γᵏTᵏ (0 ≤ γ < 1) share an eigenbasis. State any
  stationary-mode removal, diagonal removal or normalization separately.
- Shared eigenvectors do not imply the same leading modes, because squaring
  can promote negative modes. A common leading subspace needs suitable
  spectral support and order-preserving weights.
- A nonsymmetric transition operator instead gives a shared-context kernel
  TTᵀ, generally with different eigenvectors. Visual similarity is a separate
  candidate operator.
- Observation maps matter. If X = h(Z), the observed operator is the
  compression J*TJ, where J is the pullback f ↦ f∘h.
  - A bijective h gives a unitarily equivalent operator.
  - A non-bijective h can lose spectral structure. PRH's bijectivity
    assumption sits exactly here.

**P2 — An exact learning bridge for a specified surrogate objective.**

- For a symmetric kernel K = f(T), characterize the rank-r minimizers of
  ‖K − ΦΦᵀ‖_F² from the leading nonnegative eigenvalues.
- The optimum is ΦΦᵀ = Σ_{i≤r} max(λᵢ, 0) uᵢuᵢᵀ, so an indefinite K keeps
  its largest positive eigenvalues (Eckart–Young–Mirsky, plus the PSD
  constraint).
- Claim uniqueness only for the retained subspace, and only when the
  boundary eigengap is positive. Prove the optimization result without
  assuming the desired form.
- The result covers only this unweighted Frobenius objective. It does not
  solve SGNS's weighted logistic loss, unit-diagonal or nonnegativity
  constraints, or general InfoNCE.
- This is a controlled example of the objective determining the spectrum
  (here g(λ) = λ^{1/2}). It does not show that CLIP, PEcore, SigLIP2, DINOv2
  or brains optimize this loss. Each published factorization result needs its
  own bridge to T.

**P3 — Coarse stability can coexist with fine variability.**

- Let T_S = T_0 + E_S, where E_S is symmetric with zero row sums,
  ‖E_S‖₂ ≤ ε, and transition probabilities stay nonnegative. The constant
  vector is then an exact common eigenvector. Work in its orthogonal
  complement, where the coarse subspace is the leading q-dimensional
  eigenspace, with reference gap δ = λ_q − λ_{q+1} > 0 in algebraic order.
- Bounds to use (verified 2026-10-13):
  - Yu–Wang–Samworth (2015, Theorem 2) needs only δ > 0:
    ‖sin Θ‖_F ≤ 2 min(√q ε, ‖E_S‖_F)/δ.
  - Weyl's inequality keeps a separated leading cluster when ε < δ/2.
    Davis–Kahan with the mixed gap then gives ‖P̂_S − P_0‖₂ ≤ ε/(δ − ε).
  - The draft condition ε < δ/4 is valid but conservative. It gives
    ‖P̂_S − P_0‖₂ ≤ 4ε/(3δ) < 1/3.
  - Two learners each within ε of T_0 satisfy ‖P̂_1 − P̂_2‖₂ ≤ 2ε/(δ − ε).
  - Errors and gaps must refer to the same operator. A claim about
    K = h(T) needs h's spectral separation, since weights can reorder or
    collapse modes.
- Construct a stochastic block example whose perturbations act only within
  fine contrast spaces. Show that an independent, rotationally invariant
  fine-perturbation model gives chance-level expected fine-projector overlap
  while the coarse subspace is preserved.
- That distributional assumption must be stated, because small fine
  eigengaps alone do not imply disagreement.
- Compare eigenspaces across degenerate blocks, never arbitrary eigenvectors
  within them. Human taxonomic labels are proposed analysis scales, not
  established eigenmodes of experience.
- Subspace stability does not imply neighbourhood stability. A sufficient
  condition is a distance margin. If ‖K̂ − K‖₂ ≤ η, squared distances move by
  at most 2η, so a gap above 4η between an anchor's k-th and (k+1)-th
  neighbour preserves its kNN set.

**P4 — Readout invariance has a limited domain.**

- For full-column-rank training features and a fixed invertible feature
  transform, unregularized least-squares predictions are invariant once the
  coefficients are transformed accordingly, consistently for training and
  evaluation stimuli. A correspondingly transformed regularizer preserves
  this.
- Isotropic ridge, rank truncation, zero weights and finite-data estimation
  do not generally preserve it. Fixed-geometry RSA is sensitive to spectral
  weighting.
- "Encoding is insensitive to the spectrum" is therefore not an
  unconditional prediction.
- An unrestricted readout cannot identify latent geometry. For every
  invertible R, ZB = (ZR)(R⁻¹B), yet the latent Gram becomes ZRRᵀZᵀ. A full
  readout that matches brain geometry (EP22 R3) shows predictive
  sufficiency, not the native geometry.

**P5 — Pooling produces spectral weighting only under a commutation
condition.**

- Let the noise-free neural responses be N = U A Vᵀ and the measured
  responses B = N M. Then B Bᵀ = U A (Vᵀ M Mᵀ V) A Uᵀ.
- Weighting without mode mixing holds if the middle matrix is diagonal in
  the neural mode basis. Degeneracies need separate treatment.
- Under that condition, measured amplitudes identify only products of
  neural and pooling weights (log μᵢ = 2 log aᵢ + log dᵢ). Without
  independent constraints the two factors are not identified.
- Spatial low-pass filtering alone establishes neither the diagonal
  condition nor α = 2. Construct an analytic counterexample with mode mixing
  despite smoothing.
- Model noise and response estimation separately from this signal-only
  statement. With independent zero-mean repeat noise, the cross-repeat Gram
  E[B⁽¹⁾B⁽²⁾ᵀ] is unbiased for the signal Gram, whereas a same-repeat Gram
  adds the noise covariance. The cross-repeat estimate need not be PSD in a
  finite sample.
- Precedents: measurement models already show that voxel averaging and
  sampling change apparent geometry (Kriegeskorte & Diedrichsen 2016), and
  that finite neuron sampling delocalizes eigenvectors (Kang, Canatar & Chung
  2025). P5 is a conditional restatement for pooling, not a new result.

### What remains conjectural

- Natural experience gives different learners sufficiently similar
  operators.
- Their actual objectives preserve comparable leading modes.
- Fine perturbations differ enough to create the proposed scale contrast.
- Brain pooling approximately preserves those modes, and spatial frequency
  tracks their order.
- Entrywise PMI and log density-ratio transformations need not preserve an
  operator eigenbasis. Published PMI or occupancy results therefore do not
  prove this account automatically.

### Proof rules

- No step may use properties known only for the conjectured result.
- Lemma conditions are proven, or cited together with their conditions.
- Simulations, SMT searches and symbolic checks on special cases can refute
  a claim but cannot prove it.
- BR's theory tools cover finite rational affine identifiability
  (y = Mθ + b), so most of this argument will not be machine-checked. P5's
  factor nonidentifiability, in log coordinates, is the one statement inside
  that scope.

## Empirical component

The first stage is the conditional theory plus bounded exploratory tests on
existing resources. The operator in A0 is measured from human annotations of
photographs. It is a proxy for the statistics a learner might experience, not a
record of any learner's experience. This stage tests observable restrictions of
the account, not its causal learning mechanism. Data roles and counts are in
[DATASETS.md](DATASETS.md).

**A0 — Does a measured co-occurrence operator predict model and brain
subspaces?** Primary; exploratory on LAION development data. Feasibility was
checked on 2026-10-13 without model or neural data (see the
[log](outputs/experiment_log.md)).

- **Operator (primary): Visual Genome v1.4.**
  - Per-image presence of THINGS concepts, mapped through a WordNet 3.0
    offset crosswalk (exact synset names are a lower bound).
  - The shared-context PPMI kernel uses all VG synsets as contexts, excludes
    each concept's own label, and is double-centered.
  - Pooled S is 615 dev concepts with ≥10 images; S_p is 166–181 per
    participant.
  - The direct normalized operator D^-1/2 A D^-1/2 is a robustness variant.
    It is frequency-dominated (Spearman of |u₂| with log degree = 0.69).
- **Secondary operators:**
  - COCO captions, with a cleaned name matcher; this is a description
    (language) channel.
  - LVIS v1, as robustness only, with verified-pair denominators.
- **Targets:**
  - for each participant and primary sector, concept-level brain patterns
    from that participant's images, with a cross-repeat Gram;
  - for each model, concept-level embeddings.

  Both are double-centered on S_p (participants) or S (model–model).
- **Statistic:** principal-projector overlap ‖U_opᵀU‖²_F/q at q = 8 (4 and
  16 as sensitivity), plus a band profile over modes 1–8, 9–16 and 17–32.
- **Controls:**
  1. a concept-permutation null;
  2. taxonomy-residual alignment, with category27 projected out of both sides
     and a within-category permutation null;
  3. a frequency-preserving null, with permutations within log-frequency bins;
  4. alignment beyond an independently computed low-level visual-similarity
     operator from the stimulus images. Its source is pending a scientist
     decision.
- **Units:** participants for brains. Model and model–model results are
  descriptive.
- **Interpretation:** alignment beyond taxonomy, frequency and visual
  similarity supports a shared-statistics account at coarse scales. It does
  not establish identical exposure or a common learning mechanism.
  Cross-source agreement is high (0.55–0.79 against a null of about 0.03),
  but COCO, LVIS and part of VG share images, so it shows robustness to the
  annotation channel, not independent replication.

**A1 — Is weighting along native axes sufficient?** Exploratory; LAION
development data.

- **Setup:** use EP04's frozen folds and EP22's conventions. Fit PCA within
  training folds at rank 64 (rank 32 as sensitivity).
- **Metrics compared:** rank-matched native, spectral power (EP22's
  0–4 α grid), nonnegative diagonal, and full PSD. All live in the same
  retained span.
- **Fitting:** diagonal and full metrics fit the same trace-normalized
  training brain second-moment target, under the same loss. Shrinkage toward
  native is chosen on inner training folds.
- **Contrasts:** full-minus-diagonal held-out RSA is primary;
  power-minus-native is secondary. Both are reported per participant.
- **Interpretation:** a cross-axis gain shows predictive value, not its
  biological cause. Absence of a gain does not establish a shared operator.
  A1 refines exposed EP22 D2 evidence and cannot confirm it.
- **Precedents:**
  - reweighting versus remixing (Khaligh-Razavi & Kriegeskorte 2014);
  - cross-validated feature fitting (Storrs et al. 2021);
  - FR-RSA (Kaniuth & Hebart 2022). Its weights are signed; A1's nonnegative
    diagonal is a constrained relative.

  A1 is a matched-span, matched-target follow-up of EP22, not a new
  phenomenon.

**A2 — Is shared structure stronger at coarse scales?** Exploratory; THINGS
category27 labels on the development set's THINGS and THINGSplus images.

- **Views:** coarse is category centroids; fine is images after
  within-category centering.
- **Structure measure:** principal-projector overlap in stimulus space, at
  r = min(8, ⌊(C − 1)/2⌋), the same r at both scales. Overlap is calibrated
  against 200 correspondence permutations: category identities for coarse,
  images within categories for fine.
- **A2a, model–model (noise-free, descriptive):**
  - images pooled across participants: 19 categories with ≥20 images, r = 8;
  - tests directly whether fine structure differs between models;
  - model pairs are not biological units.
- **A2b, model–brain (participants as units):**
  - categories with ≥10 images per participant: 7–10 categories, r = 3–4;
  - brain eigenspaces come from cross-repeat Gram estimates (repeats split
    in two), which remove the noise bias of same-repeat Grams (P5);
  - coarse and fine agreement are compared relative to split-half brain
    reliability at the same scale and rank;
  - a scale whose brain structure is unreliable is inconclusive for that
    participant and sector, not evidence of learner-specific structure.
- **Caveat:** within a participant, a concept has one or two images, so the
  fine scale is mostly between-concept structure within a category.
- **Precedent:** Khaligh-Razavi & Kriegeskorte (2014) contrasted categorical
  divisions with within-category structure for models and IT. A2 adds three
  things:
  - matched-rank projector subspaces;
  - the noise-free model–model fine test;
  - reliability-relative comparison.

**A3 — What can measurement explain?** Analytic construction plus synthetic
simulations.

- Hold a synthetic neural spectrum and pooling operator fixed, and vary the
  neural topography.
- Compare mode-aligned and mode-mixing cases with controlled noise. Measure
  mode mixing and the resulting preferred power.
- Use at most 20 fixed seeds, never tuned to reproduce EP22's α.
- EVC-versus-high-level contrasts remain descriptive.

**A4 — Independent structural test.** Inactive; no eligible source.

- **Candidate endpoint:** each participant's reliability-relative
  coarse-minus-fine model–brain agreement, averaged equally over the four
  models. It is tested with an exact one-sided participant sign test, which
  requires at least five independent participants.
- **Prerequisites:** a source chosen before any of its outcomes are seen,
  and a frozen endpoint. EP17 and EP18 hold the existing THINGS reservations
  (see DATASETS.md).
- **Limits:** a cross-modality result would test structural generality, not
  isolate fMRI pooling.

**Deferred:**

- intermediate VLM layers and child-headcam models, which need checkpoints
  and GPU extraction;
- training-curation effects, which need matched training;
- egocentric or child-experience operators (Ego4D, EgoObjects, SAYCam,
  BabyView), which need access approvals;
- a full-ADE20K operator, which needs registration and safe parsing of its
  pickle index;
- a text-corpus operator such as Wikipedia, at an estimated 10–100
  CPU-hours;
- an unpaired brain–model alignment test.

## Competing accounts and discriminating predictions

| Account | Distinguishing prediction |
| --- | --- |
| Shared coarse operator | The measured operator's leading subspace predicts model and brain subspaces beyond taxonomy and frequency (A0). Coarse subspaces agree after null calibration, and model–model fine agreement is weak (A2a). Reliability-relative fine model–brain agreement is lower than coarse (A2b). |
| Shared taxonomy only | Operator alignment vanishes once category27 structure is projected out of both sides (A0 control 2). The audit named this the strongest competing explanation. |
| Frequency only | Operator alignment is matched by the frequency-preserving null (A0 control 3). |
| Measurement ceiling only | Model–model fine agreement is high, and reliability-relative coarse and fine model–brain agreement are equal. |
| Neural weighting | A steep preference persists after independently calibrated measurement effects; current fMRI geometry alone cannot show this. |
| Measurement weighting | A specified topography-plus-pooling model predicts changes under controlled pooling. Arbitrary smoothing has no universal direction (A3). |
| Representation-stage mismatch | Changing the extraction stage changes the effect. Needs new embeddings. |
| Stimulus selection | Effects follow the sampling policy and weaken under an independent design. Development resampling assesses only sensitivity. |
| Finite-sample regularization | Gains from flattening track unstable modes and reliability. Training-only fitting and rank controls test this. |
| Shared visual statistics | A visual operator explains structure as well as co-occurrence. Separating them requires measured operators. |

## Decision this episode informs

Does a common coarse-operator account make useful, falsifiable restrictions
beyond "representations can be linearly aligned"?

- If diagonal metrics suffice, or fail, narrow the subaccount explicitly.
- If coarse/fine agreement merely tracks reliability, drop the
  learner-specific fine claim.
- If measurement remains unidentified, specify the missing assay rather than
  assign a cause.
- The results choose which independent test, if any, is worth acquiring.

A0 decides the most direct form of the scientist's co-occurrence intuition, on
exposed data:

- If the operator predicts model and brain subspaces beyond taxonomy,
  frequency and visual similarity, an independent confirmation (A4) becomes
  worth acquiring.
- If alignment vanishes after taxonomy residualization, the account reduces
  to shared categories at the tested scale.
- If models align with the operator but brains do not, either measurement or
  the experience mismatch is implicated. Which one is not identified.

## Nearest prior work and contribution

This section was verified on 2026-10-13 by a web-retrieval audit, with spot
checks of the closest overlaps. Conditions and exact theorem statements are in
[literature_verification.md](outputs/design/literature_verification.md).
Preprints are marked.

### Convergence claims and measures

- **Huh et al., PRH, ICML 2024**
  ([arXiv 2405.07987](https://arxiv.org/abs/2405.07987)). Population-optimal
  contrastive scores recover a co-occurrence PMI ratio, in an idealized finite
  world with bijective observations. PMI need not be PSD, and there is no
  finite-data or optimizer theorem.
- **Gröger, Wen & Brbić, ICML 2026**
  ([arXiv 2602.14486](https://arxiv.org/abs/2602.14486)). After
  aggregation-aware null calibration, capacity-dependent convergence in the
  tested global spectral and distance measures largely disappears.
  Neighbourhood alignment remains.
- **You et al. 2026, preprint**
  ([arXiv 2609.27252](https://arxiv.org/abs/2609.27252)). Relational structure
  converges more than distances, at kNN and spanning-tree scales. Those
  scales are not between- versus within-category.
- **Yamins & Nayebi 2026, preprint, Contravariance**
  ([arXiv 2607.08561](https://arxiv.org/abs/2607.08561)). Conditional
  alignment results for minimal solutions to hard tasks. They do not imply
  that any two high-performing systems align.

### Co-occurrence spectra and hierarchical geometry (closest to P1–P3)

- **Nava & Wyart 2026, preprint**
  ([arXiv 2605.23821](https://arxiv.org/abs/2605.23821)).
  - Under positivity and decay of a tree-distance co-occurrence kernel,
    leading eigenvectors separate broad taxonomic branches first and finer
    ones later.
  - Tested in word2vec and in Gemma 2B unembeddings. Language only.
- **Karkada et al., ICML 2026**
  ([arXiv 2602.15029](https://arxiv.org/abs/2602.15029)). Symmetries in
  language statistics shape representation geometry; eigenvalue separation
  supports robustness.

### Objective-specific spectral results (P2)

- Spectral contrastive loss factorizes a normalized augmentation graph at
  global optima (HaoChen et al., NeurIPS 2021).
- SGNS is a weighted factorization of shifted PMI with separate word and
  context factors (Levy & Goldberg, NeurIPS 2014).
- InfoNCE is spectral clustering, in a regularized sense (Tan et al., ICLR
  2024).
- SSL objectives recover spectral embedding methods (Balestriero & LeCun,
  NeurIPS 2022).
- The contrastive-RL critic is an entrywise log occupancy ratio (Eysenbach
  et al., NeurIPS 2022).

None of these is a universal theorem, and entrywise logarithms do not
preserve eigenbases.

### Identifiability

- Roeder et al. 2021, Zimmermann et al. 2021, Daunhawer et al. 2023 and Yao
  et al. 2024 give conditional identification results.
- Nielsen et al. (NeurIPS 2025) show that vanishing KL divergence need not
  imply representational similarity.

### Readouts, metrics and measurement (P4, P5, A1–A3)

- **Khaligh-Razavi & Kriegeskorte, PLOS Computational Biology 2014.**
  Reweighting plus linear remixing of supervised features explained IT,
  including its categorical divisions. Within-category similarity was
  analysed separately.
- **Storrs et al., Journal of Cognitive Neuroscience 2021.** Cross-validated
  feature and layer fitting substantially reduced architecture differences in
  predicting human IT geometry (62 images).
- **Fitting frameworks:**
  - FR-RSA (Kaniuth & Hebart, NeuroImage 2022);
  - encoding, PCM and RSA share G = FΩFᵀ (Diedrichsen & Kriegeskorte, PLOS
    Computational Biology 2017);
  - ridge predictivity depends on eigenvalues and target alignment (Canatar
    et al., NeurIPS 2023).
- **Measurement effects:**
  - measurement models change apparent geometry (Kriegeskorte & Diedrichsen,
    Philosophical Transactions B 2016);
  - finite neuron sampling delocalizes eigenvectors (Kang, Canatar & Chung
    2025, [arXiv 2502.19648](https://arxiv.org/abs/2502.19648)).
- **Dimensions and objectives:**
  - Elmoznino & Bonner 2024;
  - universal dimensions are shared by networks and human fMRI (Chen &
    Bonner 2024, preprint);
  - the objective drives cross-dataset consistency of representational
    similarity (Ciernik et al., ICML 2025).

### Neural precedents for experienced statistics

- Bonner & Epstein (Nature Communications 2021): object co-occurrence
  predicts anterior PPA, and word2vec predicts LO/pFs.
- Stachenfeld et al. 2017, Schapiro et al. 2013, Garvert et al. 2017, and Li
  & DiCarlo 2008.

Each is a restricted precedent, not a general account.

### Measured co-occurrence, scene statistics and representation (closest to A0)

The full table is in
[operator_novelty_audit.md](outputs/design/operator_novelty_audit.md).

- **Sadeghi, McClelland & Hoffman (Neuropsychologia 2015).**
  - Method: LSA of object co-occurrence in labelled scene photographs.
  - Finding: it recovers taxonomic similarity and cross-category
    associations, closer to feature norms than text LSA.
  - Not tested: model or brain subspaces.
- **Stansbury, Naselaris & Gallant (Neuron 2013).**
  - Method: LDA topics learned from scene-object statistics.
  - Finding: they predict visual cortex responses.
  - Not tested: an encoding model is used, not an operator–brain eigenspace
    comparison.
- **Huth et al. (Neuron 2012).** A shared leading semantic space coexists with
  individual dimensions. Its PCs come from fitted brain models.
- **Magri, Elmoznino & Bonner (Cognition 2023).** Scene context predicts
  object similarity judgments. Its predictor uses CNN features.
- **Caption and dimension models of the brain:**
  - Doerig et al. (Nature Machine Intelligence 2025): LLM caption
    embeddings;
  - Wang et al. (Nature Machine Intelligence 2023): language supervision;
  - Contier, Baker & Hebart (Nature Human Behaviour 2024): behavioural
    dimensions beat categories.
- **Training diet and abstraction levels:**
  - Conwell et al. (Nature Communications 2024): training diet dominates;
  - Muttenthaler et al. (ICLR 2023; Nature 2025): alignment across
    abstraction levels. Base models showed weak global semantic organization
    in human judgments.
  - Mahner et al. (Nature Machine Intelligence 2025).
- **Theory and method analogues:**
  - Saxe, McClelland & Ganguli (PNAS 2019): the SVD of environmental
    statistics sets learned modes and the order of hierarchical
    differentiation in deep linear networks;
  - Johnson, El Hanchi & Maddison (ICLR 2023): contrastive kernels recover
    positive-pair Markov eigenfunctions;
  - Abdelnour et al. (NeuroImage 2018): structural Laplacian eigenvectors
    predict functional eigenvectors with different spectral weights.

**Verdict of the bounded search.** Each element of A0 has partial precedents.
No study combining these was found: an external concept operator's
eigenspaces, matched-rank prediction of both model and individual-brain
stimulus subspaces, and taxonomy, frequency, visual and noise-free
model–model controls.

### Unpaired alignment

- **Rosetta Stone** (Schnaus et al. 2026, preprint,
  [arXiv 2610.09411](https://arxiv.org/abs/2610.09411)). Reports several
  unpaired embedding-alignment successes, including encoded NSD brain–brain
  alignment. Its MEG–text alignment is at chance.
- **Marcos-Manchón et al. (2026, preprint).** No cross-subject pairing is
  used for fitting. Within-subject repeat correspondence is used, however, and
  the headline result selects seeds on shared-image retrieval.

### What remains open, and EP23's contribution

P1–P5 mostly restate known mathematics: functional calculus,
Eckart–Young–Mirsky, Davis–Kahan and Yu–Wang–Samworth, least-squares
invariance, and measurement mixing. Their value here is explicit conditions,
not new theorems.

The bounded search found no study that does all of the following:

- tests whether the language-derived coarse-to-fine operator account
  transfers to vision models and brains;
- separates subspace sharing from spectral metric with participant-level
  evidence;
- uses noise-free model–model comparisons to tell learner-specific fine
  structure from a measurement ceiling.

That combination, and its empirical discrimination, is the proposed
contribution. Priority is not established.

A0's framing: EP23 tests whether spectral modes of independently measured
concept co-occurrence predict the leading stimulus-space subspaces of visual
models and individual brains. It separates subspace agreement from spectral
weighting, and asks whether the prediction survives taxonomy, frequency,
visual-similarity and reliability controls. The near misses to contrast
explicitly are Sadeghi et al. 2015, Huth et al. 2012 and Johnson et al. 2023.

## Claim boundary

- Conditional mathematics does not prove that real learners satisfy its
  assumptions.
- Exploratory agreement with exposed LAION outcomes is not independent
  confirmation.
- An α near two does not identify a neural timescale, successor parameter or
  pooling law.
- Paired-stimulus analyses establish neither unpaired alignment nor recovery
  of unique correspondences.
- Subspace overlap is not neighbourhood identity, and linear predictability
  is not a learning mechanism.
- Participants are the biological units; sectors, images and model families
  are not substitutes.
- Existing data cannot establish how the brain learns its world model.
