# EP18 paper plan: when does a concept survive a change of picture?

## October 6 Pacific: current preparation readiness

The exact stable CPU runtime and all five public model assets are qualified:
allocation **46842156** passed source-integrity checks and loaded all six
selected routes offline. The separate first-64-name pilot then completed,
and the coordinator released all **1,854** unchanged concept-name prompts in
allocation **46843116**. The [label-feature report](runs/feature_preparation/all_name_text_46843116/label_feature_report.json)
records a complete **1,854 by 512 float32** matrix, **169.454814 s** elapsed,
**609,144 KiB** peak RSS, and zero truncations or provider-name fallbacks.
The [execution log](experiment_log.md) preserves the release and attempt
records; these observations supersede the execution snapshots below.

The [sequence-local fitting core](code/continuous_sequence_model.py) now
implements overlapping event expansion, direct joint nuisance/feature ridge
fitting and removal of selected fitted event contributions without refitting.
Its **ten focused algebra checks passed in 0.011 s** in coordinated session
**46845646**, and the changed ASTRA program passed validation. The
[execution record](experiment_log.md#2026-10-06-pacific--baseline_search-sequence-local-fitting-checks-pass)
links the evidence and preserves the earlier prepared/unrun state. This tests
the numerical component; it does not fit EEG or complete preprocessing,
sensor whitening, temporal-recipe selection or the evaluation runner.

The separate [training-transform implementation](code/training_transforms.py)
now encodes the existing registry's unique-image/unique-concept fitting unit,
training mean/population SD, zero-SD handling, full float64 SVD, loading-sign
rule and numerical rank cap. Every required family retains a record, including
effective rank zero; the sequence design accepts its zero-column term when
other predictors exist. No held-out statistics, PCA whitening or extra family
rescaling is introduced. Its fixed sensor basis uses Helmert column j-1 with
positive entries `1/sqrt(j(j+1))` in the first j frozen channels and negative
entry `-j/sqrt(j(j+1))` in channel j+1, for j=1..61. Common-average projection
therefore uses the same 61 coordinates across participants and folds.

The residual covariance interface is explicitly parameterized: training-only
nuisance residuals, sample weights, centering choice, covariance denominator,
shrinkage target and shrinkage strength must all be supplied. It forms the
symmetric inverse root in 61D without eigenvalue clipping and applies that
same metric to responses and predictions without nuisance residualization or
new response centering. These are available numerical operations, not adopted
shrinkage defaults. The [focused transform checks](code/test_training_transforms.py)
remain unrun pending the coordinator's grouped compute opportunity. No image
or EEG access, empirical transform fit or evaluator release occurred.

The completed label extraction supplies one required feature family. Image/caption feature extraction,
enforced pixel-worker delivery, EEG reader separation, matched false-group
construction and the continuous-EEG fitting/evaluation pipeline remain
unfinished. No image pixels, private image routing, events or EEG were read
by the label process. No complete feature freeze, access seal, EEG comparison
or scientific paper result follows from this preparation.

## October 6 Pacific: added-slot capacity definition and implementation

The policy's added-slot effective degrees of freedom now means the trace of
that fitted term's influence within the **joint** weighted ridge model. The
coordinator selected this interpretation before any EEG access; it follows
the term-wise trace definition in the
[official mgcv method documentation](https://stat.ethz.ch/R-manual/R-patched/library/mgcv/html/summary.gam.html).
Total model EDF is the trace of the complete fitted-value influence. Its
increase after adding a slot is a separate diagnostic because refitting a
penalized base can redistribute fitted influence between terms.

For one sensor contrast, let `Z0 = sqrt(W) X0` contain the common nuisance,
image and linear-label training columns, and let `Za = sqrt(W) Xa` contain
one added slot, both after the same sequence-local temporal expansion. With
the explicit diagonal base penalty `P0`, the existing joint objective is
`||sqrt(W)y - Z0 b - Za a||² + b'P0 b + lambda a'a`.
Let `A0 = Z0'Z0 + P0`, `B = A0⁺ Z0'Za`, `H0 = Z0 A0⁺ Z0'`,
`R0 = I-H0`, `E = Za-Z0 B`, `T = E'E + B'P0 B = Za'R0 Za`, and `U = E'E`.
Then the operative slot capacity is
`trace((T + lambda I)⁻¹ T)`. The incremental full-model capacity is
`trace((T + lambda I)⁻¹ U)`, and full-model EDF is the base EDF plus that
increment. These coincide for an unpenalized projection base; they need not
coincide when the base is penalized. In the one-row example `Z0=Za=[1]`
with both penalties one, base EDF is `1/2`, term EDF is `1/3`, and incremental
model EDF is `1/6`.

The [implementation](code/slot_capacity.py) computes the base identified
range from the augmented-design SVD `[Z0; sqrt(P0)]`, with an explicit caller
rank tolerance and the full singular spectrum recorded. This permits
rank-deficient unpenalized nuisance columns without inserting a penalty.
The right singular system of `[E; sqrt(P0) B]` supplies the eigenvalues of
`T`; the same vectors give the quadratic forms of `U`. No PSD eigenvalue
clipping, stochastic trace or response fitting occurs. This numerical range
still needs qualification against the production joint solver.

Every supplied slot shares one base, training row order, weights and base
penalties. Both curves use exactly `10**[-6, ..., 6]`. For the primary
`term_edf` definition, the matcher lists every pair satisfying
`df_concept <= df_label <= 1.05 df_concept`, or the symmetric five-percent
false-group bound. It reports zero-reference or unmatched grid points and
selects no penalty. A no-pair result is underidentified for those supplied
designs, not an automatic episode closure. Incremental-EDF matching is
labelled diagnostic. All quantities are per sensor contrast and conditional
on the supplied training transforms and penalties, not EDF for the entire
adaptive search.

This component consumes an explicit added-slot coefficient geometry. The RBF
feature/RKHS construction is implemented below but remains untested; passing
raw kernel columns with Euclidean ridge would impose a different model and is
not selected here.
The exact dense rectangular SVD route also has unmeasured full-size memory
and runtime costs; it is not yet a scalable matrix-free implementation.
The [five focused regression tests](code/test_slot_capacity.py) remain
**unrun**, awaiting the coordinator's next substantive compute allocation.
No EEG, source images, learned-feature payload or held-out row was read.

## October 6 Pacific: registered label-kernel coordinates

The [label RBF implementation](code/label_rbf.py) consumes the existing
training-only label-panel output. Its fitted panel record must retain all
three name/property/category families, with the same unique training-concept
IDs and column dimensions. It performs no additional centering, scaling or
PCA. IDs only align or repeat rows; they do not enter distances or features.
The bandwidth is the median of strictly positive pairwise Euclidean distances
among those training rows, times the existing multiplier `0.5`, `1` or `2`.
Distinct IDs with identical vectors remain included; an all-identical panel
has no identified bandwidth and refuses fitting.

For `K = U diag(d) U'`, the retained training coordinates are
`Phi_train = U sqrt(d)` and evaluation coordinates are
`Phi_eval = K(eval, train) U / sqrt(d)`. Euclidean coefficient ridge on these
coordinates therefore represents the retained RBF kernel. Raw `K` columns
with Euclidean ridge would instead change the kernel geometry. Event rows
repeat the fitted concept coordinates after fitting the unique concepts.
Training panel statistics, bandwidth and kernel directions are unchanged by
evaluation values or repeated event counts.

All numerical tolerances are explicit caller arguments. The rank threshold
is `max(eigenvalue_atol, eigenvalue_rtol * largest_eigenvalue)`. Eigenvalues
strictly above it are retained; a negative value below minus that threshold
refuses. The complete and discarded spectra, effective rank, relative
Frobenius reconstruction error and maximum absolute kernel-entry error are
reported. Both reconstruction errors must satisfy their supplied limits.
Discarded numerical directions receive no replacement eigenvalue or ridge
epsilon. This qualifies only the finite representation; it does not fit EEG,
choose penalties, establish capacity matching or open any data boundary.
The [four focused checks](code/test_label_rbf.py) cover kernel-ridge
equivalence, training-only application and ID joining, duplicate-value/rank
behavior, and sequence-design integration with the term-capacity calculation.
They remain **unrun** pending coordinated substantive compute.

The [finite label-preparation runner](code/prepare_label_geometry.py) is also
prepared but unexecuted. The existing structural report has the complete
block/concept checks for all 46 participants, and the frozen join establishes
their common inventory. Each permitted eight- or ten-block training set
therefore contains the same 1,854 unique concepts. Reusing a label transform
for that identical training-unit set reproduces separate training-only fits;
it is not a pooled EEG fit or a use of held-out-only labels. This equivalence
does not apply to image-family transforms, whose training images differ.

The runner reads only the completed name embeddings, aggregate human-feature
table, their identity/qualification records, and the structural/freeze reports.
It fits each label family once and retains all 27 requested rank triples and
81 rank/width mappings. Effective-rank aliases reduce computation to at most
six distinct label geometries and 18 kernels; actual numerical ranks may
reduce that count further. It writes each kernel eigensystem and releases it
before computing another. Shared PCA parameters and transformed geometry
rows are stored once; training coordinates remain derivable as `U sqrt(d)`.

The fixed numerical settings are eigenvalue relative tolerance
`1854 * float64_epsilon`, absolute tolerance zero, relative Frobenius kernel
error at most `1e-10`, and maximum entry error at most `1e-9`. These are
outcome-blind representation checks, with discarded spectrum and its ratio
to the existing minimum ridge penalty reported. That ratio is **not** an
error bound after temporal expansion and joint residualization; downstream
operator conditioning and capacity qualification remain unfinished.

The first necessary kernel is also the timing pilot and is retained as a
completed cache. Twice its observed time projects the remaining kernels,
with a 30-second completion allowance inside the explicit one-CPU, 7-GiB,
at-most-30-minute envelope. A failed projection leaves a reported partial
cache and starts no successor. These timings are not yet observed, and the
work may finish well below ten minutes. It is code-only until bundled with
substantive due work; it adds no padding, repeated equivalent fits, feature
freeze, access seal or EEG comparison.

## October 6 Pacific: selected compute-provisioning correction

**Historical selection snapshot:** the planned runtime/asset work below has
since completed as recorded above. The original stopped routes and access
requirements remain preserved.

Under the standing resume/repair instruction, the coordinator selected the
[bounded Slurm provisioning route](code/provision_exact_cpu_runtime_001.sbatch):
one CPU, 7 GiB, no GPU, at most two hours. A fresh private GroupHome environment
installs Python 3.12.1-compatible `torch==2.4.1+cpu` and
`torchvision==0.19.1+cpu` from the [official CPU distribution recipe](https://pytorch.org/get-started/previous-versions/#v241),
plus every unchanged feature-registry version and the selected text-cleaning
dependencies. Ordinary package dependencies are resolved and recorded in that
private environment; the prior environment and stopped alpha-module route
remain preserved. Installation/import compatibility and the tighter 7-GiB
loader envelope have not yet been demonstrated.

Sherlock's current `/etc/agents/storage.md` says the DTN has no interactive
login. The old instruction to run a downloader in a DTN shell was therefore
not executable. The selected correction retrieves only the existing fixed
public HTTPS files on allocated compute, without SSH, cookies or credentials,
and stops after any failure while preserving partial files. "Authenticated
assets" means matching the recorded source bytes/checksums before loading;
it does not require an account login for these public files. External rsync
or Globus remains a possible transfer mechanism, not an additional gate.

The sequential offline loader retains its 1,200-second internal bound and
exact layer/dimension checks, with no forward pass or optional text pilot.
The job supplies neither a pixel-worker boundary nor an EEG audit seal.
The failed namespace route remains stopped, and EEG-driven model comparison
still requires effective reader separation. The recommendations below
describe earlier decisions; this correction changes their transfer/runtime
implementation only, with no scientific-version or access-boundary amendment.

## October 6 public-source update — exact release recipes exist, runtime untested

**Historical public-source snapshot:** runtime execution was untested at this
point. Later compute qualification and label encoding are recorded above.

The [official PyTorch previous-version page, v2.4.1](https://pytorch.org/get-started/previous-versions/#v241)
publishes the exact `torch==2.4.1` / `torchvision==0.19.1` pairing already
selected by EP18, with CPU-only and CUDA 11.8/12.1/12.4 distributions. This
resolves public distribution availability; it does not validate those builds
against EP18's Python 3.12.1, other pinned packages or the Sherlock platform.
The stopped shared-module imports of 2.4.0a0/0.19.1a0 remain separate facts.
No installation, module repair, alpha-build substitution or compute job was
selected by finding the release recipe.

The [pinned CLIP configuration](https://huggingface.co/openai/clip-vit-base-patch32/raw/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/config.json)
was also publicly retrievable at the exact revision already in the
[asset map](config/model_runtime_asset_map_proposal.json). This verifies a
small metadata endpoint, not checkpoint-byte availability, integrity, loading
or feature inference. The attempted BLIP and DINOv2 metadata lookups did not
return usable content in this review; their availability was not disproved.
The existing exact asset list and authenticated DTN/Globus transfer route are
unchanged. No model weights, images, EEG or sealed outcomes were downloaded
or opened. Supported local runtime qualification, authorized transfer and
effective pixel/EEG reader separation remain necessary operational work.

## October 6: execution-route recommendations — unadopted

**Historical recommendations:** the later selected compute-provisioning route
superseded the transfer/runtime recommendations below without substituting
versions. The proposed weaker access boundary was not adopted; effective
image/EEG isolation remains required.

The supplied review correctly identifies completed input preparation: the
30/16 participant assignment, event join and EP17/EP18 qualified-universe
comparison are complete. Both studies retain the recorded correlated,
design-exposed interpretation. The overlap decision is no longer a missing
dependency. Completion of these metadata tasks does not qualify model assets,
the inference runtime, pixel-worker delivery or EEG evaluation separation.

The [asset/runtime proposal](config/model_runtime_asset_map_proposal.json)
already records the stopped state; its exact-version recipe is not superseded
by a different episode's runtime choice. The intended next routes are:

1. **Acquire the fixed public model assets through the authenticated DTN or
   Globus route.** The selected files total 2,142,873,276 bytes. A previous
   compute-side download elsewhere does not make a new 2.1-GB compute-node
   transfer the default replacement for the site's large-transfer route.
   Recommend the existing user-authenticated DTN handoff. No authentication
   retry, new download location or transfer is selected by this revision.
2. **Resolve the exact software recipe.** EP18 selects Torch 2.4.1 and
   Torchvision 0.19.1; the actual module imports were 2.4.0a0/0.19.1a0.
   Public matching-release recipes are now linked above; local compatibility
   is still unverified. Accepting a build for EP21 does not amend EP18. Recommend supported matching
   builds or clarification of the reported build identities. An existing
   compatible group environment can be assessed against EP18's complete
   Python/Pillow/Transformers and feature recipe before choosing a new route;
   no sibling environment is assumed compatible from its Torch version alone.
   An alpha-build substitution would need its exact consequences documented
   before feature generation; it remains unadopted here.
3. **Resolve effective reader isolation.** The recorded bubblewrap attempt
   failed before child launch with namespace `ENOSPC`. That observation does
   not establish its cause or identify it with Codex's expected agent-sandbox
   limitation. Preserve the stopped route and the existing support handoff;
   no configuration probe, weaker flags or different-node retry follows.
   A supported, appropriately configured controlled runtime can potentially
   enforce the required separation. Same-UID permissions, a subagent or an
   unqualified container alone cannot demonstrate it. The relevant test is
   whether the development process can reach or retrieve audit EEG, not the
   name of the isolation technology.

Recommend keeping the current separation contract while resolving a supported
route. A procedural-only boundary is a reviewable alternative amendment with
a disclosed weaker evaluation claim; it is not adopted, and EEG-driven
candidate comparisons remain prohibited before the present seal is verified.
The feature-worker boundary also remains separate from the EEG audit seal.
No evaluation access, software substitution, resource increase, simulation
program or empirical job is activated by these recommendations.

## Existing scientific plan

[Current conceptual figure](ep18_conceptual_question-v2.png) · [Exact image-generation record](ep18_conceptual_question-v2-prompt.md).

Status: proposed study design, 2026-09-27. No EEG result, audit result, or
paper claim is reported here.

## The intended contribution

The opening question is whether sharing EEG responses by named concept
predicts unseen exemplars better than equally flexible continuous-label
sharing and matched false groupings, under the same image model.

Named labels may already identify the concepts in the continuous inputs.
The comparison concerns the sharing and regularization basis, not a clean
separation of conceptual from visual information or newly supplied labels.

Cross-exemplar category decoding is already known, so a paper cannot stop
there. EP18 asks a more demanding question. A true-concept template must
predict held-out EEG better than a capacity-matched nonlinear label basis
built on the same image model. It must also beat false image groupings
constructed to look equally plausible in image-feature and temporal terms,
and it must repeat in participants who did not influence the analysis.

The deeper result would explain when the advantage survives. The preferred
prediction is that the true-concept contribution remains specific for
held-out images that are far from the ten training images in a prespecified
visual-only feature panel. A collapse with that distance would favor residual
visual organization over a stable concept response.

## What is already known, and what EP18 must add

This is a crowded question. EP18 should not present cross-exemplar category
information, an early-to-late visual-to-conceptual progression, or an
association between semantic models and EEG as new findings. The closest
prior work sets the following boundaries.

| Prior work | What is already established | What remains for EP18 |
| --- | --- | --- |
| [Grootswagers et al. (2022)](https://doi.org/10.1038/s41597-021-01102-7) | THINGS-EEG1 contains measurable image and concept structure across 22,248 images and 1,854 concepts. It is the same EEG dataset used here. | EP18 is a stricter reanalysis, not an independent acquisition or a first demonstration of cross-exemplar information. |
| [Holm et al. (2024)](https://doi.org/10.1016/j.neuroimage.2024.120626) | Low-level image statistics can support apparently semantic THINGS-EEG decoding, and model optimization can increase sensitivity to that confound. | A concept result must survive a strong image model and alternatives that preserve residual visual coherence; ordinary random labels are not enough. |
| [Kim et al. (2026)](https://doi.org/10.1167/jov.26.8.2) | Behavior-derived perceptual and conceptual structure explains distinct parts of time-resolved object EEG. The pattern was evaluated in THINGS-EEG1 and a separately acquired EEG dataset. | EP18 cannot claim novelty from showing that conceptual features relate to EEG or emerge later than perceptual features. It instead tests whether a categorical basis adds held-out predictive value under a common capacity-controlled fit. |
| [Rong et al. (2025)](https://doi.org/10.7554/eLife.108915) | Vision, language, and combined vision-language representations have been compared as encoding models in THINGS-EEG2. | EP18 must compare bases with matched fitting opportunities rather than infer conceptual coding from unequal model rankings. |
| [Watson et al. (2026)](https://doi.org/10.1016/j.neuroimage.2026.122012) | Continuous dimensions of natural-object variability can organize behavioral, EEG, and MEG responses in the THINGS ecosystem. | A categorical slot winning prediction does not by itself establish a discrete category code; it must beat matched continuous-feature and pseudo-group explanations. |

Against that background, EP18 adds four linked tests rather than another
generic semantic-decoding result:

1. all models predict the same continuous RSVP EEG while explicitly modeling
   overlapping responses;
2. the true-concept slot competes with a capacity-matched nonlinear readout of
   the same label information;
3. the true concepts compete with eight EEG-blind false partitions matched for
   visual structure, temporal context, and fitting capacity; and
4. the complete comparison, including the distant-exemplar prediction, is
   repeated under a frozen procedure in held-out participants.

These precedents narrow the claim; they do not decide the episode in advance.
They determine what each possible result can contribute. A reproducible true-concept advantage
would establish a narrow predictive-basis result; equivalence with the
nonlinear label model would show that the categorical basis is not needed
within the tested margin and model space; equivalence with matched false groups would challenge the
specificity of earlier concept interpretations; and failure in audit would
place a replication boundary on the complete procedure. None of these
outcomes may be renamed after release to recover a preferred story.

The contribution is therefore conditional. It is a claim about the predictive
value of a categorical basis in this inventory, not proof of abstract or
amodal semantic representation.

## Prospective sequence filter — October 12, 2026

Before EEG comparison, fix the129-tap causal low-pass described by
`scipy.signal.firwin(129,100,window=("kaiser",8.6),pass_zero="lowpass",scale=True,fs=1000)`.
Use origin0, no high-pass or temporal detrending, and decimate to250Hz at
phase0 of the original recording grid. Preserve acquisition timestamps. This
has128ms of past support, no future support and64ms group delay: the fixed
0–800ms model horizon predicts **filtered EEG**, and can miss late unfiltered
response shifted past800ms. It is not a claim of unfiltered neural latency.
Slow drift remains a joint nuisance term. Negative-lag and shifted-label
controls remain mandatory.

The actual coefficient/metadata evidence is
`runs/feature_preparation/sequence_geometry_fir_20261012_001/metadata_and_fir.json`.
It inspected headers, markers and file sizes from all30 development participants,
not EEG amplitudes. All nonempty recorded E3 intervals have the proposed window
support; exact main-event and frozen-channel binding still precedes preparation.
Use each recorded preceding E3 and its authentic segment, preserving variable
cue gaps rather than forcing one second. Every retained filtered sample must
have its complete FIR support within that same physical sequence. The policy
records the filter choice; empirical access, full span/mask integration and
resource admission are separate decisions.

## Paper sequence

### 1. Establish that the test is identifiable

Reconstruct all 12 blocks and 72 main physical sequences, verify the exact
event-image join, and show that filtering and temporal expansion do not share
samples across fitting and scoring. Confirm recoverable image, task, and
repeated-image EEG signals at the participant level.

Use simulations with the real event timing to show that the continuous model
distinguishes a concept-stable signal from image-only, neighboring-image, target,
drift, and latency effects. Demonstrate that all eight false partitions meet
their visual, temporal, and fitting-capacity targets without EEG.

If these checks fail, the study is underidentified. More model search cannot
repair a design that cannot separate overlapping responses or construct the
intended alternative.

### 2. Test the categorical-basis advantage

For each participant and outer fold, fit on ten blocks and score two. Compare
the true-concept slot with:

1. the image model;
2. the linear label model;
3. the capacity-matched nonlinear label model; and
4. each of the eight matched false-group models.

The main result is participant-level held-out prediction, averaged over six
outer folds. The true concept must exceed a meaningful margin against the
nonlinear label model and against every matched false grouping. The same
samples, nuisance terms, temporal basis, sensor space, and fitting budget are
used for every comparison.

This step can establish a reproducible predictive-basis advantage. By itself,
it does not explain whether that advantage reflects stable conceptual
structure or visual similarities that the reference models missed.

### 3. Ask whether the effect survives visual change

Measure, without EEG, how far each held-out image is from the ten training
images of the same concept in a frozen panel containing low-level,
supervised-vision, and self-supervised-vision features. Exclude captions,
concept names, human semantic features, and vision-language embeddings from
this distance definition. Divide held-out events into near, middle, and
distant thirds using the same rule in every participant and fold.

Keep the full continuous model. For distant held-out events only, remove the
fitted added-slot contribution over its complete 0–800 ms response support,
leave every other event contribution unchanged, and rescore the complete
physical sequence. Perform this post-fit ablation separately for the true
concept, capacity-matched label, and each false-group model. No model is
refitted in a distance stratum.

This is fit-dependent predictive importance: correlated bases can redistribute
their fitted contributions without changing the available information. The
ablation is not unique variance, a causal contribution, or a substitute for
the primary held-out model comparison. The existing contrast remains unchanged.

The participant-level explanatory contrasts ask whether removing the distant
true-concept contribution harms prediction more than removing the distant
label contribution and more than removing each distant false-group
contribution. The label contrast and eight false-group contrasts form one
separate simultaneous family. Overlap simulations must show that this
procedure recovers the intended event contribution without turning a
neighboring-image signal into a distant-exemplar result.

The competing predictions are:

| Explanation | Prediction in the distant third |
| --- | --- |
| Categorical sharing remains useful across visual-feature distance | Removing the true-concept contribution harms prediction more than removing the label or every false-group contribution |
| Residual visual similarity | The distant true-versus-label contribution difference shrinks toward zero |
| Generic grouping | At least one matched false-group contribution is indistinguishable from the true-concept contribution |
| Sequence artifact | Negative lags, shifted labels, or neighboring-image simulations show a similar apparent effect |

Estimate the pattern in development, freeze the display and uncertainty rule,
and repeat the directional prediction in the 16 audit participants. Failure
of this explanation does not rewrite the primary result. It narrows the paper
to a categorical predictive-basis advantage whose source remains unresolved.

### 4. Locate the signal in time without turning latency into a new search

Plot image gain, true-concept advantage, and true-versus-false separation over
positive lags under the fixed temporal model. The expected pattern is that
image features dominate the earliest response and the concept advantage
persists later. Negative-lag and long label-shift curves are shown on the same
scale.

This is a secondary diagnostic. An isolated favorable latency cannot replace
the global continuous-prediction result, choose a different baseline, or
rescue failed sequence controls.

### 5. Repeat the complete prediction in new participants

Open audit only if the released development result clears the categorical and
all eight false-group margins, its required controls pass, and development
participant vectors show that 16 participants can resolve the planned
margins. Then apply the complete frozen procedure once to all 16 assigned
audit participants. Coefficients are refitted within each participant's
training blocks, but the model family, features, folds, partitions, margins,
and decisions do not change.

Report all assigned participants. Reliability or an unfavorable effect cannot
be used to remove or replace an audit participant.

## What each result would mean

| Result | Paper-level interpretation | Next action |
| --- | --- | --- |
| Primary result and distant-exemplar prediction both repeat | A categorical response basis remains useful across substantial distance in the prespecified visual-only panel | Develop the timing and category-level boundary as the main explanation |
| Primary result repeats but the distant prediction fails | The categorical basis is predictive, but stability across the prespecified feature distance is not established | Retain the narrower result; do not claim a concept-stable mechanism |
| The true concept beats the linear label basis and is equivalent within the margin to the capacity-matched nonlinear basis | A categorical basis is not needed relative to this tested nonlinear readout and margin | Report the boundary or stop the independent-paper direction |
| A matched false grouping is equivalent or better | True concepts are not distinguished from matched image-feature and temporal organization | Report partition nonspecificity; do not rename the grouping effect as conceptual |
| Negative-lag or shifted-label effect is meaningful | Temporal leakage or sequence structure explains the apparent advantage | Stop the concept interpretation |
| The fixed development baseline does not clear the complete categorical and false-group margins | Development does not support the prespecified conjunction | Do not open the one-shot audit and do not select a friendlier baseline |
| Development effect fails in audit | The procedure did not reproduce in new participants | Report nonreplication without reopening model search |
| Precision or reliability is inadequate | The design cannot distinguish the explanations | Report underidentification; do not treat a null p-value as sufficiency |

## Figures, contingent on observed evidence

| Figure | Scientific judgment | Required content |
| --- | --- | --- |
| **1. The question and competing explanations** | What would distinguish a concept-stable response from feature sufficiency or artifact? | Synthetic concept exemplars, overlapping EEG, fair model comparison, visual-feature-distance prediction, and held-out participants |
| **2. Is there a categorical-basis advantage?** | Does the true concept beat the strongest reference and all matched false groups? | Participant-level contrasts, simultaneous intervals, every false partition, and positive controls |
| **3. Does the effect survive visual-feature distance?** | Is the true-concept contribution still specific for the most distant held-out exemplars? | Frozen visual-only distance, post-fit contribution ablations, true versus label and false groups |
| **4. Is the timing stimulus-locked?** | Does the positive-lag pattern differ from sequence leakage? | Image and concept time courses, negative lags, shifted labels, target and drift controls |
| **5. Does it repeat in new people?** | Which parts of the development story survive untouched? | All 16 audit participants, primary conjunction, explanatory prediction, failures, and uncertainty |

The first figure is a design illustration, not evidence. Later figures should
show all participant-level results rather than a selected grand-average trace.

## Planned conceptual figure

The conceptual figure should follow the logic of EP12 rather than depict a
generic workflow:

- **Panel A:** one concept represented by visibly different synthetic images;
  ten train the template and two images distant in the visual-only panel are
  held out;
- **Panel B:** a continuous EEG trace in which neighboring RSVP images have
  overlapping responses;
- **Panel C:** the same held-out data predicted by a strong image-and-label
  model, the true-concept model, and matched false-group models;
- **Panel D:** distinct evidence patterns for concept stability, feature
  sufficiency, generic grouping, and sequence artifact; and
- **Panel E:** the complete fixed comparison repeated in 16 new participants.

Use synthetic illustrations rather than THINGS source images, and label the
asset as a conceptual mockup with synthetic data.

## Claim language

If the primary and explanatory predictions pass, an abstract could say that
a categorical response basis predicted held-out exemplars better than a
capacity-matched label basis built on the same image model, remained specific
for exemplars distant in the prespecified visual-only feature panel, beat
matched alternative groupings, and reproduced in new participants.

It should not say that EEG contains abstract meanings, that vision is
irrelevant, that concepts have been causally localized, or that the result
generalizes to concepts and images outside the fixed THINGS inventory.

EP17 and EP18 reuse the THINGS stimulus ecosystem. The paper must disclose
their exact image, concept, and feature overlap; EP18 is not an independent
stimulus-family confirmation of EP17.
