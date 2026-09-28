# EP21 paper plan: do language-aligned visual representations transfer to LAION-fMRI?

Status: fixed study design initialized on 2026-09-28. No empirical input has
been provisioned, no neural result has been inspected, and no execution is
authorized. The scientist authorized contract authoring only.

## The intended contribution

[Doerig et al. (2025)](https://doi.org/10.1038/s42256-025-01072-0) reported
that high-level visual representations in the human brain align with
language-model representations of scenes. EP21 asks a narrower transfer
question: do three prespecified directional patterns from that work appear in
the regular-image portion of LAION-fMRI when the analysis is fixed in advance?

The contribution is not a claim that LAION-fMRI recreates the original Natural
Scenes Dataset (NSD) experiment. The participants, images, preprocessing, ROI
definitions, category metadata, and inference differ. A useful result would
show which parts of the original pattern meet the prespecified transfer rule
under those differences and which remain unsupported. Non-replicated, partial,
and non-evaluable results are therefore part of the intended paper rather than
reasons to search for a replacement claim.

The paper will report one verdict for each of three findings. It will not
collapse them into a single episode-wide success label. Even a fully positive
vector would support only directional transfer to these five participants,
regular images, and proxy visual-sector ROIs—not exact quantitative
reproduction, population generalization, or a universal representational
principle.

## Why this reproduction

The central scientific opportunity is to test whether the reported relationship
between scene semantics and visual cortex is tied to one dataset and analysis
context or remains visible in a substantially different image-fMRI resource.
LAION-fMRI offers participant-specific regular images for fitting and a common
regular-image set for fixed participant-by-participant evaluation.

The evidential limitations are equally important. The 1,121 shared regular
images were already opened in the EP04 lineage, so this is retrospective and
correlated evidence rather than a fresh confirmation. Of those images, 240
originate from NSD; the full regular pool remains primary, with a prespecified
exclusion sensitivity. The five participants support within-participant
adjudication but not population-level inference.

| Direction | Assessment |
| --- | --- |
| Claim an exact reproduction of the NSD study | Unsupported because the participants, stimuli, ROIs, preprocessing, metadata, and inferential design differ |
| Explore LAION-fMRI and retain whichever semantic analysis looks strongest | Invalid for this episode because the shared outcomes are already exposed and the procedure is fixed before EP21 scoring |
| Test the three source findings under one declared transfer protocol | Preferred: it gives supported, partial, and unsupported transfer outcomes a clear interpretation while keeping every adaptation visible |
| Use the 371-image OOD pool to strengthen the story | Prohibited: the entire OOD payload is reserved for EP04 and is not an EP21 input |

## What each finding would mean

| Finding | Scientific question | What prespecified support would mean | What failure would mean |
| --- | --- | --- | --- |
| 1. Caption geometry, encoding, and decoding | Do caption-derived scene representations resemble higher-level visual-cortex geometry, predict held-out responses, and remain decodable from the prespecified visual sectors? | For each supported subclaim, the corresponding original direction transfers in representational similarity, brain encoding, or embedding decoding under the participant rule | That subclaim does not meet the prespecified transfer rule; it does not establish that semantic information is absent from the brain |
| 2. Contextual integration | Do full captions align better than isolated words, nouns, verbs, and object inventories? | Full-caption geometry carries alignment not captured by the prespecified lexical and object-content comparisons, subject to the source paper's ROI exceptions | The apparent alignment may be adequately explained by simpler lexical or object information, or the adapted control may be unevaluable |
| 3. Language-trained recurrent vision models | Do released recurrent visual networks trained toward caption-MPNet representations align better than category-trained and original alternative models? | The prespecified alignment advantage of the language-trained RCNN transfers to the LAION-fMRI setting | That subclaim does not meet the prespecified transfer rule, or the comparison is not evaluable because authentic weights, comparators, preprocessing, layer, time step, or aggregation cannot be established |

The LAION object inventory is an adapted proxy for the original COCO-category
comparison. Shuffled captions are a separate adapted negative control and
cannot change the primary Finding 2 verdict.

Finding 1's original-dictionary nearest-caption lookup is qualitative. The
quantitative LAION correct-caption rank is an explicitly adapted subclaim and
runs only with a fixed candidate dictionary containing the prespecified targets.
Failure of that adapted rank cannot erase an otherwise evaluable embedding-
decoding result.

## Study sequence

### 1. Qualify the source without opening neural outcomes

Establish the LAION-fMRI release identity, permissions, risk classification,
trial-to-image joins, regular-image roles, masks, features, and official splits.
Create separate read-only views for structural qualification, non-neural
features, and post-lock regular-image scoring. The qualification-facing record
contains structural facts only, never neural values or derived scores.

Authenticate the source code, model checkpoints, tokenizers, parsers, decoder
dictionaries, RCNN instances, and original comparator panel. A missing asset is
reported as absent or unknown; it is not silently replaced.

### 2. Freeze and test the analysis procedure

Resolve every common protocol field and every field required by a runnable
module. Test preprocessing, representational similarity, encoding, decoding,
paired inference, bootstrap, multiplicity, and verdict logic on synthetic or
permuted fixtures. An unresolved branch is frozen as `not_evaluable` before any
neural result; unrelated qualified branches may continue.

A complete `protocol_lock.yaml` is necessary before separately authorized
regular-image neural scoring; the lock does not itself grant execution
authority. Its resolver cannot receive subject-unique or regular-shared neural
values, predictions, scores, or any OOD derivative.

### 3. Prepare outcome-independent features

Construct the frozen caption and control features, then the RCNN and cross-modal
feature manifests, from authorized inputs. Every model revision, layer, pooling
rule, preprocessing step, normalization, image ID, and missingness rule must
match the protocol lock. Feature availability cannot be decided from neural
performance.

### 4. Evaluate the three core findings separately

Fit predictive maps only on each participant's 4,712 regular subject-unique
images and evaluate them unchanged on the 1,121 regular images shared by all
five participants. Run the prespecified ROI analyses in participant native space.
Finding 1, Finding 2, and Finding 3 are sibling branches; an outcome in one
cannot select the method, comparator, or continuation of another.

For each directional subclaim, report every participant's effect, interval,
raw and corrected probability, null distribution, exclusions, and missingness.
A subclaim is `replicated` only when all five participants are evaluable, at
least four effects have the predicted direction, and at least three have
corrected significance in that direction.

It is `partial` when exactly three participants have the predicted direction,
or when at least four have that direction but fewer than three have corrected
significance. At most two predicted directions give `not_replicated`.
Incomplete or invalid required participant results give `not_evaluable`.

### 5. Report robustness and extensions without rescuing the core

Run the official `tau` split and five-cluster (`cluster_k5`) analyses as fixed
secondary checks. Report all ten three-of-five caption subsets for Findings 1
and 2, and repeat the applicable analyses after excluding the 240 NSD-origin
images. No subset or sensitivity replaces the primary full-pool result.

The secondary cross-modal analysis compares caption MPNet with DINOv2,
OpenCLIP, PEcore, and SigLIP2 under a common train-fold dimensionality rule. It
asks whether early visual cortex contains more vision-unique signal and whether
higher visual sectors contain more text–vision shared signal. These contrasts
form a separate family and cannot change a core finding verdict.

### 6. Integrate the result vector

Report source qualification and protocol deviations first, then the three core
verdicts, followed by robustness and extensions. Every module must end in a
result or an explicit `not_evaluable` record. The final scientific outcome is
the ordered vector `(Finding 1, Finding 2, Finding 3)`; operational completion,
technical failure, and policy violations are reported separately.

## Decisive controls and rival explanations

| Rival explanation or validity threat | Prespecified test | Consequence if it accounts for the result |
| --- | --- | --- |
| Alignment reflects generic low-level visual similarity | Compare higher-level sectors with the prespecified early-visual-cortex specificity control and report the fixed ROI pattern | Narrow the claim to the observed ROI pattern; an EVC effect cannot rescue a failed higher-level primary endpoint |
| Full captions add no information beyond their words or visible objects | Compare contextual captions with separately embedded words, nouns, verbs, and the adapted LAION object inventory | Do not claim contextual integration when a simpler prespecified representation explains the result |
| Any visual network would show the same brain alignment | Compare the language-trained RCNN with its category-trained counterpart, caption MPNet, and the authenticated original alternative-model panel | Do not attribute an advantage to language-oriented training if the prespecified comparators match or exceed it |
| A decoding result is created by dictionary construction | Keep the original qualitative lookup dictionary separate from the adapted target-containing rank dictionary and freeze targets, similarity, ties, rank direction, and chance | Mark only the affected retrieval subclaim `not_evaluable` or `not_replicated`; do not reinterpret it as an original-paper endpoint |
| Noise-ceiling correction or dependent RDM edges inflates evidence | Authenticate the participant/ROI RDM ceiling and use image-level permutation and bootstrap procedures that rebuild dependent quantities | Lock the affected contrast `not_evaluable` if a valid calibrated method cannot be established |
| The apparent transfer depends on NSD-origin images | Repeat the prespecified sensitivity after removing all 240 NSD-origin images while retaining the full regular pool as primary | Report the dependence and narrow the image-source claim; do not select the more favorable pool |
| EP04 exposure makes the result look confirmatory | Declare the shared evidence and retrospective status in every report and prohibit EP04 outputs from EP21 inputs | No EP21 result may be called a fresh independent confirmation or used to select or audit EP04 candidates |
| Missing permissions, weights, comparators, or source-code details are repaired with convenient substitutes | Qualify every required asset before scoring and freeze unavailable branches as `not_evaluable` | Stop the affected branch; do not substitute after viewing neural outcomes |

## The OOD boundary

The source paper's OOD Method 3 is distinct from EP21 Finding 3. It is not an
executable EP21 analysis. The 371-image pool contains relation, unusual,
cropped, shape, and Gabor conditions and is reserved for EP04's conditional
audit. EP21 may cite published category-level descriptions, but it may not
access the raw images,
derived features, neural responses, predictions, scores, or summaries.

Consequently, this paper cannot support an OOD-robustness claim, regardless of
the regular-image results. A future OOD analysis would require a separate
scientific and cross-episode decision; it cannot be inferred from EP21.

## When to deepen, narrow, or stop

| Evidence | Next action and permitted interpretation |
| --- | --- |
| All three findings meet their prespecified rules | Report directional transfer across the three prespecified finding families, bounded to these participants, regular images, and proxy ROIs; seek fresh data for independent confirmation |
| The three findings produce mixed verdicts | Preserve the ordered verdict vector and explain which representational claims transfer; do not average them into an episode-wide success |
| Finding 1 meets its rule but Finding 2 does not | Report semantic alignment or predictability without claiming a specific advantage for contextual sentence integration |
| Findings 1–2 meet their rules but Finding 3 does not | Separate brain–language representational alignment from the proposed alignment advantage of the language-trained RCNN |
| A prespecified control explains a positive-looking effect | Narrow the interpretation to that control or organization; do not promote a secondary analysis into a replacement primary claim |
| A required asset or calibrated inferential procedure is unavailable | Mark the affected subclaim `not_evaluable`, report why, and continue only unrelated branches whose locks are complete |
| No core finding meets its prespecified rule | Report bounded non-replication with its precision and adaptations; do not search the same outcomes for a replacement claim |
| Source roles, DUA, or the no-OOD handoff cannot be qualified | Stop before neural access; contract completion does not justify a weaker boundary |
| A consequential scientific change is proposed after a neural score | End EP21 and create a new episode rather than revising this one around the observed result |

Failure to replicate is not evidence of no effect because no equivalence margin
is prespecified. A `not_replicated` result means only that the directional
reproduction rule was not met under the specified LAION-fMRI transfer design.

## Paper figures, contingent on the results

| Figure | Claim the evidence must earn | Decisive content |
| --- | --- | --- |
| 1. The transfer design and its limits | EP21 performs a fixed, transparent NSD-to-LAION-fMRI transfer test | Dataset differences, fitting/evaluation roles, participant unit, EP04 dependence, and the closed OOD boundary |
| 2. Caption geometry and prediction | The locked adjudication rule supports Finding 1 or a named component-level subclaim | Every participant's ROI RSA, held-out encoding, embedding-decoding effects, uncertainty, the locked Finding 1 aggregation rule, and EVC specificity control |
| 3. What context contributes | Full captions outperform the prespecified simpler representations where predicted | Participant-level full-caption contrasts against words, nouns, verbs, and objects, with source exceptions and shuffled-caption control visible |
| 4. How language-trained visual models compare | The language-trained RCNN has the prespecified alignment advantage over fair comparators | Ten-instance aggregation, category-trained RCNN, caption MPNet, original comparator panel, and every participant result |
| 5. Robustness, extensions, and boundaries | The retained interpretation survives its prespecified checks without relying on selection | `tau`, held-cluster results, caption subsets, NSD-origin exclusion, cross-modal components, failures, and non-evaluable branches |

Figures are contingent summaries, not promised findings. A non-evaluable module
must remain visible rather than disappearing from the figure sequence. The
abstract should name the adapted and retrospective design, the five-participant
scope, the ordered finding vector, the strongest rival explanation addressed,
and the limits imposed by proxy ROIs, shared evidence, and absent OOD testing.
