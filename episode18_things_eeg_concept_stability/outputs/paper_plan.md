# EP18 paper plan: when does a concept survive a change of picture?

Status: proposed study design, 2026-09-27. No EEG result, audit result, or
paper claim is reported here.

## The intended contribution

The opening question is simple: if a person sees different pictures of the
same thing, is there an EEG response shared by the concept rather than by the
particular pixels?

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
nonlinear label model would show that readout capacity explains the apparent
advantage; equivalence with matched false groups would challenge the
specificity of earlier concept interpretations; and failure in audit would
place a replication boundary on the complete procedure. None of these
outcomes may be renamed after release to recover a preferred story.

The contribution is therefore conditional. It is a claim about the predictive
value of a categorical basis in this inventory, not proof of abstract or
amodal semantic representation.

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
| Stable concept response | Removing the true-concept contribution harms prediction more than removing the label or every false-group contribution |
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
| The true concept beats the linear label basis but not the capacity-matched nonlinear basis | The first advantage reflected insufficient label-readout capacity | Report the boundary or stop the independent-paper direction |
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
