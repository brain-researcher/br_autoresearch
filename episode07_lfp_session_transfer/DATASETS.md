# EP07 data: what is available and how it answers the question

EP07 uses the public motor-cortex LFP and population-spiking release from
[Dryad DOI 10.5061/dryad.xd2547dkt](https://doi.org/10.5061/dryad.xd2547dkt)
(metadata version 7, file-bearing version 5). The Dryad record lists all twelve
named session files below and describes extracellular neural and behavioral
recordings during an eight-target reaching task. That establishes that the
planned sessions and broad measurement types exist; it does not yet establish
the field-level trial correspondence, usable condition cells, or alignment
rank required by EP07. The release has been used in earlier work, so the
present study can provide a careful internal test but not an independent
replication.

No neural analysis has started for this episode. The remaining preparation is
to confirm that the named sessions support the planned time windows, reach
directions, trial counts, LFP features, and population-spike targets.

## Paper integration without a new data split

EP07 owns the history-and-population-components paper; EP06 is a supporting
recording-domain component study. This decision does not merge their data
roles. EP07 keeps its chronological three-source/three-target day assignment
and whole-trial calibration/development/held-out split. EP06's development and
reserved session positions are different and may expose trials still closed
in EP07. No EP06 outcome or fitted object may enter EP07 model development.
On this corpus, score the EP06 component only after EP07's choices are frozen
and its single primary opening is complete, unless a scientist prospectively
approves a compatible joint-role design before any relevant score is viewed.
Same-corpus component evidence is not an independent replication or a new day.

The EP06 common-time, direction-by-time, and trial-residual projections answer
a frequency-rule consequence question, not EP07's history-minus-today-only
contrast. The two are reported with their own denominators, fitting roles,
and uncertainty. EP05 reach-deviation outcomes and EP08 acquisition-policy
outcomes are likewise not inputs to this episode.

## The recordings used in the primary study

The primary corpus contains twelve simultaneous M1/PMd recording days from
Mihili and Chewie-L. The first three days from each animal supply historical
training data; the next three are new days on which transfer is tested.

| Animal | Earlier days | New days |
| --- | --- | --- |
| Mihili | 2014-02-17, 2014-02-18, 2014-03-03 | 2014-03-04, 2014-03-06, 2014-03-07 |
| Chewie-L | 2016-09-29, 2016-10-05, 2016-10-06 | 2016-10-07, 2016-10-14, 2016-10-21 |

Transfer is permitted only within the same animal, implant, and cortical
region. A trial from a new day never enters the historical fit for another new
day. M1 and PMd from the same behavioral trial always receive the same data
role.

Chewie-R represents another implant in Chewie. It may later be used to ask
whether the result is sensitive to implant identity, but it is not an
independent animal and does not contribute to the primary comparison.

## The two neural settings

Use the provider's native 30-ms bins:

- **M1 during movement:** 150--450 ms after movement onset, 10 bins.
- **PMd before movement:** 450 ms before the go cue up to the go cue, 15 bins.

Because region and task epoch change together, an M1/PMd difference cannot by
itself be interpreted as a preparation-versus-execution effect.

The response is the released native-bin population spike count. The inputs are
the released LMP and band-power features. These LFP features were processed by
the provider at the session level and raw voltage is not available here. All
models therefore use the same released features unchanged, and conclusions
are limited to that offline representation.

## Which trials and neurons are eligible?

A behavioral trial is included only when:

- the reach was successful;
- animal, implant, date, reach direction, and simultaneous M1/PMd records
  agree;
- the movement and go-cue indices support both planned windows;
- required LFP and spike-count arrays are present and finite; and
- spike counts are nonnegative integers.

The eight reach directions are assigned to the nearest multiple of 45 degrees,
with at most 0.01 radians of circular error. A failure in either M1 or PMd
excludes the whole simultaneous trial. Eligibility depends only on identity,
timing, and data availability. Neural magnitude, variability, model fit, and
prediction error never determine inclusion.

Each new day and region keeps the neurons that are consistently present across
all eligible trials for that day. Earlier days keep their own neuron sets.
Neurons and numeric electrode labels are not matched across days. If a model
uses a shared low-dimensional representation, it must learn that representation
from the permitted earlier-day and calibration data.

Every primary day must contain all eight directions and at least 15 complete
trials per direction. If either animal lacks the required six eligible days,
the study stops for redesign before neural scores are examined. Days,
directions, or time windows are not replaced after seeing results.

## How new-day trials are divided

Within each new day and reach direction, trials are shuffled once using the
predeclared ordinary seed `ep07-target-role-split-v1`. The resulting order is
saved before neural outcomes are examined. Trials are then divided as follows:

```text
n_calibration = floor(0.2*n + 0.5)
n_development = floor((n - n_calibration)/2)
n_held_out    = n - n_calibration - n_development
```

At the minimum of 15 trials, this gives 3 calibration, 6 development, and 6
held-out trials per direction. All time bins, neurons, features, and predictions
from one behavioral trial share its role.

- **Calibration trials** are visible to both the history-assisted and
  today-only models.
- **Development trials** provide the aggregate feedback used to compare model
  choices, but their spike targets are not available to model-fitting code.
- **Held-out trials** remain unseen until the model choices and statistical
  analysis are final.

Trial order is not searched. Trials are never moved between roles because of a
neural value, model score, missing result in another role, or a more favorable
split.

## What each analysis stage may use

The history-assisted model may use:

- LFP features, spike targets, behavior, and timing from the three earlier days
  assigned to the same animal (six earlier days in the study as a whole);
- LFP features, spike targets, behavior, and timing from each new day's
  calibration trials; and
- LFP features and behavioral information, but not spike targets, from that
  day's development trials when predictions are requested.

The today-only comparison receives the same new-day information and the same
number of model-development opportunities, but no earlier-day observations or
fitted state. The direction-and-time comparison uses only calibration trials
from the new day.

Held-out LFP features and spike targets are kept apart from model development.
After all choices are final, the selected models are scored once on the same
held-out trials. No partial held-out result may be used to modify a model,
exclude a day, or request another candidate.

This split does not mean the public outcomes were previously unknown. The final
report must state that the public release and earlier campaign work were
already outcome-exposed.

## Data needed to explain what transfers

The first study asks only whether earlier days help. If they do, the proposed
follow-up needs enough retained information to distinguish three explanations:

1. a better direction-and-time mean;
2. reusable low-dimensional population geometry; and
3. trial-specific LFP--population coupling after the mean response is removed.

For that follow-up, preserve for every permitted day:

- trial, direction, and time-bin identities needed to estimate means from
  training or calibration data only;
- the exact LMP and power-band identities used by the selected model;
- unresidualized spike responses used to learn source-fitted and
  target-calibrated population coordinates;
- the condition cells and direction-by-time averages used as alignment
  anchors before residualization;
- predictions needed to refit and score a genuine residual-only model;
- both frozen history-assisted and today-only whole-trial predictions, needed
  to apply the same within-direction mismatch and compare their incremental
  correct-pair gains;
- omission results for each earlier day and retained model component; and
- electrode proximity and spike-quality summaries needed to distinguish a
  low-frequency field-potential result from high-frequency spike-rich content.

The population bridge and the residual predictor use the data in a deliberate
order:

1. Learn each day's population basis and the between-day coordinate mapping
   from permitted **unresidualized** training and calibration spike responses.
   Direction-by-time averages can identify the common coordinates only at this
   stage, while those averages are still present.
2. Freeze that basis, mapping, and the new-day reconstruction loading.
3. Estimate and remove direction-and-time means from both LFP inputs and
   spike-count targets using the same permitted trials.
4. Project and predict spike residuals with the frozen bridge. Do not give the
   residual predictor the mean template, uncentered activity, direction, or
   time as a shortcut.

Subtracting a mean from an already completed prediction and target would not
test a mechanism. Likewise, learning an alignment after residualization from
direction-by-time averages is undefined because those proposed anchors have
been set to zero.

For bridge rank `r`, define `A_d` as the `K` common direction-by-time anchor
rows (`K >= 64`) after unresidualized means have been projected into day `d`'s
permitted population basis and centered across rows. If `A_d = Q_d R_d` is its thin QR
factorization, the registered cross-day support matrix is
`Q_source' Q_target`; its smallest-to-largest singular-value ratio must be at
least 0.05 for every source-to-new-day link.

The decisive trial-identity check keeps all fitted objects fixed and permutes
whole predicted residual trajectories only among held-out trials in the same
new day, region, and reach direction. Apply each derangement identically to
the history-assisted and today-only predictions. The endpoint is the history-
minus-today-only gain under correct pairing minus the mean history-minus-
today-only gain under mismatching. Correct and mismatched terms therefore
share the same estimated means, population map, trial counts, task condition,
and trial relabeling. Absolute history correct-versus-mismatch performance is
reported only as a diagnostic.

### Are the needed data available?

| Requirement | Current status | Consequence if absent |
| --- | --- | --- |
| Simultaneous trial-level LFP and spike responses with a common trial identity | The release was created for LFP--population analysis, but a common field-level trial key has not yet been checked locally | Without verified correspondence, do not claim trial-specific coupling |
| Direction and native time-bin labels on earlier-day training and new-day calibration trials | Dryad documents an eight-target task; complete cells and event indices still must be checked in every named file | Without complete common cells, the planned behavioral alignment is not identified |
| Unresidualized calibration spike vectors for the new day's retained neurons | Dryad documents extracellular recordings and sorted spikes; the required trial-wise vectors have not yet been checked locally | Without them, the target population basis and reconstruction loading cannot be learned |
| At least six held-out trials per direction for mismatching | The primary minimum supplies six | If attrition breaks this minimum, mark that day-region explanation unavailable before outcomes are scored |

These are readiness checks, not optional follow-up conveniences. Record their
result before any neural score is examined.

The current development data can be used to choose and refine this explanation,
but cannot independently confirm it. Confirmation requires a genuinely new
recording day whose non-calibration spike outcomes remain unseen until the
explanation and prediction rule have been chosen.

## Current readiness and storage

The public source has been identified, but the following scientific checks are
still required before scoring:

- confirm the twelve named days and their M1/PMd pairing;
- confirm eight directions and at least 15 complete trials per direction;
- confirm both native-bin time windows and event-index conversion;
- verify the meanings of LMP and the released band-power features;
- build the per-day neuron sets without using outcome values;
- confirm that unresidualized training/calibration spike responses meet the
  fixed 64-cell, rank, condition-number, and singular-value bridge gates;
- confirm trial-level LFP--spike correspondence and at least six held-out
  trials per direction after all eligibility rules;
- save the earlier-day, calibration, development, and held-out assignments;
- document previous access to this public corpus; and
- run one small end-to-end test showing that development and held-out spike
  targets cannot enter candidate fitting.

Large source files remain outside Git. Temporary computation belongs in
`$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/`; only compact data
descriptions, trial-role tables, model results, and reports belong in this
episode directory.
