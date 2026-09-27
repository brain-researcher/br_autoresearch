# EP07 data: what is available and how it answers the question

EP07 uses the public motor-cortex LFP and population-spiking release from
Dryad DOI `10.5061/dryad.xd2547dkt` (metadata version 7, file-bearing version
5). The release has been used in earlier work, so the present study can provide
a careful internal test but not an independent replication.

No neural analysis has started for this episode. The remaining preparation is
to confirm that the named sessions support the planned time windows, reach
directions, trial counts, LFP features, and population-spike targets.

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

This separation is procedural rather than a claim of cryptographic secrecy.
The final report must state that the public release and earlier campaign work
were already outcome-exposed.

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
- source-fitted and target-calibrated population coordinates;
- predictions needed to refit and score a genuine residual-only model;
- omission results for each earlier day and retained model component; and
- electrode proximity and spike-quality summaries needed to distinguish a
  low-frequency field-potential result from high-frequency spike-rich content.

The residual analysis must remove direction-and-time means from both LFP
inputs and spike-count targets before fitting. Subtracting a mean from an
already completed prediction and target would not test a mechanism.

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
- save the earlier-day, calibration, development, and held-out assignments;
- document previous access to this public corpus; and
- run one small end-to-end test showing that development and held-out spike
  targets cannot enter candidate fitting.

Large source files remain outside Git. Temporary computation belongs in
`$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/`; only compact data
descriptions, trial-role tables, model results, and reports belong in this
episode directory.
