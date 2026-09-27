# EP07 paper plan: what part of an earlier recording day is still useful today?

Status: proposed study design, 2026-09-26. The public corpus and earlier
campaign work have already exposed neural outcomes. No EP07 prediction result,
mechanism result, or new-day validation is claimed here.

## The intended discovery

The first question is deliberately narrow. A new recording day supplies its
own small calibration set. Do three earlier days still improve prediction on
untouched new-day trials, beyond both a direction-and-time average and an
equally tuned model that uses the same calibration trials but no history?

If the answer is yes, the paper should determine what the earlier days
contributed:

1. a better estimate of the average response for a reach direction and time;
2. a reusable low-dimensional population trajectory after day-specific
   alignment; or
3. a reusable relationship between same-trial LFP fluctuations and population
   spike-count fluctuations after the average reach response is removed.

The third result would provide the deepest explanation because it identifies
what trial-specific information survived the change of recording day. The
first two are still informative, but support narrower conclusions.

The primary test in [GOAL.md](../GOAL.md) remains the first result. Later
mechanism work cannot change that comparison or turn a negative or ambiguous
primary result into a positive one.

## A concrete example

Take two reaches to the same rightward target at 240 ms after movement onset.
Their direction-and-time mean is identical. Suppose the low-frequency LFP is
larger than usual on the first trial and smaller on the second, and population
spike counts change in the same direction.

If earlier days teach that signed trial-to-trial relationship and it persists
on a new day after limited calibration, EP07 has found reusable residual
coupling. If history improves only the common rightward-reach trajectory, EP07
has instead found reusable task structure. Similar-looking mean trajectories
cannot distinguish these explanations; a separately fitted residual-only task
can.

## What earlier work already established

| Earlier work | What it showed | What EP07 must add |
| --- | --- | --- |
| Gallego-Carracedo et al., eLife 2022, [doi:10.7554/eLife.73155](https://doi.org/10.7554/eLife.73155) | In this dataset, LFP relationships with population dynamics depend on region and frequency and can remain stable across behavioral periods. | A chronological earlier-to-new-day test with small new-day calibration, an equally tuned today-only comparison, untouched trials, and an explanation of what transfers. |
| Hall et al., Nature Communications 2014, [doi:10.1038/ncomms6462](https://doi.org/10.1038/ncomms6462) | Low-frequency motor-cortical LFPs can estimate local firing rates and support real-time biofeedback. | Show whether a low-frequency relationship survives across recording days under EP07's day and calibration rules. |
| Ahmadi et al., Scientific Reports 2021, [doi:10.1038/s41598-021-98021-9](https://doi.org/10.1038/s41598-021-98021-9) | LFP features can predict population spiking, with LMP reported as a strong predictor. | Separate genuinely low-frequency information from high-frequency spike-rich content and test whether earlier days add value beyond today-only fitting. |
| Gallego et al., Nature Neuroscience 2020, [doi:10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4) | Motor-cortical population dynamics can remain similar across days after alignment despite unstable recorded units. | Evidence about LFP-to-population prediction, especially trial-specific residual coupling, rather than another demonstration that average trajectories align. |
| Degenhart et al., Nature Biomedical Engineering 2020, [doi:10.1038/s41551-020-0542-9](https://doi.org/10.1038/s41551-020-0542-9) | Aligning low-dimensional spiking activity can stabilize an online intracortical BCI under recording instability. | A target-calibrated, offline LFP-to-spike transfer result; EP07 must not describe this as demonstrated online stabilization. |
| Sussillo et al., Nature Communications 2016, [doi:10.1038/ncomms13749](https://doi.org/10.1038/ncomms13749) | Historical neural variability can improve decoder robustness to future variability and electrode loss. | Show whether earlier LFP/spike days add information beyond an equally tuned today-only model rather than merely supplying more rows. |

These studies make “cross-day transfer is possible” and “LFP predicts spikes”
insufficient novelty claims. EP07 needs a named, testable part of the
LFP--population relationship that survives days, or a useful boundary showing
that only average task structure survives.

## Honest evidence labels

The public source, source paper, and related campaign analyses were already
known when this plan was written. The evidence should therefore be described
plainly:

| Data | Role in this project |
| --- | --- |
| Earlier days, calibration trials, and development trials | Choose the primary model and develop the explanation |
| The current untouched trials | Test the original earlier-day added-value question once |
| The same current data after that test | May describe prespecified secondary patterns, but cannot freshly confirm an explanation chosen afterward |
| Genuinely new recording days | First forward test of the chosen mechanism and the prediction of when history will help |

If a mechanism diagnostic was examined before its rule was chosen, it belongs
to development evidence. The paper should identify the evidence role behind
every figure.

## Study sequence

### 1. Answer whether earlier days help

Use the chronological day assignment, nominal 20% new-day calibration, and the
same model-development budget for the history-assisted and today-only streams.
Report both comparisons for all six new days and both M1/PMd settings.

A history-assisted model must not be called successful merely because it beats
the direction-and-time mean. It must also beat the equally tuned today-only
model. This step answers whether history adds predictive value on untouched
trials from the six fixed new days. It does not yet explain why and does not
test a wholly unseen day.

### 2. Fit a real residual-only prediction task

Subtracting the same direction-and-time mean from a completed prediction and
its target leaves their squared error unchanged. That algebraic recentering is
not a mechanism test. The follow-up instead fits new, capacity-matched models
after residualization.

For every earlier day, estimate direction-and-time means for both LFP inputs
and spike-count targets using only its training trials. For each new day,
estimate the corresponding means from its calibration trials. Remove those
means before fitting either residual model:

```text
x_res[t,q]   = x[t,q] - mean_allowed[x | direction(t), q]
y_res[t,q,j] = y[t,q,j] - mean_allowed[y_j | direction(t), q]
```

Neither residual model may see uncentered earlier-day values, an earlier-day
mean template, development spike targets, or held-out outcomes. The
history-assisted and today-only residual models receive matched features,
capacity, and tuning opportunities. Their predictions are scored directly
against held-out spike residuals; the mean component is reported separately.

### 3. Bridge changing neuron sets without pretending neurons match

Every day keeps its native neuron set. Earlier-day and new-day vectors are
never joined by unit number, waveform, electrode number, or a guessed neural
identity.

Instead:

```text
native residual spikes within each day
    -> population coordinates learned from permitted data
    -> alignment anchored by reach behavior
    -> loading learned from the new-day calibration trials
    -> residual spike-count predictions for the new day's neurons
```

The mapping aligns low-dimensional coordinates, not individual neurons. The
new-day loading and alignment may use calibration trials only. Any construction
that uses held-out spike outcomes is a new-day fit, not evidence of transfer.

### 4. Determine whether the signal is genuinely low frequency

High-frequency power can contain local spike-related energy. A field-potential
interpretation therefore requires the same residual comparison under:

- LMP or low-frequency features only;
- complete removal of 100--400 Hz power;
- a prespecified electrode/unit proximity analysis;
- matching on unit yield, spike-quality summaries, missingness, and available
  channel count; and
- separate low-frequency, high-frequency, and combined models with comparable
  capacity.

If transfer exists only in 100--400 Hz power or on spike-rich electrodes,
report that boundary. A low-frequency result still establishes prediction,
not synaptic causality or stable single-neuron identity.

### 5. Identify which part of the model carries the gain

Keep the explanation small. Before new mechanism-specific scores are examined,
choose at most two candidate explanations and the comparisons that could
disprove them. Candidate components are:

- direction/time template versus behavior-aligned population coordinates;
- LMP/low-frequency versus high-frequency feature families;
- each of the three earlier-day positions; and
- population dimensions ordered using earlier-day information only.

A component supports the explanation only if omitting it reduces residual
history benefit in both animals without causing both models to fail generally.

| Alternative explanation | Decisive comparison | Interpretation if supported |
| --- | --- | --- |
| Better average reach estimate | Score the mean separately; residual models show no history benefit | Earlier days mainly improve a task template |
| Generic regularization or more examples | Equally tuned today-only residual model, no-history ablation, and sample-size simulations | No evidence for a history-specific neural relationship |
| One influential earlier or new day | Omit each earlier day and each new day in turn | Narrow the finding or stop |
| High-frequency spike-rich content | Low-frequency-only, 100--400 Hz removal, proximity, and quality-matched checks | State the signal boundary; do not claim general field-potential coupling |
| Average aligned dynamics only | Mean trajectories transfer but separately fitted residual models do not | Claim shared geometry, not trial-specific coupling |

### 6. Predict when history will help

An explanation becomes useful when it predicts the result before the new-day
outcomes are known. Build a small rule from quantities available after fitting
the earlier days and calibrating the new day, such as:

- agreement between earlier-day and calibration reach trajectories;
- disagreement among the three earlier-day residual predictions;
- reliability of the retained low-frequency features in calibration; and
- how well the earlier-day population coordinates represent calibration
  activity.

The outcome is the later day's history-assisted advantage over the today-only
model, reported for both native counts and the residual-only task. When
developing this rule, the held-out gain from one day cannot be used to fit the
prediction for that same day. Use a simple linear or monotone rule with at most
two mechanism scores; six new days are too few for a flexible predictor or a
broad generalization claim.

### 7. Test the explanation on genuinely new days

Before viewing outcomes from a new recording day, choose the complete source
set, eligibility rules, calibration fraction, feature meanings,
residualization, population alignment, models, comparisons, effect margins,
missing-data behavior, and result table.

The strongest validation would use a third animal with enough chronological
earlier and later sessions. Neural weights would still be fitted within that
animal; what transfers across animals is the procedure and mechanism
prediction, not a common neuron or electrode map. A genuinely unseen later day
in Mihili or Chewie-L would test new-day transfer but not population
generalization. Chewie-R tests implant sensitivity in the same animal.

For every eligible new session, the chosen procedure should predict:

1. population activity on untouched trials;
2. within-direction/time residual activity;
3. whether earlier days will beat the today-only residual model; and
4. whether the result survives the low-frequency/spike-rich controls.

All eligible sessions must be reported, including those predicted not to
benefit. No compatible new source is currently selected, so this stage remains
a requirement rather than a claimed result.

## Result-dependent decisions

| Evidence after the first test | Next step | Strongest defensible claim |
| --- | --- | --- |
| History fails cleanly in both settings | Stop the mechanism story; report the fixed-corpus boundary if useful | Earlier days did not clear the added-value margin under this calibration regime |
| Estimate is positive but varies strongly by day or remains imprecise | Examine the planned day-influence checks and seek more new days | History may help some fixed days, but a stable relationship is unresolved |
| Benefit is confined to the mean direction/time component | Deepen the task-structure explanation and test it on new days | Earlier days improve stable reach structure, not trial-specific coupling |
| Residual benefit survives only high-frequency or spike-rich views | Report the signal boundary and stop the broad field-potential story | Spike-rich high-frequency features predict residual spiking in this corpus |
| Low-frequency residual benefit survives controls but does not predict new-day benefit | Report an internal mechanism result with limited scope | Residual coupling is detectable internally; forward usefulness remains unverified |
| Low-frequency residual benefit and the benefit prediction both succeed on new days | Develop the full cross-day relationship claim | A specified low-frequency LFP--population residual relationship transfers under limited calibration in the declared scope |
| M1 and PMd differ | Name the supported region/epoch setting | No preparation-versus-execution claim because region and epoch are confounded |

The explanation should be narrowed or stopped when the effect disappears
against the today-only residual model, is carried by one day, cannot be
separated from the average response, or lacks a new-day test.

## Figures, contingent on evidence

| Figure | Scientific judgment | Required content |
| --- | --- | --- |
| 1. Do earlier days help? | History adds value after same-day calibration | Chronological design, both comparisons, every new day, both animals, uncertainty, and negative scores |
| 2. What transfers? | Gain belongs to average structure, residual coupling, or both | Mean result, separately fitted residual models, and the day-specific-neurons-to-population-coordinates bridge |
| 3. Is it a field-potential result? | Low-frequency information contributes beyond spike-rich contamination | Low-frequency-only model, 100--400 Hz removal, proximity and quality-matched checks |
| 4. Can benefit be predicted in advance? | Calibration-visible measurements forecast when history helps | Day-wise cross-fitting, the prespecified score, failures as well as successes |
| 5. Does the explanation survive a new day? | The mechanism extends beyond the data that generated it | All new sessions, activity and benefit predictions, and exact animal/implant scope |

The abstract should state which M1/PMd setting was supported, whether the gain
concerned means or separately fitted residuals, which frequency boundary
survived, and what kind of new session was tested. It must also say that the
study is offline, uses provider-preprocessed features, and does not establish
clinical BCI performance, synaptic causality, or stable single-neuron or
electrode identity.
