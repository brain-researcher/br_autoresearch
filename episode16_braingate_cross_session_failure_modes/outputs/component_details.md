# EP16 components — what remains usable when an old mapping fails?

Design refinement, 2026-09-30. This document makes the proposed comparisons
concrete; it does not report an experiment or select a winning explanation.
The current `GOAL.md`, `DATASETS.md`, and `SEARCH_POLICY.yaml` retain their
participant roles, estimands, trial packets, numerical anchors, and budgets.
Choices explicitly left open below are not silently activated amendments.

The scientific object is one earlier-session mapping and the signal still
recoverable in a later session. Each component asks a different question:

| Component | Comparison that matters | What it distinguishes |
| --- | --- | --- |
| Session-local benchmark | Frozen source versus target-local models on the same target trials, with matched-label learning curves | Transfer mismatch versus finite-data local prediction difficulty |
| Recording-state emulator | Observed changes versus changes produced by a label-blind transformation of source recordings | Whether specified recording changes can reproduce the failure pattern |
| Recalibration | Adapted source versus both frozen source and target-only models at the same target-data budget | Recoverable mismatch, and whether source history adds value |

These explanations can coexist. The experiment is not required to nominate a
single cause, and the search need not begin with a preferred remapping model.

## 1. Common trials, coordinates, and information exposure

For chronological pair `s -> t` and outer fold `k`, denote the 64-trial
source fitting side by `S_k`, target fitting side by `T_k`, and opposite
evaluation sides by `S_not_k` and `T_not_k`. Whole blocks stay together. Reverse
the fitting/evaluation sides for fold two. Every observed target comparison
uses the same `T_not_k` trials. Average each score equally over the two folds
before differences, normalization, or recovery fractions.

Keep the registered first-16/32/64 nested target packets. A method does not
get to replace them with easier directions or a packet selected from its
decoding performance. Direction coverage and missing-octant handling still
need an explicit rule before scoring; do not silently change the packet rule.

| Quantity | Source fitting labels | Target fitting activity | Target fitting labels |
| --- | --- | --- | --- |
| Frozen source `F_st` | 64 | None for fitting | None |
| Target-local primary `L_t(64)` | None | 64 trials | 64 |
| Target-local curve `L_t(n)` | None | Same nested `n` trials | `n` |
| Unlabelled calibration `U_st(n)` | 64 | Same nested `n` trials | None |
| Supervised adaptation `A_st(n)` | 64 | Same nested `n` trials | `n` |
| Target null `B_t` | None | No direction-dependent neural fit | Fitting labels only if estimating an intercept |

Evaluation labels enter scoring, not fitting or candidate selection. The
unchanged-source prediction consumes each evaluation trial's neural features,
but does not estimate target normalization statistics from them. A method
receiving more unlabelled activity must be reported as a different exposure
setting. Whole-session normalization remains the existing descriptive upper
bound, not an equal-budget calibration method.

Use the structural physical-array/electrode axis. For source-fitted predictor
coordinates, write `z_sj = (x_sj - mu_sj) / sigma_sj`, with both statistics
estimated only from `S_k` using the candidate's declared stable scaling rule.
Transport applies those same source statistics to the target. A missing target
channel has `z_tj = 0`, the source training mean, with its missingness retained
for diagnostics. No target centering is hidden inside the frozen comparator.
Target-local ridge can estimate its own statistics on `T_k`.

The survivor-electrode control refits models on the common physical-electrode
subset; deleting coefficients from a full-channel model is not equivalent.
Adding mask indicators as predictors would change the primary model and is not
part of this clarification.

The common score has the form `Q = -sum_i w_i loss(y_i, prediction_i)`, with
each direction octant receiving equal total evaluation weight. Fitting weights
come from fitting labels. The implementation must settle whether the
two-coordinate squared-error convention is a sum or a coordinate mean before
setting any raw-score margin; normalized ratios alone cannot settle those units.

### Cross-fitting is not automatically chronological calibration

Reversed whole-block folds are an offline cross-fitted comparison. In the fold
where a later block fits an earlier evaluation block, fitting-side-only activity
is not literally past-only. Keep those descriptions separate. The existing
past-only normalization requirement needs an explicit time-exposure rule: only
activity genuinely preceding a prediction can support that label, and the
allowed history must be declared and matched across methods. Neither the
reversed primary estimand nor the whole-session descriptive bound is replaced
here. A forward-only sensitivity would require an explicit scientific amendment,
not a relabeling of the primary folds.

## 2. Session-local benchmark: how strong is the reference?

The primary reference stays the capacity-matched **64-label target-local
ridge**, not the most flexible model tried on the target. Define its fold-
averaged above-null performance as `J_t = L_t(64) - B_t`.

Place two learning curves on the same evaluation trials:

- `L_t(16), L_t(32), L_t(64)`: what target-only fitting can recover;
- `A_st(16), A_st(32), A_st(64)`: what source-informed recalibration recovers.

`A_st(n) - L_t(n)` tests the value of source history at the same target-label
budget. `A_st(32)` versus `L_t(64)` tests recovery relative to the registered
local reference. Neither comparison substitutes for the other. Also retain
source-local learning curves: a weak reference in both sessions can reflect
limited data or model fit rather than a new transfer problem.

Keep the target-local reference fixed when evaluating an adaptation method;
do not redefine the denominator to favor that method. The required latent
and development-selected ceiling models remain sensitivities. They can reveal
that the ridge reference was restrictive, but do not estimate all information
present in motor cortex.

Capacity matching should specify predictor dimension, fitting exposure, and
the development-chosen regularization procedure. Identical ridge penalties
need not imply identical effective flexibility under changed covariance or
missingness; report effective degrees of freedom where well-defined. Global
tuning and selection use development participants. Any permitted pair-specific
fit uses its designated fitting packet only, never evaluation scores or the
64-label local fit to tune a 32-label method.

The local-signal route still uses the registered chronological
`(J_t - J_s) / J_s` and raw transported-above-null contrasts, their common
component set and weighting, and joint uncertainty. It is adjudicated even
when extra normalized long-gap transport loss is not established.

## 3. Recording-state emulator: fit the recording change, then test its effect

### Inputs and fitting objective

For each fold, estimate `q_s` and `q_t` from source and target fitting sides
only: physical channel availability, finite-value rates, and prespecified
marginal channel-quality summaries. Same-day yield/impedance are optional
supplements; their absence cannot define the primary cohort. No direction
labels, cursor trajectories, decoder weights, prediction residuals, `L`, `F`,
or recovery scores enter transformation fitting.

Optional quality measurements obey the same declared fitting/time exposure.
An untimed whole-day yield summary is an ancillary descriptor unless its
permissible exposure is established; “same-day” does not itself mean past-only.

Choose the mapping from summaries to transformations using recording-summary
fit, stability and complexity, not closeness to the observed decoding deficit.
A transform with label-blind inputs can still be outcome-tuned if one selects
its noise or attenuation settings after looking at `L - F`. That is not the
proposed reproduction test.

Keep both registered dropout and dropout-plus-noise families. A common
illustrative notation on source-standardized features is

`z_emulated[i,j] = m[i,j] * (a[j] * z_source[i,j] + epsilon[i,j])`.

`m` implements missing channels or label-independent invalid-sample patterns;
`0 <= a[j] <= 1`; added noise is zero-mean and generated independently of source
feature realizations and labels, conditional on the fitted recording summaries.
Missing values are zero in source coordinates. Blockwise versus trialwise
missingness and permitted noise parameterizations can remain development
candidates. No
missing source channel is invented, and no information-amplifying transform
is introduced.

### What the available measurements do and do not specify

Channel disappearance and invalid samples are directly observable operations.
An attenuation/noise decomposition from marginal activity is different:

`Var(z_emulated) = a^2 * Var(z_source) + Var(epsilon)`

on an available channel under independent additive noise. One marginal
variance does not identify both unknowns. Lower `sbp` variance is not itself
evidence of lower gain or SNR;
higher variance is not itself evidence of more noise. Activity mixtures can
also change the summaries. A nonzero attenuation is invertible in exact
coordinates: it can break a frozen decoder without necessarily destroying
available information.

Specify a label-blind partial order for the existing “later state no better in
both folds” condition. New usable target electrodes or improvement in a
required quality dimension can make a transition incomparable. A direction
of variance change cannot alone establish degradation. Attenuation/noise
settings that fit the same summaries remain assumption-dependent alternatives;
do not choose the one with the most persuasive decoding result. Report whether
the reproduction conclusion changes across those compatible alternatives.

This branch retains its required six near and six long eligible transitions
per participant. Incomparable, improvement-like and fold-discordant transitions
remain controls, rather than being forced into degradation or used to replace
an unfavourable participant.

### Where the emulated scores come from

After fixing `E_st,k` from `q_s,q_t`, transform **source** fitting and evaluation
features. Fit an emulated-local ridge on transformed `S_k` using the existing
source labels. On transformed `S_not_k`, evaluate both that ridge and the
unchanged clean-source ridge. Keep source trial labels and the source null
fixed. Thus emulated local, transported and null scores use the same source
evaluation trials; no emulated direction labels are invented.

Reuse the fitting-side-fitted transformation unchanged on source evaluation
trials; do not re-estimate parameters from their activity. Report recording-
summary fit as well as decoding reproduction. A transform that cannot represent
the target summaries is not rescued by a coincidentally similar decoder score.

After fold averaging, let `L_e` be emulated-local performance, `F_s,e` the
clean-source decoder on emulated features, and `B_s` the source null. Then

```text
J_s = L_s - B_s
J_e = L_e - B_s
emulated_local_change = (J_e - J_s) / J_s
emulated_transport_loss = (L_e - F_s,e) / J_e
observed_local_change = (J_t - J_s) / J_s
observed_transport_loss = (L_t(64) - F_st) / J_t
```

Apply the registered positive-denominator rules without clipping or epsilon
replacement. Compare observed and emulated quantities on the **same eligible
pair graph**, with the existing target/schedule/component weighting. Recompute
both observed and emulated near/long summaries on that graph rather than
comparing emulation-subset results with a different full-cohort contrast.

Within each system, benchmarks share evaluation trials. Across systems,
observed scores use real target trials and emulated scores use transformed
source trials. This is a recording-state sufficiency test under preserved
source trial structure, not a same-target-trial counterfactual. Differences in
unmeasured behavior or neural state can limit reproduction.

Test classwise absolute reproduction errors for local change and transport
loss, plus the error in the extra-long-gap contrast. Report joint uncertainty
on those errors, not just a correlation. Exact error aggregation, tolerances
and uncertainty still require the existing pre-scoring scientific choices.

### Identity-aware versus matched-random degradation

The random control preserves the entire degradation bag: affected-channel
count, attenuation/noise settings, missingness patterns, and repetition count.
Permute its assignment within physical arrays and prespecified source-quality
strata, without consulting channel tuning or decoder weights. Use paired
randomness where appropriate. The control matches degradation severity while
breaking which physical electrodes are affected.

If both assignments reproduce the changes, generic degradation may suffice.
Better identity-aware reproduction would add spatial specificity. The policy
contains an unset identity/random separation margin; whether that separation
is necessary for any sufficiency claim or only for spatial specificity must be
decided explicitly. The required control is not removed here.

## 4. Recalibration: repair what, using how much information?

Treat the existing families as required coverage, not a claim that one
architecture is the answer. Keep source predictions, source-coordinate
features and latent representations distinguishable:

- Output correction fits a bias/gain/rotation of frozen source predictions.
- Feature or latent remapping states exactly where the transformation occurs.
- Low-rank updates and bounded supervised alignment state what is trainable,
  what remains fixed, and which target packet is visible.
- Unlabelled alignment states its actual fitting objective. Unrelated trials
  cannot be treated as paired examples; direction-based correspondence spends
  labels. Covariance matching can leave orientation ambiguous.

Broader development proposals can be considered within the episode's allowed
model families and shared budget. A new primary model, data role or estimand
would be a separate amendment. All proposals retain equal target exposure,
target-only learning curves and the supervised shuffled-label controls.

After fold averaging, define

```text
J_t = L_t(64) - B_t
D_st = L_t(64) - F_st
G_st(32) = A_st(32) - F_st
recovery_fraction = G_st(32) / D_st
relative_gain = G_st(32) / J_t
```

These give concrete meaning to the existing proposed 0.50 gap-recovery and
0.10 relative-gain anchors. `J_t` must clear its readiness minimum and `D_st`
its separate positive raw-gap minimum. Negative gain means harm; recovery
above one means exceeding the 64-label ridge reference. No absolute values,
clipping or denominator constants are used. An unsupported gap is unresolved,
not discarded to make recovery look favourable.

The final rule still needs to specify which gap classes it applies to and
whether it aggregates pair ratios or takes a ratio of aggregated gains/gaps.
Those estimands differ. Retain the registered weighting and dependence-aware
uncertainty once that choice is made.

Recovering half of `L_t - F_st` does not by itself show that calibration removes
the **extra** long-versus-near deficit. Display recalibrated near/long loss
contrasts alongside learning curves. Unless a separately specified comparison
supports the stronger statement, the interpretation is repairable offline
mismatch, not specifically repaired long-gap deterioration.

## 5. Relation to existing stabilization methods

[NoMAD (Karpowicz et al., 2025)](https://www.nature.com/articles/s41467-025-59652-y)
uses latent dynamics for unsupervised alignment. EP16's current single-window
ridge design is not a reproduction of that dynamic architecture.
[PRI-T (Wilson et al., 2025)](https://www.nature.com/articles/s41551-025-01536-z)
uses task/cursor structure to infer training targets. It is not equivalent to
the activity-only unlabelled information tier specified here. Incorporating
that extra information would require a declared branch and matching exposure.

These sources motivate distinctions between information tiers and objectives;
they do not establish novelty for EP16. The proposed contribution remains a
common operational diagnostic panel, not another claim that alignment or
recalibration can stabilize a decoder.

## 6. Choices left open by this refinement

Before scoring, resolve together: the raw loss convention and octant-support
rule; the literal time exposure for past-only methods; the channel-state
partial order and permissible attenuation/noise assumptions; classwise
reproduction-error aggregation and identity/random separation; and the
population/aggregation for the recovery decision. Existing unset onset,
support, complexity, stability and numerical margins remain unset here.

The six/three participant split, chronological near/long primary comparison,
128-trial packets, two reversed folds, 16/32/64 calibration curve, 32-label
primary, 40–96-panel search, resource limits and one final evaluation remain
unchanged. No neural payload was read and no experiment was run for this
component-design revision.
