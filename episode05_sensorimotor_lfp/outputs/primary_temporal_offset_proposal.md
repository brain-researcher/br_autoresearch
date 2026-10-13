# Exact finite within-trial temporal-offset control of stationary LMP

Prepared 2026-10-03 while the joint LMP/high panel remains pending. No joint
pilot/grouping scientific scores, new source values or new offset results
were opened. **Root adopted this entire specification at 2026-10-04 00:00 UTC
before any offset computation. New core/runner/seven focused fixtures are
prepared. Root static review passed, all seven tests passed in 0.040 seconds
in46440428, and ASTRA schema/semantics passed with five existing optional-format
warnings. The complete report finished 2026-10-04 00:10:37 UTC in 0.714216s;
the directional timing screen passes, while the aligned anchor still fails
its practical margins. No promotion.**
Recommend one informative control comparison, without an extra candidate.

## Question and fixed anchor

Does the stationary full-channel LMP decoder's trial-specific velocity
information depend on alignment within the fifteen-bin movement window, or
does similar information survive an arbitrary nonzero cyclic phase offset
of its neural contribution within the same trial?

Use the preserved `lmp_full_channel_ridge_v1` anchor from
`runs/adaptive_search/primary_multichannel_attempt001/<session>/`, with its
complete sixteen-session independently selected panel in
`runs/adaptive_search/primary_multichannel_summary_attempt001/primary_multichannel.json`.
Use all original trials, physical electrodes, eight held-direction folds,
native XY at **0:+14 inclusive**, fifteen 30 ms bins and the original
session/animal aggregation. Chewie implants remain one animal. This control
does not depend on which later candidate scores best.

The existing saved predictions are held-direction predictions: each trial's
neural coefficient map, physical-channel means/SDs, direction/time template
and shared residual intercept were fit using the other seven directions.
All those fitted quantities remain fixed. There are no new fitted parameters,
source windows, channel/trial selections, source values, or candidate fits.

## O: exact mean over fourteen nonzero cyclic offsets

Recipe ID: `lmp_fixed_residual_circular_offsets14_v1`.

For session s and original trial i, let `B[i,t,:]` be its preserved baseline
prediction, **including the shared fitting residual intercept**, and let
`P[i,t,:]` be the preserved full stationary LMP prediction at the anchor's
selected lambda. Define the neural correction only as `R = P - B`.

For every integer offset `k` in the fixed set `{1,...,14}`, define

`q_k(t) = (t + k) mod 15`

`P_offset[k,i,t,:] = B[i,t,:] + R[i,q_k(t),:]`.

The same map moves X and Y corrections together in every original trial.
The recipient trial ID, direction, actual XY target, baseline and intercept
never move. No donor trial is introduced. No time bin is discarded, padded,
interpolated, replicated or moved into another fold. Every original correction
bin appears exactly once in each offset trajectory. Offset zero is the
preserved anchor reference and is not included in the control mean.

Because the anchor's standardized linear coefficient map is constant across
time within a held-direction fold, this is exactly its prediction under the
same within-trial permutation of evaluation LMP feature vectors, with the
baseline evaluated at the original target time. No source reread or refit is
needed to construct that frozen-model feature-offset counterfactual. Moving
the complete prediction, including its direction/time baseline, would test a
different question and is forbidden here.

Compute native correct SSE and the existing exact within-direction
wrong-trial SSE separately for each `k`, using unchanged actual XY and the
complete offset trajectories. Then average the fourteen correct SSEs and the
fourteen wrong SSEs equally. Use the original denominator `D_s` and baseline
SSE to calculate

`delta_O = (SSE_baseline - mean_k SSE_correct[k]) / D_s`

`identity_O = (mean_k SSE_wrong[k] - mean_k SSE_correct[k]) / D_s`.

Do not average predictions into an ensemble before scoring: averaging all
cyclic predictions would suppress temporal variation and answer a different
question. These fourteen offsets are the entire declared finite set, not
Monte Carlo replicates or biological observations. Retain each offset's
correct/wrong scores descriptively, with no favorable-offset selection,
additional lag comparisons, p-value or confidence interval from the offsets.

## Lambda, reference replay and aggregation

For each session, use exactly the independently selected lambda/index already
recorded in the stationary LMP panel. That selection used only the other
fifteen **unaltered anchor** score banks, equal-session/equal-animal means and
larger-lambda tie breaking. No offset outcome enters that selection. Do not
rerun a selector on offset scores, on the evaluated session, on later joint
or history scores, or separately by animal. This is evaluation sensitivity of
a fixed selected decoder, not a new tuned lagged model.

Before evaluating the offsets, replay that selected unshifted prediction's
correct/wrong SSE, baseline SSE and denominator against the preserved bank;
verify the saved original trial-ID/direction support and selected score-row
identity. A mismatch stops the attempt without changing the anchor, lambda,
window or original records. Keep the reference paths and selection metadata
in the new report. Offset zero must reproduce the stored full prediction.

Collapse all bins, coordinates, trials, held directions and offsets to one
control score per original session. Aggregate sessions equally within animal,
then give Mihili and Chewie equal weight. Report O and the fixed anchor, their
paired primary/identity contrasts, both animals, primary-positive session
counts and the same fixed-prediction session-omission influence diagnostic.
The omitted session is removed only from that descriptive aggregation; do
not retune lambdas in the influence calculation.

## Proposed directional timing screen and limits

Let A denote the preserved aligned anchor. The directional timing screen
passes only if both animal means of both `A-minus-O` primary and identity
contrasts are strictly positive, and the primary contrast is positive in at
least 4/6 Mihili and 6/10 Chewie sessions. Ties fail. This is not a significance
test. There is no requirement that O be zero or negative: trial-constant or
slow within-trial information can survive this control.

A passing screen supports dependence on within-trial alignment for this
particular fixed decoder. It does not rescue A's failed **0.01 primary /
0.005 identity** margins, satisfy the other required controls, or establish
an optimal physiological lag. A failing screen leaves timing specificity
unsupported under this diagnostic; it does not establish absence of neural
information or identify the source of the retained prediction.

The circular construction has a wrap boundary. A single cyclic offset is not
a uniform signed physical lead/lag across every target bin, and reaches need
not be stationary across the window. Timing differences can therefore reflect
movement-phase mismatch as well as disruption of trial-specific fluctuations;
reporting both primary and identity contrasts limits, but does not remove,
that ambiguity. Retaining every source bin exactly avoids crop/padding and
capacity changes, not these interpretation limits.

A native -120 ms shift inside the qualified wider source window would be a
plausible predictive lead, especially if refit, and is not substituted for
this control. The current diagnostic neither tests a causal neural lead nor
supports online use. All stored features are offline processed, and all
sessions are exposed exploratory development. Any eventual finalist with a
different temporal/nonlinear recipe needs its own applicable frozen-model
timing assessment rather than inheriting this anchor's pass.

## Prospective implementation, checks and count

Implement a new finite report runner after adoption, reading only the
completed anchor score banks and saved prediction arrays. Preserve all old
science and source files. No source provider/payload access is necessary.
Require all sixteen sessions and all fourteen offsets before scientific
completion; partial reports remain preserved and are not scientific results.
Root selected **1 CPU, 4 GiB, 150-second internal / 180-second external**
bounds for this report-only comparison inside its verified allocation
46440428, after the full joint panel. Workers do not run tests or read
prediction arrays. No new worker allocation or submission is needed.

Focused fixtures must verify: exact offset indexing and complete X/Y vector
movement without changing trials; zero-offset identity; a varying baseline
with zero neural correction is unchanged for every offset; a constant neural
correction is unchanged even with a time-varying baseline; changing evaluation
scores cannot change the frozen anchor lambda; exact selected-anchor replay;
mean correct/wrong SSE rather than ensemble predictions; and unchanged
target/denominator with the original session/animal aggregation. An explicit
linear-feature fixture should verify equivalence to permuting held-out LMP
vectors under a fixed stationary affine neural map while keeping B at target
time.

This is **one new temporal alternative-explanation control**, not fourteen
offset trials, a new candidate or another charge for the existing wrong-trial
readout. If the pending joint/control pair completes, it advances twelve to
thirteen valid comparisons, with four controls among seven post-initial
comparisons. Its scientific distinction from prior trial shuffles is that
trial identity and the fixed decoder are retained while within-trial temporal
alignment is changed. All 28–64 depth, patience, coverage and promotion rules
remain in force; no minimum is met by splitting this finite control into parts.
