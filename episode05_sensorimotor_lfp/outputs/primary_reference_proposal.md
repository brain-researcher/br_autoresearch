# First primary comparison: cyclic behavior template plus mean LMP

Prepared2026-10-03T22:02:46Z on the coordinator's explicit request. **Adopted
prospectively by the coordinator**, recorded22:12:25Z. No benchmark numerical result was read to choose these
settings. Root accepted the complete scoped publication benchmark at22:20UTC
and selected the first primary pilot; the decision is recorded in the execution
log. The complete fixed sixteen-session comparison finished at22:26UTC,
failed every first-reference margin screen and is retained as one charged
scientific trial; see the result and interpretation linked in `README.md`.
The settings below remain the executed recipe. This document does not change GOAL,
DATASETS or SEARCH_POLICY.

Independent equations/nesting review by agentA found no concrete conflict in
the normalized ridge objective, shared-intercept comparator, whole-session
outer exclusion, pooled SSE, exact wrong-pair mean or animal grouping. That
review accessed no source/benchmark numerical results and ran no tests; it is
not benchmark acceptance. The coordinator subsequently adopted the recipe
and the proposed0.01/0.005 aggregate margins, both-animal positivity and strict
primary-positive session majorities. Inclusive aggregate boundaries are `>=`;
positivity remains strict. These are adopted operational choices, not a
benchmark acceptance or permission to promote a candidate.

This is one simple reference recipe in the existing open grammar, not the
complete minimum model set, search or promotion decision. Its baseline is a
necessary comparator, not a second scientific trial. Later low/high-frequency,
reduced-rank, nonlinear, influence and falsification work remains required.
No future-session source exists; the entire released corpus remains exposed
exploratory development. Leave-one-session-out estimates do not make it fresh.

## Existing decisions retained

Use the original Dryad v5 source, the sixteen M1 identities and original
trial IDs established by `runs/source_support/attempt002`, and the unchanged
animal/implant grouping. The source's nominal direction assignment is the
nearest octant within0.01rad, preserving the original source angle separately.
No EP07/EP08 outputs or learned states are inputs. Native LFPs are already
processed offline; this recipe neither filters raw voltage nor claims online
causality. Integer-spike validity is not a kinematic inclusion gate.

Reuse `tools/method_components.py`'s equal-training-direction cyclic template,
training-only input standardization, residual ridge and joint X/Y SSE score.
Those equations passed their existing focused checks; their earlier synthetic
parameter settings were never a primary scientific choice. The publication's
all-trial spike PCA, three-bin lag, cubic calibration and correlation-squared
score do not carry over to this primary recipe.

## Source support, window and first feature

Use the native movement-onset offsets **0 through+14 inclusive**: fifteen
30ms samples, timestamps0 through420ms. This is an explicit execution window
wholly contained in the already qualified paper/figure window(-4:+14). It
includes onset and late execution without introducing a time-window search or
reading an additional source interval. It is not a claim to cover an entire
variable-duration reach; “complete trajectory” here means every sample in this
fixed fifteen-bin analysis window, without favorable-bin selection.

Retain all previously qualified whole trials, subject to the fixed minimum
of **two complete trials in every one of eight directions**. This is the
structural minimum for a wrong-trial comparison, not a precision guarantee.
Confirm its counts from the existing trial metadata before the first fit. A
failure stops this fixed sixteen-session first comparison for review; it does
not silently drop a session or pool directions. Finite LFP and velocity support
in this subwindow follows from the completed wider-window check, but the
reader still fails on unexpected nonfinite data without rowwise deletion.

The first LFP feature is the arithmetic mean of stored M1 LMP, guide band
`(0,0)`, over all physical LMP electrodes with stable unique guide identity
across the selected trials in that session. Freeze this metadata-derived
intersection before fitting. Include no spike-based unit-quality or numerical
channel selection; spike values are unnecessary. Report channel IDs/counts and
any structural intersection losses. An empty or ambiguous LMP set stops the
comparison. Do not require equal channel counts across sessions or borrow a
favorable electrode subset. The later matched-channel sensitivity remains open.

One current-bin scalar LMP mean predicts two velocity coordinates; no lag,
history expansion, PCA, temporal averaging, coefficient transfer between
sessions, or polynomial output transform is introduced. Its simplicity tests
whether even a fixed low-frequency summary contains useful residual signal;
failure would not reject the larger permitted LFP grammar.

## Cyclic direction-by-time behavior baseline

Use canonical target angle `theta_d = d*pi/4` from the frozen direction label.
Keep the measured angles as source metadata, but do not use their small within-
octant serialization differences as trial-specific covariates. This choice
makes the behavior-only prediction identical for all trials to the same target.

At each native time index t and coordinate k, fit

`B0_k(theta,t) = a_0kt + a_1kt*cos(theta) + b_1kt*sin(theta)`.

The first harmonic is fixed, not selected from held-direction scores. It
represents smooth circular directional modulation with three coefficients per
time/coordinate and is identified by seven directions. Coefficients vary
freely over the fifteen fixed time bins; there is no temporal basis fitting,
smoothing or time-bin CV. Weighted least squares assigns every training trial
weight `1/n_d`, so each of the seven training directions has equal total
weight. Predictions at the missing direction evaluate the same periodic
function; they never substitute a held-direction empirical mean or nearest
neighbor trajectory. Require full column rank; no regularized rank repair or
adaptive harmonic reduction is allowed.

For each held direction d in an evaluated session, fit B0 using only the other
seven directions. The baseline and candidate use identical complete evaluation
trials. The baseline's rank, means and coefficients are recomputed for every
direction split without the held direction. A time-only intercept baseline can
be retained descriptively from the same permitted training observations; the
later required behavior/elapsed-time falsification comparison is not declared
complete by this initial anchor.

## Residual LMP ridge and nested tuning

For an allowed fitting trial i, time t and coordinate k, first form
`r0_itk = velocity_itk - B0_k(theta_i,t)`. Let `c_k` be its mean over permitted
fitting trials and bins, as already returned by the component's residual
intercept. Use **B=B0+c as the comparator**, and the centered residual target
`r_itk=velocity_itk-B_k(theta_i,t)` for the LFP term. The same training-only
intercept is present in both predictions. Otherwise the candidate could gain
from a pooled residual-mean correction alone when direction trial counts are
unequal. This is a common intercept calibration of the cyclic function, not a
held-direction correction; weighted circular shape estimation remains as
specified above. A zero-valued standardized LFP feature predicts exactly B.
The existing component's full candidate prediction is unchanged, but a runner
must add `model.residual_mean` to `model.baseline.predict` for its fair
comparator; the old component helper's bare baseline is not sufficient here.

Average LMP electrodes first, then
estimate that scalar feature's mean and population SD over the allowed
fitting trials and their fifteen bins only. An exactly zero SD maps to1 and
is reported; it does not justify dropping trials. Fit X/Y ridge slopes
against the centered residual target above. No target scaling
changes the native equal-coordinate squared-error loss.

Use five dimensionless penalties
`lambda in [0.001,0.01,0.1,1,10]`, with objective

`(1/N_fit) * sum_it ||r_it - z_it*C||_2^2 + lambda*||C||_F^2`.

Thus the existing component receives `ridge_alpha=N_fit*lambda`, where
`N_fit=15*(number of fitting trials)`; the common intercept is unpenalized. This
normalization makes a global lambda comparable across differently sized
sessions. The logarithmic grid spans weak through strong shrinkage without a
zero-penalty branch or adaptive extension. Exact tuning ties choose the larger
lambda. All centering, the baseline and coefficients refit on the seven legal
directions for every scored direction and lambda.

The outer table below has sixteen singleton session folds. In outer fold s,
only the other fifteen sessions select lambda. For each possible inner
validation session v among those fifteen, evaluate each fixed lambda using
eight leave-one-direction-out predictions in v. These are session-local
calibration fits: coefficients use v's seven allowed directions, while its
eighth direction is evaluation-only. No numerical coefficients are transferred
from the other fourteen sessions. The fixed univariate recipe has no additional
cross-session learned parameters requiring an inner training fit. The outer
fifteen inner-validation session scores choose one lambda by equal-session,
then equal-animal mean delta_R2; no outcome from outer session s enters that
choice.

Evaluate that selected lambda in s by fitting eight models on s's respective
seven directions and predicting the eighth. Access to those calibration
directions is explicitly part of the contract; holding out a whole session
holds it out of **policy/hyperparameter selection**, not all session-local
coefficient calibration. No held-direction response fits or tunes its model.
The same global recipe applies to all sessions; lambda differences arise only
from the prescribed outer-training selector, not per-session winner picking.

For this fixed family, one reusable score bank of16sessions×5lambdas×8direction
fits is algebraically sufficient; the sixteen outer selectors index only their
fifteen allowed session rows. Reuse does not create additional observations or
permit the held session to affect lambda. Preserve the selected lambda and
the exact allowed inner-session IDs per outer fold. The fixed baseline needs
only16×8 fits and can be reused across lambdas. No additional primary comparison
is charged for these nested hyperparameter fits.

This is a nested comparison for the **specified LMP family**, not an unbiased
estimate of a future winner adaptively chosen after viewing all development
results. A later cross-session learned selector must train on the fourteen
inner-training sessions, and final candidate-family selection must preserve the
same outer exclusion. No final global policy is locked from this first result.

## Exact outer fold identities

Every row's outer-training set is **all other fifteen rows**; its inner
validation folds are the fifteen singleton members of that set. Day groups
never split across folds. Chewie-L and Chewie-R remain one animal.

| Fold | Outer evaluation session | Animal | Implant |
| --- | --- | --- | --- |
|00|Mihili_20140217|Mihili|Mihili|
|01|Mihili_20140218|Mihili|Mihili|
|02|Mihili_20140303|Mihili|Mihili|
|03|Mihili_20140304|Mihili|Mihili|
|04|Mihili_20140306|Mihili|Mihili|
|05|Mihili_20140307|Mihili|Mihili|
|06|Chewie_20160929|Chewie|Chewie-L|
|07|Chewie_20161005|Chewie|Chewie-L|
|08|Chewie_20161006|Chewie|Chewie-L|
|09|Chewie_20161007|Chewie|Chewie-L|
|10|Chewie_20161014|Chewie|Chewie-L|
|11|Chewie_20161021|Chewie|Chewie-L|
|12|Chewie_20150313|Chewie|Chewie-R|
|13|Chewie_20150319|Chewie|Chewie-R|
|14|Chewie_20150629|Chewie|Chewie-R|
|15|Chewie_20150630|Chewie|Chewie-R|

## Score, pairing and animal collapse

Concatenate the out-of-direction predictions so every selected trial appears
once, retaining its fifteen bins and both coordinates. For session s, let
`mu_sk` be the observed coordinate mean across its complete evaluation-trial
pool and all fifteen bins. This is a **scoring-only** reference; it enters no
fit, prediction or tuning of that held session. Use the existing component's
joint denominator

`D_s = sum_i,t,k (velocity_itk - mu_sk)^2`.

`R2_s(model)=1-SSE_s(model)/D_s`, and
`delta_s=(SSE_s(baseline)-SSE_s(candidate))/D_s`.

Use the same positive finite D_s and identical rows for candidate/comparator.
Keep negative R2 and increments. A zero/nonfinite denominator or incomplete
prediction panel is not evaluable and blocks this full comparison; no session
is removed. This pooled session score weights each complete trial and native
bin equally, so directions with more trials contribute more. It does **not**
average independently centered per-direction R2 values; such averaging would
define a different estimand. Direction/time summaries are descriptive only.

For the trial-specific readout, hold all full predicted trajectories fixed and
pair them with another observed trial **within the same evaluation direction**.
Use the exact mean over the finite uniform derangement distribution as the
null center. If a direction has m trials, each ordered mismatched pair has
probability1/(m-1); therefore

`SSE_wrong_d = sum_i [sum_{j != i} ||velocity_i - prediction_j||_F^2]/(m-1)`.

Sum the eight direction contributions and keep D_s unchanged. Compute the
off-diagonal sum algebraically from vector sums/norms, without quadratic pair
storage or random resampling. This equals averaging all products of the eight
direction-wise derangement distributions; report each m and subfactorial !m
as its conceptual finite support. There is no arbitrary Monte Carlo draw
count, random seed, permutation-tail p-value or biological precision claim.
All observed/native-trial trajectories remain grouped. The identity increment
is `(SSE_wrong-SSE_correct)/D_s`, equivalently observed delta minus the exact
wrong-pair mean delta. This is an evaluation-pairing control, not permuted-target
training; its no-fixed-point design is deliberate.

Compute one selected delta and identity increment per outer evaluation session.
Within each animal take the arithmetic mean of its sessions; average the two
animal means with weights1/2. Mihili has six sessions; Chewie has ten, with both
implants nested in Chewie. Chewie implant profiles are descriptive, not a third
replication unit. The same available-session-then-animal weighting selects
lambda within each outer fold. Report every session, both animal means, all
eight direction predictions and exact leave-one-session means; no image/bin/
electrode/trial-level p-value is an animal-level test.

## Adopted choices and remaining promotion boundaries

The coordinator adopted the operational choices above under the existing
pre-score authority: window0:+14, first-harmonic canonical-angle baseline,
equal-direction baseline shape fit with a shared pooled residual intercept,
all-common-LMP mean, five normalized ridge
penalties, singleton outer folds, pooled joint SSE, equal-session/animal
aggregation and exact derangement-mean identity readout. These are consequential
choices recorded before this recipe's first score, not new user-permission
requirements.

The coordinator selected **0.01 R2** as the minimum animal-balanced primary
increment and **0.005 R2** as the trial-identity increment: respectively one and
one-half percentage point of this defined full-trajectory variation scale.
These are practical operating margins, not thresholds justified by statistical
power, benchmark agreement or prior EP05/other-episode outcomes. Require
strictly positive primary and identity summaries in both animals and
a strict majority of positive primary increments within each animal (at least4/6 Mihili
and6/10 Chewie), with equality to a positivity boundary unresolved/nonpassing.
The aggregate margins use inclusive `>=`.
The mandatory influence and other falsifier rules remain to be fixed before
they can promote any candidate. This initial recipe cannot claim promotion.

Root's explicit acceptance of the complete publication benchmark satisfies
the preceding execution gate. Operational recipe and practical-margin adoption
precede the first primary-search score, as required by GOAL. Other feature families and the
later population-explanation plan need their own specifications when reached;
they are not reasons to invent a new gate for this reference comparison.

## Implementation and compute handoff after adoption

`tools/primary_reference.py` and `tools/run_primary_reference.py` implement
a selected-LMP/velocity-only source view using the existing fixed
trial table; deterministic session split records; normalized-lambda score-bank
and outer selector; the exact wrong-pair mean; and compact prediction/score
outputs, reusing the passed baseline/ridge/SSE components. No primary source
execution was authorized before benchmark acceptance. The three focused checks
in `tools/test_primary_reference.py` establish that changing an outer session's tuning-score row cannot
alter its selected lambda; that a constant LMP adds zero prediction beyond the
same calibrated behavior baseline; and that the algebraic wrong-pair mean
equals a small exact trajectory-pair construction. No generic requalification chain.

The full first recipe contains640 univariate ridge fits plus128 reusable
behavior-template fits, not thousands of high-dimensional models. Process one
session at a time and retain only the LMP mean/velocity arrays and compact
score/prediction banks. Reading native source LMP channels, rather than these
small linear solves, should dominate elapsed time. Initial planning is1CPU,
4GiB, numeric threads1; measure one predetermined large session
Chewie_20150313 using the unchanged full recipe in a bounded5-minute step,
then size the remaining full sixteen-session run from that evidence. The
representative is part of the same predeclared comparison and cannot produce a
winner or stop unfavorable science; preserve/reuse its completed score bank.
No source files are copied or modified. Root owns tests and any Slurm command.

Charge **one** valid scientific trial only after the complete sixteen-session
LODO comparison and identity readout succeed. The baseline, five lambdas,128
direction folds, outer selectors, reference timing, fixtures and technical
retry do not multiply the scientific trial count. An incomplete attempt is not
a valid comparison. The first actual primary evaluation starts the existing
120-hour search clock; the aggregate2000core-hour ceiling and28–64/12patience
contract remain unchanged.
