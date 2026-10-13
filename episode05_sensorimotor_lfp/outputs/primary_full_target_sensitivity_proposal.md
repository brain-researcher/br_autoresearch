# Does the current-LMP result depend on fitting a residual target?

Prepared 2026-10-04 while the adopted history-plus-repeatability panel runs.
**Adopted by root 2026-10-04 07:27 UTC after root and independent written review.**
New implementation and fixtures are prepared for coordinated validation; this
paragraph makes no execution claim. Actual clock checked 07:28:01 UTC during
implementation handoff.
No incomplete HW or elastic-net score motivates this sensitivity. Root owns
adoption and all execution records. The current-LMP branch below is the
existing GOAL-required “train-only target-residualization versus full-target
fit” sensitivity; it does not change the primary residual-target definition.

## Scope and prior-recipe audit

The permitted prior proposal files and primary fitter definitions were checked
for a direct full-native-XY branch. The initial full-channel LMP implementation
in `primary_multichannel.py` fits Y minus its training-derived direction/time
baseline and shared residual intercept, then adds that baseline at prediction.
The later primary branches retain a residual target. No identical completed
direct-full-target primary recipe is documented in those definitions. The
publication decoder uses different splits and scoring/mapping conventions and
does not establish this grouped sensitivity or contribute scientific depth.
No scientific result file was opened for this audit.

Test only the already qualified **current-bin, all-physical-channel LMP ridge**
family and its preserved residual-fit anchor R, `lmp_full_channel_ridge_v1`.
Do not add the repeatability prior, history, other bands or another capacity
grid. This isolates one interpretation of the original LMP family. It does
not automatically complete the residualization sensitivity for a later
finalist with a different feature representation or predictor.

The direct model must learn typical movement as well as trial departures
through the LMP-to-XY map. The residual model is supplied a separately fitted
direction/time prediction. Their difference therefore concerns that entire
two-stage recipe, not just a harmless reparameterization or proof of leakage.
In particular, worse direct-target accuracy can reflect the burden of learning
the typical trajectory. Neither sign of the comparison alone establishes a
trial-specific biological mechanism.

## F: direct full-native-XY ridge with one fitting-global intercept

Recipe ID: `lmp_full_target_ridge_sensitivity_v1`.

Preserve all sixteen original sessions, complete trials, frozen physical LMP
electrodes, direction identities and the native 0:+14 LMP/XY window, fifteen
30-ms bins. There is no new source interval, lag, target rotation, smoothing,
time basis, spike access, trial selection or channel selection. Chewie implants
remain one animal. X is read from the native released LMP values and then
standardized using the original fitting-only feature normalization below.
Y remains native and unscaled; XY receive equal loss weight within the
original pooled joint SSE denominator.

For each of the eight held directions, fit the original first-harmonic
equal-direction template B0 from the other seven directions and true Y. Let
`c=mean_fit(Y-B0)` over fitting trials and bins, and retain **B=B0+c** as the
unchanged comparator. Save its prediction on the actual evaluation trials.
This baseline computation is separate from the direct neural head.

For the direct head:

1. Use exactly the original full-channel anchor's numerical X preprocessing:
   `flat=X[train].reshape(-1,E)`, per-column fitting population mean/SD,
   zero SD mapped to one, with evaluation transformed by those same values.
   Preserve the reader/reshape path; do not substitute a differently ordered
   multidimensional reduction, extra centering or post-weight transformation.
2. Define `mu=Y[train].reshape(-1,2).mean(axis=0)` once from true fitting XY.
   This is one two-coordinate, trial/bin-weighted global intercept, not a
   direction-specific or time-specific mean. It is unpenalized and shared
   across the paired model and every training-label control.
3. Fit `V=Y[train].reshape(-1,2)-mu` from the standardized fitting matrix Z:

   `beta_lambda=(Z'Z+n*lambda*I)^(-1) Z'V`,

   at exactly `lambda=[0.001,0.01,0.1,1,10]`, with the existing direct positive
   ridge solve and all physical columns retained. No output whitening,
   target-SD weighting, extra intercept, iterative tolerance or rank cutoff.
4. Predict **`mu+Z_eval beta_lambda`** in native XY. Do not add B, B0, c,
   a direction template or a time template to this direct prediction.

Direction labels determine the allowed split and the separately evaluated B;
they are not inputs to the direct neural head. Elapsed time determines the
fixed source bin alignment but does not enter as an added predictor.

Score the complete held-out predictions relative to the unchanged B with
the same pooled session denominator and retained negative R² values. Report
the original exact mean within-direction whole-trial evaluation derangement
score, applying the entire fixed F prediction to wrong held-out trials. This
evaluation-pairing readout is distinct from the training refits below.

Each outer session chooses F's lambda from only the other fifteen complete
F banks, using the original equal-session/equal-animal primary objective.
Exact ties prefer the stronger lambda. No control, anchor or held-session
score chooses lambda; no new tie tolerance is introduced.

## C: eight within-direction whole-trajectory training-target label controls

Recipe ID: `lmp_full_target_training_labels8_sensitivity_v1`.

For each held direction and each of replicates 0..7, independently permute
the fitting target trials **within each of the seven fitting directions**.
Order recipients by their validated session-qualified numeric source trial
IDs. Seed one ordinary Python version-2 shuffle for each fitting direction
with literal

`ep05-lmp-full-target-training-label-v1|20261004|<session>|<held>|<replicate>|<direction>`.

Save recipient and donor native IDs. Keep the same map across every electrode,
all five penalties, all fifteen bins and both XY coordinates. Reassign the
entire 15-by-2 target trajectory as a unit; never permute X, split bins or
coordinates, cross direction boundaries, or map a held-direction trial.
Ordinary maps may contain fixed points or be the identity; accept them
without redrawing. Do not select a favorable seed or invent derangements
after examining any result.

Compute the true-target mu once as above and reuse it exactly. Center true
training XY by that mu, then permute the complete centered trajectories.
Reuse the same standardized X and exact same Gram matrix Z'Z in every arm;
each control needs its own `Z'V_permuted` and ridge coefficients. Its held-out
prediction is `mu+Z_eval beta_control`, using unchanged evaluation X and Y.
Do not shuffle evaluation inputs/targets, refit the comparator, or add B to
the direct control. There is no need to recompute mu in a shuffled order.

These assignments preserve, algebraically, the collection of target
trajectories, native XY scale, global mean, and every fitting-direction-by-
time target mean. They remove the original within-direction neural/behavioral
trial correspondence to the extent permitted by each ordinary finite map.
They **do not remove** the task-locked direction/time relationship or all
neural dependence; identity and fixed-point maps can preserve some pairing.
This is a conditional trajectory-pairing sensitivity, not an unconditional
null, exact p-value, or eight independent biological replications.

At each F-selected session lambda, average the eight correct SSEs and the
eight exact evaluation-wrong-pair SSEs with the common denominator. Do not
average predictions, tune the controls, or treat seed spread as uncertainty
over animals. Retain every seed's full five-lambda bank and selected scores.

## Comparators, fixed reports and interpretation

Reuse the completed residual-LMP R banks and independently selected R panel:

- `outputs/runs/adaptive_search/primary_multichannel_attempt001`;
- `outputs/runs/adaptive_search/primary_multichannel_summary_attempt001/primary_multichannel.json`.

Report F-minus-R under both R's independent outer-selected lambda and the
same original lambda selected by F. Same X preprocessing and Gram make the
matched-lambda comparison use the same linear smoothing operator on different
fitting responses; R additionally supplies B at evaluation. This is not equal
overall model information, a new nested-model theorem, or a requirement that
the direct model beat R.

Report both F/C B-relative primary and identity increments, F-minus-mean-C,
both R contrasts, every session/animal, all eight seeds and every fixed-
prediction session omission. Preserve the original displayed practical
screens (0.01 primary, 0.005 identity) and directional screens (strictly
positive primary and identity in both animals, primary-positive counts
at least 4/6 Mihili and 6/10 Chewie). Ties fail directional screens. A claim
that training trial correspondence helps F requires the same directional
screen on F-minus-C and the original fixed-omission sign reporting.

No new F-minus-R noninferiority margin or minimum acceptable loss is invented.
If direct-target primary or identity support is absent, report the lack of
robustness beyond residual fitting; a positive R result alone cannot be
described as residualization-independent. If the F-minus-C screen fails,
the direct fit has not established the proposed training-pairing dependence.
Such outcomes cannot be tuned away or used as a passed required check.

F and C are **sensitivity recipes, not promotable alternatives** to the
registered residual-target objective. Their scores cannot rescue failed
primary gates, replace a primary leader, or establish future-session evidence.
The R contrasts quantify the effect of the original baseline/residualization
choice, without silently converting a sensitivity into a target search.

## Exact identities, degeneracies and scientific depth

Save fitting mu, original X means/SDs and zero-SD columns, the unchanged B
predictions, target-map IDs, common Gram trace/rank, every arm's cross-product
and its equality flags, and selected-lambda df. All arms have the same X Gram
and nominal ridge capacity; only their target cross-products differ.

Zero feature energy or zero fitted coefficients is a valid constant-prediction
result, not a reason to drop a fold or silently replace F with B. If fitting
targets are constant, keep the unpenalized mu and the exact same scoring.
No near-zero cutoff, nonnegative-R² clipping, alternative window or retry is
authorized by a negative or degenerate result.

If every fitting and evaluated original B vector equals the fitting-global
mu throughout every required fold/session, F and the residual-LMP anchor R
are algebraically the same prediction recipe, and F adds zero scientific
depth. This is an exact sufficient structural criterion; no near-equal score,
fitted-performance match or rounding tolerance declares a duplicate. Rounding
differences between centered numerical paths do not create a new hypothesis.

If every control's fitting cross-product is exactly the paired cross-product
throughout all eight replicates, folds and sessions, shared Gram/mu makes
the complete mean control an exact duplicate of F, and it adds zero depth.
This includes identity maps or within-direction identical target trajectories
when they yield exact equality. Preserve such maps and report the degeneracy;
do not resample to force novelty. If the control also meets the constant-B
duplicate condition through its paired equivalence, it adds zero as well.
Other exact algebraic duplicates must be documented before assigning a count;
finite near-equality in scores is not a duplicate criterion.

Otherwise each distinct complete recipe adds one valid comparison, at most
two for the pair; negative results count. Conservatively add at most one to
the post-initial control tally, for the mean-eight training-label recipe.
Do not award additional control credit merely because F is a component
sensitivity. Individual seeds, lambdas, folds, anchor contrasts, or exact
engineering repairs do not add depth. The final incoming count is determined
only after the running HW panel is closed; this draft assumes no HW result.

## Prospective bounded implementation

Only after root adoption and independent written review, use new modules
without altering the completed residual-LMP core or banks. The existing
current-LMP reader suffices. Replay original native trial/target/physical
support and B exactly after fitting. Fixtures must distinguish B from mu,
test full-target prediction without double-adding B, retain native XY units,
verify original flat X normalization, exclude held values from fitting,
preserve whole XY trajectories and direction/time means under the maps,
check own cross-products with common Gram/mu, stronger-lambda ties, other-
session exclusion, mean SSE and the declared duplicate cases.

Propose the same predetermined complete Chewie_20150313 resource pilot,
all eight controls and five lambdas, one CPU within 4 GiB, 270 seconds
internal and 300 external. Select a finite remaining-panel bound only from
complete pilot resource/support facts while scientific scores stay closed.
No pilot-result model choice or automatic retry. Intended fresh outputs,
not yet created or selected for execution, are
`primary_full_target_sensitivity_control_attempt001`,
`primary_full_target_sensitivity_summary_attempt001`, and
`primary_full_target_sensitivity_panel_attempt001` under
`outputs/runs/adaptive_search`.

Preserve 28–64 scientific depth, 12-trial post-minimum patience, at least 40%
post-initial controls, required model/falsifier coverage, two adaptive rounds
and two challenger comparisons, 2,000 core-hours, zero GPUs and the 120-hour
clock starting 2026-10-03 22:23:02.824 UTC. This is exposed-corpus development
evidence for two animals; it cannot support causal, online, spike-free,
cross-session coefficient-transfer or new-animal claims.
