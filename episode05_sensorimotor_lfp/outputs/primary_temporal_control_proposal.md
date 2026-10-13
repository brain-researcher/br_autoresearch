# Time-varying LMP map and matched training-pairing control

Prepared 2026-10-03 after the complete six-recipe development panel.
**Adopted prospectively by the coordinator, including the directional pairing
screen, before implementation or fits. Both complete comparisons finished
2026-10-03 23:30 UTC after seven focused fixtures and ASTRA validation passed.**
The temporal candidate failed its practical margins and Mihili primary sign;
the paired-training directional screen passed. See the complete result and
interpretation in `experiment_log.md`. The executed specifications remain below.
This specifies two comparisons in the existing grammar. It does not amend
GOAL, DATASETS, SEARCH_POLICY, source roles or the adopted 0.01/0.005 margins.

## Scientific decision

Full-channel stationary LMP ridge remains the strongest aggregate primary
anchor: delta_R2 0.001937385 and identity increment 0.003396325. Rank-one
ridge lowers both readouts. The fixed Nyström reference increases identity
to 0.004254813 but lowers the primary to 0.001562080, with a near-zero
Mihili primary mean (0.000078989). Neither capacity recipe passes the primary
or identity margin. This does not reject nonlinear LFP maps generally; it
does not motivate a cosmetic extension of their parameter grids.

The next interpretable hypothesis is that a single electrode-to-velocity
coefficient map across the entire reach suppresses a useful phase-dependent
relationship. Test one smooth, fixed quadratic time interaction. In a matched
refit, destroy individual training-trial pairing while preserving direction,
elapsed time, full neural trajectories and their electrode covariance. This
asks whether learning from the correct training trials contributes beyond
the common direction/time relationships. It is a different control from the
already reported fixed-prediction evaluation-pairing null.

## Unchanged data and evaluation

Use every frozen M1 session (six Mihili, ten Chewie), original complete trial,
physical LMP electrode, canonical direction and eight leave-one-direction-out
splits. Chewie implants remain one animal. Read the same current-bin stored
LMP and native XY values at offsets **0 through +14 inclusive**, fifteen bins
at 30 ms spacing. There is no new source interval, temporal lag, smoothing,
filtering, phase reconstruction, spike access or endpoint selection.

Reuse the equal-training-direction first-harmonic behavior template, its
shared residual intercept, native XY residual target, full-trajectory joint
SSE denominator and exact within-direction evaluation derangement mean.
The baseline and candidate use identical evaluation trials. All transforms
and model coefficients use only the seven permitted fitting directions.
The outer-session selector sees only the other fifteen session banks.

## T: fixed quadratic time interaction

Recipe ID: `lmp_timepoly2_ridge_v1`.

At native bin t in {0,...,14}, let u_t=t-7, a=mean_t(u_t^2), and
b=mean_t((u_t^2-a)^2), where these means use the fixed fifteen-bin design,
not source values. Define

`q0(t)=1`, `q1(t)=u_t/sqrt(a)`, `q2(t)=(u_t^2-a)/sqrt(b)`.

These three fixed columns are mutually orthogonal with mean square one on
the complete fifteen-bin grid. No degree, knot, basis coefficient or time
window is selected from an outcome. q2 is centered using the discrete grid,
not an assumed continuous Legendre normalization.

Within each fitting-direction split, standardize each original LMP channel
using its fitting-trial/bin population mean and SD, with exactly-zero SD
mapped to one as in the stationary linear anchor. Let Z be this matrix. Form
the tensor features in `(basis degree, physical-electrode order)`:

`F_it,(j,e) = qj(t) * Z_ite`, for j=0,1,2.

Subtract each F column's fitting-row mean and apply that same mean to
evaluation F. Do **not** whiten, restandardize product columns, remove null
columns, perform PCA truncation or fit an additional intercept. The shared
behavior residual intercept remains in both comparator and candidate.
Centering the products is necessary because a centered electrode can still
have a nonzero time-weighted mean.

Fit the same centered native XY residual R with objective

`(1/N_fit) ||R - F_centered C||_F^2 + lambda ||C||_F^2`,

using lambda in `[0.001,0.01,0.1,1,10]`. Exact tuning ties prefer the larger
lambda. Add the resulting residual prediction to the unchanged behavior
baseline. Each outer session selects lambda from animal-balanced primary
scores on the other fifteen complete session banks, exactly as before.

The stationary full-channel anchor is nested in this feature family: setting
the q1 and q2 coefficient blocks to zero leaves the same standardized LMP
map, penalty and behavior prediction at the same lambda. The q0 feature
mean is mathematically zero; final product centering changes it only by
floating-point roundoff. This is model nesting, not a requirement that a
higher-dimensional penalized model improve held-out performance. Reuse the
already completed stationary anchor rather than charging another trial for
its evaluation. Report its independently nested lambda selection alongside
the temporal recipe, without choosing an anchor separately for each session.

## S: eight whole-trial training derangements of the same temporal recipe

Recipe ID: `lmp_timepoly2_train_derangement8_v1`.

Use fixed replicate IDs 0 through 7. For every session, held direction,
permitted training direction and replicate, sort the original trial IDs.
Initialize Python `random.Random` via `seed(literal, version=2)` with

`ep05-lmp-timepoly2-train-derangement-v1|20261003|<session>|<held_direction>|<training_direction>|<replicate>`.

Use the canonical integer direction labels 0 through 7 and unpadded decimal
replicate labels 0 through 7. Starting from the ordered integer positions,
draw a fresh uniform permutation with `shuffle` until no position is fixed.
Accept the first derangement. Impose a finite 10,000-draw implementation
bound; exhausting it stops for diagnosis rather than changing seed or
method. All directions already require at least two trials. Rejection from
uniform permutations produces a uniform derangement conditional on success;
eight draws are a fixed Monte Carlo control, not its exact finite mean.

Map each fitting target-trial position i to the complete centered tensor
feature trajectory at its permuted position. Move all fifteen bins and all
electrodes/basis columns together; preserve the within-trial time index.
The permutation never crosses a direction or imports an evaluation trial.
Use the same permutations for every lambda. Record the actual original-ID
maps, not only the seed strings.

Compute channel standardization and tensor centering once from the legal
unpermuted fitting feature pool, then permute the completed feature rows in
whole-trial blocks. The fitting pool, feature moments and Gram matrix are
unchanged by this permutation; sharing those computations is allowed.
**Keep training XY, evaluation XY, the behavior fit and the residual target
unchanged.** Refit only the tensor slopes against the incorrectly paired
training features. Evaluation features retain their original correct trial
pairing; there is no evaluation shuffle in this training-control recipe.
Calculate the usual separate evaluation wrong-pair readout with each fitted
prediction held fixed, as for every other recipe.

### Fairness of lambda selection

For outer session s, first select the temporal recipe's lambda using only
the other fifteen correctly trained temporal banks. Apply that identical
lambda to the correctly trained model and all eight shuffled refits in s.
The control does not choose another lambda, seed, time basis or capacity from
its own favorable scores. Thus its scientific contrast holds regularization
and representation fixed and changes training pairing only. The common
lambda may depend on permitted paired outcomes from the other fifteen
sessions; this is explicitly a matched control of that paired policy, not a
separately optimized competing null model. No score from s tunes either arm.

For each session, average the eight **SSE scores** and the eight corresponding
wrong-pair SSE scores before forming its control primary and identity
increments with the unchanged denominator. Equivalently, average its eight
increments. Do not average predictions and score the average: that would
create an unintended eight-model ensemble. Retain all eight seed-level
results and report their spread descriptively; there is no permutation-tail
p-value or eight-fold increase in biological sample size.

Report session-level `paired_delta_R2 - mean_shuffled_delta_R2` and
`paired_identity - mean_shuffled_identity`, then the unchanged equal-session,
equal-animal summaries. Also report each animal, positive-session counts,
the stationary anchor contrasts and existing fixed-prediction influence
summaries. All sixteen sessions and every seed must complete; a failed cell
stops this comparison rather than being discarded or replaced.

### Adopted interpretation and promotion screen

The paired-training screen passes only if both animal means are strictly
positive for both paired-minus-shuffled contrasts and the primary contrast
is positive in strict majorities of both animals (at least 4/6 and 6/10).
Ties fail. This is a prospective directional control screen, not a new
practical-gain margin or a significance test. The candidate must separately
meet the unchanged 0.01 primary and 0.005 identity margins and every other
required check before promotion; this pair cannot promote it on its own.

A failure of this screen blocks promotion under the adopted screen and
bounds the claim that learning from correctly paired training trials is
needed. It does **not** establish absence of trial-specific neural
information: the shuffle deliberately preserves neural/behavior mean
relationships by direction and time, which can teach a map that still
predicts correctly paired evaluation-trial variation. Conversely, a positive
contrast does not identify a population mechanism or causal relationship.
Do not demand that the shuffled score itself be zero or reinterpret a
positive shuffled score automatically as leakage.

## Count, budget and implementation handoff

Each complete sixteen-session recipe adds one valid scientific comparison:
T is a model-family comparison and S is one alternative-explanation control.
Their complete pair would bring six to eight valid trials, with one control
among the two post-initial comparisons (50%). Eight seeds, five lambdas,
pilots, the reused anchor and evaluation wrong-pair calculations add no trials.
No trial is charged for an incomplete panel. The 28 minimum, 64 maximum,
12-trial patience, 40% post-initial control requirement, adaptive/challenger
requirements and existing CPU/wall-clock limits remain in force. The exposed
corpus supplies development evidence only; no future-session source exists.

The separate temporal runner preserves existing recipes and
banks. One LMP/XY source read can serve both arms. Shared feature moments,
Gram matrices and a five-penalty eigensolve are valid efficiencies because
whole-trial row permutations leave the design covariance unchanged. Each
control still needs its own cross-product, fitted slopes and scored
predictions. With at most about 15,570 rows and 288 tensor columns for a
96-electrode session, the tensor design is about 36 MB in float64; this is
a working-array estimate, not measured total RSS or runtime.

The coordinator selected the same complete Chewie_20150313 representative, both arms/all eight
replicates, one CPU, 4 GiB and five minutes, numeric threads one. Root must
use a suitable verified allocation with enough time remaining and select
the remaining-panel bound from measured cost without consulting pilot
scientific scores. No source cache, extra interval or replay is needed.

Focused implementation checks should establish stationary embedding at a
common lambda, correct tensor centering and held-direction exclusion,
whole-trial/direction-preserving no-fixed-point maps and invariant Gram
matrix, unchanged evaluation targets, mean-of-scores rather than
score-of-mean aggregation, and exclusion of the outer row from the common
lambda selector. Reuse the passing SSE and exact wrong-pair equations.
Root owns every compute test, source read and scheduler action. Implementation
is `tools/primary_temporal.py` and `tools/run_primary_temporal.py`, reusing the
unchanged LMP-only source reader and post-fit reference replay. Seven focused
fixtures are in `tools/test_primary_temporal.py`; no worker executed them.
