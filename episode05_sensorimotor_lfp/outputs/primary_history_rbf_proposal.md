# Nonlinear prediction on the fixed weighted historical LMP state

Prepared 2026-10-04. **ADOPTED by root after scientific review at 20:35 UTC.**
Implementation and tests were unverified at adoption; the subsequent measured
resource decision is recorded below without releasing pilot effects. Root owns
adoption, compute, accounting and the experiment log. This comparison stays
within the existing history/reliability/RBF search grammar. It does not adopt
the separate independent-penalty blend amendment.

## Question and scientific boundary

Forty valid comparisons are complete, patience is seven of twelve, and no
candidate meets the original practical criteria. The stationary 105-landmark
RBF reference improved identity relative to stationary ridge while losing
primary performance. Subsequent HW work supplies a qualified four-bin history
representation, but its ridge head and the W/HW blends remain linear in that
history. Ask whether a fixed nonlinear map of **this same weighted historical
state** improves full native-XY residual prediction.

This is a predictor change on HW, not a bandwidth/capacity search, and not a
claim specifically about cross-lag interactions. RBF also changes within-bin
electrode interactions, shrinkage and effective capacity. A gain would not
identify a mechanism, neural latency, spike-free LFP or online capability.
The existing stationary RBF, linear HW and constant blend remain independent
fixed anchors. No unseen outcomes selected this proposal. All recordings are
exposed exploratory development from the original two animals.

## HW105: fixed candidate

Recipe `lmp_history4_repeatability_weight_nystrom105_v1` uses all sixteen
original sessions, original complete trials and physical LMP channels. Read
LMP offsets -3:+14 once per session, keep target offsets 0:+14, and preserve
lag order [0,-1,-2,-3]. No spikes, new band, new window, new target or source
selection enters. Reuse the exact HW `prepare_fold` and
`history_trace_weights` helpers: current-only W mean/population-SD and
T/P/r/a with a=0.1+r; separate history mean/population-SD and final column
centering; one physical weight repeated over four lags, normalized by the
original full-history trace. Keep the zero-Gram/equal-a conventions. Do not
restandardize after applying weights or choose a new floor.

Reuse the original RBF operator from `primary_capacity.py`:

1. Choose exactly one fitting trial per legal direction using the existing
   `landmark_rows` helper, original qualified IDs and literal
   `ep05-rbf-landmark-v1|20261003|<session>|<held_direction>|<training_direction>`.
   Reuse its exact ordering and Python seed version 2. All fifteen bins of
   each chosen trial yield 105 landmarks. No landmark or seed search occurs.
2. Set fitting-zero-SD lag/electrode coordinates to zero in both fitting and
   evaluation kernel inputs, retaining their identities. Activity is determined
   from each lag/electrode's original fitting SD, never current-only T or a.
   This preserves the old RBF-specific rule; linear HW's treatment of an
   evaluation-varying fitting-constant column is not falsely relabeled identical.
3. Use sigma squared equal to the number of positive-SD history columns and
   k(z,z')=exp(-||z-z'||^2/(2*sigma_squared)). The positive physical weighting
   preserves total fitted history energy mathematically, though individual
   weighted coordinates no longer have unit variance. This is the inherited
   dimension rule, not a new exact floating-point trace rule or median search.
4. Use the existing Cholesky Nyström map with fixed 1e-8 diagonal regularizer;
   center its features by their fitting mean. If no active columns exist,
   use the exact zero-feature branch. No jitter sweep, rank cutoff, whitening,
   feature standardization or alternate bandwidth is permitted.
5. Fit native centered XY residuals at the unchanged five penalties
   [0.001,0.01,0.1,1,10], normalized loss giving n*lambda in normal equations.
   One symmetric Gram/eigensystem serves candidate and controls; no clipping
   enters the solver. Diagnostic PSD conventions remain diagnostic only.
   Add the original direction/time baseline and common residual intercept once.

Every fitted object excludes the held direction. Select each outer session's
lambda using only the other fifteen complete candidate banks, equal sessions
within animal then equal animals, with the original stronger-lambda exact tie.
Identity, anchor contrasts and held-session results cannot select lambda.

## S105: one mean of eight training-pairing controls

Recipe `lmp_history4_repeatability_weight_nystrom105_training_surrogate8_v1`
uses the existing eight HW training-surrogate maps and their exact literal
seeds, ordinary permutations, fixed points and repeated draws. Freeze the
Nyström landmarks, feature mean and complete fitting/evaluation features
before applying any map. Move complete fifteen-row feature trajectories
within each permitted fitting direction; keep native residual targets and
evaluation features correctly paired. Do not reconstruct landmarks from
shuffled trial labels or refit preprocessing.

For centered fitting feature matrix Phi and whole-trial row permutation P,
the control design is P*Phi. Its Gram equals Phi'Phi algebraically. Reuse
that single actual Gram and eigensystem for all nine heads; only the
cross-product with true residual targets changes. Therefore conditional
ridge-head Gram spectrum and df are matched at each lambda. This does not
match RBF capacity to linear HW, and the control retains direction/time
structure rather than being a signal-free recording.

Fit all five penalties, then evaluate all eight controls at the **candidate's**
outer-selected penalty. Average their correct SSEs and exact within-direction
whole-trajectory wrong-pair SSEs, never predictions. Keep every seed result.
No seed selection, independent control tuning or eight-draw p-value is allowed.

## Replay, comparisons and accounting

After all new fits for a session are fixed, replay original HW physical state,
current statistics, history normalization, weights, covariance, native IDs,
directions, Y/B and every old five-penalty prediction score exactly. Saved Y/B
are restored with the existing original time-fast layout helper; predictions
retain literal C-order storage. No allclose, alternative layout or refit is
allowed to rescue a mismatch.

Before complete-panel release, independently replay the old HW and stationary
RBF outer selectors and the constant blend's original 25-cell selector. Replay
saved original selected arrays/scores and common native support. Compare the
new selected candidate with all three original selected policies, and with
HW and stationary RBF at its selected nominal lambda (descriptive only).
Report candidate-minus-mean-control and every candidate-minus-seed contrast.
Preserve every session, both animals, majority counts, all fixed-prediction
within-animal omissions and the unchanged 0.01/0.005 practical screens.
These comparisons do not match RBF to linear model capacity or transfer old
recipe-specific finalist falsifiers to a new positive candidate.

Compare the complete selected candidate literally with each fixed selected
HW/RBF/blend panel and the baseline. Equality in every session to one same
panel is a duplicate with zero charge. Similar scores, matching different
anchors by session or one zero-feature fold are insufficient. Each control
draw's selected prediction is compared separately with the candidate, the
three fixed anchors and B. A mean-control recipe is an exact duplicate only
if every draw in every session equals one same complete fixed panel (including
the new candidate); otherwise it remains one distinct mean-control recipe.
This conservative literal rule also catches zero-feature and all-identity-map
cases without treating similar mean scores as duplicate science.

Process candidate first, control second. At most two distinct valid recipes
are charged; only the control earns literal-null and alternative-check credit.
Only a strict comparable primary improvement by the complete candidate can
reset patience. No parameter cells, seeds, folds, anchor contrasts, pilot,
technical failures or exact duplicates increase depth. Root supplies actual
incoming counters and applies the original minimum28/maximum64, patience12,
>=40% post-initial alternatives, 2000core-hours and original120-hour clock.
No promotion, stopping, future-session opening or spike follow-up is implied.

## Implementation and prospective resources

Use only new `primary_history_rbf` source/runner/test filenames. Preserve all
old files and attempts. Existing history source reader supplies both recipes
once per session. Publish session score banks and complete prediction arrays;
do not expose partial scientific scores. A complete sixteen-session summary
is published only after all new and original replay checks pass.

Root's prospective resource-only pilot is the fixed largest
Chewie_20150313 session (1038 trials,86 electrodes), one CPU,6GiB, one numerical
thread, at most600seconds internal and a15-minute outer envelope. It includes
all eight directions, five penalties, candidate/eight controls and original
HW replay. Source/fit/replay/write durations and peak RSS are recorded; pilot
effects remain closed. Preserve the complete pilot if successful. The
remaining fifteen-session worker count and aggregate deadline require root's
decision from measured pilot cost and actual allocation, before launch.
No automatic retry, solver modification, extension or source workaround is
selected. Tests run only in root's verified Slurm allocation.

Focused fixtures cover preservation of HW state; held-direction exclusion;
the inherited landmark/constant-column rules; fixed nonlinear-map row
permutation and shared-Gram heads versus independent solves; baseline/add-back;
whole-trial evaluation pairing; mean-SSE control aggregation; outer-session
exclusion/ties; exact saved-layout replay; and global duplicate/count rules.

### Root-selected remainder resources after the fixed pilot

Root reports nine fixtures passed in 0.416 seconds and the fixed
Chewie_20150313 pilot completed in 61.118 seconds overall (child 60.239;
source 54.246; fitting 5.358; original-HW replay 0.247 seconds), with peak
541,404 KiB, all 1,038 trials and 86 electrodes, and every original-HW replay
passing. Pilot scientific effects remain closed. These measurements select
resources only; they do not select a recipe, parameter, session or outcome.

Preserve `primary_history_rbf_attempt001/Chewie_20150313` and finish exactly
the other fifteen original sessions in one batch: normal partition,
CPU_SKU:7502, one CPU, 6 GiB, 35 minutes, no GPU, no requeue, one worker and
one numerical thread. Fifteen pilot-cost sessions project about fifteen
minutes. The aggregate internal deadline is 1,980 seconds; the external
envelope is 2,040 seconds with at most ten seconds for timeout termination,
within the 2,100-second allocation. Each child retains the original 600-second
internal/900-second external bounds, and no child starts without a full
900-second aggregate reserve. This is a resource choice from the successful
pilot, with no scientific recipe change or automatic retry.

The executable is `outputs/tools/run_primary_history_rbf_remainder.sbatch`.
Its fixed paths reuse the original scalar bank, HW attempt002, stationary K
`primary_capacity_attempt001` and its original summary, and constant-blend
attempt001 with its original summary. All sixteen new banks remain in
`outputs/runs/adaptive_search/primary_history_rbf_attempt001`; the fresh
wrapper directory is `primary_history_rbf_remainder_attempt001`, and the
complete scientific summary directory is `primary_history_rbf_summary_attempt001`
under the same adaptive-search parent. All complete-panel replay, duplicate,
counting, stopping and effect-release rules above remain unchanged. Root
records actual job execution and charges its consumed allocation to the
original 2,000-core-hour/120-hour limits; this proposal does not claim that
the remainder has launched or completed.
