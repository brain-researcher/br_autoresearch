# Does the marginal-selection result survive using only high-frequency power?

Prepared after32 completed development comparisons; **root adopted at
2026-10-04 06:18 UTC** after root and independent scientific review passed.
The fixed implementation and bounded full-cohort run are authorized. Four
held sessions and named explanatory endpoints remain closed.

## Distinct spectral question

Greedy conditional selection retains useful initializer correspondence but
does not beat the original independently ranked marginal selector in either
animal mean. The completed LMP-only branch also preserves residual prediction
yet has a small, omission-sensitive selection contrast and loses to the
three-band incumbent. The next question concerns the available feature
information, not another setting of the same conditional-proxy penalty:
does the marginal selection and correspondence result survive **without LMP**,
using only native100–200 and200–400Hz power?

This completes a complementary high-power-only sensitivity to the existing
LMP-only and three-band panels. It does not isolate spike contamination,
prove a physiological cause, or establish conditional complementarity.
Removing a band changes input information, bandwidth and regularized model
capacity. Neither high-band sufficiency nor a performance loss identifies
what physiological process supplies the predictive signal.

Candidate: `initial_marginal_high_power_only_v1`.
Control: `initial_marginal_high_power_target_labels5_v1`, fixed seeds0through4.
One complete mean-five control is one comparison.

## Actual two-coordinate representation

Keep all eight original development sessions, two animals,96 physical
electrodes,96 fitting and40 evaluation trials/session,32-row initializer,
four24-row inner views, original balanced queues, all native target units,
and the ten-bin150–450ms mean. Preserve electrode prefixes8/16/32 plus96,
trial totals32/48/64/80/96, all20 cells and the closed four held sessions.

For every permitted electrode/trial, select only the native numerical guide
rows **(100,200)** and **(200,400)**, in that order. Each electrode has exactly
two input coordinates, its two ten-bin power means. Do not read LMP numerical
rows merely to zero/drop them later. The full guide may be read as metadata
to verify identities. Do not reconstruct or filter a waveform, infer an
undocumented filter, logarithmically transform power, change targets, add an
unselected band, duplicate columns or pad the arrays back to3E.

The new reader/trajectory must use actual2E column indexing and retain native
target means and unit-guide equality throughout its current run. Numerical
exposure counts20 actual-dtype values per electrode-trial, not the old30 or
LMP-only10. The physical-channel cost is unchanged, but actual acquired
feature bandwidth differs across spectral branches. Missing one declared
band or inconsistent support stops the comparison without fallback/exclusion.

## Marginal objective and matched maps

Every information view uses only its true fitting rows to compute the
existing direction means, residual-X population SD, raw-Y population SD and
residual intercept. Zero SD becomes one; no channel or unit is dropped.
Let X and Y denote the resulting standardized direction-residual matrices.
For each electrode's **two-coordinate** block e use the original alpha1
mean-loss objective in its actual dimension:

`G_e = X_e' X_e / n`, `C_e = X_e' Y / n`,

`gain_e = trace[C_e' (I_2 + G_e)^(-1) C_e] / U`.

Symmetrize G as in the qualified marginal selector, compute all gains once
from the empty set, and rank decreasing gain with exact ties resolved by
ascending physical electrode ID. Keep a single nested top32 ordering.
There are no greedy conditioning updates, selector penalty search, band
weights, whitening or target-guided dimensionality changes in this branch.

For each control view/replicate, reuse the exact completed ordinary original-
trial-ID permutation and seed literal:

`ep08-acquired-marginal-target-label-v1|20261004|<filename>|<view>|<replicate>|<direction>`.

View is outer or0/1/2/3; each direction has four outer or three inner rows.
Move whole multivariate target rows within each direction, preserve identity
and fixed points, and never redraw. Normalize on true fitting Y first, then
use mapped Y only in C_e for ranking. X, its Gram and normalization remain
fixed across seeds. Save full recipient/donor original IDs, literal seeds,
gains and selected physical IDs. The same physical maps pair the sensitivity
with prior spectra without making their representations/capacities equal.

## True-target decoder and information barriers

Freeze all30 candidate/control orders per session before any post-initial
source row. Each inner decoder uses its own24-row view's selected columns;
an outer ranking cannot replace it. The outer set stays fixed through all
later checkpoints, observing only its retained two-coordinate channels.
No reranking, discarded-channel backfill, adaptive direction or future-value
peek is introduced.

Every decoder and CV fit uses **true, unpermuted Y** with the qualified
direction-residual reduced-rank ridge objective, original alpha grid and
existing lower-rank/stronger-alpha tie rule. Compute legal ranks1/2/4/8 using
the actual **2E** input dimension, target-unit count and initial/later fitting
row bounds, plus the unchanged full option. No hidden3E menu or padded
feature assumption. Later conditioned folds include every initializer row
and validate only newly acquired rows as before.

Candidate and its own high-power-only control match actual dimension,
information state, true targets, tuning opportunity and evaluation support.
The three-band/LMP-only comparators have different feature information and
capacity, so their differences are descriptive whole-pipeline sensitivities.
Share five identical all96 high-power fits across the six branches. All95
distinct models/120 references must be fixed before evaluation/session.

## Scoring, references and decisions

Use unchanged Q/SSE, wrong-pair and native-target error. Mean the five seed
SSEs, never their predictions; preserve each seed and common denominator.
Keep all20 equal resource cells, equal sessions within each animal and equal
animal weighting. Report every session/animal/cell and all fixed-prediction
session omissions. Do not choose an exclusion or winning seed.

Two prospective screens are separate:

- **Correspondence usefulness:** high-power true marginal Q minus its own
  mean-five control Q is strictly positive in each animal mean; ties fail.
- **Residual information:** high-power true marginal Q minus direction-only
  Q is strictly positive in each animal mean; ties fail.

Keep the existing0.001 search-progress resolution, with no new biological
margin, significance/equivalence claim or automatic promotion threshold.
Report small, seed-sensitive or omission-sensitive effects even if their
full means pass. These screens do not constitute a joint acquisition result.

Only after all new fits, read the completed original initializer target-label
packet and `lmp_correspondence_attempt001/lmp_correspondence.json`. Reuse the
three-band and LMP-only candidate/control results without refitting or
recharging them. Check exact source-role/trial/physical-channel records,
target-unit counts/exposure, all20 cells and exact25 original control maps.
Historical unit-guide IDs were not serialized; verify current within-run
guides and recorded historical counts/support without asserting unrecorded
unit-ID equality.

Target denominators and direction-only Q are representation-independent;
require their existing1e-10 relative/absolute denominator and1e-10 absolute
Q replay tolerances. All96 high-power score/capacity/channel records replay
**within the new six-branch run**. Do not require new high-power predictions,
rank/alpha choices or Q to equal old three-band/LMP models whose features
differ intentionally. No old outcome is a fitting, selector or seed input.

Report high-power true Q minus three-band true Q and minus LMP-only true Q,
and the corresponding differences between true-minus-own-control contrasts,
overall and by animal/session. They are fixed descriptive comparisons, not
post-score endpoints. Historical uniform/repeatability and cheaper fixed
random need no extra reference here: the new own-label control already
matches high-power representation and initializer cost.

If both screens survive, this supports residual prediction and useful
initializer correspondence using the two native power features under this
recipe. It does not prove that LMP lacks information, that power carries
independent physiology, that spikes contaminate it, or that a new animal
would reproduce the result. Failure limits this two-band pipeline without
establishing universal absence or a contamination mechanism.

## Bounded work and accounting

Each method pays `32*96 + (n-32)*E` physical electrode-trials, now times20
LFP numerical values; target exposure stays `n*U*10`. Report actual repeated/
unique logical source bytes, CPU/wall time, peak memory and retained report
bytes. No raw array or target-decoder coefficient serialization.

The full pair has240 marginal orderings,1,600 control direction maps,760
target fits,15,200 inner fits,76,000 capacity scores,9,984 fitting-trial views
and320 evaluation trials. The completed LMP-only, three-band repeatability
and conditional pairs used the same target-model/read support in approximately
87,107 and103seconds respectively. Two-coordinate marginal ranking has no
new greedy workload. Propose one full-cohort execution on1CPU within4GiB,
270seconds internal/300seconds external, after the focused representation
checks and actual root allocation/reserve verification. This is an estimate,
not measured high-power runtime; no generic pilot or automatic retry is needed.

If adopted, preserve existing files and add narrowly scoped two-band reader,
trajectory, marginal ranker and selected-rank indexing. Reuse qualified
normalization, later folds, ridge/projection, scoring and original map helpers.
Focused fixtures should prove only selected power numerical rows are read,
actual2E feature/byte accounting and legal ranks, direct2x2 marginal objective,
unchanged target maps and true-Y decoders, inner-view exclusion, both
information barriers and appropriate target-only/within-run full96 replay.
Do not rerun unrelated unchanged general tests.

A complete candidate plus one mean-five control adds **two comparisons**,
32to34. Seeds, spectra comparisons, full96 fits, cells and reused references
are not new evaluations. Shared40–112 allowance, patience, alternative-check
accounting and resource budgets persist. Four held sessions and named
explanatory endpoints remain closed.
