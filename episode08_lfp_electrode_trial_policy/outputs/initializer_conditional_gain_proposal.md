# Does greedy conditional gain improve a frozen initializer electrode set?

Prepared 2026-10-04 after30 complete development comparisons. **Root adopted
this recipe at06:02UTC** after root and independent scientific review. Root
selected the complete first-session resource pilot described below, with new
implementation/checks before execution. Adoption is not an empirical launch
or result; no held opening or named endpoint is activated.

## Question and scope

Target-free initial repeatability beats its paid uniform-order control in
both animal means, but its Mihili effect is small and omission-sensitive.
It loses to the target-linked marginal incumbent in both animal means; that
incumbent advantage is itself strongly influenced by Chewie20161006.
The next distinct question is whether individually useful electrodes are
redundant: does conditioning the selection proxy on already selected
electrodes improve the later residual population decoder under exactly the
same32-trial information state?

Use the existing **algorithmic** direction-residual conditional-gain proxy
with fixed alpha1. The earlier residual-conditional panel recomputed sets
on each full training prefix while all96 electrodes remained visible. This
proposal freezes the set from the initializer and observes only selected
channels later. It is not a repeat charge for that offline panel.
Training-objective conditional gain is not a newly activated biological
complementarity, electrode-removal-value, next-direction-value or sensor–trial
coupling endpoint. Those named explanatory decisions remain unactivated.

Candidate: `initial_residual_conditional_v1`.
Control: `initial_residual_conditional_target_labels5_v1`, with fixed seeds
0,1,2,3,4 and one mean-five control comparison.

## Unchanged information and decoding

Preserve the eight original development sessions, two exposed animals, all96
physical electrodes, original96 fitting/40 evaluation trials, same balanced
queues, initial32 rows and four24-row inner-fitting views. Use the unchanged
three native features per electrode, ten-bin150–450ms means, every native
population unit and true target scaling. Four held sessions stay closed.

For each candidate/control branch, one outer greedy order supplies nested
prefixes8/16/32. It is frozen through later totals32/48/64/80/96. Include the
common96-electrode branch in all20 cells. No post-initial reranking, lookahead,
discarded-channel backfill, unit exclusion, window search or trial adaptation.

At each initial inner view use only its24 fitting rows for direction means,
residual-X population SD, raw-Y population SD and residual intercept. Exact
zero SD becomes one as before. The outer view uses its32 rows. Fit the
unchanged reduced-rank residual decoder with **true, unpermuted Y** and the
existing alpha/rank grid, tie rule and tuning opportunity. Initial CV uses its
own view-specific selected sets; later conditioned CV includes every initial
row in fitting and validates only later acquired rows.

All30 orderings per session (five candidate views and25 control views) must
precede every post-initial source row. Every95 distinct target models/120
references must precede the40 evaluation rows. Five identical all96 models
can be shared across the six branches after support equality. Original
outcomes cannot choose a hyperparameter, favorable seed or stopping time.

## Exact inherited conditional-gain algorithm

Use the equation and algebra already implemented in
`tools/residual_selection_training.py:selector_order` for
`residual_conditional`. After the qualified true-fitting normalization, let
X contain three coordinates per physical electrode and Y the native-unit
target residuals. For selected set S define

`A_S = (I_n + X_S X_S' / n)^(-1)`.

For an available electrode's three-coordinate block e,

`G_e(S) = X_e' A_S X_e / n`,

`C_e(S) = X_e' A_S Y / n`,

`gain_e(S) = trace[C_e(S)' (I_3 + G_e(S))^(-1) C_e(S)] / U`.

At each step choose the unselected electrode with maximum gain; exact ties
use ascending physical electrode ID. Continue exactly32 steps, with no
outcome-dependent early stop or selector penalty search. This produces one
nested order, not separately optimized sets at8/16/32.

Reuse the existing rank-three update rather than a new objective: initialize
`AX=X`, `AY=Y`, symmetrize each computed G as in the qualified routine, and
let `M=I_3+G_e` and `H=(AX)_e` for the chosen electrode. Compute both right
hand sides from the **pre-update** state, then

`AX <- AX - H solve(M, X_e' AX / n)`

`AY <- AY - H solve(M, X_e' AY / n)`.

Mask already selected electrodes before the next choice. Preserve float64
equations and the current finite-score check; no epsilon, new ridge scaling,
gain clipping, tie perturbation, whitening or hidden rank change. Save each
view's selected32 physical IDs and32 selected gain values, with exact tie
diagnostics. No raw arrays, full per-step matrices or decoder coefficients
need serialization. A direct-inverse small fixture should verify that the
update realizes this objective across several choices.

## Matched target-label controls

Use the exact completed initializer target-label control's ordinary original-
ID maps and literals, independently for each matching24/32 view:

`ep08-acquired-marginal-target-label-v1|20261004|<filename>|<view>|<replicate>|<direction>`.

The inherited literal remains unchanged intentionally so physical trial maps
are paired across marginal and conditional selectors; it does not mean the
new selector is marginal. Sort original integer recipient IDs inside each of
eight directions, shuffle one fresh donor list once with Python version2,
map back to supplied row order, and move complete target vectors together.
Identity/fixed points are allowed. No derangement, redraw, cross-direction
mapping or bin/unit shuffle is introduced.

Compute the same true-fitting normalization first, then replace Y by the
permuted normalized Y **only for conditional ranking**. Each control starts
its own AX/AY state and follows its own32 choices. After selected sets differ,
the controls' conditional geometry also differs; do not share candidate AX
updates or pretend the comparison preserves identical selected covariance.
Every final decoder/CV fit still uses true Y.

Within-direction target permutation preserves the view's direction means,
marginal target variance and X. It disrupts the correspondence used by the
greedy objective but can leave finite-sample associations and overlapping
sets. It is not an exact biological null test, causal intervention or an
assumption that the three/four repeats are independent animals.

## References, scoring and decisions

Use unchanged Q/SSE, wrong-pair and native-target error. Average five seed
SSEs, never predictions. Retain all20 equal cells, equal sessions within each
animal and equal animal means. Report each session/animal/cell, every seed,
all selected gains and all fixed-prediction session omissions without refit.

The **incumbent challenge screen** requires conditional Q minus completed
true initial marginal Q strictly positive in both animal means; ties fail.
The **correspondence usefulness screen** separately requires conditional Q
minus its own mean-five target-label control Q strictly positive in both
animal means. An algorithmic claim that greedy conditioning improves the
practical policy requires both screens, with session/seed fragility disclosed.
Keep the existing0.001 search-progress resolution; it is not a new biological
margin, significance threshold or automatic promotion criterion.

Use the just-completed matched-exposure uniform panel from
`runs/development_search/initializer_repeatability_attempt001/initializer_repeatability.json`
as a descriptive fair reference after all new fits; no refit or extra
evaluation charge. Also display its repeatability candidate and the existing
initial marginal reference. The cheaper historical fixed-random panel remains
explicitly unequal in initializer cost. These references cannot determine
conditional orders or replace the two prospective screens.

After all new fits, read the completed initializer target-label packet and
repeatability packet. Verify exact shared role/trial/physical-channel records,
target-unit counts/exposure,20-cell support and all25 control view/seed maps
against the original target-label packet. Historical unit-guide IDs were not
serialized; current within-run guides remain exact, but no unrecorded old/new
unit-ID equality is claimed. Require all six branches' five common all96
channel/rank/alpha/candidate-count records to match the qualified three-band
reference, and Q/Qwrong/pairing to replay within1e-10. Match common target
denominators and direction-only Q under the existing1e-10 relative/absolute
and dimensionless tolerances. Do not claim historical prediction-array identity.

If conditional selection robustly beats marginal selection, that supports
this fixed greedy redundancy adjustment for future decoding from a paid
initializer. Even positive screens do not establish the unactivated
electrode-removal endpoint or prove biological complementarity. If it fails,
distinguish limitations of this alpha1 greedy proxy and32-row estimation from
absence of all conditional population information.

## Finite execution and shared accounting

Per-method exposure is unchanged:
`32*96 + (n-32)*E` electrode-trials, times30 LFP numerical values, plus
`n*U*10` native-target values. Record actual repeated/unique source bytes,
wall/CPU time, peak memory and retained artifact bytes. No exposure is free
because a source array was previously available.

A full cohort contains240 greedy orderings, **7,680 greedy selection steps**,
1,600 direction-level control maps,760 target models,15,200 inner ridge fits,
76,000 capacity scores,9,984 fitting-trial views and320 evaluation trials.
The latest three-band repeatability pair took107.166seconds with the same
target-model/read counts, but its score was cheap and nonsequential. The
32-step greedy solver multiplies selection work and must not inherit that
runtime as measured performance.

Propose one complete, preselected first-session resource pilot,
**Mihili_CO_20140217.mat**, including all six branches/views, true targets,
95 models and40 evaluation rows, bounded270seconds internal/300seconds
external on1CPU within4GiB. Its scientific scores remain withheld while
only fit/source timing, memory, support/counts and numerical validity are
used to choose the finite remaining-seven envelope. This is a representative
measurement of the added greedy cost, not a generic qualification exercise
or favorable-session scientific test. Root may instead select a justified
finite full-cohort bound before source reads; that concrete choice must be
recorded before execution, without changing the recipe or comparison count.

The pilot is reused after the remaining fits, with exact source-role and
recipe equality and no recomputation or favorable omission. Reference
comparisons and full scientific release happen only for the complete eight-
session cohort. Incomplete work consumes resources but adds no partial
candidate/control comparison. No automatic retry/requeue or unlimited bound.

If adopted, preserve old files and reuse qualified reader, normalization,
decoder, later folds and scoring helpers. Focus new fixtures on direct-inverse
equivalence of greedy updates, selected-block masking and tie behavior,
exact old conditional order when target maps are identity, matched original-
ID maps, fitting-only24/32 statistics, independent control geometry, true-Y
decoder inputs,30-before-future/95-before-eval barriers and full96 replay.

One complete candidate and one mean-five control add **two comparisons**,
30to32. Greedy steps, five seeds, views, cells, pilot and historical uniform/
repeatability/marginal references are not additional evaluations. Shared
40–112 allowance, patience, alternative-check accounting and resource
budgets persist. The four held sessions and named explanatory endpoints
remain closed.
