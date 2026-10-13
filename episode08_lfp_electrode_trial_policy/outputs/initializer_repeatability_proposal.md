# Can target-free initializer repeatability identify useful electrodes?

Prepared 2026-10-04 after28 completed development comparisons. **Root adopted
this recipe at05:51UTC**, after root and independent scientific review.
Implementation and focused checks precede bounded execution. Adoption is not
an empirical launch or result; no held opening or named endpoint is activated.

## Distinct question

True initializer target correspondence improves three-band marginal selection
on average, but its Mihili effect is small and omission-sensitive. The native-
LMP correspondence advantage is smaller still and reverses on one Chewie
omission, despite residual population prediction beyond direction means.
Those results motivate a simpler competing explanation: useful electrodes
might be recognizable from direction-repeatable LFP structure in the paid
initializer, without using the population target to rank them.

The earlier direction-repeatability panel recomputed selection from each
entire32/64/96 fitting prefix, with all96 electrodes still available, and used
the earlier raw ridge decoder. This proposal uses only the once-paid32-row
initializer and the currently qualified direction-residual reduced-rank
decoder, then acquires only its frozen selected channels. It is a distinct
sequential-information comparison, not another charge for the old panel.

The control is a **uniform random ordering of physical electrodes** with the
same declared initializer exposure. It is not called a score-label permutation:
that operation plus deterministic ties would generally fail to give uniform
random orderings when many repeatability scores equal zero. The proposed
control isolates the practical value of the frozen repeatability ordering
against an equal-exposure metadata-only choice, including cases with tied
scores. It also addresses the unequal-initializer-cost limit of the historical
fixed-random reference.

## Preserved data, model and action state

Candidate ID: `initial_direction_repeatability_v1`.
Control ID: `initial_uniform_order5_v1`, replicate labels0,1,2,3,4.
One complete mean-five control is one comparison.

Keep all eight development sessions, original animals/trials/electrodes, the
three native features per physical electrode (LMP,100–200 and200–400Hz power),
the150–450ms ten-bin mean, every original native population unit, the same
32-row all96 initializer and four24-row inner-fitting views. Keep original
balanced queues through32/48/64/80/96 trials, nested electrode counts8/16/32
plus96, all20 resource cells and all40 reserved evaluation trials/session.
Four held sessions stay closed. No new frequency, window, target averaging,
direction allocation, unit alignment or exclusion is introduced.

Only the ranking function is target-free. Its public pure-array API accepts
X, directions and physical-channel identities, **with no Y argument**.
The trusted source reader still acquires initializer population targets,
because the downstream population decoder needs them. These targets remain
available to decoder fitting/CV; do not claim zero-target acquisition,
target-free decoding or OS-level isolation. Later true targets remain part
of each selected-channel acquisition history.

## Exact original repeatability formula

Use precisely the existing direction-repeatability equation in
`tools/selector_training.py:selector_order` and
`selector_comparison_proposal.md`, with a target-free entry point. For each
view separately, standardize each **raw** LFP feature by its view mean and
population SD; exact zero SD becomes one. Do not use direction-residual SD,
target scaling or future trials for this statistic.

There are eight direction groups, each of size m=3 in an inner24-row view or
m=4 in the outer32-row view. For standardized feature x, let μ_d be its mean
within direction d and μ its global mean. Define

`MS_between = m * sum_d (μ_d - μ)^2 / 7`

`MS_within = sum_i (x_i - μ_direction(i))^2 / (n - 8)`

`tau2 = max((MS_between - MS_within)/m, 0)`

`repeatability = tau2/(tau2 + MS_within)` when the denominator is positive;
the exact zero/zero case is **zero**, with no epsilon or adaptive fallback.
Average the three feature repeatabilities per electrode. A fully constant
feature therefore contributes zero; do not drop it or reweight the other
bands. No shrinkage, target-linked weighting or additional selector tuning.

Rank decreasing mean repeatability; exact ties use ascending physical
electrode ID, exactly as the original rule on its canonical ascending input.
Take one nested order through32. Save per-band repeatability, between/within
components, per-electrode means, zero-SD columns, exact tied-score counts and
selected physical IDs so a result dominated by ties is visible.

This score describes direction-repeatable LFP structure. It is not direct
within-direction trial information, population-target association, broad
artifact quality, line-noise suppression, neural causality or an independently
certified signal-to-noise ratio. Direction means are removed by the residual
decoder, so useful residual information from this ordering is an empirical
question rather than guaranteed by the score's definition.

## Uniform, tie-independent control

For every filename/view/replicate, start from the full canonical ascending
list of96 physical electrode IDs. Seed Python `random.Random().seed(literal,
version=2)` with

`ep08-initial-uniform-order-v1|20261004|<filename>|<view>|<replicate>`.

View is outer or0/1/2/3; replicate is0through4. Shuffle the fresh list once,
using its exact resulting order directly. There is no score sorting after
the shuffle, rejection of identity/fixed points, target-dependent redraw,
geometry stratum, channel-quality restriction or call-history dependence.
Save the complete physical-ID permutation and literal. Every E-prefix is a
uniform E-subset conditional on the available96 channels, irrespective of
candidate repeatability ties. Controls are nested across E and independently
seeded across information views; each inner decoder uses its own declared
view's set, and the outer set is used for final refit/later acquisition.

This is an ordinary Monte Carlo random-selector comparator, not a calibrated
randomization test of biological exchangeability. Five seeds are not five
animals. The candidate's own exact-tie preference for low physical IDs is
preserved; the uniform control does not inherit that deterministic preference.
Report this difference and the observed tie counts, rather than attributing
all performance differences uniquely to distinct repeatability values.

The uniform rule does not need all96 initializer LFP values to make its
choice, but receives/pays that same initializer in this explicitly matched-
exposure comparison. Historical fixed random remains a useful cheaper
alternative. Do not suggest that random selection intrinsically requires the
extra observations or that equal experimental exposure proves efficiency.

## Frozen ordering, true targets and matched decoding

Build five candidate orders and25 control orders before any post-initial
source row, for30 total orders per session. An inner order must use only its
24 fitting rows; no outer order is substituted into inner CV. Each outer
ordering remains fixed after the initializer. Acquired channel sets cannot
change and discarded-channel observations cannot be backfilled.

Reuse the qualified three-coordinate selected-rank initial fitter and
initializer-conditioned later folds, always with true unpermuted Y. Use the
same training-only direction means/scales, alpha grid, rank menu, true-target
loss and existing lower-rank/stronger-alpha tie rule as the completed marginal
selector. Every branch gets the same tuning opportunity. Directions, true
targets and future queues are unchanged by the random electrode ordering.

Share the five identical all96 fits across candidate and five controls after
within-run support equality. Preserve all95 distinct models/120 references
per session, with every actual model fixed before evaluation. The old outcome
packets do not supply selections, hyperparameters, favorable seeds or stopping.

## Scores, replay and prospective interpretation

Use the unchanged Q/SSE and wrong-pair readouts. Average the five control
SSEs, not predictions; retain every seed, the common denominator and native
target error. Preserve all20 equal cells, equal sessions within animal and
equal animal means. Report all session/animal/cell contrasts, seed means and
fixed-prediction session omissions without refitting.

The **repeatability usefulness screen** requires candidate Q minus mean
uniform-order control Q strictly positive in both animal means; ties fail.
The **incumbent challenge screen** separately requires candidate Q minus the
completed true initial marginal Q strictly positive in both animal means.
Neither screen implies the other. Report residual improvement beyond the
direction-only decoder descriptively and retain the existing0.001 search
progress resolution. There is no new biological margin, p-value, equivalence
threshold or automatic promotion rule.

Read the completed initializer-target-label control packet and its original
acquired-only reference only after all new fits. They provide the existing
three-band true-marginal, target-label-control and cheaper fixed-random panels.
Require exact current source-role records, session/animal/trial/physical IDs,
target-unit counts/exposure and twenty cells; historical unit-guide IDs were
not saved, so do not assert that unrecorded identity. The current reader must
still preserve native unit-guide identity throughout the new run.

Unlike the LMP spectral branch, the common all96 representation/decoder is
unchanged here. Require every new branch's five all96 channel/rank/alpha/
candidate-count records to match the completed three-band reference, and
Q/Qwrong/pairing to replay within1e-10. Match target denominators and direction-
only Q under their existing tolerances. Other old scores remain comparisons,
not fitting inputs or purported old prediction-array identity.

If repeatability beats its equal-exposure uniform control, direction-repeatable
initializer LFP is a useful electrode-ranking proxy under this recipe; this
does not prove a causal mechanism or make its proxy specifically residual.
If it additionally beats the target-linked incumbent robustly across animals,
sessions and seeds, that would motivate it as a simpler practical challenger.
Failure limits this32-row repeatability recipe, not every reliability metric,
signal-quality measure or target-free selector. A mean-positive but omission-
or seed-sensitive result remains explicitly fragile.

## Finite work and comparison count

All branches pay `32*96 + (n-32)*E` LFP electrode-trials, times30 numerical
values for the three ten-bin features; targets cost `n*U*10` values. Candidate
ranking does not use Y, but source exposure and decoder supervision include
it. Report actual repeated/unique source bytes, wall/CPU time, peak memory
and retained output bytes. No raw arrays or decoder coefficient export.

The cohort contains40 repeatability fitting views,240 total orderings,
200 full physical-ID control permutations,760 actual target models,15,200
inner ridge fits,76,000 capacity candidate scores,9,984 fitting-trial reads
and320 evaluation trials. The previously completed three-band control took
99.608seconds for640 models and8,448 fitting reads; this is approximately19%
more model work with cheap fixed-form repeatability scoring. Propose the same
1CPU/4GiB,270seconds internal/300seconds external full-cohort envelope,
subject to root's actual prelaunch resource selection. No generic new pilot,
automatic retry/requeue or successful-subset charge is needed.

If adopted, new pure selector/orchestration/runner files reuse the qualified
three-band reader, selected-rank fitter, later-fold and scoring helpers without
changing old files. Focused checks should establish exact agreement with the
original repeatability formula including constants/ties, absence of a Y
selector interface, inner validation exclusion, uniform full-ID permutations
and nested subsets independent of tied scores, true-target decoding, both
information barriers and common96 replay. Do not rerun unchanged general
numerical tests without a new concern.

A complete candidate and mean-five control add **two comparisons**,28to30.
Seeds, views, nested resource cells, full96 reuse and historical references
are not extra evaluations. This is one target-free acquired-only selector
family plus its matched randomized comparator, not five new families or a
repeat of the old all-prefix raw-decoder panel. Shared40–112 allowance,
patience, alternative-check accounting and1200CPU-hour/120-hour budget persist.
The four held sessions and named explanatory endpoints remain closed.
