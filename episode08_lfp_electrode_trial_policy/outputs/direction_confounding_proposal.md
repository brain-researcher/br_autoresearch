# Next development comparison: direction means versus trial identity

Proposed October 3, 2026 after the completed five-selector batch and adopted
prospectively by the coordinator at **21:25:58 UTC**. All three panels and the two
focused checks completed in job 46430982, step 9, at 21:32:46 UTC. The execution
log records actual results and three evaluation charges. This is an alternative-explanation comparison under the existing
open exploration, not activation of named removal-value or coupling endpoints.

## Why this is next

Direction repeatability has the largest aggregate Q (0.148469), followed by
marginal target association (0.144147) and conditional target association
(0.128715). All three improve each of the eight session means over random
selection. The conditional selector beats its global target-permutation control,
but that permutation destroys direction structure as well as individual-trial
linkage. Therefore current Q does not identify which part of the useful signal
belongs to the particular trial. More elaborate selectors would not resolve
that ambiguity.

## Common roles, support and scoring

Retain exactly the current eight development sessions, forty evaluation trials
per session (five per direction), nested balanced 32/64/96 fitting trials and
four whole-trial inner folds. The four held sessions remain closed. Features
are the existing three ten-bin electrode means and targets are all native M1
population means; no time window, target column, support, source preprocessing,
or direction is changed. Numerical candidate components receive copied fitting
arrays and fitting direction labels only. The trusted evaluator completes all
selected models for a session before reading its evaluation arrays. Its use of
evaluation direction metadata selects an already-fitted direction mean; it
never estimates a mean or selects a model from evaluation targets.

Every outer score remains
`Q = 1 - SSE(prediction) / SSE(outer-training-global-mean null)`, with both
errors in the original outer-training population-SD target coordinates.
That denominator and target weighting are identical for every method at the
same fitting amount. Do not replace them with an evaluation or direction-
residual variance. Zero target SD becomes one and all target columns remain;
a zero/nonfinite total denominator is unresolved, as in the existing recipe.
Keep all session/cell scores, average replicates within cell, cells within
session, four sessions within animal and the two animals equally. No final
biological margin or held-session decision is introduced.

## Panel D: direction-only population null

For each permitted fitting prefix and each of eight directions, compute the
arithmetic mean of every raw target column. Predict that fitted mean for
each evaluation trial's direction. No neural input, selector, hyperparameter
or shrinkage is used. Every direction has four/eight/twelve fitting trials;
support is never borrowed from evaluation. There are three distinct models
per session, one per trial amount. Reuse their values across the electrode
axis only for paired comparisons; do not count those copies as extra fits or
evaluations. This panel directly measures how much existing Q is explainable
by direction labels alone. It uses the same training spike observations and
direction metadata, with zero LFP electrodes used by the predictor.

## Panel W: fixed within-direction wrong-trial control

Apply one new falsification panel to the completed **direction-repeatability
leader and conditional-target challenger**, chosen now, across every existing
resource cell. Reconstruct their existing fits with the unchanged recipes,
folds, seeds and five-penalty tuning; no new selection choice is permitted.
The earlier reports did not save models or predictions, so this replay is
necessary for the new pairing question, not a new independent replication.
Correct-pairing cell Q must reproduce the retained report within absolute
1e-10; a discrepancy is an engineering issue, not a seed/model search. The
all96 model is shared between these two unchanged reference branches.

After all fits are fixed, permute complete predicted population vectors only
among the five evaluation trials of the same direction, excluding fixed
points. Keep all units together and do not refit transforms or coefficients.
Use the exact mean wrong-pair loss, not a noisy Monte Carlo approximation:

`SSE_wrong = sum_d [sum_(i != j in d) ||(Y_i - prediction_j)/SD_train||^2 / (m_d-1)]`.

Here every `m_d=5`. This equals the mean SSE over all 44 derangements within
each direction and hence over their Cartesian product across directions,
without enumerating that enormous product. The off-diagonal average is a
whole-trial pairing control; it is not independent shuffling of target units.
Report `Q_wrong` and `pairing_increment = Q_correct - Q_wrong`, along with
the cell/session/animal summaries. A direction-constant predictor has exactly
zero pairing increment. No random seed, permutation p-value, biological
significance or new search objective is claimed. Exact expected wrong-pair
loss is sufficient for the question whether prediction follows trial identity.

## Panel R: direction mean plus within-direction residual ridge

Use the same five fixed random-electrode permutations, nested 8/16/32/all96
sets, five decoder penalties and four inner folds as the original random
baseline. No new selector is introduced. Within **each inner fitting fold**,
compute raw X and Y means separately for each direction. Subtract those
fitting means from fitting rows and from inner-validation rows of that
direction. Scale each X residual column by its fitting residual population SD;
exactly zero becomes one. Scale Y residuals by the **raw fitting Y population
SD**, not residual SD, retaining the original target weighting. Use ordinary
mean-loss ridge with unpenalized intercept. Select alpha by true-target error
on the inner validation rows after adding back the fitting-only direction
mean, in the raw fitting target-SD coordinates. Exact ties choose larger alpha.

Refit all direction means, scales and coefficients using the complete allowed
outer prefix. An evaluation prediction is its fitted Y direction mean plus
the ridge prediction from X after subtracting that direction's fitted X mean.
The held evaluation row never contributes to any mean or scale. All96 is fit
once per trial amount and reused across random replicates, but is a different
model from the original raw-feature all96 fit and must not inherit its Q.

Report Q and its paired increment over Panel D. Also apply Panel W's exact
wrong-trial formula to these same already-fitted residual predictions as a
secondary control; this does not require another fit or evaluation charge.
Direction residualization can still leave errors in estimated direction means;
the pairing increment distinguishes trial correspondence from a common
direction-level correction. It does not imply causation or raw spike recovery.

## Cost, counting and next decisions

This proposal contains **three** substantive method/control panels: D, W and R.
Completing all three would move the shared total from six to nine. W's two
preselected reference branches, random seeds, cells, inner fits and the repeated
correct-pair scores are not additional panels. The fixed-prediction pairing
readout is descriptive and cannot reset patience as a newly optimized objective.
For comparable candidate Q, retain the already adopted 0.001-Q progress rule
and all shared 40–112, patience, coverage and alternative-check requirements.

Read only the same 96 fitting and40 evaluation rows per development session;
the trusted reader can supply this one grouped comparison without copying
source files. The offline all96 training exposure remains explicit. Planned
ridge workload is 3,528 fits for the two unchanged reference branches and
8,064 for the residual-random panel, plus direction means and training-only
selectors. Direction-only uses no ridge fits. No neural arrays or coefficients
need be returned to candidate components after evaluation. The existing
same-UID ambient-filesystem limitation remains disclosed.

Selected coordinator bound after implementation/source review: one CPU,
2 GiB, ten minutes, numeric threads one, bundled with substantive work in an
attended allocation. The preceding 14,112-fit plus selector batch took39.7 s;
this estimate adds headroom for the new residual bookkeeping. No allocation,
tests or source access was launched by adopting the document. The prepared focused numerical
checks verify the exact wrong-pair formula against
five-trial derangement enumeration and that validation rows cannot fit
direction means; unchanged ridge/selector checks need not be repeated.

Interpretation remains open. If direction means account for most Q and pairing
increments are negligible, investigate task predictability rather than claim
individual-trial information. If residual prediction and correct pairing add
consistent value, next compare temporal representations and capacity-matched
linear/low-rank/nonlinear decoders on that same target. Trial-coverage and
acquired-only replay questions remain distinct later work, requiring their own
legal-action and actual-exposure recipe. These results do not predetermine a
forty-panel menu, activate the named explanatory proposal, or open held data.
