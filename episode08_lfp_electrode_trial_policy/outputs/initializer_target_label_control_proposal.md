# Does the paid initializer need its true trial-to-population correspondence?

Prepared 2026-10-04 after 25 complete development comparisons. **Root adopted
this exact recipe at 05:22 UTC**, before new implementation or source reads.
The new implementation is being prepared for root's independent checks and
bounded execution. Adoption is not an empirical launch or completed result;
no held-session opening or named explanatory endpoint is activated.

## Why this control is the next distinct question

Initial-only residual marginal electrode selection remains stronger than the
seven-donor and reverse-animal learned rules. The latter do not consistently
beat their own donor-label controls. Their three-feature rankers nevertheless
retain a current-target-initializer feature computed from its **true** marginal
LFP-to-population association. Randomizing donor training labels does not test
whether this current-session association is responsible for marginal
selection's advantage.

The earlier [residual-selector comparison](residual_selection_proposal.md)
randomized a **conditional**, greedily updated selector fitted anew on every
complete training prefix, with all electrodes visible throughout. It did not
randomize the **marginal** selector whose once-only 32-trial ordering must
survive later acquisition after other channels become unavailable. The
[acquired-only protocol](acquired_only_proposal.md) changed both this information
timing and later CV. Its fixed-random reference also had a cheaper initializer.
That earlier control is relevant background, not a replacement for this test.

Test one matched-information control of the existing marginal policy: keep
all acquired initializer information and decoder targets, but randomize only
the within-direction target correspondence used to rank electrodes. This
addresses an alternative explanation for the surviving selection result,
without changing decoder capacity, introducing a more complex transfer model,
or adding an outcome-selected grid. The operative open-exploration rules allow
such matched selection checks; the conditional-removal, next-direction-value
and sensor–trial-coupling endpoints remain unactivated.

## One fixed five-seed control panel

Control ID: `initial_marginal_target_labels5_v1`.
Fixed replicate labels: **0,1,2,3,4**. Their mean is one control comparison.
The previous true-pairing `initial_marginal` result is a reused reference, not
a new candidate panel or a newly charged evaluation.

Keep all eight development sessions, the original 96 fitting trials per
session and their direction queues, all 40 reserved evaluation trials,
ascending 96 physical electrodes, and the same three mean LFP features per
electrode. Targets remain separate ten-bin time means for every original
native unit; there is no neuron averaging, unit alignment across sessions,
new time window or target transformation. Four held sessions remain closed.

Every control pays the original 32-trial all-96 initializer. It freezes one
nested electrode ordering through32 positions with retained prefixes8/16/32,
then acquires only its retained channels along the same balanced original
queues. Evaluate totals32/48/64/80/96 and include the common all-96 branch,
for the unchanged20 cells. No post-initial reranking, discarded-channel
backfill, channel revision or adaptive trial action is introduced.

## Exact randomization and marginal ranking

For each session, form the five already qualified information views: each of
the four24-row initial inner-fitting sets and the outer32-row initializer.
Each inner view has exactly three trials per direction; the outer has four.
Use only that view's X, true Y, direction labels and original trial IDs to
compute the existing direction means, residual-X population SD, **raw-Y**
population SD and global residual intercept convention. Zero SD becomes one
without dropping a channel or unit. An inner view excludes its eight initial
validation rows from all these statistics and its selection computation.

Let X and Y below denote the resulting standardized, direction-residual,
centered matrices for a permitted view. For every direction separately, sort
the original integer trial IDs in ascending numeric order. Seed Python
`random.Random().seed(literal, version=2)` using

`ep08-acquired-marginal-target-label-v1|20261004|<filename>|<view>|<replicate>|<direction>`.

View is `outer` or unpadded`0`/`1`/`2`/`3`; direction is unpadded`0`through`7`.
Shuffle a fresh canonical donor-ID list once, align it with the sorted
recipient IDs, and map back to the supplied row order. Ordinary uniform
permutations include identity and fixed points: **no rejection, derangement,
adaptive redraw or dependence on RNG call history**. Keep one donor mapping
for the entire target vector, so all native units move together. Do not shuffle
units, LFP, direction labels or bins independently, and do not cross directions
or information views. Save exact recipient/donor original IDs and seed literals.

Use `Y_selector = Y[donor_map]` only in marginal ranking. Computing the qualified
normalization on true fitting Y before permuting preserves that exact metric
for every seed. Within-direction whole-vector permutations preserve direction
means and raw-Y marginal variances mathematically. They randomize the residual
row correspondence rather than generating different marginal targets.

For every three-coordinate electrode block e, use the unchanged penalty1
mean-loss proxy:

`G_e = X_e'X_e/n`, `C_e = X_e'Y_selector/n`,
`gain_e = trace(C_e' (I + G_e)^(-1) C_e)/U`.

Compute all gains from the empty retained set once. There is **no conditional
update** after selecting an electrode. Sort decreasing gain, with exact ties
resolved by ascending physical electrode ID, and retain one nested ordering.
Reuse the identical view's X Gram across its five seeds where appropriate;
only the selector cross-product changes. Preserve the same original score
formula and float64 conventions, not a target-whitened or retuned proxy.

Ordinary permutations have zero expected centered cross-covariance, unlike
forced derangements in groups of three or four. The squared gain is still
positive on average, and its null distribution can depend on X covariance,
finite-group support and target variability. The control therefore asks about
useful trial correspondence under this recipe, not an absence of all neural
information or a biological causal effect.

## True-target decoder and information barriers

Generate all25 orderings (five views times five seeds) before reading any
post-initial target row. At checkpoint32, each inner decoder uses its view's
seed-specific retained columns, but its training residuals, rank/alpha
validation loss and final refit always use **true, unpermuted Y**. An outer
ordering must not replace an inner ordering. The outer ordering then stays
fixed across all later checkpoints and nested electrode budgets for that seed.

Preserve the qualified mean-feature reduced-rank decoder, finite rank/alpha
grid and tie rules. Each control receives the same tuning opportunity as the
original marginal policy, using true-Y CV; do not force the old policy's chosen
hyperparameters onto a different retained set. At later checkpoints, retain
the existing initializer-conditioned folds: all32 initial rows are present in
each inner fitting set, with only newly acquired trials partitioned into
validation folds. No evaluation feature or target enters any selection or fit.

All80 distinct target models for a session must be fixed before its evaluation
rows are read. Five identical all-96 models can be shared across the five seeds
after exact within-run source/support equality; scarce-electrode histories and
fitted coefficients remain separate. There are100 model references per session,
not100 distinct fits. Preserve every seed, trial, direction and session.

## Scoring, reference replay and interpretation

For each matched resource cell, retain all five correct/wrong/native SSEs and
their common scoring denominator. Average the five SSEs, then derive Q and
pairing increment. **Never average predictions**, select a favorable seed or
use a different seed subset for an animal. Report seed means and all session
and animal contrasts; five controls are neither independent animals nor a
calibrated sampling distribution or confidence interval.

Use the unchanged acquired-only20-cell normalization and equal-cell,
session-within-animal, then equal-animal aggregation. After every new session's
fits are complete, read
`runs/development_search/acquired_only_attempt001/acquired_only.json`
for the existing initial-marginal and fixed-random references. The qualified
metadata/role reader enforces the current original source roles. Check the old
report's complete session/animal identities, original recipe, all resource
cells and target-size/exposure records, then require all five common all-96
score/capacity replays within the existing 1e-10 Q tolerance, including exact
physical electrode, rank, alpha and candidate-count agreement. The old
acquired-only artifact does not serialize a source-role record or prediction
arrays; do not claim bytewise role-record or prediction-array identity from
it. Save the current role record in the new report. No old outcome is a
fitting or seed-selection input.

The principal contrast is **true initial marginal minus mean target-label
control**, reported overall, for each animal, session and cell. A claim that
true initializer correspondence contributes requires this Q contrast to be
strictly positive in both animal means; ties fail. Keep the existing0.001
search-progress resolution on this unchanged support, without inventing a
biological margin, p-value or automatic promotion rule. Report original
correct/wrong pairing readouts descriptively and all fixed-prediction session
omissions without new fitting. Full-96 identity cells remain in the20-cell
aggregate rather than being selectively dropped.

The candidate and control match initializer exposure, retained-channel/trial
cost, decoder family and tuning opportunity. Each uses
`32*96 + (n-32)*E` LFP electrode-trials plus`n*U*10` target values. The fixed-random
reference pays a cheaper initializer, so its raw Q difference remains an
unequal-acquisition-cost comparison. There is no donor supervision in this
new control, unlike the learned-rule comparisons.

A positive contrast supports the usefulness of within-direction target
correspondence for this fixed, early marginal ranking. It does not identify
conditional complementarity, a biological mechanism, calibrated uncertainty
or a successful joint electrode/trial rule. A weak or inconsistent contrast
would limit attribution of the marginal policy's advantage to that
correspondence; it would not prove exact equivalence or reject all LFP
population information. The control leaves the final decoder's true signal
available, and many selected sets may overlap.

## Finite work and comparison count

If adopted, use new files and the existing qualified reader, residual scaling,
selected-rank fitting, later-CV and scoring helpers; change no earlier recipe
or output. A complete run has200 selector orderings,1,600 direction-level
permutation maps,640 actual target models,12,800 inner ridge fits and64,000
capacity candidate scores. It reads8*1056 = **8,448 fitting-trial views** plus
320 evaluation trials. The unique logical support remains the original
96 fitting plus40 evaluation trials per session. Report repeated/unique
logical bytes, CPU/wall time, memory and retained artifact bytes; no source
array or target-decoder coefficient serialization is needed.

The just-completed reverse-animal run took72.198s for520 target models and
6,912 fitting reads. This control has about23% more of that qualified workload
and small initial marginal-score work. Propose a finite full cohort on1CPU
within4GiB,270s internal/300s external, with at least five minutes remaining
after review/checks. Runtime remains unmeasured for this implementation; root
chooses the actual allocation and final bound. No new generic pilot, cache,
automatic retry/requeue, favorable-score stop or omitted seed/session is needed.

Focused fixtures should verify original-ID/call-order-independent ordinary
maps with accepted identity, direction/view boundaries, exact transformed-Y
cross-product with unchanged X Gram, exclusion of inner validation information,
true-Y decoder inputs, all25 orderings before future reads, mean-five-seed SSE
and the shared all-96 identity. Tests must address those scientific risks,
not duplicate established generic numerical checks.

One complete new control advances25 to26 valid comparisons. The reused true
marginal policy, five seeds, resource cells, inner views and reference replays
are not additional evaluations. The shared40–112 allowance, patience and
alternative-check fraction persist, and broader family/robustness coverage
remains incomplete. The four held sessions stay unopened.
