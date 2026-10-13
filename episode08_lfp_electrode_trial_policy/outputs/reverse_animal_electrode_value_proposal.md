# Reverse-animal transfer of the fixed learned electrode-value rule

Prepared 2026-10-04 after 23 complete development comparisons. **Adopted by root
at approximately 00:20 UTC before implementation or new source reads/fits.**
The exact recipe below is fixed for the selected complete comparison. The four
held sessions and optional named explanatory endpoints remain unopened.

## Authority and the distinct question

The operative [open-exploration amendment](open_exploration_design.md) allows
transfer and development-trained selection, and requires session-cross-fitted
offline training labels. `SEARCH_POLICY.yaml` names reverse-animal transfer
and training on one animal and applying to the other among the original
reference's alternative checks. That reference remains optional; this proposal
instantiates the same useful check in the adopted 32-trial acquired-only branch,
without importing its 16-trial grid, endpoint or 0.005 margin.

The completed seven-donor learned rule underperformed initial marginal
selection and its label control in both animal means. This proposal asks a
different question: does the fixed relationship between early electrode
summaries and later single-electrode gain transfer when **every training donor
belongs to the other animal**? The prior leave-one-session-out rule still used
three sessions from the target animal and cannot answer this question.

Both transfer directions are required together: four Chewie sessions train
the rule applied to all four Mihili sessions, and four Mihili sessions train
the rule applied to all four Chewie sessions. Neither direction, target session
nor seed is chosen from its score. These are the same two source-exposed
biological animals, not eight independent animals or newly unexposed data.
The current-session decoder still requires acquired target-session responses;
only the ranking relationship transfers. This is not zero-shot decoding.

Removing same-animal donors also changes donor count/composition from seven
to four. A difference from the previous ranker cannot isolate animal identity
from training amount, donor distribution or estimator variance. No extra
same-animal-only arm or donor-count grid is added here. This is a bounded
transfer check, not a rescue of the failed seven-donor recipe or proof of
conditional electrode complementarity.

## Fixed candidate, information views and control

Candidate ID: `learned_electrode_value_reverse_animal_v1`.
Control ID: `learned_electrode_value_reverse_animal_labels3_v1`.

Keep the three predictors, signed single-electrode value labels and all
numerical conventions in the adopted
[learned-value proposal](learned_electrode_value_proposal.md): residual marginal
gain, mean three-feature direction repeatability, and mean residual block
redundancy with the other 95 electrodes. These are not artifact measurements
or conditional removal values. There is no new feature, transform, penalty
search or model family.

Use the completed full report's `donor_training_records` for all eight sessions
and five views (four 24-row initializer inner-training views and the outer
32-row view). For each recipient animal and view, select exactly its four
other-animal records **before accessing numerical predictor or label fields**.
Exclude all same-animal records, including the target's own, from training,
weighted means/SDs, labels, coefficients and control permutations. Reject
missing/duplicate/wrong-view donors; do not replace one with a same-animal
record. Animal identity comes from the fixed role manifest, not filename
guessing or a score-derived grouping.

The stored donor labels were fitted on each permitted 24/32-row view and
evaluated on that donor's separate later 64 fitting trials. They never used
its 40 evaluation trials. Preserve the exact qualified donor objective
`||Y_residual-X_residual B||_F^2/n + 1*||B||_F^2`, implemented by
`ridge_training.ridge_coefficients` with the existing direction-residual X
scaling and raw per-unit Y SD. The signed label is the direction-only minus
electrode residual SSE divided by the original view-scaled later-target
normalizer. Preserve the original separate ten-bin time mean for each native
target unit; never average neurons or align units between animals.

Fit the same three-predictor ridge ranker with unpenalized intercept and
`weighted mean squared label error + ||beta||_2^2`. With one donor animal,
give its four sessions equal weight and each session's 96 electrodes equal
weight: each record has weight `1/(4*96)`. Weights sum to one; do not retain an
extra factor one-half that would silently change effective regularization.
Ranker predictor centering/SD are donor-only; zero SD becomes one. Keep
descending predicted value and ascending physical-channel-ID exact tie breaks.
The five donor-view states can be shared by all four target sessions of the
recipient animal because their training inputs are identical.

For the three label-control replicates 0/1/2, use the identical ordinary
within-donor/view label permutations from the previous recipe, with literal
`ep08-learned-electrode-label-v1|20261003|<donor_filename>|<view>|<replicate>`
and Python version-2 string seeding. Shuffle a fresh ascending-channel index
list once; accept identity and fixed points. View is `outer` or unpadded
`0`/`1`/`2`/`3`. The target filename and recipient scores never enter the seed.
These common donor permutations make the new donor-composition contrast easier
to interpret without generating favorable new control labels. Record all maps.
The control retains the same four donors and target acquisition costs and
changes only their predictor-to-value-label association.

## Target acquisition, fitting and scoring remain fixed

Keep the exact eight development sessions, original 96-trial fitting queues,
40 evaluation trials per session, all 96 eligible physical electrodes, current
three-band ten-bin LFP summaries, and original native population targets.
Acquire 32 initial trials on all channels, choose nested retained prefixes
8/16/32 once, and use balanced original queues through totals
32/48/64/80/96. Include the common all-96 branch and all 20 cells. No later
reranking, channel backfill, extra initializer or adaptive trial action occurs.

For each initial inner fold, target selection sees only its 24 fitting rows
and the matching donor-view ranker; neither target validation X nor Y enters
the ordering. The outer ordering uses all 32 acquired rows and the outer
donor view. Freeze all 20 target orderings (five views times four methods)
before any later target source read. Subsequent CV keeps the original
initializer-conditioned fitting protocol. Preserve the reduced-rank decoder,
rank/alpha grid, tie rules, training-only transforms and true target labels.
Every scarce retained set has its own fitted decoder coefficients.

Fit all 65 distinct target models for a session before reading its evaluation
rows. Share only the identical five all-96 models across the new candidate
and controls, with exact within-run source/support equality before reuse.
Average the three control correct/wrong/native SSEs within a matched cell,
using the identical fixed normalizer; never average predictions. Keep every
seed score, all cells and original correct/wrong trial pairing diagnostics.

Use the existing acquired-only Q normalization and equal-cell, equal-session
within animal, then equal-animal aggregation. Report transfer minus its
own label control, initial marginal selection, previous seven-donor learned
rule and previous seven-donor label control. Fixed random is context with a
cheaper initializer, not an equal-acquisition-cost comparator. Report both
transfer directions, every session and all fixed session omissions without
refitting. Omissions are sensitivity descriptions, not independent training
replicates or a newly fitted leave-one-donor-out analysis.

For a claim of useful learned-label transfer, require the candidate-minus-own-
control Q contrast to be positive in both recipient-animal means; ties fail.
For a claim of improvement over the paid within-session marginal rule, require
the corresponding contrast to be positive in both animal means. Report the
existing 0.001 search-progress resolution on the unchanged support; introduce
no significance test, new biological margin, automatic promotion or stopping
rule. Positive transfer alone would not establish utility over marginal
selection. Failure addresses this fixed feature/label/ranker recipe under
these two directions, not all transferable neural information.

## Reuse, reference access and honest cost accounting

The declared numerical training input is
`runs/development_search/learned_electrode_attempt001/learned_electrode_value.json`,
specifically its 40 `donor_training_records`. Validate the completed status,
recipe, all eight fixed identities, 96 channel IDs, five exact views, original
label objective/support and finite required training fields. A narrow loader
returns only these records. The combined JSON must be parsed before fits, so
do **not** claim the old report or its outcome bytes remained unopened.
Its outcome fields are not allowed ranker or decoder inputs; use them for
comparison only after every new target fit has completed. This is a logical
numerical boundary, not OS isolation or newly blinded data.

Read the old acquired-only outcome reference only after all new target fits.
Authenticate source roles, electrode identities, folds, checkpoints and target
support; require the common all-96 scores/rank/alpha choices to replay at the
existing `1e-10` Q tolerance against both applicable completed packets.
They contain scores, not old prediction arrays; make no prediction-array
identity claim. No old target model is silently reused from a score packet.

This reuses already paid donor supervision rather than making it free.
For each target the rule has access to four other-animal full 96-trial,
all-channel fitting pools through their fixed derived labels/features,
including the later 64 trials. Record that inherited calibration information
separately from new source reads. The full stored donor pass required 3,840
single-electrode fits; the previous separate pilot caused another 3,360,
so the shared history retains all 7,200 incurred donor fits and repeated reads.
Do not recompute or recharge those as new fits; do not erase their historical
cost. The new ranker uses no same-animal training data, but the shared evaluator
has previously encountered every development session in other roles.

Candidate versus own control matches donor information, compute opportunity
and target cost. Versus marginal selection, only target-session acquisition
cost matches, namely `32*96 + (n-32)*E` LFP electrode-trials plus `n*U*10`
target values; total training information/cost does not match. Versus the
previous seven-donor rule, target cost matches but donor amount/composition
does not. Record new/inherited bytes, actual repeated target reads, unique
source exposure, CPU/wall time, memory and retained artifact size separately.

## Finite implementation proposal and search accounting

Use new files while reusing unchanged qualified reader, donor-feature,
target-fitting and scoring math. There are 40 distinct donor ranker fits
(two recipient animals times five views times candidate plus three controls),
zero new donor-label fits, 520 distinct target models and 10,400 inner fits.
With the existing trajectories, new source reads are 8*864 = 6,912 fitting
trial reads and 320 evaluation trials. Target-specific rankings still number
160 despite sharing identical donor ranker states. Keep all counts explicit.

The complete previous family cost was 83.538 seconds including donor fitting
and all 520 target models. Reusing its donor records removes that fitting work
while preserving the target workload. A new generic pilot is unnecessary.
Root selected one finite full-cohort run on one CPU within 4 GiB,
270 seconds internal / 300 seconds external, requiring at least five minutes
remaining after review and focused checks. This estimate is not measured runtime for
the new implementation. Preserve partials and failures; no automatic retry,
requeue, omitted direction/seed or score-based early termination.

Focused fixtures should establish other-animal exclusion before numerical
access, matching 24/32 views, unit-sum four-session weights with unchanged
penalty, unchanged ordinary donor permutations, target-only ordering timing,
mean-SSE controls and exact common all-96 replay. Existing qualified math does
not need a duplicate generic qualification chain.

Two complete panels advance 23 to 25 valid comparisons. Transfer directions,
rankers, seeds, views and cells are not separate valid evaluations. This
addresses reverse-animal transfer of the learned selector; shallow learned
ranking, quality-matched swaps, sensor-dependent trial benefit and remaining
coverage still require distinct work. The shared 40–112 allowance, patience,
alternative-check fraction and four unopened held sessions remain unchanged.
