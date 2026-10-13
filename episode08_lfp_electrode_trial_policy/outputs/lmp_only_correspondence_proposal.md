# Does initializer correspondence remain useful using only native LMP?

Prepared 2026-10-04 after26 completed development comparisons. **Root adopted
the clarified recipe at05:40UTC** after root and independent scientific review.
New implementation and focused checks are being prepared before bounded
execution. Adoption is not an empirical launch or completed result; no
held-session opening or named explanatory endpoint is activated.

## Question and limits

The completed three-band initializer control gives true-minus-randomized
selection Q+0.005381 overall, positive means in both animals, but a small
Mihili effect that reverses after one session omission or against one seed.
Selection and decoding both currently use native LMP plus100–200 and200–400Hz
power. Their positive correspondence contrast does not establish whether that
result survives without the two high-frequency power features.

Run one native-LMP-only marginal-selection candidate with its matched five-seed
initializer target-label control. Both selector and decoder receive only the
native guide pair **(0,0)** already designated LMP in the qualified reader.
Do not reconstruct a waveform, infer an undocumented filter cutoff, refilter
data, replace native LMP with another low-frequency feature, or modify targets.
This is sensitivity to removing the two selected high-frequency power bands.
Native LMP is not certified spike-free by this operation. A retained contrast
would show that these power features are unnecessary for this narrow result;
loss of the contrast would not establish spike contamination as its cause.

This branch is within the operative open-exploration allowance and requested
low-frequency/spike-contamination alternatives. It does not activate the named
conditional-removal, next-direction-value or sensor–trial-coupling endpoints.

## Fixed candidate, control and information state

Candidate: `initial_marginal_lmp_only_v1`.
Control: `initial_marginal_lmp_only_target_labels5_v1`.
Fixed control replicate labels are0,1,2,3,4, averaged as one comparison.

Preserve all eight development sessions, both animals, original96 fitting
trials,40 evaluation trials, initial32 trials, four initial24-row inner views,
same balanced direction queues, all96 physical electrodes, nested retained
counts8/16/32 and common96, and checkpoints32/48/64/80/96. Preserve each
original native population unit's ten-bin time mean over150–450ms. Do not
average neurons, align units across sessions, choose a new window, omit an
unfavorable seed/session, or inspect any of the four held sessions.

At each initializer information view, every physical electrode has **one real
input coordinate**, its native LMP ten-bin mean. The reader selects only guide
(0,0) numerical rows for the exact permitted trial IDs/channels. It can read
the unchanged full guide metadata to verify physical identities; no100–400Hz
neural values are needed. No three-column padding, high-band zeroing,
duplication of LMP coordinates or silently retained factor-of-three indexing.

The original32-trial initializer remains paid on all96 physical electrodes,
now one band each. All30 orderings per session (five true views and25 control
views) are frozen before any later target row or LMP observation is read.
Each outer ordering stays fixed through later acquisition; no discarded-channel
backfill, reranking or trial adaptation is introduced. All scarce-channel
histories acquire only their own selected native LMP rows.

## Scalar-group selection and matched randomization

Reuse the qualified true-fitting direction means, residual-X population SD,
raw-Y population SD and residual intercept convention. Zero SD becomes one;
no channel/unit is dropped. Inner normalization and selection see only that
view's24 fitting rows, excluding its eight validation rows.

For standardized, direction-residual and centered matrices X and Y, native
LMP electrode e has scalar `g_e = X_e'X_e/n` and row vector
`c_e = X_e'Y/n`. Its exact alpha1 mean-loss marginal gain is

`gain_e = (c_e c_e') / ((1 + g_e) * U)`.

All scores are computed from the empty retained set once. Sort decreasing
gain, with exact ties resolved by ascending physical electrode ID, and take
one nested top32 order. This is the one-coordinate version of the original
marginal objective, not the same three-coordinate model with hidden zero
features or a newly optimized regularization choice.

For every control view/seed, reuse the completed initializer control's exact
ordinary original-ID within-direction mapping and seed literal:

`ep08-acquired-marginal-target-label-v1|20261004|<filename>|<view>|<replicate>|<direction>`.

View is outer or0/1/2/3; direction is0through7. Canonical sorted recipient
IDs are paired with one ordinary shuffle of their donor-ID list, preserving
identity/fixed points and entire target vectors across all native units.
Only `c_e` uses permuted normalized Y. The true view's normalization and
scalar X Gram remain fixed. Permutations are not redrawn to improve a result.
Save all original recipient/donor IDs and literal seeds. Pairing the maps
across spectral representations improves comparability without making the
two fitted model capacities identical.

## True-target decoder and capacity

Each candidate/control model uses true, unpermuted Y for fitting, CV and refit.
The initial nested decoder uses the view-specific electrode order; an outer
order may not replace an inner order. Later initializer-conditioned folds
retain every initial row in fitting and validate only later acquired rows,
using the same original queue-position/direction fold assignment.

Use the existing reduced-rank ridge objective, alpha grid
0.0001/0.01/1/100/10000, training-only scaling and rank projection. With E
retained electrodes the real input dimension is **E**, rather than3E. Legal
ranks remain1/2/4/8 when no larger than the minimum of actual input dimension,
target-unit count, each inner-fitting size minus8, and outer-fitting size
minus8, plus `full`. Exact CV ties use the established smaller-rank then
larger-alpha rule. At E8, rank8 and full may represent the same untruncated
solution; preserve the declared grid and disclose that structural redundancy.
Do not pretend the full three-band capacity is matched by this sensitivity.

Candidate and its LMP-only control have the same real input dimension, target,
training information, tuning grid and evaluation support. Across LMP-only and
the completed three-band panels, feature information, dimensionality and
regularized model capacity differ. Their Q difference is a whole-pipeline
spectral sensitivity, not an isolated effect of frequency content or spike
contamination.

The common all96-LMP fit at each checkpoint can be shared across the six
branches after identical within-run source/support is established. There are
95 distinct target models and120 model references per session. All95 models
must be fixed before reading the40 evaluation rows. No prior three-band
coefficients or scores select any LMP hyperparameter, electrode, seed or stop.

## Outcomes, references and interpretation

Use the unchanged true-target Q/SSE, exact expected wrong-pair score,
20 equal resource cells, equal sessions within animal and equal animal means.
Retain all seed correct/wrong/native SSEs and common denominators; average
five SSEs, never predictions or favorable seeds. Keep direction-only means
and the common full96 cells in the20-cell aggregate.

Two distinct prospective screens are descriptive development decisions:

- **Correspondence usefulness:** LMP true-selector Q minus mean LMP control Q
  is strictly positive in each animal mean; ties fail.
- **Residual information:** LMP true-selector Q minus matched direction-only
  Q is strictly positive in each animal mean; ties fail.

Report every session, animal, seed and resource cell and all fixed-prediction
session omissions. Preserve the existing0.001 search-progress resolution;
neither screen creates a biological margin, calibrated p-value, equivalence
decision or automatic promotion rule. A small or omission-sensitive animal
contrast is reported as such even when its mean passes the sign screen.

Only after all new fits, read the completed
`runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json`
and its already declared original acquired-only reference. These provide the
unchanged three-band true/control/random display panels and recorded source
roles. Require matching original session/animal/trial IDs, physical electrodes,
native target-unit count and numeric target exposure, checkpoints and CV role
specification. The historical role record does not serialize native unit-guide
IDs; do not claim old/new unit-ID equality from it. The current qualified
reader still checks identical native unit guides across its fitting and
evaluation trials, with no averaging or reordering of those units.
The initializer report saves current role records, all seed maps, scoring
denominators and direction-only Q, so those equal-role/target-only quantities
can be checked against the new run. Use the existing1e-10 absolute tolerance
for dimensionless score comparisons and1e-10 relative/absolute tolerance for
denominators. For every view and seed, require exact original-ID map equality.

All96-LMP models must replay **within the new run** across candidate and all
five controls, with identical channel/rank/alpha/candidate count and scores.
They cannot be required to reproduce old three-band predictions or Q because
their representation is intentionally different. Compare the three-band and
LMP true/control contrasts descriptively, including the difference of those
contrasts; no new favorable comparison becomes an endpoint after scoring.
The historical fixed-random reference still has a cheaper initializer and a
different three-band decoder, so it is not called a matched LMP comparator.

If both screens survive and both animal/session sensitivities are convincing,
the result supports initializer correspondence and residual population
prediction using native LMP under this recipe. It does not show which
physiology supplies the signal, establish causal channel importance, certify
an independent third animal, or rescue the unsupported joint rule. If a
screen fails, distinguish limited LMP-only evidence from universal absence
of low-frequency information or proof of a high-frequency artifact.

## Exposure, finite execution and accounting

Each method pays `32*96 + (n-32)*E` LFP electrode-trials. Its actual LFP numeric
element cost is that quantity times10, versus times30 for three bands.
Target numeric elements remain `n*U*10`. Separate physical-channel exposure,
numeric bandwidth, source bytes and compute; the same E does not imply the
same acquired feature bandwidth. Save repeated and unique logical source
bytes using actual one-band dtype/shape, along with wall/CPU time, peak memory
and retained artifact bytes. The original Exposure helper hardcodes three
bands per unique channel, so a new local one-band counter must use10 values
per LMP electrode-trial and the actual dtype; do not reuse its factor of30.
No raw arrays or target-decoder coefficients
need serialization.

A complete cohort has240 selector orderings,1,600 direction permutation maps,
760 distinct target models,15,200 inner ridge fits and76,000 capacity scores
when all recorded native-unit supports retain the five legal ranks. It reads
`8 * (32 + 19*64)` = **9,984 fitting-trial views**, plus320 evaluation trials.
The common physical support stays96 fitting and40 evaluation trials/session.
Unexpected missing support or a reduced legal rank menu must stop with an
explicit report, not silently remove a session or pad features to match counts.

The completed three-band control took99.608seconds for640 models. This plan
has about19% more target fits, one third of the LFP feature dimensions and
about18% more fitting reads. Propose one finite full-cohort execution on1CPU
within4GiB,270seconds internal/300seconds external, after a focused reader/
scalar-selector/rank/barrier review. This is an estimate, not a measured LMP
runtime; root selects the actual bounded allocation before launch. No generic
new pilot, full-band source copy, automatic retry/requeue or partial scientific
charge is needed. Earlier successful code paths and tests are reused.

The new reader/trajectory and selector-aware initial fit require explicit
one-coordinate indexing; preserve all old three-band files. Focused fixtures
should establish LMP-only numeric source selection, true raw-unit target
identity, direct scalar alpha1 gain, unchanged exact control maps, inner-view
boundaries, true-Y fitting, legal rank based on E rather than3E, no high-band
padding, all30 orders before future reads, and95 models before evaluation.
Existing alpha/rank projection, later-fold and scoring mathematics are reused.

One complete candidate plus one mean-five control adds **two** comparisons,
26to28. Seeds, spectral contrasts, shared full96 fits, unchanged references,
checkpoints and replay checks are not extra evaluations. Consumed resources
remain in the shared40–112 search allowance and1200CPU-hour/120-hour budget;
family coverage and stopping conditions remain incomplete. No held session
opens and no new independent-animal evidence is created.
