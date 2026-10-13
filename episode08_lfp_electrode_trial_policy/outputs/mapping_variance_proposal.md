# Acquired-LFP design-variance reduction versus target-only trial allocation

Prospective proposal, 2026-10-03. **Adopted; implemented, unrun.** The proposal
was prepared before the arrival-order outcomes. After that complete comparison,
root adopted this unchanged scientific recipe before any mapping fit, including
the stable squared-factor numerator below. The current development count is
19; this complete comparison would add one, reaching 20. No held opening or
named explanatory endpoint is activated.

## A concrete new scientific comparison

The current value/error utilities depend only on acquired population targets
and direction counts. They cannot identify sensor-dependent trial value. Add
one mapping-informed rule, `mapping_design_value`, whose next-direction score
uses only acquired LFP rows on that policy's fixed random electrode set.
Compare it with the existing balanced, mean-value and mean-error policies.
These are matched count/target-variance competitors, not a new family of
near-duplicate regularization settings.

Keep the exact eight development sessions,96fitting/40evaluation roles,
32balanced initializer, fixed random electrodes8/16/32/all96, existing five
scarce-set repetitions, native3bands×10bin means, population target and
32/48/64/80/96checkpoints. Use the original queue order **and all three already
declared post-initial order variants**, without selecting a favorable order
after the pending result. The first four trials per direction are unchanged.
Each policy can request only the next trial of a legal direction with count<12,
and only retained electrodes. No future neural feature, target, quality flag
or realized next-trial benefit enters a choice.

## Fixed proxy coordinates and regularization

For a selected set of Eelectrodes, x is its3Evector of mean LFP features.
From the common32initializer alone, retain the existing residual model's
direction input means, residual global mean and residual-feature population
SDs. The proxy coordinate of an acquired row of direction d is

`z = (x - mean_init,d(x) - residual_mean_init(x)) / residual_SD_init(x)`.

Exactly zero initial SD becomes1 as in the qualified fitter; no feature is
dropped. These **proxy** coordinates never change later. Neither future
features nor evaluation data update their scale, reference distribution or
regularization. Let Z0 be the32initial rows in these coordinates, and set

`M = Z0' Z0 / 32`,
`eta = 32 * alpha_init`,
`A_n = eta I + sum_(i acquired at n) z_i z_i'`.

Alpha_init is the same finite-grid alpha chosen by the unchanged initializer
decoder, once per fixed set. Eta remains that fixed positive number; no new
proxy tuning grid is introduced. M is a fixed empirical reference second
moment: it weights prediction support on the initializer's residual-feature
distribution. It is not an estimate obtained from unseen evaluation features.
All calculations use float64.

## The pre-choice score

Define the nonnegative regularized design-variance index

`V_n = trace(M A_n^{-1})`.

For a prospective row z, the exact algebraic decrease in that index after one
rank-one addition is

`g_n(z) = z' A_n^{-1} M A_n^{-1} z / (1 + z' A_n^{-1} z)`.

This follows directly from the Sherman–Morrison identity. To estimate the next
row distribution for direction d, use **only that direction's already acquired
rows**, equally weighted:

`u_d = (1/n_d) sum_(i acquired, direction d) g_n(z_i)`.

Choose the legal direction maximizing u_d; exact ties prefer smaller actual
n_d and then smaller frozen direction index. Read its next row only after
choice, update A with its observed z, then recompute the next decision. Do not
divide u_d by n_d again: A already incorporates acquired design support, and
the1/n_d factor forms the empirical expectation rather than an observation
count bonus. M and eta stay fixed. The finite same-direction empirical
distribution is a plug-in assumption about future rows, not access to those
rows or proof of stationarity. It effectively approximates a new independent
draw by sampling acquired rows with replacement, whereas the real finite queue
reveals a distinct observation. That mismatch is part of the test.

For a stable implementation, the numerator can be evaluated exactly as
`||Z0 A_n^{-1} z||² / 32`. This uses M's known initializer factor and avoids
subtracting nearly equal trace values or obtaining a tiny negative quadratic
from floating-point cancellation. It introduces no truncation, approximate
rank choice or change of proxy. An independent static equation review agreed
with the trace identity, expectation, units and disclosed model mismatch;
no numerical implementation or test has run.

The index uses standardized, dimensionless features and a **unit-noise
convention**. Multiplying every direction's index by the same positive noise
scale would not alter its choices. It does not estimate output-specific noise
or calibrated uncertainty in native spike units. Under a specified linear
homoskedastic prior model it resembles integrated posterior predictive
variance; this experiment makes no such probabilistic calibration claim.
In particular it is not entropy, Bayesian information gain, expected realized
test improvement or conditional sensor–trial coupling.

## Decoder fairness and the deliberate proxy mismatch

Use exactly the current reduced-rank direction-residual decoder, with initial
numeric rank/alpha frozen, including the resolved full-rank cap. Refit its
ordinary acquired-history direction means, residual scales, response projection
and coefficients at each checkpoint; no later CV. Every policy uses the same
fitting recipe. Evaluation uses the common initializer raw-target SD and mean
null, with the same forty trials, native-target SSE and wrong-pair diagnostic.

The allocation proxy is **full linear design support in frozen initial feature
coordinates**, whereas the actual decoder has a selected response rank,
updated fitting transforms and the normalized-loss penalty n*alpha_init.
The proxy instead fixes eta=32*alpha_init. It also omits uncertainty in the
decoder's direction means and any heteroskedastic/noise covariance. Thus its
algebraic index reduction is exact, but interpreting it as the actual decoder's
predictive-variance reduction would be wrong. A win would support usefulness
of this bounded mapping-informed heuristic; a loss could reflect these
mismatches, noisy same-direction distribution estimates or irrelevance of the
selected design directions. This first comparison cannot distinguish them all.

The target-only mean-value and mean-error competitors assess whether feature
design information helps beyond direction variance/count balancing. They are
not perfect mechanistic controls for M, shrinkage or decoder mismatch. A
formal feature-association control or matched uncertainty model would require
a later explicit comparison; none is quietly included as a new endpoint here.

## Reuse, reporting and accounting

Run the new policy on all four fixed order conditions. Reuse the **completed,
matching** balanced/mean-value/mean-error result packets for each order rather
than refitting their unchanged models. The original packet is
`runs/development_search/trial_allocation_attempt001/trial_allocation.json`;
the three-order packet will be
`runs/development_search/arrival_order_attempt001/arrival_order.json` only
after it is complete and root permits interpretation. Prior scores enter
trusted reporting/replay after all new fits, never the utility or fitter.
Require identical roles, initial numeric rank/alpha, normalization, electrode
seeds and orders before using a reference. Verify new32/96endpoint replay in
the existing1e−10 tolerance; final acquired arrays must equal the original
canonical96-ID pool. Preserve all old attempts unchanged.

Report all four orders, every20cell/session/animal, the informative intermediate
profile, fixed-prediction influence, and contrasts against each existing policy.
Give orders equal weight, disclose the original and new-order histories, and
never select the best order, animal or electrode set. Utilities and chosen
directions can differ with Ebecause their features differ; that is an input
dependency, not evidence that the differences predict conditional trial value.
No named coupling margin, causal interpretation or participant-population
claim is activated. Repeated use of the same development evaluation remains
selection-exposed. Four held sessions stay closed.

At E,n all policies use n*E LFP electrode-trials and n*U*10target values;
charge the new proxy's actual matrix work and source reads separately. Reuse
does not erase the reference policies' recorded acquisition costs. Propose
**one** new scored mapping-informed comparison against the shared112ceiling,
not four orders or twenty cells as separate methods. Its eventual count depends
on the pending robustness completion and must be adopted before launch.

Sharing new-policy initial/final fits across four orders requires128initial
CV fits (2560inner systems),1792actual refits,36,864fitting-trial reads and
320evaluation reads. The selected four-order policy performs
**32,768 pre-choice proxy updates** (8sessions×16sets×4orders×64choices).
Use rank-one inverse/index updates or equivalent stable solves, with a focused
independent trace-difference check before execution; no generic preflight.
Prepared implementation: `tools/mapping_variance.py` and
`tools/development_mapping_variance.py`, with four focused fixtures in
`tools/test_mapping_variance.py`. The inverse, the initializer-to-acquired
cross-products and acquired quadratic diagonals receive exact Sherman–Morrison
updates. Utilities use squared cross-product columns, without trace subtraction,
clipping or an adaptive regularizer. The decoder is unchanged. Reference files
are opened only after every new fit, and their channels, queues, numeric rank,
alpha, scoring normalization and shared endpoints are checked before reuse.

The completed arrival panel cost 570.718 seconds. This new panel has 1792
refits rather than 3712 and 36,864 fitting reads rather than 77,824, plus the
32,768 bounded matrix updates. Prepare 1 CPU / 2 GiB / 15 minutes with an
840-second internal bound; actual cost remains unmeasured. Root owns the
execution/resource selection, tests and final implementation review.
