# Multichannel LMP and fixed low/high-frequency ridge comparisons

Proposed2026-10-03 after the complete first spatial-mean LMP result. **Adopted
prospectively by the coordinator; implemented.** The unchanged three-bank pilot
completed22:44:55UTC; the full sixteen-session panel completed22:48:58UTC,
adding3 charged trials (total4), with no pilot-score-based recipe choice. None
of these three recipes met the adopted practical margins. No new source values were read to prepare this
proposal. The failed mean-LMP recipe remains charged and unchanged. These are
three prospective comparisons within the existing channel-PCA/band/ridge
grammar, not a replacement of the first result or a search termination rule.

## Scientific change and documentary motivation

The first spatial mean gave primary−8.77e−7 and identity−2.78e−7 R2. Test
whether preserving electrode contrasts supplies information that averaging
discarded, before changing the baseline, window or predictor family. The
paper's [LFP-processing Methods](https://elifesciences.org/articles/73155)
describe common-average de-referencing followed by temporal centering/notch
filtering and linear LMP averaging. Under a common reference set and identical
linear operations, averaging that complete set removes its signal exactly.
This is an algebraic possibility, not a measurement that the released86/other
retained channels have zero mean. Missing/reference channels and implementation
details can leave a nonzero mean. Standardizing a nonzero small signal alone
cannot establish that it is meaningful or only roundoff.

The stored power bands are nonlinear energy summaries; common referencing
does not imply their across-channel mean is zero. A multichannel low/high
comparison therefore also distinguishes slow field-like features from the
100–400Hz branch for which spike contamination remains a required concern.
No low-frequency label will depend on spike quality or high-band values.

## Three fixed recipes proposed for selection

| Recipe | Guide-defined block | Change from its reference |
| --- | --- | --- |
| `lmp_full_channel_ridge_v1` | LMP `(0,0)` | Replace the scalar spatial mean with the complete electrode feature span. |
| `low_full_channel_ridge_v1` | `(0.5,4)`, `(4,8)`, `(8,12)` Hz | Replace LMP with the fixed0.5–12Hz block under the same spatial/model recipe. |
| `high_full_channel_ridge_v1` | `(100,200)`, `(200,400)` Hz | Replace that slow block with the fixed100–400Hz block. |

The low block collects the three published bands below13Hz; the high block
matches the contract's explicit100–400Hz contamination boundary. The12–100Hz
bands are not silently included, optimized or declared uninformative. Those
bands and all-band/combined variants remain possible later declared trials.
The literal guide pair `(0.5,4.0)` and every listed companion band are present
in the existing completed `runs/source_support/attempt002/source_support.json`
metadata, including the predetermined Chewie_20150313 pilot; the source adapter
records guide pairs directly without reclassifying their lower edge. The
reader nevertheless requires the exact pair per original channel/trial, not
an assumed `(0,4)` substitute.
The existing five lambdas are unchanged; no rank, channel-count or bandwidth
grid is added. Different feature widths are disclosed; capacity-matched and
matched-budget tests remain required before a strong spectral conclusion.

For each session use the identical stable LMP physical-channel set from the
first recipe. Before reading values, require each proposed block to have
exactly one guide entry per named band for every such electrode throughout
the selected trials. A missing/ambiguous band stops that proposed comparison
for review; do not drop channels, trials or sessions to improve support. This
preserves the LMP anchor's electrode opportunities and fixes the same channels
across low/high comparisons without using any high-band values to select the
low feature. Report exact electrode IDs and guide rows.

At each of the eight direction splits, flatten only the legal fitting trials
and fifteen bins; estimate a separate population mean/SD for each electrode-
band feature. Exactly zero SD becomes1 and is reported, with no column deletion.
Compute a full, unwhitened orthonormal PCA rotation on these standardized fitting
features and keep **every feature-space basis vector**, including null-space
directions. Do not whiten or restandardize PCA scores: retain the standardized
input metric exactly. Apply that same rotation to evaluation features. No variance
cutoff, numerical-rank selection or outcome-dependent truncation is used.
Ridge over this complete orthonormal basis is algebraically equivalent to
ridge on the standardized per-electrode features; the PCA representation
fits the declared grammar while preserving all spatial contrasts. It adds no
effective complexity relative to full-channel ridge. **The coordinator explicitly
selected the equivalent direct standardized-feature solve. No numerical PCA
rotation is computed or claimed.** The rank-deficient orthogonal-invariance
fixture checks the selected equivalence, including retained null directions.

Current-bin inputs, native values, source feature definitions and fifteen
movement-onset samples0:+14 remain fixed. No further filtering, squaring,
log-power conversion, smoothing, spike selection or interpolation is introduced.
This continues to be an offline released-feature analysis.

## Everything else remains fixed

Use the exact sixteen-session outer folds, original complete trials, canonical
angles and all eight leave-one-direction-out splits in
`primary_reference_proposal.md`. Use the same equal-direction first harmonic,
common training residual intercept, native X/Y residual target, pooled joint
session SSE denominator, whole-trajectory exact wrong-pair mean and equal-
session then equal-animal aggregation. All five lambdas retain the mean-loss
objective and ties prefer stronger shrinkage. Session-local transforms and
coefficients see only the seven fitting directions. Each outer session's
lambda uses only the other fifteen score-bank rows.

The complete data width changes, not the biological unit, outer selection
authority or denominator. Reuse a baseline only after exact trial/order/
direction/window equivalence; always retain its common residual intercept.
Use the same0.01 primary and0.005 identity operating margins, animal signs and
session majorities. No candidate promotion until every outstanding required
falsifier, influence rule and minimum coverage condition is satisfied. A
high-band-only success cannot establish general low-frequency LFP information.

Report every session/animal for every recipe, even if unfavorable. Each complete
sixteen-session recipe is **one additional valid scientific trial**; three
complete recipes would bring the running count from1 to4. Their baseline,
five penalties, PCA rotations, pilots and technical retries do not inflate the
count. These fixed panels cover spectral/linear contrasts, not the required
reduced-rank/nonlinear families or the whole open search. Family selection
after viewing the panel remains development adaptation, not fresh confirmation.

## Source-scale diagnostic selected within the next feature preparation

The coordinator selected a bounded diagnostic emitted during the next
authorized feature preparation, not a separate source replay or prerequisite
to the multichannel test. For each session's exact primary LMP channel/trial/
window set let X have rows for all selected trial/bin observations and columns
for the frozen physical electrodes, and m be its rowwise spatial mean. Report
stored source dtype(s), `sqrt(mean(m^2))/sqrt(mean(X^2))`, and
`SD_population(m)/sqrt(mean_e[Var_population(X_e)])`, including each numerator
and denominator and exact-zero flags. A zero denominator yields a disclosed
undefined ratio, not channel exclusion or a fitting rule. These ratios quantify
observed spatial cancellation relative to electrode amplitude/variation; no
threshold, float-precision claim or outcome-based rank cutoff is introduced.
No per-trial signal, behavior correlation, fit or session selection is required.
The diagnostic is selected and implemented but remains unrun. The documentary
CAR argument motivates the comparison without needing a perfect cancellation
proof. Diagnostic output must not alter model preprocessing or eligibility.

## Selected implementation and resource handoff

Preserve `primary_reference.py`, its runner and all first-recipe banks. The
separate `multichannel_reader.py`/`run_primary_multichannel.py` reuse checked metadata/event
mapping and baseline/score helpers. Retain guide-only qualification before
each recipe's values; source spike fields remain unopened. One session at a
time needs at most about15,570rows×288features (roughly36MB float64) for the
largest observed trial count with96electrodes, plus small fitting workspaces.
This estimate is not evidence that every session has96channels or that source
I/O/runtime scales identically. Propose one CPU,4GiB and a five-minute full
Chewie_20150313 pilot for the three fixed banks, preserving all completed banks.
Measure actual compute/I/O cost before selecting the remaining fifteen-session
resource bound; do not use the pilot's scientific scores to abandon a recipe.

Four focused tests in `test_primary_multichannel.py` verify full-basis PCA/direct-ridge equivalence with a
rank-deficient multichannel fixture; exact guide-band/electrode ordering; and
retention of the common behavior intercept under a constant multichannel
feature, plus each newly summarized recipe's outer-row exclusion. Reuse the
passing wrong-pair equation. Independent proposal and four-file implementation
reviews by agentA found no concrete defect. No tests or source values were
opened during review. Root owns
all tests, source-value access and scheduler execution.
