# EP05 coverage after the completed two-component PLS comparison

Prepared 2026-10-04 from the released complete PLS compact report, the existing
coverage audit, the adopted PLS proposal and the original contract.
**ADOPTED by root at 21:53 UTC on 2026-10-04** for implementation, focused
checks and the fixed resource-only pilot after review. The fifteen-session
remainder requires its later measured resource selection. No amendment,
promotion or stopping is adopted. Root owns scientific execution and accounting.

## Current evidence and what is no longer an open coverage question

The [earlier coverage audit](primary_post_intersection_coverage_audit.md)
ended at comparison 37. Root confirms the subsequent complete chronology:
38 tested early/late blend allocation; 39–40 tested the actual blend's I/R
support stress; 41–42 tested weighted-history RBF and its pairing control;
43–44 tested genuinely deflated two-component PLS and its full target-label
control. None improved the constant W/HW blend's original-support primary
incumbent, 0.0026788544310999856. These options are completed evidence, not
missing families to repeat. The earlier audit's original coverage,
adaptive-round and leader/challenger findings remain applicable.

The [released PLS evidence](runs/adaptive_search/primary_history_pls_summary_attempt001/compact_evidence_attempt001.json)
has primary increment 0.00036033660315523165 and identity increment
0.001405621668816818. Both original practical gates fail. Primary gains are
positive in 5/6 Mihili and 8/10 Chewie sessions, both animal summaries, and
all within-animal omissions. All 16 outer selectors choose the original
largest penalty, 10. All 128 candidate and 1,024 shuffled-target fold fits
retain two components; this result is not a rank-zero fallback.

PLS loses to independently selected HW by 0.0019809178700217526 primary and
0.004208762298526082 identity, and to the constant W/HW blend by
0.0023185178279447544 primary and 0.00315377304995635 identity. Both animal
means and every within-animal omission favor each preserved anchor on both
endpoints. This retires the tested compact predictor as an improvement over
those references. It does not reject every PLS rank or all supervised
compression. Choosing lambda 10 does not authorize a larger-penalty grid,
score normalization, component expansion or a different deflation algorithm.

PLS exceeds its mean full target-label control by 0.000673050383954537 primary
and 0.0011233698170851326 identity. Both animal contrasts are positive for
every one of the eight individual draws. As the
[adopted PLS proposal](primary_history_pls_proposal.md) explicitly states,
that comparison changes the labels used for **both** supervised projection
construction and the ridge head. It does not tell us whether the correctly
paired projection itself adds useful information once a downstream head can
learn from true fitting targets.

## Recommended remaining question: projection correspondence with a true head

Recommend one narrow alternative-explanation comparison: retain the completed
eight shuffled-target PLS projections and fit their ridge heads on the original,
unshuffled native fitting residuals. This is a **basis-only shuffle diagnostic**,
not a new candidate or a signal-free decoder.

The question is consequential for the interpretation of the completed result:
does selecting the two-component projection with correctly paired training
targets provide an advantage over the already specified correspondence-shuffled
projections when both downstream heads receive true targets? Separately, how
much of the existing full-target control deficit disappears when its head is
allowed to learn the true targets on exactly the same projected features?
This explains what the observed positive PLS/control contrast can support;
it is not an attempt to rescue PLS's failed absolute gates.

The motivating directional prediction is that restoring true head targets will
improve primary and identity effects over the existing full-target controls in
both animals. Whether the actual PLS projection additionally passes the frozen
directional comparison against this diagnostic is the unresolved test. No
equivalence, negligible-difference margin or projection-mechanism attribution
is introduced if that comparison fails.

This use is distinct from the completed stationary, unweighted
[covariance-SVD basis-only control](primary_covariance_svd_proposal.md).
The current transform operates on the original HW weighted historical state
and performs successive regression deflation. The old result cannot isolate
the present PLS projection's contribution. It is also distinct from rerunning
the completed HW capacity control or changing the PLS rank. The original
training-only PLS/ridge and alternative-explanation grammar permits the
comparison prospectively; it is optional science, not a missing mandatory
family or a requirement created by the patience count.

## Exact proposed operator and reuse boundary

Proposed diagnostic ID:
`lmp_history4_repeatability_weight_pls2_basis_labels8_true_head_v1`.

Let X be the unchanged weighted HW fitting input, U the original native fitting
residual, T the full fitted sequential PLS transform including score centering,
and U_r the exact existing whole-trial label permutation for draw r. The three
arms are:

| Arm | Projection | Ridge-head fitting targets | Status |
| --- | --- | --- | --- |
| P: paired PLS | T(X,U) | U | Completed; reuse without credit |
| F_r: full target-label control | T(X,U_r) | U_r | Completed; reuse without credit |
| B_r: proposed basis-only diagnostic | **The same stored T(X,U_r)** | **U** | The only new fitted head |

Preserve all 16 sessions, 6 Mihili/10 Chewie grouping, complete original trials,
physical channels, LMP -3:+14, native XY 0:+14, four lag-major blocks, original
HW means/SDs/weights, baseline/intercept, eight held-direction folds, native
target scale, denominator and full-trajectory scoring. Only the final head's
target changes relative to each stored F_r arm. No new neural feature, time
window, source, target, baseline or role is introduced.

Reuse each **stored completed PLS transform**, not a new fit or a new SVD:
weights, input and target deflation loadings, rank decisions, exact-tie
convention and projected fitting means remain the original serialized values.
Use the preserved sequential inference arithmetic on fitting and held inputs;
do not substitute raw XW, whiten scores, use an alternate effective-rotation
implementation, choose new components or normalize the projected features.
The completed transforms are in
`runs/adaptive_search/primary_history_pls_attempt001/<session>/score_banks.json`.
No bank was opened to write this review; their authorized execution-time use
would be limited to the already complete states and replay artifacts.

Read the original source once per session through the existing source reader
to reconstruct the unchanged HW fitting/evaluation state, U and baseline.
Before releasing any new diagnostic score, require literal original HW state,
weighted covariance, source/trial/physical support and baseline replay. Reuse
the exact recipient/donor maps already saved for replicates 0–7. Each map stays
within its fitting direction and moves the entire 15-by-2 trial residual.
Keep the old seed literals, fixed points, identity maps and repeated draws;
there is no redraw or new seed bank. Held-direction targets enter scoring only.

For every stored candidate/control transform, reconstruct its score matrices
and projected covariance diagnostics. Refit the **old** heads against U for P
and U_r for F_r with the original five-penalty arithmetic. Require literal
equality with all five saved candidate and all eight-by-five saved F predictions,
and exact original scoring replay. These are reconstruction checks, not new
comparisons. Preserve the original time-fast Y/B layout and prediction layout.
A mismatch stops technically: do not recompute the transform, change array
layout, relax equality or substitute a fresh fit to rescue the diagnostic.

Then, for each stored shuffled projection, fit B_r's original native-U ridge
head at the unchanged penalties [0.001, 0.01, 0.1, 1, 10]. Evaluation is the
original B plus that correction once. Use the same deterministic ridge solver,
finite checks and no-rescue behavior as completed PLS. Retain K=0/1 conventions
unchanged even though all actual preserved transforms have K=2.

The selected penalty is **the already completed PLS candidate's outer-session
choice**, currently lambda 10 in every session and shared across its eight
held-direction folds. Replay that original other-15-session selection using
the completed P bank. New B or old F scores cannot select, change or retune it.
There is no new selector, independent component penalty, grid or recipe choice.
All five rows are retained only for reconstruction/transparency; only the
preserved selected row enters the diagnostic.

## Comparisons and interpretation to freeze before outcomes

Average the eight B_r correct SSEs and eight exact wrong-pair SSEs, with the
unchanged session denominator. Do not average predictions, select a favorable
draw or treat the draws as biological replicates or a permutation p-value.
Report the new mean diagnostic against baseline, both endpoint animal means,
every session, positive-session counts and fixed-prediction omissions.

Keep the two explanatory contrasts separate:

1. **P minus mean B**, with every P-minus-B_r draw also shown. This asks whether
   correctly paired projection construction helps when heads learn the same
   true U. A positive projection-correspondence interpretation requires strictly
   positive primary and identity contrasts in both animals, primary-positive
   majorities at least 4/6 and 6/10, and positive within-animal omission
   contrasts on both endpoints. Failure is lack of that demonstrated advantage,
   not proof of equivalence or of zero projection information.
2. **Mean B minus mean F**, with matched B_r-minus-F_r draw contrasts. Each
   paired draw now has the identical projection, feature Gram and numeric lambda;
   only head targets change. Apply the same directional screen to the mean
   head-correspondence contrast. This identifies the contribution of true head
   targets conditional on those frozen shuffled projections, not the effect of
   projection supervision in the candidate.

Preserve all eight draw results even when signs differ. Report original P,
full-target F, HW and W/HW-blend summaries as completed context, without
reselecting or charging them. New diagnostics cannot override the original
0.01 primary/0.005 identity gates or alter the failed PLS candidate's status.

Although B_r and F_r share a score Gram exactly, **P and B_r are not matched in
effective capacity**: their learned projections, score spectra and ridge degrees
of freedom differ, and projection learning itself uses training information.
Reuse the existing diagnostic-only ranks, spectra, traces and selected-lambda
head df to make this limitation visible; do not normalize them away. Ordinary
within-direction shuffles retain direction/time structure. These are neither
uniformly random two-dimensional subspaces nor zero-signal recordings. No
physiological mechanism, spike-independence, causal or population claim follows.

## Complete-panel validity and accounting

All 16 sessions and all eight new head draws must finish and replay before any
scientific panel is released. A resource pilot reports resource/replay facts
only. Preserve complete earlier evidence and all failed attempts; no partial
cohort, partial seed average or technical retry earns a scientific comparison.

Perform literal global duplicate checks on the new selected prediction arrays:
all eight draws across all 16 sessions must be compared against the preserved
P candidate, original HW/blend, baseline and the previously required original-
support anchors. Also check exact draw-for-draw identity with the completed F
ensemble and any structurally relevant completed eight-draw diagnostic,
including the old covariance-SVD basis-only control when its support agrees.
Recognized equivalence with another completed recipe must be resolved before
counting. Equality of average scores or averaged predictions is insufficient;
matching different reference recipes in different sessions is insufficient.

If valid and distinct, this is **one mean-loss alternative-explanation
comparison, no new candidate and no literal-null credit**. The true head still
learns correctly paired U. The eight draws, two explanatory contrasts, replayed
P/F panels and old anchors add no further comparisons. It is unpromotable and
cannot reset or update the comparable-primary incumbent even if its aggregate
score happens to exceed a candidate.

Incoming counts are 44 valid comparisons, patience 11/12, alternatives 23/38
post-initial comparisons and literal nulls 15/38. A distinct complete new
diagnostic would give 45, patience 12, alternatives 24/39 and literal nulls
15/39. An exact duplicate or technical failure changes none of these counts.
This conditional arithmetic is **not the scientific rationale for the run**.
Root must still reconcile applicable coverage, the >=40% alternative allocation
and consumed resources before any stopping decision. The existing 28–64 count
limits, strict-improvement patience rule, 2,000 core-hours and 120-hour clock
from 2026-10-03 22:23 UTC remain unchanged.

## Cost and decision requested from root

This reuses all supervised projections and adds only the true-target ridge
heads. For a complete panel, the new work is 16×8×8×5 = 5,120 two-component
head fits plus scoring. Old candidate/full-target head reconstruction and
original-support replay remain required. Source reading and saved-bank I/O
will likely dominate; no fresh PLS factorization is necessary.

Root observed the complete PLS remainder batch finish in 153 elapsed seconds.
That establishes feasibility of the surrounding source/replay pipeline, not
the measured runtime of this new diagnostic. Recommend a fixed
Chewie_20150313 resource-only pilot with one CPU, 6 GiB, one numerical thread,
no GPU, at most 600 seconds internally and a 900-second external envelope.
These are the prior complete-session envelopes, not a deadline-driven speed
assumption. The pilot must demonstrate exact stored-transform/old-head replay
before scale-up. Select the other 15 sessions' aggregate resources from its
measured facts, preserve the pilot, and expose no new scientific effects until
the complete panel succeeds. No automatic retry, resource expansion or new
source copy is proposed. Actual shared-allocation costs belong in root's ledger.

Recommend adopting this single conditional diagnostic because it resolves a
specific limitation explicitly left by the completed PLS controls. Do not add
another rank, band, lag, capacity grid or blend penalty merely because this
diagnostic could fail or duplicate a prior recipe. The independent-penalty
amendment remains unadopted. If root decides the remaining conditional question
is not worth its cost, the honest alternative is an explicit prospective
scientific resolution of the still-unmet patience obligation, rather than
declaring a stop at 44/11 or inventing a replacement experiment.
