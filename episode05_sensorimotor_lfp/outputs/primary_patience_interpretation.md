# Prospective conservative execution of the existing patience requirement

Adopted by root after independent written review, 2026-10-04 07:56 UTC,
with 25 completed comparisons and before the all-band source pilot.

GOAL.md and SEARCH_POLICY.yaml retain a minimum of 28, maximum of 64 and
12 valid trials without material constrained primary improvement after the
minimum. They do not quantify a material improvement or explicitly initialize
the counter. The initial primary proposal leaves that contract unchanged.
The absolute 0.01 primary and 0.005 identity eligibility margins are not
search-progress deltas. No numerical patience history is reconstructed here.

Root chooses the following conservative execution rule within the existing
allowance. This is an explicit prospective choice, not the only possible
reading of the inherited text and not a claim that it mandated 40 trials.

1. Track the full-precision best animal-balanced primary increment among
   complete true-pair residual-XY candidate recipes on the identical original
   sixteen-session, trial, target and scoring support. Eligibility for this
   progress ledger means valid comparable prediction, not passed promotion.
   Initialize it from the completed history-plus-repeatability candidate:
   `0.0023412544731769847`, stored in
   `runs/adaptive_search/primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json`.
   Continue updating the best score through comparison 28.
2. Patience is **not started** for comparisons 1–28. Beginning at comparison
   29, each distinct complete valid scientific recipe adds one to the counter,
   unless a qualifying candidate establishes a strictly higher primary score;
   that resets it to zero even if practical or identity gates still fail.
   Recognizing every strict improvement is conservative relative to ignoring
   small gains under an unspecified materiality threshold. No added numerical
   tolerance or meaningful-effect threshold is introduced.
3. Null controls and nonpromotable or different-support diagnostics count as
   valid comparisons when otherwise qualified, but do not update this
   candidate incumbent. Technical failures and exact duplicates add no count.
   For a paired completion, process the candidate first, then the mean control,
   including when a pair crosses the 28/29 boundary. Each later proposal must
   state its recipe order if it contains more than those two recipes.
4. No unregistered frontier criterion resets this counter. A scientific
   interpretation, passed sensitivity or newly established first score on a
   different endpoint does not silently create a progress reset.
5. A patience stop requires 12 consecutive counted evaluations without such
   progress **and** completion of applicable required coverage, stress tests,
   control allocation and adaptive/challenger obligations. Matching trial and
   electrode support remains required work; a failed positive-result gate
   cannot be relabeled passed. The conditional spike-population follow-up
   remains unactivated while the primary result is not positive.

The maximum of 64, 2,000 core-hours, 120-hour clock, all other resource limits,
original promotion criteria and source boundaries remain unchanged. Reaching
a ceiling cannot make unfinished requirements complete or authorize an
extension. Existing outputs and failed attempts are preserved. The live log
records future counter transitions with their actual complete evidence.
