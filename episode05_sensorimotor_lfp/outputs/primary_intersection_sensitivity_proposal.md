# Does the fixed primary model depend on the recorded unit-guide intersection?

Resource amendment by root,09:22UTC: the complete metadata-selected
Chewie_20150313 numerical pilot passed at09:20:49.345353–09:21:13.268025
in job46472346. Elapsed23.922664s, global metadata1.641233s, source10.067647s,
reduced fits11.756837s, peak323,424KiB. All nine supports required unique
fits; original target/B/trial/scoring replay passed, with no full-HW refit.
Scientific pilot scores remain closed. Preserve that pilot and run fifteen
fresh sessions sequentially,one thread/6GiB,720s internally/750s externally,
unchanged270/300 children. Fifteen pilot equivalents358.839957s plus a full
300s child reserve leave61.160043s for summary and variation. Require780s
actual allocation time remaining, including cleanup. Root freezes live
counts/patience/alternative/null inputs only at remainder launch, after any
earlier complete panel; no future H2W outcome or ordinal is assumed. Fresh
summary/parent roots are `primary_intersection_sensitivity_summary_attempt001`
and `primary_intersection_sensitivity_panel_attempt001`. No further pilot,
source requalification, automatic extension or scientific-policy change.

Prepared 2026-10-04 during the already selected matched-budget/elapsed-only
work. **Adopted by root at 2026-10-04 08:46 UTC after root and independent
scientific review.** Implementation starts with isolated guide-only metadata
qualification and focused fixtures; root reviews and executes those before
releasing the resulting metadata for numerical implementation. This paragraph
makes no execution or score-release claim. Root owns metadata extraction,
compute, resource decisions and the scientific record. Neither unfinished
panel scores nor new spike values were read to choose this proposal.

## Required check and limited question

[GOAL.md](../GOAL.md) lines 232–239 retain applicable development stress tests;
line 250 names a same-electrode/unit-intersection sensitivity. The adopted
08:08 coverage reconciliation in [the log](experiment_log.md) distinguishes
that primary check from the conditional spike-population explanation. The
positive-result criteria separately require that a promoted gain not be
confined to high-frequency or same-electrode spike contamination. Performing
a metadata-defined restriction cannot, by itself, establish that latter claim.

The existing qualified publication join is explicit in
`tools/run_publication_reproduction.py`: it verifies unchanged LFP and unit
guides across all selected trials, then intersects their physical-electrode
IDs. The primary HW fits have not applied that restriction. The source-support
inspection recorded guide/count facts, but a publication join or a count is
not a primary refit on the intersection.

Freeze the strongest complete original-support primary recipe, HW
(`lmp_history4_repeatability_weight_ridge_v1`), as specified in
[its adopted proposal](primary_history_repeatability_weight_proposal.md).
This choice uses the released completed evidence through all-band comparison
27. Its primary increment remains below 0.01. No later matched/elapsed score
selects the anchor or changes this design.

The narrow question is whether that fixed primary calculation changes when
LMP inputs are restricted to electrodes represented in the released unit
guide, relative to ordinary LMP subsets of the same size. This is a
**nonpromotable primary-support sensitivity**, not a new winner, spike-value
readout, spike-richness ranking, contamination-removal procedure or causal
test. No complement-of-intersection model is selected. No new scientific
success threshold or primary endpoint is introduced.

## Qualify all sixteen joins before new neural values

Use the same sixteen M1 sessions, original complete trial whitelist and source
identities as HW. Existing authorized inputs are the original trial table
`runs/source_support/attempt002/trial_identity.csv`, the original scalar
source records, complete HW source records and the read-only source's
`M1_lfp_guide` and `M1_unit_guide` metadata. The provider's physical electrode
and unit identifiers are distinct from signal amplitudes or unit quality.

The current source adapter already knows the guide shapes and the publication
path already required unchanged guides. A new root-owned, bounded metadata
qualification must nevertheless save the exact joins used here; per-trial
matched-count ranges alone cannot identify the chosen physical set.

Before any new LMP/XY value read or fit, reconcile all sixteen original session
and trial records and perform the following guide-only pass:

1. Read `M1_lfp_guide` and `M1_unit_guide` for every original retained trial.
   Check their shapes against dataset dimensions without indexing numerical
   `M1_spikes`, LFP or velocity values. Reading a dataset object's shape is
   permitted metadata access; `np.asarray`, slicing, moments, rates, PCA,
   integer-count tests or unit-quality evaluation of spike values are not.
2. Require finite guides, unique full guide rows, positive integral physical
   electrode IDs, and the original publication-qualified constant guide arrays
   throughout that session. A changed guide is a qualification failure, not
   permission to invent a smaller across-trial unit intersection or drop trials.
   Keep the finite stored unit labels exactly as released; do not discard a
   label zero, impose a sorted-unit quality criterion, or weight an electrode
   by its number of units. Multiple distinct units on an electrode map to one
   physical LMP electrode.
3. Reconcile the original stable LMP electrode set `E_s` against the scalar
   and HW source records, using the literal LMP band `(0,0)` and original
   physical-ID order. Define `U_s` as the unique physical electrode IDs in
   `M1_unit_guide[:,0]`, and freeze `I_s = E_s intersection U_s`, in ascending
   original physical order. Record `E_s`, full unit-guide identity pairs,
   `U_s`, `I_s`, `E_s minus I_s`, and unit-guide electrodes outside `E_s`.
4. Require `1 <= |I_s| <= |E_s|` for every original session. An empty or
   ambiguous intersection stops this complete-cohort proposal before values;
   no session/animal omission, smaller cohort or replacement channel set is
   selected. Every original direction must retain its original trial count.

Save a new immutable metadata artifact under
`runs/source_support/primary_intersection_metadata_attempt001/` with exact
session/trial identities, source paths, guide invariance checks, joins and
`spike_values_read=false`, `LFP_XY_values_read=false`, `scores_computed=false`.
No scientific comparison is charged for this qualification. Its implementing
tool/command and resource bounds require root review before compute execution.

Guide membership is an attribute of the released recording/processing output.
Absence of a listed unit does not establish absence of spiking or contamination;
presence does not measure activity, waveform amplitude, isolation quality or
contamination. Equal numerical electrode IDs across sessions/implants do not
identify the same tissue. This metadata check does not activate the separately
conditional LFP-to-spike-latent follow-up in DATASETS.md/SEARCH_POLICY.yaml.

## I and its eight count-matched ordinary subsets

Freeze recipe order:

1. `lmp_HW_unit_intersection_v1` (I): use exactly `I_s` in each session.
2. `lmp_HW_unit_intersection_random_support8_v1` (R): mean of eight ordinary
   random physical LMP subsets of size `k_s = |I_s|` from the original `E_s`.

For R replicate `r=0,...,7`, start from the ascending original `E_s` list,
initialize Python `random.Random`, seed with version 2 and the exact literal

`ep05-HW-unit-intersection-random-support-v1|20261004|<session>|<replicate>`,

perform one ordinary `shuffle`, take the first `k_s` IDs and restore their
ascending original order for numerical fitting. Save the whole queue, literal
and selected IDs. Never redraw because a subset equals I, another replicate
or full support. Do not force overlap, disjointness or a complement. All nine
support sets are metadata-only, fixed before values, and shared across held
directions, bins and penalties. The control preserves true neural/behavioral
pairing; it is a random-support reference, not a literal pairing-null recipe.

The count-matched reference separates recorded guide membership from simply
using fewer electrodes. It does not equalize anatomical coverage, signal
quality, fitted covariance spectra or effective ridge degrees of freedom.
That limitation prevents a membership contrast from being labeled a direct
spike-contamination effect.

## Unchanged target and training-only HW reconstruction

Retain every original trial, every held direction and both animals. Neural
input stays native LMP offsets -3:+14; targets stay native XY offsets 0:+14,
fifteen 30-ms bins. No power-band or spike values, alternate interval, new
filtering or velocity outcome is requested. The qualified history reader may
load the original LMP session once; only the frozen physical subsets enter
their respective fits. Reconcile its guide, source window and trial records
against the completed metadata pass before fitting.

Use the fixed channel-only indexing operation `x[:,:,positions]`, with
positions in original physical order and no forced C-order conversion.
Pass the original y, direction labels and trial IDs through untouched: do
not route this through the matched-budget helper that also copies trial/Y
arrays. Preserve W's current raw slice `x[:,3:18,:]` on the selected input.
For every genuinely new support, refit all existing HW states separately on
the seven fitting directions: direction-balanced first-harmonic template B0,
common native XY residual intercept, current-channel mean/population SD and
unchanged T/P/r/a prior, lag/channel means/SDs, original history fitting-column
center, full-history trace normalization and native residual ridge head. Keep
lags `[0,-1,-2,-3]`, the 0.1 prior floor, zero-SD handling, exact zero-Gram
branches, no post-weight standardization/centering and the existing solver.
No full-support weight or normalization is borrowed for a reduced fit.

The behavioral comparator `B=B0+c` uses unchanged Y, trials and directions,
so it must replay the original full-HW B exactly for every support. Native Y,
trial IDs, directions, time windows and pooled joint-XY denominator D must
also replay. If a supposedly unchanged component differs, stop rather than
relaxing equality or changing the preprocessing recipe.

Fit all five original normalized-loss penalties `[0.001,0.01,0.1,1,10]`.
For each outer held session, I selects one lambda using I's other fifteen
session banks, with the original equal-session/equal-animal weighting and
stronger-lambda exact tie rule. R uses that same I-selected lambda in every
replicate. Do not tune R separately, tune by electrode group or inspect the
held session when selecting lambda. Using one candidate-selected penalty is
the existing controlled-comparison convention, not a claim of equal capacity.

## Original scoring, full-HW anchors and descriptive readouts

For I, score full native XY predictions `B + neural residual` on the identical
original evaluation trials. Preserve the original pooled session denominator
and exact mean over within-direction whole-trial derangements for the identity
readout. For R, average eight correct SSEs and eight exact wrong-pair SSEs with
that same B and D; never average predictions or select a seed. Then collapse
to one session estimate and the original equal-animal summary. Seeds, folds,
units and electrodes do not increase the biological sample size of two animals.

Reuse the completed full-HW candidate banks/predictions under
`runs/adaptive_search/primary_history_repeatability_weight_control_attempt002`
and the complete summary
`runs/adaptive_search/primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json`.
Load saved full-HW prediction arrays only after the new supports and fitting
choices are fixed. Require exact target, baseline, original-ID, direction and
source-window replay. Do not refit full HW or retune it on the evaluated
session. Compare I with full HW at both:

- full HW's original independently selected other-fifteen-session lambda;
- full HW at I's selected lambda from the unchanged five-lambda bank.

The direct prediction contrast is `(SSE_fullHW - SSE_I)/D`. Because B and D
are exactly shared, it also equals the difference of primary increments.
The identity contrast is the difference in original identity increments on
the same retained trial pairs. Report I minus R at I's lambda, both full-HW
anchor contrasts, all sixteen sessions, both animal means, positive-session
counts, fixed-prediction session omissions, each R replicate and finite seed
ranges. Report k_s, guide membership, subset overlap, fitting rows, selected
lambda, Gram trace and selected effective df for every support/fold.

Report the unchanged 0.01 primary and 0.005 identity margins and existing
directional summaries for the restricted calculation. These do not rescue
HW's failed original primary margin or create a new promotion gate. Do not
require I to outperform unrestricted full HW as a new pass condition. Seed
ranges and session omissions are descriptive rather than confidence intervals.

A persistent I gain shows that the specified calculation also works on this
metadata-defined intersection. A loss bounds its support robustness. I−R
describes recorded-membership association at equal channel count. Neither
result establishes whether the LMP features contain spike bleed-through.
**Failure to identify contamination remains a scientific limitation even
when every planned computation completes.** The stronger contamination and
causal claims are not relabeled passed.

## Exact reuse, duplicate accounting and resource pilot

If `I_s == E_s`, reuse that session's original full-HW five-penalty arrays and
baseline after qualification/replay; R then has the same full set in every
replicate. If a random subset equals I or a prior random subset, reuse the
corresponding fitted state/predictions exactly. Count physical-set equality,
not numerical score similarity, and preserve every requested replicate record.
No favorable subset is selected from duplicates.

I adds one scientific stress comparison only if its support differs from
original full HW in at least one session. R adds one only if its eight support
sets are not all exactly I across all sixteen sessions and it is not itself
a global full-HW duplicate. If every I is full, both recipes are structural
duplicates: qualify the join, reuse complete original artifacts and report
zero new scientific charges rather than execute redundant fits. Any further
algebraic-duplicate assertion needs an exact proof, not a small-variance or
score tolerance. Partial panels and metadata qualification add zero depth.

Process I first and mean R second. At most two complete distinct recipes add
to the valid and post-initial alternative/stress counts. The literal pairing-
null count is unchanged. Their final ordinals depend on actual completion of
already selected work; no unfinished comparison is assumed complete here.
Both are nonpromotable changed-column diagnostics and cannot update the
same-support candidate incumbent or reset the prospectively adopted patience
counter. Root records actual charged transitions under
[the existing rule](primary_patience_interpretation.md), preserving all depth,
control-allocation, resource and stopping requirements.

Select a bounded resource pilot from the completed guide-only metadata, with
no score input. For each session let u_s be the number of distinct genuinely
new physical supports among I and its eight R subsets after full-HW/exact
support reuse, N_s its original trial count, and p_s=4*k_s. Use the fixed
workload proxy `u_s * (105*N_s*p_s^2 + 8*p_s^3)`; the first term counts seven
training appearances of fifteen bins per trial, and the second represents
eight square-system factorizations. This is a planning proxy, not a measured
runtime or a scientific selection criterion. Choose its maximum, breaking
ties by the original `SESSIONS` order; freeze the resulting session before
value access. If every u_s is zero, no numerical resource pilot is needed.

Proposed nonduplicate pilot: one CPU, 6 GiB, 270 seconds internal and 300
external, complete I plus all eight R supports/five penalties for that session.
Keep scores closed while selecting the finite remaining-session bound from
complete resources. Width is at most the already qualified 384-column HW
design, but new timing must be measured; no automatic retry, limit extension
or concurrency escalation follows. Root owns any remaining allocation.

Use fresh proposed roots `primary_intersection_sensitivity_attempt001`,
`primary_intersection_sensitivity_summary_attempt001` and
`primary_intersection_sensitivity_panel_attempt001` under `runs/adaptive_search`.
Preserve all prior tools, scores and failed attempts. Independent focused
fixtures must cover guide-only access, all-sixteen/no-empty qualification,
multiunit-to-physical joins without quality filtering, ordinary fixed subsets,
training-only HW states, exact unchanged B/Y/D, common-lambda mean SSE,
full/support-duplicate reuse and ordered zero/one/two accounting. The adopted
guide-only extraction is bounded at 60 seconds internally and 90 externally,
one numerical thread/CPU; root reviews its exact tool/command and fixtures
before execution. All later value reads retain the separate pilot boundary.
