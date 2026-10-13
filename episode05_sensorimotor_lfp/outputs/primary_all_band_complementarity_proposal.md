# Do the remaining low and middle bands add information to LMP plus high power?

Prepared 2026-10-04 from completed J, H, W and HW interpretations and the
existing source-support metadata. **Adopted by root 2026-10-04 07:44 UTC after
root and independent written review.** Implementation and focused fixtures
are prepared for coordinated validation; this paragraph makes no execution
claim. No incomplete
full-target or elastic-net score, new source value, or new fit was consulted.
Root owns source execution and the complete result record for this pair.

## Why this comparison now

The completed LMP-plus-high-power model J passed its training/evaluation
alignment screen but failed both LMP-complementarity screens. H, W and HW
changed the LMP representation; HW improved on H in both animals, but its
primary increment over W was negative in Mihili. None passes the primary
0.01 margin. These results do not test whether the unexamined 12–100-Hz
representations, together with the registered low bands, supply useful
information conditional on the existing LMP/high block.

The next useful comparison is therefore **J plus the fixed remaining six
stored power bands**, rather than another LMP weighting/lag search. Keeping
J's coordinates retains previously useful information and supplies a direct
conditional comparator. A power-only model would ask a different question
by removing LMP at the same time. This pair fills the missing all-band
representation in the planned spectral coverage; it does not declare each
middle band independently useful or make the earlier J failures disappear.

## Fixed source universe and qualification

The permitted completed metadata record
`outputs/runs/source_support/attempt002/source_support.json` has sixteen
session records, all declaring exactly these nine guide pairs:

`(0,0), (0.5,4), (4,8), (8,12), (12,25), (25,50), (50,100), (100,200), (200,400)`.

The `(0,0)` representation is LMP; the other eight are stored power features.
This was a bounded metadata inspection, not a raw-neural read. A session-level
band list does **not** prove each frozen physical electrode has every band
in every selected trial. After adoption, qualify one exact guide column per
original electrode and each named band on every original trial before using
new values. Require the original frozen LMP physical-electrode IDs, not a
new all-band intersection. Ambiguity, missing support or nonfinite values
stops the pair; do not drop a band, electrode, trial, direction or session.

Keep the original sixteen exposed M1 sessions, complete trial identities,
native offsets 0:+14 (fifteen 30-ms bins), XY targets, direction labels and
both-animal grouping. Chewie implants remain one animal. Only the three
12–100-Hz bands are newly requested relative to the previously read LMP,
low and high blocks. Record actual reader scope. Do not read spike values,
expand the source window or alter feature definitions.

## A: LMP and all eight fixed power bands

Recipe ID: `lmp_all_band_joint_ridge_v1`.

Use a fixed 9E-column layout that preserves J as its first block:

1. LMP: E coordinates in frozen ascending physical-electrode order.
2. High power: 2E coordinates, electrode first, then `(100,200),(200,400)`.
3. Added low/middle power: 6E coordinates, electrode first, then
   `(0.5,4),(4,8),(8,12),(12,25),(25,50),(50,100)`.

The first two blocks together are the original 3E-column J representation.
At each held direction, use only the other seven directions to estimate
the original J per-coordinate population mean/SD, zero SD mapped to one.
Preserve J's separate raw LMP/high array and normalization paths. Apply the
same fitting-only per-coordinate procedure to the added block. Concatenate
the three standardized blocks; subtract the resulting fitting-column means
from both fitting and evaluation arrays, preserving J's final-centering
convention. Save all normalization and centering state.

Do not log/square stored power, filter it again, whiten, truncate rank, select
channels/bands, or rescale blocks by their widths. Each standardized coordinate
receives the same ridge penalty; six added bands contribute six times the
nominal column count of LMP. This is a real change in feature energy, metric
and available capacity, not an equal-width comparison to J or LMP.

Keep the original equal-direction first-harmonic direction/time template B0
and common fitting-residual intercept c. Fit the native, unscaled XY residual
`U=Y-B0-c` from the full centered A design F using

`mean_row ||U-F beta||² + lambda ||beta||²`,

at exactly `[0.001,0.01,0.1,1,10]`, and add B0+c at prediction. Reuse the
qualified positive ridge eigensystem routine over the full feature span,
with one arm-specific decomposition serving all five penalties; no rank
cutoff, whitening or negative-eigenvalue clipping in the solve is introduced.
All five penalties use the same native-XY loss. Zeroing the six added-band
coefficient blocks embeds J algebraically at a common lambda; preserved
centering paths need not produce bitwise identical floating-point arrays.

Select each outer session's lambda from only the other fifteen complete A
banks using the original equal-session/equal-animal primary objective and
stronger-lambda exact ties. No control, anchor or held-session result chooses
lambda. Retain the unchanged pooled session SSE denominator and exact
within-direction evaluation wrong-trial prediction pairing readout.

## C: eight added-block training-trial label controls

Recipe ID: `lmp_all_band_added_training_labels8_v1`.

For each session, held direction, replicate 0..7 and fitting direction,
sort the fitting recipient IDs by validated native source trial number.
Use one ordinary Python version-2 shuffle of that list, with literal

`ep05-lmp-all-band-added-training-label-v1|20261004|<session>|<held>|<replicate>|<direction>`.

Save complete recipient/donor native IDs. Accept fixed points and identity
maps without redrawing. Keep the map fixed across all five lambdas and every
time, electrode and added band. Move the entire **15-by-6E standardized added
block** of each donor as a unit, within its fitting direction only. Do not
shuffle individual bands, electrodes, bins or coordinates. Neither held-
direction donors nor targets participate in a map.

The fitting recipient's LMP/high J block, true native residual U and behavior
baseline remain unchanged. Physical means/SDs are the paired A fitting state;
do not refit SDs after permutation. Each arm centers its assembled fitting
columns and applies those means to its evaluation design. **Evaluation LMP,
high and added bands all retain their original correct-trial values.** This
is a training-correspondence control; it differs from J's earlier high-block
control, which also reassigned the evaluated high trajectories.

Within the fitting directions, a whole-block permutation preserves the added
block's direction/time distributions and all within-block electrode/band/time
relationships. J's within-block Gram also remains unchanged. The **J-by-added
cross-Gram generally changes**, as does the added-block-to-U cross-product.
Fit every control using its own complete centered 9E Gram and true-U head;
never reuse the paired full Gram or fit only the added coefficients. Report
traces, spectra/rank and selected-lambda df for all arms. Shared width and
five-lambda grid do not imply equal spectra or df.

At A's selected outer-session lambda, average the eight correct SSEs and the
eight exact evaluation-wrong-pair SSEs. Do not average predictions, select
seeds or optimize a control penalty. Seeds are finite descriptive refits,
not additional animals or confidence intervals.

This control tests whether correctly paired added-band fitting trajectories
help the complete decoder beyond information learnable with the disrupted
training correspondence. It removes both added-to-target and J-to-added
trial alignment during fitting. It does not isolate their causal roles or
establish nonlinear synergy: the predictor remains additive and linear.
Because evaluation is unchanged, it is not an evaluation corruption test.

## Preserved anchors and prospective interpretation

Reuse complete J and original full-channel LMP banks without refitting:

- J: `primary_lmp_high_control_attempt001` and
  `primary_lmp_high_control_summary_attempt001/primary_lmp_high.json`;
- LMP: `primary_multichannel_attempt001` and
  `primary_multichannel_summary_attempt001/primary_multichannel.json`.

All paths above are under `outputs/runs/adaptive_search`.
Report A-minus-J and A-minus-LMP with both independently selected anchor
penalties and A-matched original lambda. These four contrasts reuse frozen
anchor banks and add no scientific depth. The J contrast asks the direct
incremental low/middle-band question; the LMP contrast retains the original
useful reference. Different widths and common-lambda penalties are disclosed.

Report A, mean C, A-minus-C, every seed/session/animal, and every fixed-
prediction omission for both original readouts. A claim that the added bands
improve on J requires positive primary and identity contrasts in both animals
under both J reference selections, primary-positive counts at least 4/6
Mihili and 6/10 Chewie, and the same directional screen for A-minus-C. A claim
of improvement over LMP additionally requires both LMP screens. Ties fail.
No new contrast magnitude or noninferiority threshold is invented.

Promotion still requires the original 0.01 primary and 0.005 identity margins,
both-animal signs/majorities, omissions and every other required test. Passing
an added-block alignment screen cannot rescue failed A-versus-J or primary
screens. No outcome establishes that 12–100 Hz alone is responsible: the added
block includes all six declared low/middle bands. High-band contamination
remains relevant to the complete representation; all-band success would not
establish spike-free field-potential information.

## Exact degeneracies and trial accounting

If every added centered fitting column has exactly zero energy in every
required fold/session, its ridge coefficients are zero and A duplicates J
algebraically; all controls also duplicate J, so this pair adds zero depth.
Use exact energy, not a variance floor. Zero-variance support in one fold
never deletes a coordinate or invalidates otherwise evaluable predictions.

Record each control's exact fitting- and evaluation-design equality to A after
its own fitting centering. If both designs equal A throughout all required seeds/folds/sessions,
its prediction recipe is the same and the mean control adds zero depth.
This includes identity maps and repeated identical donor blocks; do not
redraw to create novelty. Other exact structural duplicates require explicit
documentation, not near-equal score matching. Candidate and control charges
are independent; otherwise at most two distinct complete comparisons and
one mean-control increment are added. Failed new scientific results still
count. Individual bands, seeds, lambdas, contrasts and engineering repairs
do not count separately. The incoming count is taken from closed panels only.

## Bounded qualification and implementation, only after adoption

Use new reader/core/runner files and preserve old J/LMP code and banks. Guide
qualification precedes any newly requested values within a session. After
all session fits, replay the original physical/trial IDs, true XY, direction
labels, behavior baseline and source window exactly. Verify the retained J
source blocks and their normalization conventions independently of outcomes.
Focused fixtures must cover nine-band ordering/support refusal, fitting-only
normalization, J-prefix nesting, true-U heads, whole-added-block maps, unchanged
evaluation, changed full cross-Grams, own-arm centering/df, mean SSE, outer
session exclusion, stronger ties and exact duplicate accounting.

Propose one complete fixed Chewie_20150313 pilot, all controls and penalties,
one CPU within 6 GiB, 270 seconds internal and 300 external. The largest frozen
pilot has 1038 trials and 774 columns: one float64 trial/bin design is about
96.4 MB, before temporary matrices and outputs. A 96-electrode session has
864 columns. These are shape estimates, not measured peak memory or timing.
Process arms sequentially and reuse each arm's decomposition for its five
penalties. A resource failure stops for diagnosis; no automatic longer retry,
feature deletion or new numerical approximation is selected here. Choose a
finite remaining-panel bound from complete pilot resource facts while scores
remain closed. Proposed fresh roots are `primary_all_band_control_attempt001`,
`primary_all_band_summary_attempt001`, and `primary_all_band_panel_attempt001`.

## Search coverage is distinct from eligibility

The absence of a currently eligible model does not itself authorize stopping
before the planned minimum or invent a material-improvement cutoff. This pair
addresses missing all-band coverage with a concrete scientific hypothesis;
it is not a pair of extra counts chosen merely to reach 28. Reaching 28 would
not automatically satisfy the 12-trial post-minimum patience requirement,
alternative-explanation coverage or finalist comparisons.

Matched channel/trial support and other finalist falsifiers remain distinct
from the all-band question. This pair does not mark them passed, and a positive
all-band candidate would need those tests before promotion. Conversely, it
does not claim a positive-candidate gate has been completed by running an
unrelated negative recipe. Root retains the exact coverage and stopping audit.
Preserve 28–64 depth, at least 40% post-initial controls, two adaptive rounds,
two challenger comparisons, 2,000 core-hours, zero GPUs, and the 120-hour clock
starting 2026-10-03 22:23:02.824 UTC. No sealed future sessions are created by
this exposed-corpus comparison, and no new-animal or causal claim is available.
