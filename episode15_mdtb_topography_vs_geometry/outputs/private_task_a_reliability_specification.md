# EP15 private Task-A reliability preparation

Adopted by root at20:53UTC on2026-10-04 after root and independent code review
and nine passing compute fixtures. No private values have been read. This
implements the Task-A-only reliability statistic already required by the
12/12 role-assignment contract. It does not assign roles, open Task B, fit a
candidate, activate scientific margins, or choose an eligibility threshold.

The completed first-person mapping pilot used the lexicographically first
participant, sub-02, and all 464 noninstruction Task-A betas. It obtained 58
finite, nonconstant half maps for 29 conditions at the unchanged 208 support
locations in 3.141773945 seconds, with 74,528 KiB peak RSS. It decoded
628,604,928 beta bytes and 13,984,380 forward-field bytes, with no Task-B stream.
Those are operational observations; no correlation was calculated or released.

## Fixed numerical operation

Process all 24 metadata-verified participants in lexicographic ID order.
Reuse the completed metadata plan, prior headers, and all24 intersection from
the pilot. Reuse its exact-member allowlist, full 736-row Task-A plan checks,
29-condition/run multiplicities, live beta-versus-A/mask grid checks, actual
NIfTI scaling, and strict positive-weight trilinear stencil. Read only each
person's forward transform and exact noninstruction Task-A beta members.
No mask, GM, Task-B, instruction beta, pixels, or model features are added.
Retain the identical 208-point ordering, without exclusions or support changes.

For each condition and run, average its planned regressors arithmetically;
then average the eight run maps with equal weights for runs 1–8 and, separately,
9–16. These are the pilot's unchanged arithmetic operations. For each of the
29 conditions, center each half map over all 208 locations and calculate

`r_c = sum_v x_c(v)*y_c(v) / (sqrt(sum_v x_c(v)^2)*sqrt(sum_v y_c(v)^2))`.

Compute in float64 with ordinary equal-location sums, no weighting, clipping,
Spearman–Brown correction, Fisher transform, epsilon or new variance floor.
The participant scalar is the ordinary median of all 29 `r_c` values. Negative
correlations remain negative. Floating-point values are not clipped to [-1,1].
A nonfinite input, centered energy, denominator, correlation or median stops
the whole cohort. Exactly zero centered energy likewise stops because Pearson
correlation is undefined. No condition/person is omitted, replaced or imputed,
and no magnitude threshold is used for eligibility.

This statistic will balance the future participant split; it is not a noise
ceiling, evidence of independent run errors, or a screening rule. In particular,
the scalar does not establish that a participant can be excluded for low
reliability. Task-A and B remain confounded with session as in the contract.

## Private artifact and public report

Accumulate only the 24 participant medians in memory; discard each person's
maps, native coordinates and condition correlations after computing its median.
No private file is published until all 24 complete successfully. The selected
private destination is a fresh attempt under the already Git-ignored
`outputs/cache/task_a_private_reliability/`, with a single CSV named
`task_a_reliability.csv` containing `participant_id` and
`Task_A_median_centered_spatial_correlation` only. It has exactly 24 data rows
in lexicographic ID order. No condition values, half maps, native coordinates,
anatomy, derivatives, ranks or percentile summaries are retained.

Create the private attempt directory with mode 0700 and private CSV with mode
0600 to reduce accidental disclosure. These permissions do not separate
same-UID agents or establish an OS seal. The artifact is a declared private
steward handoff: the coordinator and candidate workers must not read its
contents. Only the later trusted role-assignment operation may consume it.
Write the complete CSV to a temporary file only after cohort success, then
publish it with no-overwrite semantics. A preexisting attempt path refuses
execution. A failed cohort produces no partial-median artifact; incomplete
temporary publication is removed on a handled failure.

The public JSON report lives in a fresh
`outputs/runs/source_qualification/` attempt. It reports completion, all24
finite-computability flags, operational counts, bytes, elapsed time, peak RSS,
and the private artifact's path. It reports no reliability values, minima,
maxima, signs, quantiles, ranks, per-condition values, or participant-specific
reliability diagnostics. On failure, report the operation stage and a fixed
error category; do not print exception payloads containing scientific values.
Existing successful counts may be reported, but failure does not publish the
already computed medians. All role/margin/Task-B/candidate-authority flags stay
false. The preexisting pilot report and files remain untouched.

## Resources and execution

One CPU, 4 GiB, no GPU; one numerical thread. Use an internal 540-second
deadline and external 600-second envelope. The pilot projects approximately
24 × 3.1418 = 75.4 seconds for mapping before overhead; the internal envelope
provides approximately sevenfold headroom without changing original episode
budgets. The 2 GiB preparation scratch cap remains; no images are extracted.
Read members sequentially using the existing 32 MiB uncompressed member bound.
There is no automatic retry, participant replacement, or resource expansion.

After specification adoption and attempt recording, coordinator command:

```sh
python outputs/tools/build_private_task_a_reliability.py --output-dir outputs/runs/source_qualification/private_task_a001 --private-dir outputs/cache/task_a_private_reliability/attempt001 --max-seconds 540
```

The root coordinator owns allocation verification, tests, execution, recording
and any future role-assignment plan. Focused tests check known centered Pearson
values, median rather than mean aggregation, reuse of equal-regressor/equal-run
halves, undefined/nonfinite cases, no private publication on partial-cohort
failure, restrictive creation and no-overwrite publication, and public report
disclosure. Existing passing source/geometry guards are reused rather than
rewritten or turned into another qualification sequence.
