# Uncertainty-block variance amendment — ADOPTED October12

The scientist approved this exact amendment limited to small-block variance
through the recorded question `call6o13giUJ`. GOAL.md,
SEARCH_POLICY.yaml and the operative qualification specification now record
the same rule. This decision is not an audit release or a performance result.
A fresh complete fit requires the coordinator's finite resource selection and
focused checks; old fit001 and both diagnostics remain preserved.

## Completed evidence

`fit001` completed all 352 PCA fits, then stopped before any private inner
model score or checkpoint. The first diagnostic reproduced
`response_partition_variance_invalid` for sub-02/right, stage B, uncertainty
block04, with 101 images and 942 unchanged calibrated columns.

The coordinator's complete count-only
[quality census](runs/nested_prediction/fit001_diagnostic002/diagnostic_report.json)
finished at **2026-10-12 23:14:12.869 UTC**, taking **367.383 seconds**, with
peak RSS **1,309,784 KiB**. It evaluated every fixed participant/ROI/stage:

- All **1,248 variance partitions** were evaluated: 24 group/stage panels,
  each with 22 training partitions, six validation partitions and 24
  uncertainty blocks. Every training and validation partition passed.
- There were **18 failing partition-column occurrences**, all classified as
  **EXACT_CONSTANT_ZERO_VARIANCE**. There were no nonfinite-input, moment-
  overflow, nonconstant-underflow or positive-relative-threshold failures.
- Failures occurred only in sub-02/right at stages B, C and D: block04,
  block08, block07 and block11 had respectively one, one, three and one
  invalid columns per stage. Each stage had three unique failing columns.
  Cross-stage membership was not exported; this does not establish that
  the same three column identities fail across stages.
- All **1,632 training-baseline SST partitions/pools**, comprising
  **1,745,832 partition-column occurrences**, were finite and strictly
  positive. This includes every individual uncertainty block, not just
  pooled selection and outer out-of-fold denominators.

The census preserved every original column, block and participant. It
computed no candidate predictions, SSE, R², model objective or endpoint, and
exported no response magnitudes or column identities. It establishes the
reason for this eligibility refusal, not model performance or valid future
confidence intervals.

## The single adopted rule

**When assembling primary R² statistics, do not use centered response
variance within an individual uncertainty block as an independent veto.**
Keep every other existing validity requirement. Apply this rule uniformly
to the fixed supports, not only to the observed failing blocks or columns.

An exactly constant block can still have strictly positive SST around its
frozen training mean: the block's constant level can differ from that mean.
The primary denominator uses the latter quantity. The completed census
shows that all existing per-block SST requirements pass for development.

This is a scientific protocol amendment to the explicit block-variance
requirement in
[the qualification specification](prediction_qualification_specification.md#calibration-and-finite-response-rules),
not a numerical repair. Its scope is precisely:

| Component | Rule after adoption |
| --- | --- |
| Calibration and fixed support | No change to calibration-image or half-image variance rules, gain/ceiling operators, weights, original column identities, family001, or H inapplicability. No support repair or deletion using development responses. |
| Development training and validation | Retain both existing centered-variance requirements, `V_population > 0` and `V_population > 1e-12 * mean(y**2)`, in every training and validation partition. Preserve all finite-input and finite-moment checks. |
| Individual uncertainty blocks for primary R² | Preserve finite responses and finite computed moments. Neither zero centered variance nor failure of its relative positive-variance threshold independently rejects primary R². Still require strictly positive finite SST for every required column in **every individual block**, around each image's matching frozen outer-training mean. No denominator floor or pooled-only exception. |
| Fitting and primary scoring | Preserve training-only means, all 352 PCA fits, the full 84-setting grid, shared B/C/D selection, finite fitted quantities, finite residual sums and every existing selection/OOF SST check. Do not perturb responses, substitute held-out means, alter units or drop columns. Retain negative R² and normalized values above one. |
| Bootstrap and uncertainty | Keep the same block order, 10,000 draws, seed 1729, simultaneous max-t families, margins and decision rules. Each required statistic must remain defined. An undefined draw or nonfinite/zero bootstrap standard error still invalidates its comparison: no skipping, redrawing, substituted seed, variance floor or reduced family. |
| Correlation and CKA | Keep the existing validity rules on every actual scored/resampled vector, including observed/predicted centered-variance safeguards for Pearson/Fisher-z and positive finite centered Gram-norm products for CKA. A required secondary statistic that is undefined remains a failure; primary R² availability does not waive it. |
| Sealed audit | Only after the original predictor/relation freeze, apply the same narrow uncertainty-block rule inside the existing one-shot evaluator. Retain its whole-audit response-variance check, all finite checks, and strictly positive finite SST in each of its 12 blocks around the frozen development-training mean. No audit access or pre-screening is authorized by this draft. |

## Minimal implementation under the adopted rule

The operative specification and policy record this exact dated amendment.
The primary source changes are bounded:

- In `outputs/tools/nested_prediction.py::_prepare_responses`, separate the
  current combined validation/uncertainty-block variance loop. Keep training
  and validation calls to `_qualify_response_variance`; uncertainty blocks
  retain finite-input/moment checks and the existing `qualify_block_SST`
  path. Do not weaken the shared variance helper used by other statistics.
- In `outputs/tools/audit_prediction.py::apply_frozen_audit`, make the same
  change only to the `qualify_audit_block_variance` admission. Preserve
  `qualify_audit_response_variance` on the whole audit and `_block_sst`.
- Preserve the positive per-block SST check in
  `outputs/tools/development_endpoints.py::_group_stage` and all downstream
  endpoint, secondary, bootstrap and undefined-draw checks.

Focused verification must distinguish a constant uncertainty block with
positive training-baseline SST from nonfinite inputs, a zero-SST block and
invalid training/validation variance. It must also preserve the analogous
audit boundary and unchanged secondary failure rules. The policy amendment
does not itself launch fitting or select an allocation.

## Remaining scientific limits and decision

All four participants, both ROIs, all B/C/D stages, four models, image/concept
rosters, supports, controls, five edges and registered endpoint families stay
fixed. F1, size/support/ceiling/beta controls, secondary comparisons, exposure
limits, adaptive search obligations and independent one-shot audit requirements
remain. No failed secondary comparison may be silently omitted. An available
primary R² does not complete initial coverage, select a candidate, establish
robustness or support a mechanistic or training-recipe claim.

**Decision recorded:** adopt this uncertainty-block-only variance amendment,
with all existing per-block positive-SST and other requirements retained.
Preserve the failed attempts. Any selected fit002 recomputes the complete
registered pipeline because fit001 published no reusable checkpoint; it is
an explicit fresh attempt, not an automatic retry or partial resume. The
original240-hour deadline remains epoch1791916149.
