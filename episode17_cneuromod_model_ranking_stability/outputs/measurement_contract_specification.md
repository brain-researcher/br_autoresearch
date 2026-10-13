# Endpoint families and audit counterparts

**Adopted October 4, 23:50 UTC, before candidate scores.** Root and independent
review agree on these prospective implementation conventions. This table makes the existing
[search policy](../SEARCH_POLICY.yaml), [goal](../GOAL.md) and adopted
[prediction/calibration conventions](prediction_qualification_specification.md)
concrete. It authorizes no source access, fitting, scoring or audit. No
contract is amended by these choices; actual calibration eligibility and the
role-separated evaluator remain unfinished.

## One signed comparison, seven primary endpoints

For one registered pair, A minus B is the fixed model order from the model
registry. Within participant s, ROI r and original native column v, define
`d_stage(v) = R²_A,stage(v) - R²_B,stage(v)`. R² uses pooled held-out image
SSE/SST around the corresponding **development-training** mean. Primary
endpoint effects below are model advantages, not changes between endpoints.

First calculate the specified voxel/support mean within each of the eight
participant×ROI groups, then average left/right fusiform equally within each
person, then average sub-01/02/03/06 equally. Every endpoint requires all four
people and both ROIs; no participant/ROI replacement or available-case average.
Native voxels, image repetitions and the32 anatomical subsets are not extra
biological replicates, bootstrap units or trials.

| Endpoint ID | Stage; support | Within-participant/ROI effect | Practical margin | Exact audit counterpart |
| --- | --- | --- | ---: | --- |
| `B_Aall_raw` | B; frozen A_all | Mean `d_B(v)` |0.01| Audit B image means, final frozen B predictor, same A_all |
| `C_Aall_raw` | C; identical A_all | Mean `d_C(v)` |0.01| Audit C image means, final frozen C predictor, same A_all |
| `D_Aall_raw` | D; identical A_all | Mean `d_D(v)` |0.01| Audit D image means, final frozen D predictor, same A_all |
| `D_AN_raw` | D; frozen32-member A_N bank | Arithmetic mean of32 N=256 subset means of the **same** `d_D(v)` |0.01| Same32 subsets and arithmetic weights on the final D prediction bank |
| `D_RN_raw` | D; frozen R_N | Mean of the same `d_D(v)` on256 selected columns |0.01| Same R_N membership on the final D prediction bank |
| `D_VNC_raw` | D; frozen V_NC | Mean of the same `d_D(v)` on V_NC |0.01| Same V_NC and final D prediction bank |
| `D_VNC_normalized` | D; identical V_NC | Mean `d_D(v) / max(NC_D(v),0.10)` |0.05| Same V_NC, final D predictions and frozen calibration denominators |

A_all is the calibration-qualified common finite/nondegenerate anatomical
universe across B/C/D, with original-column identity preserved. A_N and R_N
have q=32 in each of the same eight frozen native-geometry bins, N=256 per
participant/ROI. The anatomical bank is independent of reliability magnitude;
R_N uses finite D reliability and original-column ties. V_NC is the frozen
D-stage A_all subset with a valid finite calibration ceiling **before**
flooring. Negative ceiling estimates are retained and floored only in the
positive denominator. Do not average ceiling ratios by dividing ROI means,
clip negative R², renormalize inverse-ceiling weights to sum to one, or clip
normalized values above one.

## Five directed edges, with their isolation requirements

| Edge ID | Start → end | Only allowed measurement change | Explanation/audit target |
| --- | --- | --- | --- |
| `B_to_C` | `B_Aall_raw` → `C_Aall_raw` | Response stage and its fitted voxel coefficients; one layer/rank/penalty per model selected jointly across B/C/D | Beta block-mean companion forecast, plus signed high-minus-low gain component |
| `C_to_D` | `C_Aall_raw` → `D_Aall_raw` | Same rule for C→D | Corresponding beta block-mean companion and high-minus-low component |
| `A_all_to_A_N` | `D_Aall_raw` → `D_AN_raw` | Membership/count reduction using the single D prediction bank | Required size control; cannot be relabeled the headline mechanism |
| `A_N_to_R_N` | `D_AN_raw` → `D_RN_raw` | Membership only, with identical N, bin quotas, predictions and voxel scores | Frozen development advantage reaggregated under R_N minus mean-bank inclusion weights |
| `raw_to_ceiling_normalized` | `D_VNC_raw` → `D_VNC_normalized` | Positive voxel weights only | Frozen development advantage reaggregated under `(1/max(NC_D,0.10)-1)/|V_NC|` |

Every pair receives the same images, concept folds, weights and registered
selection budget. Support-specific predictions, beta-stage-specific primary
hyperparameter tuning, differing support quotas or differing ceiling-edge
voxels make an edge composite/invalid; they are not alternative primary
contracts. Shared endpoints are computed once from their common prediction
bank, not refitted to suit each edge.

## Explicit simultaneous families and classification subsets

Define two additional raw-scale explanatory components per pair:

`H_BC = mean_high_gain(d_C-d_B) - mean_low_gain(d_C-d_B)`;

`H_CD = mean_high_gain(d_D-d_C) - mean_low_gain(d_D-d_C)`.

Their high/low sets come only from the frozen calibration gain maps and
original-column median ties. Each uses pooled-image primary R² and the same
eight-group aggregation above. Each practical margin is0.01 raw-R² units.
They are not the block-mean forecast companion, nor primary model-relation
endpoints. The two-sided intervals allow the required signed direction to
be specified for the final relation without changing the family or formula.

| Family | Fixed eligible component roster | Intended inference |
| --- | --- | --- |
| `F_raw` | All six raw primary endpoint IDs, plus each of `H_BC` and `H_CD` that passes its prespecified calibration/design eligibility | Ordinary raw-scale comparisons, including beta/support/size edges |
| `F_raw_normalized` | The same complete `F_raw` roster plus `D_VNC_normalized` | Raw→normalized comparison and full-contract robustness spanning all seven primary endpoints |

Use one common set of10,000 participant×concept-block resamples across every
component in the chosen family and the adopted studentization/max-absolute-t
procedure. Raw and normalized effects and margins retain their own scales;
only their standardized deviations enter the common maximum. No numerical
average of raw and normalized effect values is an endpoint.

The complete primary classification set is the frozen eligible subset of the
**seven primary endpoint IDs**. Full-contract stability requires every such
endpoint to clear the same signed practical margin using the joint family's
simultaneous intervals whenever the normalized endpoint is eligible. If it is
structurally/calibration-inapplicable under the frozen rule, the family reduces
to F_raw before any candidate scores. Raw-only stability may be reported as
raw-only; it cannot silently stand in for the complete eligible contract family.

The H components contribute to the max-t critical value but are **not**
extra primary endpoints for stable-positive, stable-negative or primary
equivalence classification. For a beta *explanation*, the selected edge's
H component must separately clear its frozen signed margin. A stable model
relation does not logically require nonzero reliability-associated stage
changes. Conversely, a significant H component cannot rescue endpoint
nonreplication or create a reversal. A reversal requires its two primary
endpoints to cross opposite margins, not merely a nonzero endpoint difference.

These are within-registered-pair families. P1/P2/P3 are dependent comparisons,
not independent replications. The policy's family language is relation-specific;
this proposal does not claim an experiment-wide multiplicity guarantee across
adaptive model-pair/search choices. One selected pair/relation receives the
write-once audit with its entire frozen eligible family. The required random
falsifier remains nonpromotion-eligible and cannot become a replacement pair.

Fisher-z held-out correlation and linear CKA retain separate scale-specific
component families and their own0.02 practical margins/0.04 full-width
adequacy limits. Correlation is a sensitivity; CKA is a secondary linking
falsifier. Neither receives noise-ceiling division, joins the raw numerical
effect scale, replaces R² or becomes an independent candidate-ready terminal.
The fixed secondary contract roster on each scale is `B_Aall`, `C_Aall`,
`D_Aall`, `D_AN`, `D_RN`, and `D_VNC`, with the same stages/supports as their
raw-R² counterparts. Each scale has its own per-pair max-t family over that
complete eligible roster. Fisher-z prediction correlation is calculated per
voxel before support/ROI/participant averaging; A_N averages the 32 subset
means. Rank-64 CKA is calculated for each whole support and A_N averages the
32 subset CKAs. CKA uses the adopted encoding-selected layer and development
training transform. There is no ceiling-normalized duplicate on either scale.
Eligibility freezes before any secondary score; a model-specific undefined
statistic invalidates the comparison and cannot delete images, voxels or a
family member. These secondary calculations remain unexecuted. At final lock,
name at most two secondary checks from the fixed rosters. A linking-dependence
claim needs opposite-margin encoding and CKA components on the same contract,
with their respective family bounds; neither rescues a failed primary claim.
No joint cross-scale secondary multiplicity guarantee is asserted.

## Freeze eligibility before scores; preserve failures

Before the first candidate-discriminating score, record each endpoint/edge/H
component as eligible or inapplicable, with its exact original columns,
calibration artifact, image/role mapping, score scale, required groups and
reason. This primary subset and all eligibility rules then remain fixed.

| Scope | Prespecified inapplicability/refusal source | What must not happen |
| --- | --- | --- |
| Any primary endpoint | Structural source/ROI/axis failure or absent required calibration-qualified support in any required group | No dropping a person/ROI, relabeling anatomy, replacing columns or treating a technical failure as scientific null |
| A_N or R_N endpoints | Fewer than32 eligible columns in a fixed bin; R_N also requires32 finite D reliabilities | No lower N, moved bins, altered quotas or bank reroll |
| Both V_NC endpoints | Invalid/unavailable finite calibration-ceiling support in any required group | No positive-only deletion after scores, public mixed-role ceiling substitution or denominator repair |
| H_BC or H_CD and the corresponding beta explanation | Undefined calibration gain, rank-deficient eight-bin-plus-gain design or residual standardized gain variance below1e-8 | Keep the beta primary endpoints if they otherwise qualify; do not choose a different gain model or strata from development effects |

A later missing prediction, invalid frozen-support development denominator,
nonfinite model score or failed audit counterpart is a failed paired trial /
incommensurate audit as applicable. It cannot retrospectively remove that
component from the frozen family. All bootstrap draws and components must be
defined; zero/nonfinite studentization does not permit dropping a component,
redrawing blocks or shrinking the family.

The policy permits prespecified structural/calibration inapplicability, but
also separately requires15 valid initial pair×edge trials and32–72 valid
trials overall. An inapplicable endpoint/edge is **not automatically a valid
coverage trial**, and family bookkeeping does not waive that budget. If an
actual missing edge makes required coverage impossible, report the concrete
conflict for prospective design resolution rather than silently count it.

## Exact audit behavior and forecast counterparts

For every frozen primary endpoint, apply the final all-development feature
transform, target mean and appropriate B/C/D coefficients directly to eligible
audit images. Retain the same source-column IDs, calibration supports,32-member
bank, q/N, ceiling denominators and aggregation. No audit response mean enters
the R² baseline. No layer/rank/penalty selection, response refit, support update,
reliability recalculation or second audit is permitted.

For support and ceiling explanations, the scalar forecast is frozen from
development d and the fixed operator weights before audit. The evaluator
obtains the observed counterpart by applying those same weights to audit d.
The forecast must have the registered direction and absolute error≤0.02 for
support or≤0.05 for the specified raw-to-normalized edge-change quantity.
Development reaggregation is an identity; audit agreement tests repeatability.

For a beta explanation, use the registered12/12 development-block OLS
cross-fit, then refit the same equation on the24-block development mean and
freeze its predicted **block-mean** change. The observed audit target is the
equal mean of the12 audit-block R² stage changes, each using the frozen
development predictor/mean before native-voxel/ROI/participant aggregation.
Require its registered direction and absolute error≤0.02. Do not substitute
pooled-image primary R² for that companion target. The selected H contrast
continues to use pooled-image primary R² on its frozen high/low sets and must
clear its signed0.01 margin.

The existing audit gates still apply: simultaneous required-component bounds,
the same intersection of at least three participants clearing every required
signed component, and preserved directions in every leave-one-participant and
leave-one-audit-block mean. Maximum full interval widths remain0.02 for raw
R²/H and0.10 for normalized R²; inadequate precision is underidentification,
not evidence of a tie or adequately precise nonreplication.

For the participant gate, first average each participant's two ROIs equally.
For the frozen relation-specific required component set J and its pre-audit
directions sign_j, participant s passes exactly when every
`sign_j * theta_sj > margin_j`. Require the same intersection of at least three
participants to pass every component; equality fails. These are participant
point estimates, not newly introduced within-person confidence intervals.
All four people remain in the group estimand and crossed bootstrap. J contains
all primary endpoints for robustness; the two opposed endpoints for a reversal
and the selected H component additionally for a beta explanation; or the
matched encoding/CKA components for secondary linking dependence. Forecast
sign/error checks are separate. Unselected H components still contribute to
their joint bounds but do not become required participant stability components.

Every leave-one-participant and leave-one-audit-block check must retain each
required signed direction (`sign_j * theta_without > 0`); no extra margin
crossing is required there. Recompute the specified statistic on retained
observations, pooling image SSE/SST before primary R². The beta block-mean
forecast companion remains a distinct estimand. No available-case averaging,
participant removal or block-R² substitute for pooled primary R² is permitted.

The seven endpoints/five edges directly enumerate the policy's initial
contracts. Including both eligible H contrasts in max-t while reserving
stability classification for primary endpoints is consistent with its
separate `contract_robust_relation` and `predicted_measurement_reversal`
component lists. Per-pair multiplicity, secondary rosters and the participant
gate are now fixed. Actual calibration eligibility remains unobserved. If it
makes required valid-trial coverage impossible, that concrete conflict needs
prospective resolution; this table does not waive the coverage requirement.
