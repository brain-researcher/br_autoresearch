# Proposed EP09 donor-supported primary population — October 18, 2026

**Unadopted scope proposal, not an executable complete design.** The recommendation
is a donor-supported own-versus-donor primary with a separate all-eligible
benchmark, 1,000 µm primary matching and a report-only 500 µm sensitivity. This
implements the [reviewer's direction](../README.md#companion-modeldonor-contract-draft-v01-unadopted),
with the population, target and pseudo-audit consequences made explicit below.
It does not approve the complete companion draft or authorize an outcome opening.
Primary measurement-error qualification and a complete final-mask validity
envelope remain unresolved pre-lock prerequisites, as specified below.

The original [companion draft](companion_contract_v0_1_draft.yaml) remains
unchanged. If adopted, this amendment would replace its D01/D02 recommendations,
every-animal primary support rule, affected fixed-group inference assumptions and
D12 split construction. All other unadopted companion choices remain proposals.
CEBSIT, Cux2/Wfs1 L2/3–L4 eligibility, 27 development/16 audit roles, measurement
v1.2, the exact vocabulary and stage-A freezes remain as adopted. No animal moves
between roles; an animal outside the primary scoring population retains its role.

## Why this choice is proposed

The [existing outcome-free census](../README.md#recipient-and-donor-support-outcome-free)
reports the following after same-hemisphere and depth-difference ≤0.10 restrictions:

| Distance | Supported cells, development / audit | Animals with ≥4 supported cells | Telencephalic V6 targets passing the proposed anatomical floor |
| --- | --- | --- | --- |
| 1,000 µm | 733 / 345 | 24 / 16 | 119 / 170 |
| 500 µm | 610 / 241 | 21 / 13 | 121 / 170 |

The 1,000 µm value is the uncapped animal-balanced 90th percentile of the
development fourth-peer distance, 987 µm rounded upward to 25 µm. The earlier
500 µm cap loses more recipients and animals; both widths can exceed the existing
12-development/8-audit minimum after correcting the population rule. These are
feasibility counts, not final qualified donor populations or observed target
support. The different target counts also reflect different included animals.

Wider matching improves sample support while weakening the local-neighborhood
control. Neither width is established here as biologically sufficient matching.
Exact animal × corrected source area × layer × genetic-line pools, hemisphere,
depth and donor diagnostics still matter. A positive own-dendrite advantage would
weaken interchangeability with the specified peers; it would not rule out a finer
unmeasured subtype or neighborhood. A meaningful equivalence requires a known
positive effect detected by the same pipeline at the same margin. Otherwise it
remains underidentified. The 500 µm sensitivity assesses dependence on this width
without selecting the more favorable result.

## Exact proposed populations and scores

Let `E_g` be the frozen eligible cells of animal `g`; `A_pre(i,t)` is the adopted
soma/atlas support indicator; `O(i,t)` is the adopted final observability indicator.
Neither the donor graph nor the animal set may use `O`, axonal geometry or labels.

For each animal, build the companion's no-self donor graph using valid primary
dendritic representations and the exact pools above, same hemisphere, absolute
depth difference ≤0.10 and Euclidean distance ≤1,000 µm. Retain the draft's
iterated four-donor degree pruning, perfect-matching requirement and removal of
edges absent from every feasible matching. Call its retained recipients `R_g`.
Missing/invalid primary dendrites cannot be recipients or donors. The independent
quality limitation remains: the constant release flag does not measure individual
reconstruction quality. Statistical diagnostic failure blocks the primary; it
does not authorize discarding offending animals or recipients.

For role `r`, define `G_C,r = {g in role r: |R_g| >= 4}`. Require at least 12
development and 8 audit animals, using the existing policy minimums. Define
`G_D,dev` as all 27 development animals and `G_D,audit` as all 16 audit animals.
The census suggests `|G_C,dev|=24` and `|G_C,audit|=16`; it does not freeze those
numbers. Development `R_g` and `G_C,dev` must freeze without development outcomes.
Audit `R_g` and `G_C,audit` are instantiated only at authorized stage E, from
audit dendrites using the locked rule, before audit axonal/target information is
used. Existing audit soma-based graphs are feasibility information only. No new
audit dendrite access is requested by this document.

All eligible development cells remain training observations where observable.
M0/M1 selection uses the existing all-eligible native-prediction objective and
identical opportunities; donor advantage never selects the model. Unsupported
cells remain available to the all-eligible benchmark under the specified missing
feature rules. Primary and benchmark use the same fitted correct-M1 pipeline.

For an analysis target set, average squared error over its eligible observable
recipients within animal and target, then equally over targets, then equally over
the appropriate animals. For `C`, recipients are `R_g`, animals are `G_C`, and
the paired contrast is each donor-plan score minus the correct-M1 score, averaged
over frozen plans **after scoring**. For `D` and nuisance comparisons, recipients
are `E_g` and animals are `G_D`. Within each contrast, models use identical cells,
targets and masks; no model-specific row deletion is allowed. `C` and `D` now
have explicitly different populations. A broad benchmark win is not evidence
that the same gain holds in the narrower primary population, and cannot replace
an unavailable primary. Also report `D` on the primary population as a supporting
description, without making it a new success gate.

## Prospective target and support-filter order

1. Retain the frozen all-eligible candidate vocabulary `V=V6` (666 targets).
   Do not rerun n_coverage, change the vocabulary or require primary support in
   animals outside `G_C`. A_pre primary-support counts are feasibility bounds.
2. At authorized development measurement, construct `T_D,dev` using the adopted
   rule: all targets in `V` with ≥6 observable eligible cells in every development
   animal and ≥4 development animals with detections and ≥4 with valid
   non-detections. A mixed animal may count in both categories. Define
   `N_C(g,t) = sum_{i in R_g} O(i,t)` and
   `T_C,dev = {t in T_D,dev: N_C(g,t)>=4 for every g in G_C,dev}`.
   Include every qualifying target; log the support reason for every exclusion.
3. Preserve 27 outer development folds and all-eligible model training. Each
   training fold alone determines the existing ≥3-positive/≥3-negative target
   eligibility; held-out labels never do. Apply the primary four-recipient
   support rule to all primary animals participating in a comparison. Generate
   the required out-of-fold predictions over the complete `T_C,dev` inventory
   when its training eligibility holds; verify that condition rather than
   imputing absent predictions. Primary summaries/template use only `G_C,dev`.
4. Before the shared lock, freeze prediction procedures for every `T_D,dev`
   target, and primary explanatory entries/scales/templates/sign choices over
   `T_C,dev`. Freeze the filtering algorithm; final target IDs can still depend
   on the already-adopted, outcome-blind audit observability stage.
5. At authorized stage E, finalize
   `T_D = {t in T_D,dev: sum_{i in E_g}O(i,t)>=6 for every audit g}` and
   `T_C = {t in T_D intersect T_C,dev: N_C(g,t)>=4 for every g in G_C,audit}`.
   This is a target-level intersection over a single fixed animal set, not a
   different animal subset for each target. Audit detections, prevalence,
   predictions and effects never select targets. Target removal here is the
   proposed deterministic support rule, not permission to repair the result.
6. Freeze final sets before scoring. Empty `T_C` or failed animal minimums means
   primary unavailable; `D` cannot rescue it. Four observable recipients is only
   a structural candidate-support floor. The finalized `G_C`, `T_C` and complete
   observation mask must also pass the locked measurement-error/validity envelope
   below before any primary effect score is computed. An out-of-envelope mask
   rejects the primary; it does not trigger another target or animal filter.
   After finalization, a missing pair or support violation is a technical failure,
   with no further deletion or renormalization. Donor/diagnostic failures cannot
   be relabeled as ordinary target-support attrition.

## Uncertainty, explanation and pseudo-audits

Use each contrast's actual fixed number of animals for its sample SD, standard
error, Student-t floor, calibrated bounds and leave-one-animal-out influence
rules. Never reuse `G=16` merely because 16 audit animals were assigned. Calibrate
and independently validate the locked inference recipe for the outcome-free
support counts to which it may be applied; an unqualified group count makes that
analysis unavailable. Before the shared lock, qualify each allowed primary audit
count from 8 through 16; the broad benchmark keeps 16. Qualify the development
and pseudo-audit counts for their respective procedures, including any allowed
pseudo-audit primary count from 8 through 15. Cells, targets, donor plans and
repeated pseudo-audits do not supply additional independent units. Scientific
margins remain unset. Count-specific calibration alone is insufficient: final
target identity, recipient masks, target overlap and weighting also determine
the statistic, and must be covered by the pre-lock envelope below.

E0/E1 comparisons and scale-by-target matrices use the primary population,
`G_C` and `T_C`; they cannot borrow the broader benchmark's group count. Compute
development template averages, scales and sign support using `G_C,dev` only.
Project locked columns from `T_C,dev` to `T_C` without reselection or refitting.
The existing family gate is assessed on the target set of the reported analysis:
`T_C` for the primary explanation/hierarchy, `T_D` for broad benchmark summaries.
No family failure removes exact targets or rescues a failed primary.

For each of the draft's three pseudo-audits, select 12 discovery animals from
the development-only, outcome-free `G_C,dev`, using the draft's pseudo-audit seed
1900921 and stream convention. Seed-permute its sorted IDs once; take 12 cyclic
consecutive IDs starting at `floor(k*|G_C,dev|/3)`, for `k=0,1,2`. All other development animals form the
pseudo-audit, retaining a 12/15 split for all-eligible model/benchmark work. At
the census support of 24, the primary initially has 12 discovery/12 pseudo-audit
animals. Repeat the uncapped width derivation and donor qualification on the
12 discovery animals only; freeze that recipe before pseudo-audit outcomes.
Then apply the same `R_g`, `G_C` and target rules within each role. Require the
existing ≥12 primary discovery/≥8 primary pseudo-audit minimums; recomputation
may fail them. Report every failure without redrawing a split or changing width.
Calibrate actual effective group counts, not the draft's automatic `G=15`.
These overlapping development checks reuse declared outcome-free geometry and
support for split construction; they are not independent biological replications.

## Measurement-error and final-mask prerequisites — unresolved

Retain the adopted measurement-error tolerances for the primary `C` statistic:
**error SD ≤0.05 and the 95th percentile of absolute error ≤0.10 in every
required scenario**. Error is the simulated paired contrast minus its conditional
expectation, using the exact recipient-then-target-then-animal weighting. These
are measurement-error limits, distinct from biological uncertainty and power.
The [v1.2 rule](../../attempt005/measurement_contract_v1_2_adopted.yaml)
(`n_coverage.simulation.pass_for_every_scenario`) supplies both tolerances.

The [original report](../ncoverage/ncoverage_report.json) records n=4 failing
even in the broader design: worst error SD **0.0586831**, P95 absolute error
**0.1111111** (`per_n`, n=4); n=6 passes at **0.0473703/0.0888889**.
The four-recipient floor therefore supplies no precision certification. Nor can
the existing n=6 result certify fewer animals, different masks or a smaller
target set. Keep n_coverage=6 and its original scope unchanged. Some actual
four-or-more-recipient configurations might meet the same tolerances, but no
such result exists; the proposed primary may remain infeasible. Required
same-pipeline positive-control/margin recovery is **additional**, not a
replacement for these measurement-error limits. A failure cannot be repaired by
lowering tolerances, lowering margins, changing n_coverage, or selecting a more
favorable subset after audit information is available.

Before the shared lock, the companion must contain an executable, deterministic
`qualify_mask(mask, frozen_context) -> class_id | reject` specification and its
independent validation. This is a hard prerequisite, not an undertaking to
calibrate after seeing the final audit mask. Its required contents are:

1. **Exact domain and representation.** Freeze the allowed animal-count domain,
   target-ID inventory/order, recipient-ID universe from existing manifests,
   admissible donor-eligible subset rule and pool membership, binary-mask encoding
   and resulting cell/target/animal weighting formula. Freeze allowed mask classes,
   not the still-closed actual audit mask or audit-dendrite-qualified recipient
   set. Class membership must account for per-animal/per-target counts, uneven
   counts across animals, shared recipients across targets, retained target sets
   and pool composition. Count G or minimum N alone cannot define membership.
   Pre-lock bounds/allowed classes are presently unspecified.
2. **Complete mask coverage.** Prespecify the stress-mask generator and each
   class's membership predicate before validation, including target loss,
   concentrated coverage loss, unequal support and complete target dependence.
   Use development information and synthetic designs only; audit masks are not
   simulation inputs. A finite list of randomly tested masks cannot certify all
   other masks with similar summaries. Each accepted class must have exhaustive
   coverage or a justified conservative bound covering every mask it admits;
   otherwise admit only its explicitly validated finite mask configurations.
   Do not interpolate or extrapolate to an unqualified target/mask configuration.
3. **Measurement and inferential validation.** For every accepted class, retain
   the adopted measurement scenarios as required stresses and show that the
   exact `C` weighting meets both tolerances above. Cover the registered
   prediction/probability domain rather than assuming the old fixed prediction
   pairs bound every fitted-model contrast. Separately calibrate and independently
   validate the group-level inference, influence and any intended explanatory
   matrix claims over the class's support/dependence configurations; include
   target-set projection and recipient-overlap effects. Freeze the applicable
   critical values and a deterministic class-to-rule mapping. No numerical
   envelope, coverage guarantee or passing result is supplied by this proposal.
4. **Score-blind stage-E decision.** After applying the fixed support rules,
   provide the checker only `G_C`, `T_C`, recipient masks, outcome-free pool/donor
   structure and frozen context. It receives neither target detection labels,
   predictions nor effect scores. Require exactly the registered class membership
   and every validity condition; missing or ambiguous qualification is rejection.
   Return pass/class ID or reject/reason before joining predictions to labels for
   scoring. Rejection makes the primary unavailable, with no extra deletion,
   recalibration, simulation, refitting or alternative envelope fitted to that
   audit mask. An explanation outside its separately qualified domain is
   unavailable even if the primary remains qualified.

The same prospective validity treatment applies to pseudo-audit primary claims
and any inferential interpretation of the 500 µm sensitivity. Until these
specifications and qualifications exist, this document supports a scope decision
only. The unchanged measurement tolerances can coexist with the structural
four-recipient rule because both are necessary gates; their joint feasibility
has not been established. If no defensible nonempty envelope can meet them,
report the proposed primary infeasible rather than silently increasing support
thresholds or dropping the precision requirement.

## Fixed 500 µm sensitivity and remaining dependencies

The proposed 500 µm analysis uses the same fitted primary model, residualization
recipe and other calipers, with its own qualified donor bank and prospective
`R_g`, `G_C` and `T_C` from the rules above. It is report-only, cannot replace
1,000 µm, and cannot rescue a primary decision. Its different population makes
raw effect differences ambiguous. Also report both banks on their common
observable recipient/animal/target intersection using identical paired masks,
equal animal weights and each bank's original plan frequencies; disclose any
further support loss. This is a descriptive comparison, not a new decision gate.
If 500 µm loses support or diagnostic validity, report it unavailable rather than
loosening its rule. No new threshold is introduced for this sensitivity.

The following existing bindings resolve the original draft's obsolete stage-A
placeholders; the frozen files themselves are unchanged:

| Binding | Existing authoritative artifact/value |
| --- | --- |
| Atlas, role/cell manifests, relabel QC, A_pre and target/family resources | Exact paths and existing digests in [stage-A manifest](../stage_a_freeze_manifest.json) |
| Measurement rule | [Adopted v1.2](../../attempt005/measurement_contract_v1_2_adopted.yaml) |
| Precision | [Completed report](../ncoverage/ncoverage_report.json): n_coverage=6; V6=666 |
| Dendrite blocks | [Completed boundaries](../dendrite_boundaries/dendrite_boundaries.json): R_cap=600 µm, radial edges 50/100/150 µm, branch orders 1–3/4–5/>5 |
| Depth/axis | [v2 report](../laplace_v2/cortical_laplace_report.json) and v2 field/frame paths in the manifest; v1 remains superseded |

Before execution readiness, the companion still needs its non-population choices
reviewed/resolved, a qualified donor graph/bank and diagnostic implementation,
the complete final-mask envelope and measurement-error qualification above,
effective-group inference and pseudo-audit propagation, scientific margin
proposals and recovery, and the existing stage/access handoffs. These are dependencies,
not reported completions. In particular, this proposal does not adopt D03–D11's
numeric gates or add an axon-independent individual quality measure that the
release lacks. No new source, role, measurement rule, outcome access, job or
calibration run is activated here.

**Single scope decision:** should EP09 adopt the donor-supported primary and
support-based target rule above, with 1,000 µm matching, the separate all-eligible
benchmark, supported-discovery pseudo-audits and report-only 500 µm sensitivity?
Recommendation: adopt only this scope, conditional on the unchanged precision
limits and completed pre-lock mask qualification. This is not approval of an
executable design or a promise that a usable primary population exists.
