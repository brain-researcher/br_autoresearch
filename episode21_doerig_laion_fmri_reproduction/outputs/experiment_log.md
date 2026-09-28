# Experiment log

Recorded on: 2026-09-28.

Current handoff:

- Access stage: contract-only; no EP21 neural outcome has been accessed.
- Active analysis: none.
- Blockers: empirical provisioning and execution have not been authorized,
  and the four role-specific interfaces documented in `inputs/README.md` have
  not been provisioned.
- Next analysis: `source_qualification` (blocked on authorization and
  provisioning).
- Exact next action: after an explicit scientist instruction authorizes
  empirical work, provision and review the read-only redacted structural
  manifest and outcome-blind model-asset inventory, then run source
  qualification without mounting either later-stage handoff.
- Active jobs or processes: none.
- Neural scoring: forbidden until `pipeline_conformance` emits a valid
  `outputs/protocol_lock.yaml`.
- Program: `outputs/astra/v0.0.14/astra.yaml`.

## source_qualification

Status: blocked before first attempt.

Blocker: empirical work is not authorized, and
`inputs/laion_regular_structural_manifest.json` and
`inputs/upstream_model_asset_inventory.json` are absent.

Next action: after explicit authorization, provision those two outcome-blind
interfaces; verify release, DUA, risk classification, joins, role membership,
structural coverage, prior exposure, and exact asset identities or verified
absence. Emit structural QC only, with no feature handoff, scoring handoff,
neural value, or OOD payload mounted.

## pipeline_conformance

Status: waiting for `source_qualification`.

Blocker: no qualification report exists.

Next action: after qualification, resolve every shared field and each runnable
module or subclaim in `PROTOCOL_TABLE.yaml`; lock unavailable units as
`not_evaluable`, run synthetic/permuted conformance, and emit
`outputs/protocol_lock.yaml`.

## caption_control_feature_preparation

Status: waiting for the protocol lock and feature handoff.

Blocker: neither prerequisite exists.

Next action: after both exist, prepare the locked caption/control features and
the separate source-lookup and adapted-rank dictionaries.

## rcnn_feature_preparation

Status: waiting for the protocol lock and feature handoff.

Blocker: neither prerequisite exists; required RCNN assets also remain
unauthenticated until qualification.

Next action: prepare only authenticated released RCNN/comparator features and
record unavailable required assets as `not_evaluable` without substitution.

## cross_modal_feature_preparation

Status: waiting for the protocol lock, feature handoff, and caption manifest.

Blocker: none of those prerequisites exists.

Next action: bind all five feature blocks to the locked identity and
preprocessing map, or record the extension as `not_evaluable`.

## finding1_alignment_encoding_decoding

Status: waiting for the protocol lock, scoring handoff, and caption manifest.

Blocker: no scoring prerequisite exists.

Next action: run the locked Finding 1 endpoints only after all three direct
inputs are available.

## finding2_context_controls

Status: waiting for the protocol lock, scoring handoff, and caption manifest.

Blocker: no scoring prerequisite exists.

Next action: run the locked Finding 2 controls and prespecified exception table
only after all three direct inputs are available.

## finding3_rcnn_alignment

Status: waiting for the protocol lock, scoring handoff, caption manifest, and
RCNN manifest.

Blocker: no scoring prerequisite exists.

Next action: run the locked Finding 3 comparison and its owned 240-NSD-origin
exclusion sensitivity only after all four direct inputs are available.

## method1_tau_robustness

Status: waiting for the protocol lock, scoring handoff, and caption manifest.

Blocker: no scoring prerequisite exists.

Next action: run the fixed secondary tau-split analysis without consuming core
finding results or changing their verdicts.

## method2_cluster_k5_robustness

Status: waiting for the protocol lock, scoring handoff, and caption manifest.

Blocker: no scoring prerequisite exists.

Next action: run the fixed held-cluster analysis without consuming core
finding results or changing their verdicts.

## cross_modal_extension

Status: waiting for the protocol lock, scoring handoff, and cross-modal feature
manifest.

Blocker: no scoring prerequisite exists.

Next action: run the separate secondary family and its owned 240-NSD-origin
exclusion sensitivity without consuming or revising core finding results.

## regular_pool_sensitivities

Status: waiting for the protocol lock, scoring handoff, and caption manifest.

Blocker: no scoring prerequisite exists.

Next action: report all ten three-of-five caption subsets and the Findings
1--2 240-NSD-origin exclusion without selection or primary-result replacement.

## final_reproduction_report

Status: waiting for all preceding reports or explicit `not_evaluable` records.

Blocker: no upstream analysis has run.

Next action: integrate the ordered Findings 1–3 verdict vector, robustness,
extensions, deviations, failures, and claim limits after every required input
is present.

The source paper's OOD Method 3 is intentionally absent from the executable
program. EP21 may not open its raw images, derived features, neural outcomes,
predictions, scores, or summaries.
