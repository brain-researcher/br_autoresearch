# EP11 Stage-0 steward handoff

**Run:** `ep11-stage0-20260925T000506Z`
**Scope:** identity, role isolation, redaction, atlas validation, and source
support only. No candidate-visible projection outcome is authorized.

## Why a trusted handoff is required

The acquired workbook mixes safe identity/soma metadata with the forbidden
axon-derived `Projection class`, and all morphology archives mix future
development and audit animals. No authenticated mapping currently establishes
whether the 39 provider Brain IDs represent 39 independent animals. Directly
mounting these sources into EP11 would therefore fail both the outcome firewall
and the independent-animal design.

Mechanical hashing, container validation, identity joins, and allow-list
redaction do not by themselves constitute candidate outcome exposure when no
projection values or diagnostics leave the trusted boundary and the complete
operation is logged. Candidate-visible axonal geometry, projection fields,
summaries, errors, or plots do constitute exposure. Missing access records are
handled conservatively: affected animals cannot be presented as fresh audit.

## Required trusted products

1. **Acquisition and license receipt.** Preserve provider bytes, MD5 and local
   SHA-256 values, license/API response, and source paths.
2. **Authenticated identity crosswalk.** Resolve provider cell and Brain IDs to
   independent animals/specimens using external provenance; identify linked
   IDs, duplicate reconstructions, reconstruction versions, and unresolved
   cases.
3. **Allow-listed metadata views.** Implement
   [`redacted_metadata_contract.yaml`](redacted_metadata_contract.yaml), fail
   closed on unknown fields, and prove that no projection-derived value or
   row-level audit identity entered a candidate-visible product.
4. **Shared EP09--11 exposure/role ledger.** Assign whole verified animals to
   `development`, `audit_sealed`, or `excluded`; include duplicate families,
   assignment algorithm/seed, sibling exposure history, first-outcome-access
   times, and row/ledger hashes. An existing provider Brain ID is not enough.
5. **Source-support census.** For every candidate source, report verified
   animals and eligible cells by role, position/depth/layer coverage, local
   cross-animal overlap, missing intermediate locations, experimental strata,
   exclusions, and precision inputs. Candidate output may contain aggregate
   audit support but not audit cell identities.
6. **Frozen role-safe views.** Emit a content-addressed development metadata
   view and an evaluator-only audit view. Do not symlink the mixed-role source
   archives into the episode.
7. **CCF binding receipt.** Validate NRRD orientation, axes, spacing, label IDs,
   ontology version, and compatibility with SEU registered coordinates before
   any regional projection outcome is constructed.
8. **Access receipt.** For every operation, record actor/service, executable and
   environment hashes, exact inputs/outputs, emitted schema, row/join counts,
   visibility, start/end UTC, and failures without leaking forbidden values.

## Completion and failure semantics

Stage 0 succeeds when it yields at least one source that passes every frozen
identity, cell-support, spatial-overlap, missingness, precision, and
experimental-stratum rule, plus an isolated development set on which genuine
projection exploration may safely begin. The present 12-development/8-audit
minimum is a design-specific requirement to be justified by precision and
synthetic recovery, not a universal threshold.

If no source passes, `closed_insufficient_source_support` means only that the
current resource cannot execute the current EP11 design. Before projection
outcome access, the team may prospectively version a narrower question,
different sample-size rule with a new precision basis, or additional data.
After exposure, such a change requires a new round and exposure record. Neither
case is evidence that reusable projection groups do not exist.

## Current blocker

No SEU-specific trusted builder or permission-separated evaluator handoff was
found. The current candidate process must stop at this specification rather
than opening `Full_morphometry.xlsx` or any morphology ZIP.
