# Episode 20 inputs

`inputs/` is read-only after provisioning. In this canonical checkout it
contains only this README: no NeuroCam payload, metadata contract, anchor
manifest, qualification contract, or reference example bundle is provisioned.

The root
[`DATA_LOCATION_MANIFEST.json`](../../DATA_LOCATION_MANIFEST.json) records the
canonical private steward root as `private_steward_acquisition` at
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition`
and identifies the NeuroCam PDF as `asset_ep20_neurocam_documentary`. The
steward tree was relocated by same-filesystem rename on 2026-09-24 and is
`canonical_in_place`. No source has been handed off to this directory.

Two JSON examples exist only at `legacy_ep20_reference_bundles`, whose
destination is unassigned, and are not canonical EP20 inputs. If a future
trusted provisioning step copies, hashes, and role-labels
them under `reference_bundles/`, their intended identities are:

- `neurocam_paper_direct_v1.json` is labeled `reference` and `example`;
- `neurocam_figure_derived_v1.json` is labeled `reference`, `example`, and
  `estimation`.

Both would be nonbinding inputs for documentary context and
non-candidate-discriminating simulation sanity checks only. Neither may
initialize model parameters or the search space, enter reference calibration,
reference qualification, candidate scoring, or audit evaluation. They cannot
substitute for raw I-V/C-V, a compact model, PDK, netlist, mask/layout, or
fabrication outcomes.

Absent contracts and legacy-only examples are not configuration-lock eligible
and do not provide the data or contracts required for a fit, qualification run,
candidate score, or audit opening. No
NeuroCam article payload, raw trace, simulator, design-rule package, empirical
replay, sealed audit asset, compact model, PDK, or fabrication outcome is
present here.

[`../DATASETS.md`](../DATASETS.md) requires content-addressed, role-filtered
handoffs for documentary reference, reference calibration, reference
qualification, development signal and device ensembles, virtual audit,
empirical replay, and hardware rules. Do not mount one mixed source tree for
development and audit roles. Published documentary qualification values remain
visible but role-locked out of fitting; public availability does not grant the
controller access to an independently acquired evaluator-only raw trace, a
sealed plausibility partition, or an audit asset.

Before candidate scoring or audit access, this directory must contain frozen,
hashed versions of at least:

- `SOURCE_MANIFEST.json` and `NEUROCAM_ANCHORS.yaml`;
- `REFERENCE_ANCHOR_SPLIT.yaml`, `REFERENCE_MODEL_CONTRACT.yaml`, and
  `REFERENCE_QUALIFICATION.yaml`;
- `FROZEN_HARDWARE_GRAMMAR.yaml`, `DESIGN_RULES.yaml`, and
  `RESOURCE_MODEL.yaml`;
- `SOFTWARE_BUDGET.yaml`;
- `DEVELOPMENT_ENVIRONMENTS.yaml`, `DEVELOPMENT_DEVICE_ENSEMBLE.yaml`, and
  `DEVELOPMENT_DRAW_MANIFEST.json`;
- `AUDIT_COMMITMENT.json` and `EVALUATOR_CONTRACT.yaml` for the independently
  implemented audit;
- disjoint `EMPIRICAL_DEVELOPMENT_REPLAY_MANIFEST.json` and
  `EMPIRICAL_PLAUSIBILITY_REPLAY_MANIFEST.json` records;
- `EXPOSURE_LEDGER.json` identifying every role and permitted reader.

Their required semantics and unresolved fields are specified in
[`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml). These files are planned
interfaces, not present inputs. A filename, reported aggregate, legacy example,
or estimate is not evidence that a primary payload, fitted model,
qualification result, or real device asset exists.

The repository intentionally ignores per-episode `inputs/` payloads except for
this README. Any future contract or example bundle must therefore be
provisioned and verified separately with the trusted provisioning environment;
committing the tracked documentation alone neither supplies a self-contained
reference bundle nor promises a repository-local qualification check.

No candidate output, cache, fitted model, generated field, qualification
residual, audit QC, partial score, or result may be written under `inputs/`.
After a trusted qualification run, the qualifier writes
`outputs/reference_model_ensemble_manifest.json` and
`outputs/reference_qualification_receipt.json`; those immutable runtime
artifacts are never input manifests and do not exist in this draft.
