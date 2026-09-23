# Episode 20 inputs

`inputs/` is read-only after provisioning. This scientist-authorized
provisioning step added five metadata-only contracts for the paper-derived
NeuroCam reference lane. They record source identity, reported aggregate
anchors, a proposed fit/qualification split, the permitted reference-model
scope, and a pre-run qualification gate.

A follow-up provisioning step adds two EP20-local JSON examples under
`reference_bundles/`:

- `neurocam_paper_direct_v1.json` is labeled `reference` and `example`;
- `neurocam_figure_derived_v1.json` is labeled `reference`, `example`, and
  `estimation`.

Both are nonbinding inputs for documentary context and
non-candidate-discriminating simulation sanity checks only. Neither may
initialize model parameters or the search space, enter reference calibration,
reference qualification, candidate scoring, or any launch gate. They cannot
substitute for raw I-V/C-V, a compact model, PDK, netlist, mask/layout, or
fabrication outcomes.

These contracts are not configuration-lock eligible and do not authorize a
fit, qualification run, candidate score, or launch. No NeuroCam article
payload, raw trace, simulator, design-rule package, empirical replay, sealed
audit asset, compact model, PDK, or fabrication outcome is present here.

[`../DATASETS.md`](../DATASETS.md) requires content-addressed, role-filtered
handoffs for documentary reference, reference calibration, reference
qualification, development signal and device ensembles, virtual audit,
empirical replay, and hardware rules. Do not mount one mixed source tree for
development and audit roles. Published documentary qualification values remain
visible but role-locked out of fitting; public availability does not grant the
controller access to an independently acquired evaluator-only raw trace, a
sealed plausibility partition, or an audit asset.

Before launch, this directory must contain frozen, hashed versions of at least:

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
[`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml). The five present reference
contracts instantiate metadata interfaces only, while the two JSON bundles are
explicitly nonbinding examples. A filename, reported aggregate, or estimate is
not evidence that a primary payload, fitted model, qualification result, or
real device asset exists.

The repository intentionally ignores per-episode `inputs/` payloads except for
this README. These five contracts are therefore durable OAK-provisioned inputs,
and the two example bundles follow the same provisioning boundary. They are not
files carried by a code-only checkout. Any future release must provision and
verify them separately with the trusted provisioning environment; committing
the tracked documentation alone neither supplies a self-contained reference
bundle nor promises a repository-local preflight check.

No candidate output, cache, fitted model, generated field, qualification
residual, audit QC, partial score, or result may be written under `inputs/`.
After an authorized qualification run, the trusted qualifier writes
`outputs/reference_model_ensemble_manifest.json` and
`outputs/reference_qualification_receipt.json`; those immutable runtime
artifacts are never input manifests and do not exist in this draft.
