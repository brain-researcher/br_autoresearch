# Episode 14 inputs

`inputs/` is read-only after provisioning.

The ignored directory `openbhb/exposed/` contains the development feature
matrices, train labels, fixed development split manifest, and atlas reference
text. `openbhb/audit_features/` contains feature-only public-validation rows.

Raw `participants.tsv`, original participant identifiers, and validation
targets are not retained here. They are fetched into an external temporary
directory only by the trusted provisioning/audit process. This reduces
accidental leakage but does not make publicly downloadable labels secret; a
final audit also requires a network-disabled worker.

Do not write candidate artifacts, predictions, or caches under `inputs/`.
