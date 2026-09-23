# Episode 19 inputs

`inputs/` is read-only after provisioning. No role-filtered WAY-EEG-GAL or
self-paced EEG handoff is currently present here. Their verified, unextracted
mixed source archives remain outside the episode in steward quarantine. The
deferred AJILE12 materialization also remains outside the episode and is not a
primary-task fallback.

`../DATASETS.md` requires content-addressed, role-filtered structural,
development, no-feedback lock, and evaluator-only audit handoffs. Do not link a
mixed public download or shared source tree into this directory. Public access
does not grant held-out access to a candidate or controller.

No candidate output, cache, fitted model, onset result, audit QC, or score may
be written under `inputs/`.

Before launch, this directory must also contain
`FROZEN_ADMISSIBLE_SPACE.yaml` and its signed SHA-256 record. That manifest
enumerates every legal search operator and exact configuration range; its hash
is frozen before any candidate EEG outcome is returned. It is intentionally
absent in this unregistered draft.
