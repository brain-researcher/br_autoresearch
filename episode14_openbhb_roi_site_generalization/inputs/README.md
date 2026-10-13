# EP14 inputs

`inputs/` is read-only after provisioning.

`openbhb/exposed/` contains the development ROI matrices, development age and
`siteXacq` labels, the three fixed RAMP robustness views, and atlas reference
text. These inputs may support the fixed four-arm development comparison.

`openbhb/audit_features/` contains 757 public-validation feature rows with
opaque identities. They are feature-only and must not be used before one
candidate is locked and the exposure review establishes a valid one-shot role.

Public-validation age, `siteXacq`, and internal/external role labels are absent
from this workspace and remain closed to candidate code. Do not retrieve,
search for, reconstruct, or join them during design or development work.

Do not write code, predictions, caches, or reports under `inputs/`. Durable
artifacts belong under `outputs/`; transient execution belongs in the
episode-specific Scratch directory.
