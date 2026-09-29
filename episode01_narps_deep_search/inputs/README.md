# EP01 inputs

This directory contains only the immutable declaration of the two fully
exposed historical NARPS runs:

- `prior_lineage/PRIOR_LINEAGE_MANIFEST.json`

Those records are lineage only and provide no new evidence. EP01 does not read
live sibling outputs.

The active scientific sources remain at their established read-only locations:

- OpenNeuro `ds001734` original event tables provide trialwise gain, loss,
  four-category response, RT, deadline status, run, and sequence structure.
- The complete fMRIPrep 21.0.2 derivatives provide the 432 runwise MNI 2-mm
  BOLD series, confounds, and masks for the controlled fMRI factorial.
- The curated NARPS v2.0.1 archive provides the concatenated unthresholded
  group maps and team method metadata for the many-team estimand audit.

The analysis must use `*_events_ORIGINAL.tsv`, not the transformed
`events.tsv` files produced for earlier analyses. Existing FitLins, switch,
pupil, and other collaborator products are not substituted for the new
runwise models.

No participant-level or many-team result has been generated under the current
cognitive-estimand contract. See [`../DATASETS.md`](../DATASETS.md) for exact
locations, roles, availability, and claim limits.
