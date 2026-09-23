# Verification

Validated on 2026-09-21 as a design and source-availability record, not as a
scientific result.

## Source acquisition

- The required CNeuroMod-THINGS 1.0.1 neural/structural subset is present in
  steward-managed, read-only OAK storage. It covers all four frozen
  participants and the B/C/D, structural, annotation, and anatomical source
  classes required by the episode.
- The frozen root and subdataset revisions were checked during provisioning,
  and the official release archive is retained with the source as provenance.
- The final source tree is restricted and read-only. Its annex links resolve
  within that tree; no neural or behavioral array was parsed during source
  acquisition.
- Transfer logs, integrity inventories, and checksum receipts are retained by
  the steward beside the provisioned source and intentionally omitted from
  this code repository. These verification notes do not make a code checkout
  a data handoff.

## Contract checks

- The old NSD episode directory and old EP04/EP17 shared-data labels have no
  remaining references. The only neural source in EP17 is CNeuroMod-THINGS
  1.0.1.
- All 19 formal registry paths resolve. Strict duplicate-key YAML parsing, JSON syntax,
  EP17 role/fold/block/budget/no-refit cross-field checks, shell syntax,
  control-character, trailing-whitespace, and stale-reference checks pass.
- Counts agree: 480 development + 120 calibration + 120 audit concepts; six
  80-concept development folds; 24 development uncertainty blocks; twelve
  audit blocks; and 15 mandatory initial trials for three pairs × five edges.
- The terminal-driving contract universe must be frozen before scoring, and
  the taxonomy release, crosswalk, and label rules are explicit launch
  blockers. This prevents post-score contract deletion and informal semantic
  regrouping.

## Governance and remaining boundary

- No MAT, NIfTI, HDF5, or annotation payload was parsed. No neural score,
  candidate result, audit outcome, canonical transition, or Git operation was
  produced. The pinned CNeuroMod subset is a provisioned, read-only DataLad
  tree. The separately acquired THINGS stimulus archive remains encrypted and
  unextracted in steward quarantine; a central-directory-only
  reconstructability audit read no image pixels.
- The restricted raw bundle deliberately co-locates all concept roles. Read-
  only storage is not audit blindness: it must never be mounted to a search
  worker. Frozen, materialized, role-filtered handoffs remain mandatory.
- The latest read-only canonical refresh found profile
  `codex_autoresearch_v1` and 19 loops, with no EP17 match. No registration,
  checkpoint, reward, authorization, or launch was submitted.
- EP17 remains launch-blocked on the frozen taxonomy, controlled stimulus
  extraction and exact per-image/event hash alignment, exact common-image
  eligibility and 480/120/120 roles, separated
  handoffs, B/C/D alignment and stage-specific ceilings, ROI/support/model/
  exposure/margin manifests, the terminal-contract universe, EP17/EP18
  exposure governance, and a permission-separated evaluator.

These checks establish acquisition and contract integrity only. They do not
establish model-ranking stability or any other scientific conclusion.
