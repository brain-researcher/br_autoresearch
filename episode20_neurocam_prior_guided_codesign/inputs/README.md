# Episode 20 inputs

This directory currently contains documentation only. No NeuroCam article,
fitted paper-derived reference, cortical-field model, electronics model,
physical-signal source, phantom source, or design rules have been prepared as
EP20 inputs.

2026-10-03: one Rat5 acquisition file from DANDI 000932 is staged outside Git
under [`outputs/cache/physical_signal_development/dandi000932/`](../outputs/cache/physical_signal_development/dandi000932/).
It is not yet a qualified input: the NWB stores 5-kHz voltage data but zeroed
electrode coordinates and incomplete acquisition-history metadata. See the
[source selection and limitations](../DATASETS.md#selected-acquisition-candidate-dandi-000932)
and [observed header checks](../outputs/experiment_log.md#2026-10-03---model_specification---physical-source-acquisition-attempt-001).
Rat3 is a whole-animal final-source candidate and remains undownloaded/unopened.

## What must be ready before candidate scoring

- approved article and supplement use, with clearly sourced published values;
- a calibration subset and a distinct held-aside subset of those values;
- a paper-derived reference that passes the held-aside checks;
- finite pad, scan, software, endpoint, random-draw, and resource rules;
- executable development cortical-field and device models;
- independently implemented field and electronics models for the final test;
- separate development and final physical-signal or phantom data; and
- a pre-set catastrophic-plausibility threshold.

These can be short, human-readable files. Particular filenames are not
required.

## Read and write rules

- Prepared scientific inputs are read-only.
- Development and final-test sources must not share a readable mixed directory.
- Development code may not inspect final-model parameters, traces, previews,
  scores, ranks, or candidate-specific quality summaries before one design is
  selected.
- Generated fields, fitted models, caches, and scores belong under `outputs/`,
  never `inputs/`.
- Large sources may remain outside the repository when `../DATASETS.md` states
  their scientific role clearly.

For the full source-role explanation, see [`../DATASETS.md`](../DATASETS.md).
For the search rules, see
[`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml).
