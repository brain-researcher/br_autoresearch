# Episode 20 inputs

This directory currently contains documentation only. No NeuroCam source
file, fitted reference model, simulator, empirical replay, hardware rule deck,
or audit material has been provisioned for EP20.

## Documentary source

The NeuroCam article and supplement are currently stored outside the episode:

    /oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/neurocam_documentary_20260922

That location is steward storage, not permission to use the material. Before
EP20 opens it, record the citation and rights, confirm the article and
supplement version, and assign each extracted value to documentary,
calibration, or qualification use.

Two legacy JSON examples also exist outside this directory. They may be used
only as nonbinding documentary context if a scientist approves that role.
They may not fit or qualify the reference model, initialize the search, score
a candidate, or substitute for raw device evidence.

## What must be ready before candidate scoring

- an approved article and supplement with clearly sourced anchors;
- a calibration-versus-qualification assignment and trusted qualification
  result;
- fixed hardware, resource, software, endpoint, and randomization rules;
- executable development signal and device models with predeclared draws;
- separate development and sealed empirical sources or partitions; and
- a separately permissioned audit owner and location, opening time, and
  allowed final report.

These records may be short, human-readable files. EP20 does not require
particular filenames.

## Access and write rules

- Provisioned scientific payloads are read-only. This README may be updated as
  source roles and availability change.
- Development and audit material must not share a mount or readable directory.
- The adaptive runtime may not read sealed parameters, traces, previews,
  scores, or quality-control output before candidate lock.
- Generated fields, fitted models, caches, scores, qualification results, and
  audit reports belong under `outputs/`, never `inputs/`.
- Large governed sources may remain outside the repository when this file and
  `../DATASETS.md` give a clear location and scientific role.

For the full source-role explanation, see [`../DATASETS.md`](../DATASETS.md).
For the search rules, see
[`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml).
