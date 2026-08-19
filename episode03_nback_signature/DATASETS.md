# Candidate Data for the N-Back Signature Episode

Workspace:
`/oak/stanford/groups/russpold/users/zijiao/autoresearch/episode03_nback_signature`

This file records what was visible before launch and what still needs to be
checked. It does not assign final discovery or confirmation roles. The active
run may inspect metadata and schemas outcome-blindly, but must not use sealed
neural or behavioral outcomes to choose the signature.

## Candidate sources

| Source | Sherlock path | Known now | Must be checked before use |
| --- | --- | --- | --- |
| HCP-YA BIDS derivatives | `/oak/stanford/groups/russpold/data/HCP_YA/HCP-YA-BIDS` | The derivative root is readable and includes task-WM LR/RL event-file names. Its project README describes HCP-YA derivatives, including fMRIPrep output. | Exact subjects, event labels, usable task GLM/stat maps, confounds, spaces, behavior variables, access terms, and whether a genuine retest is present. |
| HCP PTN resources | `/oak/stanford/groups/russpold/data/HCP_YA/HCP1200_PTN` | Parcellation-timeseries/netmat resources are visible. | Whether any resource represents task-WM data and matches this episode's estimand. Resting-state or unrelated connectivity cannot be substituted. |
| AOMIC `ds002785` and `ds002790` | `/oak/stanford/groups/russpold/users/zijiao/fmri_encoding/openneuro_fitlins` and `/oak/stanford/groups/russpold/users/zijiao/fmri_encoding/_claude_parc/statvox` | Both datasets have top-level roots in `input/`, `fmriprep/`, `analyses/`, `bold_pt_full/`, and the separate statistical-target root, and both include a working-memory task. | Current subject completeness, task/event compatibility, contrasts, spaces, confounds, behavior, exclusions, and whether either can serve as discovery or transport without leakage. |
| NDA candidate root | `/oak/stanford/groups/russpold/data/NDA/fmri_data` | The directory and a manifest-like CSV at `/oak/stanford/groups/russpold/data/NDA/nda_20260715_image03.csv` are visible. Their contents were not inspected during preparation. | Dataset identity, authorization, n-back presence, age/site/session structure, longitudinal or retest availability, derivatives, event semantics, and HCP/AOMIC compatibility. |

HCP-YA remains subject to its data-use agreement. Do not persist restricted
subject-level identifiers or covariate values in this Git repository, reports,
logs, or Society material.

## Roles to decide outcome-blindly

Before opening candidate-discriminating outcomes, assign non-overlapping roles
and record the evidence for each assignment:

1. discovery/model-selection source;
2. development reliability split;
3. sealed same-subject independent-session or retest source;
4. sealed task-compatible external transport source;
5. behavior construct variable and covariates for every relevant source;
6. motion, category, site/scanner, and difficulty controls.

Repeated LR/RL runs from one visit may support run-to-run reliability but do
not satisfy the independent-session role. A dataset used to select features,
parameters, thresholds, exclusions, or metrics is development data and cannot
later be called independent confirmation.

## Outcome-blind preflight

The initial inventory should check:

- dataset release, provenance, access terms, and authorized use;
- subject, visit, session, run, phase-encoding, scanner, and site identifiers;
- exact event definitions for 0-back, 2-back, cues, categories, responses,
  errors, and missing trials;
- BOLD, mask, confound, design/stat-map, space, resolution, and coverage files;
- whether categories can be balanced or held out without changing the load
  estimand;
- behavior definitions, scale direction, missingness, and permitted covariates;
- motion and censoring variables, exclusions, and site handling;
- availability of a meaningful general-difficulty comparator;
- task, age, timing, stimulus, scanner, preprocessing, and behavior differences
  across candidate datasets;
- non-overlapping subject and outcome roles; and
- storage, software, scratch, and Slurm requirements.

Preserve unknowns. A readable directory or one representative filename does
not establish completeness, task compatibility, estimability, or scientific
suitability.

## Data-use boundary

At preparation time, `inputs/` contains only documentation and optional
planning context. It contains no neural or behavioral data and no confirmation
symlink. The run should not write into `inputs/` or modify any source dataset.

Large source trees stay in their existing OAK locations and are read-only.
Write outcome-blind manifests, compatibility notes, analysis code, and durable
results under `outputs/`. Put high-frequency intermediates in
`$SCRATCH/autoresearch/episode03_nback_signature/`, then copy only the compact,
documented final artifacts back to `outputs/`.

Before any independent-session or external outcomes are accessed, the run must
freeze one signature, all learned parameters, the exact preprocessing and
score, eligibility/exclusions, behavior endpoint, metrics, practical margins,
uncertainty method, and one-shot failure rule. If no untouched compatible
source remains, report that fact and do not claim transport readiness.
