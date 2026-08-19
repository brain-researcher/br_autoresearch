# Dataset facts: NARPS `ds001734` successor workspace

## Workspace contract

This workspace is located at
`/oak/stanford/groups/russpold/users/zijiao/autoresearch/episode02_narps_analysis`.
Its `inputs/` entries are source references or a curated historical packet and
must be treated as read-only. All new artifacts must be written under
`outputs/`. No source data should be modified, copied into `outputs/`, or
replaced by a download.

## Supplied source references

| Workspace reference | OAK target | Intended contents |
|---|---|---|
| `inputs/raw` | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/input/ds001734` | OpenNeuro raw dataset checkout, including available metadata and events |
| `inputs/fmriprep` | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep/ds001734/derivatives` | fMRIPrep derivatives |
| `inputs/fitlins` | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds001734` | FitLins analysis products and run-level designs where materialized |

The references are absolute OAK symlinks for use on Sherlock. Their current
readability, exact file layout, and semantic compatibility must be rechecked in
Stage 0; a directory name or symlink alone is not evidence that a needed
artifact is usable.

## Historically observed dataset facts

The parent episode's source audit reported 108 participants, split 54
equalIndifference (EI) and 54 equalRange (ER), with four task runs per
participant and 432 runs in the parent analysis. It used
`MNI152NLin2009cAsym` at 2 mm for the relevant fMRIPrep-derived analysis.
These are prior inventory observations, not a guarantee about the current
source tree; Stage 0 must independently verify them before analysis.

EI and ER are task versions with different gain support and potentially
different regressor scales. Therefore the source data do not make a pooled
EI/ER gain-minus-loss comparison intrinsically interpretable. Any direct
task-version association requires explicit labeling and separate provenance.

The parent audit had complete checkpoint coverage for 1,944 derived
subject-by-confound-by-smoothing-by-contrast fixed-effect variance arrays.
That documents parent checkpoint completion, not current source readiness.
Its FitLins tree was used for frozen run-level designs, but source
materialization and design compatibility are still facts to verify now rather
than assumptions to inherit. The raw checkout may be incomplete as a general
OpenNeuro annex; do not assume assets absent from the local tree can be fetched
or substituted.

No multi-team NARPS analysis bundle is supplied by this workspace. Claims about
cross-team pipeline variance are out of scope unless a separately authorized,
readable source is added later.

## Curated parent evidence packet

`inputs/prior_episode/` contains a small read-only subset from the completed
v1 NARPS workspace:

- frozen analysis contract (`.md` and `.json`);
- parent terminal result and candidate bundle;
- parent source audit;
- variance-attribution, conclusion-stability, matched-size-calibration, QC,
  and multiverse manifest tables;
- edge-scaling diagnostic summary and a short parent memory note; and
- `LINEAGE.md`, which records how this evidence may be used.

The packet is prior exploratory evidence and provenance only. It does not
constitute an independent test set, fresh confirmation data, or a valid basis
to bypass re-inventory of the supplied sources.

## Stage 0 unknowns to resolve

Before any new source neural-outcome analysis, inventory and record:

- exact participant/run availability and EI/ER labels;
- readable events, confounds, masks, BOLD images, first-level designs, and
  gain/loss contrast definitions;
- spaces, resolutions, temporal coverage, and whether maps/designs are
  comparable across subjects and task versions;
- current FitLins artifact availability versus dangling or incomplete links;
- which proposed mechanisms can use existing statistics and which would need
  a bounded recomputation; and
- storage and compute availability on Sherlock.

Durable products stay in this OAK workspace. If a future launched analysis
needs high-frequency temporary I/O, it should use an explicitly named scratch
location and copy only final durable artifacts back here. This preparation step
does not itself authorize or start computation.
