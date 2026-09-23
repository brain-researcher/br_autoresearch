# Design example: Adaptive Search for Transportable Smoothing Mechanisms

## Status and authority boundary

This is an unnumbered design reference, not an episode proposal. It does not
authorize data access, computation, canonical Goal creation, Society review,
reward, launch, or a scientific claim. The two source examples remain frozen
and are never edited or reinterpreted as confirmation.

If instantiated, this design must run as a **registered search program** whose
immutable policy is `SEARCH_POLICY.md`. Do not run it through generic bounded
Goal discovery: that route would again collapse the work to at most two
one-shot tests.

## Research question

Which spatial-operator, masking, scaling, first-level estimation, and group-
aggregation mechanisms explain the change from unsmoothed to 8 mm-smoothed
gain/loss effect maps, and does the selected explanation transport from NARPS
`ds001734` to an independently collected mixed-gambles dataset, `ds000005`?

The objective is not to rediscover that smoothing changes maps. It is to build
and falsify mechanistic predictors of **where the S0-to-S8 change comes from**,
separate the deterministic consequence of an 8 mm kernel from noncommuting
pipeline operations, and test transport on a dataset that was not used to
choose the mechanism.

## Why this goes deeper than the historical examples

Episode 01 completed a frozen `confound × QC × smoothing` multiverse and closed
when its directional candidate rule failed. Episode 02 deliberately retained
only two tests and stopped after one terminal candidate. Both are useful
historical controls, but neither implemented repeated

```text
hypothesis -> experiment -> falsifier -> belief update -> successor
```

This template uses their exposed evidence only as development lineage. A
candidate is not terminal merely because it becomes the current incumbent.

## Scientific objects

For each dataset, contrast, subject grouping, and frozen analysis pathway, let

- `M0` be the group effect map from the unsmoothed pathway;
- `M8` be the group effect map from the 8 mm pre-GLM pathway;
- `K8(M0)` be the map obtained by applying the exactly matched spatial
  operator to the unsmoothed result at a declared pipeline stage; and
- `Mhat8(h)` be the prediction from mechanism hypothesis `h`.

The primary map metric is explained smoothing change in a fixed comparison
mask:

```text
ESC(h) = 1 - SSE(M8, Mhat8(h)) / SSE(M8, M0)
```

`ESC=0` is no better than predicting no change; `ESC=1` is exact prediction.
The primary comparison is the paired improvement over the deterministic
matched-kernel baseline:

```text
DeltaESC(h) = ESC(h) - ESC(K8)
```

Spatial correlation, normalized RMS difference, spatial-frequency residuals,
edge-distance profiles, sign agreement, matched-rank overlap, and thresholded
sets are diagnostic or secondary unless explicitly frozen before audit.

All metrics are computed separately for gain and loss. NARPS EI and ER are
separate environments, not interchangeable observations and not an
independent replication.

## Predeclared conclusion classes

The search may end in one of four scientifically distinct conclusions:

1. **Kernel-sufficient:** an exact matched spatial operator explains most of
   the S0-to-S8 change and more elaborate mechanisms add less than the frozen
   useful-effect threshold.
2. **Pipeline-noncommutative:** a specific scaling, mask, estimation, or
   aggregation mechanism reproducibly improves prediction beyond the exact
   kernel baseline.
3. **Development-only:** a mechanism is stable inside `ds001734` but fails the
   external `ds000005` audit.
4. **Unresolved:** the search budget is exhausted without a stable mechanism,
   or audit precision cannot distinguish the predeclared classes.

Only classes 1 or 2 may support a `candidate_ready` bundle, and only after the
one-shot audit. Classes 3 and 4 are informative negative terminals, not
permission to inspect audit outcomes and resume tuning.

## What is immutable before search

Freeze and hash before any development trial:

- dataset roles and the audit access firewall;
- eligible subjects, runs, contrasts, spaces, and comparison masks;
- the definition of S0, S8, `K8`, `ESC`, and `DeltaESC`;
- grouped development folds and aggregation across environments;
- the mechanism grammar below;
- trial ledger schema, incumbent rule, promotion rule, resource ceiling,
  minimum search depth, patience, and terminal rules;
- synthetic tests and outcome-blind audit feasibility checks; and
- the exact audit procedure, uncertainty method, useful-effect thresholds,
  and failure conditions.

Do **not** freeze the exact sequence of development hypotheses. Adaptive
selection inside the declared grammar is the object being evaluated.

## Development hypothesis grammar

Every scored hypothesis must make a directional prediction, name a falsifier,
identify the parent trial, and change one interpretable operator family at a
time unless it is a declared composition or ablation.

### A. Spatial operator and stage

- matched Gaussian operator applied to BOLD, run-level effect, subject fixed
  effect, or group map;
- FWHM dose-response chosen from the role-specific frozen grid
  `{0, 2, 4, 5, 6, 8, 10}` mm;
- boundary convention, resampling order, and normalization;
- commutation tests between smoothing and GLM/group aggregation.

### B. Mask and boundary support

- fixed coverage masks and kernel-aware erosions;
- distance-to-boundary shells defined in millimetres;
- coverage and low-intensity support diagnostics;
- whole-mask versus interior prediction with the same scoring rule.

### C. Scaling and numerical stabilization

- native versus explicitly matched signal-scaling behavior;
- prespecified variance floors calibrated on synthetic/null data;
- exclusion or modeling of invalid constant/near-constant series using
  outcome-blind rules;
- floating-point and resampling precision controls.

### D. First-level and group estimation

- the historically used C1 nuisance model as the anchor;
- C0/C2 only as declared sensitivity environments, not an unrestricted
  confound leaderboard;
- equal-run, inverse-variance, stabilized inverse-variance, and robust group
  aggregation;
- AR(1) and declared alternative noise-model checks when executable.

### E. Residual representation and reliability

- spatial-frequency decomposition of `M8-K8(M0)`;
- split-half and leave-one-subject-out reliability weighting;
- low-dimensional residual summaries used only when learned inside grouped
  development folds;
- negative controls created by spatial phase randomization, subject-label
  permutation, and deliberately mismatched kernels.

Arbitrary ROI fishing, threshold tuning, post-audit feature creation, neural
network architecture search, and unlogged hand edits are outside the grammar.

## Multi-round development program

### Stage 0: readiness and synthetic calibration

Inventory all sources without opening audit neural arrays. Validate map algebra,
kernel implementation, resampling, masks, uncertainty, and trial-ledger logic
on synthetic images with known amplitude, edge, and spatial-frequency effects.
Freeze audit thresholds after this outcome-blind calibration.

### Stage 1: broad, cheap mechanism screen

Run 12--20 valid trials using cached NARPS derivatives and bounded subject/map
subsets. Every operator family A--E must receive at least two valid trials or a
recorded input-readiness rejection. Promote a trial only from grouped
out-of-fold performance across contrast, task-version, and subject-fold
environments. The adaptive kernel grid is S0/S4/S8; mechanisms may use these
levels to learn a dose-response relationship.

### Stage 2: targeted recomputation and composition

Run 12--24 valid trials that recompute only what the surviving mechanisms need.
Compose at most three compatible operators. Every composition requires
single-operator ablations and a negative control. Development failures retire
that branch; they do not end the program while another admissible branch and
budget remain.

### Stage 3: full-development replication and ablation

Run 6--16 full-data trials on `ds001734`. Compare the incumbent, up to two
challengers, the no-change baseline, and the exact-kernel baseline. Require
stability across gain/loss, EI/ER, deterministic subject halves, and
leave-one-subject-out influence. A locked 6 mm interpolation check is opened
once for the shortlist and used to select the final configuration. The 2, 5,
and 10 mm levels may be reported only as a predeclared dose-response
sensitivity series; they cannot repeatedly change the shortlist or audit
thresholds. Recompute the primary metrics through an independently implemented
operator/metric path under frozen tolerances before configuration lock.

### Configuration lock

When the stop rule fires, freeze exactly one selected mechanism, one runner-up
for diagnostic comparison, all baselines, code, environment, manifests,
thresholds, and the complete append-only trial ledger. The runner-up cannot
replace the winner after audit access.

### Stage 4: one-shot external transport audit

Open `ds000005` exactly once through the audit runner. Refit the frozen S0/S8
paths for `paragain` and `paraloss` without adapting the mechanism. Report
point estimates, paired subject-bootstrap intervals, deterministic halves, and
leave-one-subject-out guards. Audit feedback may change the conclusion class
but may not trigger a new trial in the same program run.

## Incumbent and challenger policy

- The first valid improvement is an `incumbent`, not a terminal candidate.
- A challenger replaces the incumbent only if it improves the frozen aggregate
  development score by at least `0.01 DeltaESC`, or matches it within `0.01`
  while resolving a prespecified falsifier with lower complexity.
- Ranking aggregates environments by the median and penalizes the worst
  contrast/task-version environment; a large gain in one environment cannot
  compensate for direction reversal in another.
- Any trial that changes data eligibility, audit roles, metrics, or thresholds
  is invalid rather than a new challenger.

## Search and compute budget

- Minimum valid development trials: **30**.
- Maximum valid development trials: **60**.
- Maximum failed engineering attempts: **15**, each logged.
- Patience: stop after **12 consecutive valid trials** without at least `0.01`
  aggregate improvement, but only after the minimum trial count and branch-
  coverage requirement are satisfied.
- Requested compute ceiling: **2,500 CPU core-hours**, **1.5 TB scratch**, and
  **96 wall-clock hours**.
- GPU budget: **zero**.

Stop only when the maximum trial count, compute ceiling, qualified patience, or
exhaustion of every admissible branch is reached. A failed single hypothesis,
one negative contrast, or the appearance of an incumbent is not a stop event.

## Draft audit decision thresholds

These values are provisional until outcome-blind synthetic calibration and
scientist approval, after which they are immutable.

### Kernel-sufficient candidate

For both `paragain` and `paraloss` in `ds000005`:

- exact-kernel `ESC >= 0.80`;
- no admissible nonkernel mechanism has point `DeltaESC >= 0.05`;
- the conclusion is stable in deterministic halves and all leave-one-subject-
  out analyses; and
- mismatched-kernel and spatial-phase controls fail.

### Pipeline-noncommutative candidate

For both audit contrasts:

- selected mechanism point `DeltaESC >= 0.10`;
- paired 95% subject-bootstrap lower bound of `DeltaESC > 0`;
- direction agrees in deterministic halves and all leave-one-subject-out
  analyses; and
- the corresponding operator ablation and negative controls behave as
  predicted.

If neither rule passes, report `development_only` or `unresolved` without
substituting the runner-up or changing thresholds.

## Required search artifacts

The registered program must produce, at minimum:

- immutable source, split, and exposure manifests;
- `experiments.jsonl`, append-only and one row per attempted trial;
- machine-readable incumbent history and hypothesis genealogy;
- code/environment hashes for every scored trial;
- Stage 0 synthetic-calibration report;
- full development score table with failures and retired branches;
- configuration-lock manifest;
- audit-access receipt and one-shot audit report;
- ablations, negative controls, uncertainty tables, and resource usage;
- a conservative `RESULT.md`; and
- the schema-valid terminal bundle plus the seven local workspace projections.

## Claim boundary

Even a successful audit supports a transportable **methods mechanism** across
two small public mixed-gamble datasets. It does not establish a neural mechanism
of gain/loss processing, universality across tasks or pipelines, scientific
acceptance, or permission to update the Landscape. `ds000005` has only 16
participants, so precision and leave-one-subject sensitivity must remain
visible.
