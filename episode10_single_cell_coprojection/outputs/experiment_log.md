# Experiment log

Before the 2026-09-26 Stage-0 launch, no outcome-modeling experiment had
started under the v2 `GOAL.md`, `DATASETS.md`, and `SEARCH_POLICY.yaml` design.
The evidence-chain figure contains synthetic marks only and is not an
experiment. The 2024 axonal BARseq novelty overlap and the conditional
A+B-versus-A+C terminal-profile endpoint were added as design constraints; no
morphology outcome was inspected.

The paper plan was added as a design document. Creating it did not open data,
run an analysis, or change any scientific outcome.

On 2026-09-24, the first analysis stage was clarified as scientific data
qualification. No outcome was opened during this documentation change.

A final consistency pass clarified development-only A+B/A+C eligibility,
pooling-discrepancy inference, model-search anchors, negative-result reporting,
and route-specific held-out predictions. No data or outcome was opened.

## 2026-09-26 — Stage 0 launch and SEU qualification

- Run: `ep10-stage0-20260926T063634Z`.
- Authority: the scientist explicitly requested EP10 launch in this Codex
  task. The run is a standalone episode-managed exploration; no matching
  canonical Brain Researcher loop or Goal handoff was observed, and no MCP
  mutation was made.
- Scope: read the frozen Goal, dataset contract, and search policy; inventory
  the three EP10 steward assets; execute outcome-blind qualification of the
  compact SEU metadata workbook, CCFv3 ZIP member names, Allen NRRD headers,
  and structure hierarchy.
- Durable code and contract:
  `outputs/stage0/qualification_contract.yaml`,
  `outputs/stage0/source_manifest.json`, and
  `outputs/stage0/tools/qualify_resource.py`.
- Leakage tests: 9/9 passed. The durable qualifier skips `Projection class`
  cells before value decoding and rejects duplicate identities, identity-set
  mismatches, and ZIP path traversal.
- Engineering repair: an initial pre-artifact identity-join trial treated
  SpreadsheetML `_x005F_` escapes literally and reported 596 false mismatches.
  The parser was corrected to decode one pass of SpreadsheetML escapes; the
  same inputs and scientific configuration were rerun. The repaired result has
  zero identity issues and a 1,876/1,876 filename join. No SWC member was
  opened.
- Prelaunch probe deviation: an earlier one-off trusted probe materialized
  complete workbook rows transiently while emitting only allowed aggregate
  fields. It printed and persisted no `Projection class` value. The durable
  qualifier eliminates that unnecessary decoding. The workbook is therefore
  treated as touched by a trusted builder, not as pristine or candidate-visible.
- Observed support: 1,876 cells and 39 conservative `fMOST Brain ID` specimen
  groups overall. MOp has 75 cells across 17 groups; after the required 12
  development groups, at most five groups remain for final evaluation rather
  than the required eight. No listed source reaches 20 groups.
- Decision: `revise_resource_strategy_before_outcome_search`. This is a
  resource-feasibility decision, not a biological null result. The shared
  EP09--EP11 role ledger remains unfrozen, and no morphology, target-set,
  bouton, voxel, or final-group outcome was opened.
- Leakage hardening: the first mechanically clean result was retained as
  `qualification_run001`, then superseded after review showed that a global
  shared-string table was retained in memory. Run 002 resolved only header and
  allowed-cell shared-string indexes. Run 003 additionally rejects every
  named non-allowlisted metadata field, instruments ZIP-member non-read
  behavior, and omits forbidden-column occupancy. Run 004 closes the remaining
  schema edge case by rejecting populated columns with no header before
  shared-string lookup. Before handoff, the superseded run-002 durable copy was
  redacted so no retained result publishes forbidden-column occupancy. Run 005
  binds the executable explicitly to module Python 3.12.1 and upgrades the
  NRRD test to fail on any read beyond the header terminator. All five runs
  give the same support decision; run 005 is the current evidence.
- Evidence:
  `outputs/stage0/launch_identity.json`,
  `outputs/stage0/qualification_run005/qualification.json`, and
  `outputs/stage0/qualification_run005/source_group_support.csv`.

## 2026-09-26 — Lean Gao MOp qualification

- Official sources: Gao et al., *Neuron*, DOI
  `10.1016/j.neuron.2025.10.019`; the provider whole-cortex portal and public
  neuron/sample JSON endpoints; Zenodo record `16909126` for the full archive.
- Action: downloaded only the two public metadata JSON files to transient
  scratch and selected exact provider soma acronyms beginning with `MOp`.
  Retained output fields were identity, source/layer, hemisphere,
  reconstruction scope, fMOST filename prefix, sample join, project, and
  transgenic line. The JSON transport contains an outcome-derived `class` key,
  but its values were not queried, emitted, retained in an artifact, or used.
- Support: 837 MOp cells in 76 unique fMOST filename-prefix groups; all 76 join
  once to distinct nonempty sample IDs. Layers are MOp1 27, MOp2/3 322, MOp5
  330, MOp6a 151, and MOp6b 7; hemispheres are 226 left and 611 right.
  Provider reconstruction scope is 833 `Axon_and_dendrite` and 4 `Axon_only`.
- Identity limit: public records expose no animal, mouse, or donor key. The 76
  groups are therefore conservative released brain/specimen units, not 76
  certified independent animals.
- Cross-episode check: 15 fMOST units overlap EP11's outcome-blind SSp-tr
  inventory. EP11 has no frozen role assignment, but a shared Gao exposure
  ledger is required before either episode opens outcomes.
- Decision: Gao MOp passes the released brain/specimen-count screen, but
  independent-animal identity, exact position support, reconstruction QC,
  complete-target observation, and roles remain unresolved. Outcome search
  remains closed.
- Transfer boundary: an exact 837-filename targeted downloader was prepared in
  transient scratch. The required DTN rejected noninteractive authentication,
  so no Gao SWC was downloaded from the login node. No modeling or morphology,
  target, branching, terminal, voxel, or final outcome access occurred.
- Evidence: `outputs/stage0/gao_mop_qualification.md` and
  `outputs/stage0/gao_source_group_support.csv`.

## 2026-09-26 — Gao MOp soma geometry and prospective role freeze

- Soma access: Slurm job `45396282` requested only bytes 0--4095 for each of
  the 837 selected canonical SWCs and retained only the first validated
  46--51-byte soma-root row. It did not materialize or inspect any axon tree.
- Geometry: Slurm job `45398661` found all 837 cells atlas-geometry eligible
  (833 primary `Axon_and_dendrite`, 4 `Axon_only`). Of 48 prespecified
  radius/depth conditions, 40 were feasible and 28/36 finite-depth conditions
  were feasible. The Pareto-minimal finite conditions were 500 um/0.20 (825
  candidate cells) and 750 um/0.10 (828); the prespecified radius-first rule
  selected 500 um/0.20. Exact-distance and LineString sensitivity agreed at
  that condition.
- Role freeze: outcome-blind Slurm job `45401314` froze 12 development, 8
  final, and 56 unassigned fMOST/sample units. Development contains 209 primary
  cells, 200 supported (95.69%); final contains 54 primary cells, all 54
  supported. A same-build rerun reproduced the assignment exactly.
- Cross-episode boundary: all 15 units shared with EP11 were forced unassigned
  before optimization. EP11 roles remain unfrozen.
- Access boundary: no complete axon tree, projection target, branch, terminal,
  voxel payload, or role-specific outcome was opened. Final-role morphology
  remains sealed.
- Evidence: `outputs/stage0/gao_mop_soma_inventory.csv`,
  `outputs/stage0/mop_geometry_support.csv`,
  `outputs/stage0/mop_geometry_summary.json`,
  `outputs/stage0/gao_mop_frozen_roles.csv`, and
  `outputs/stage0/gao_mop_role_freeze_summary.json`.

## 2026-09-26 — Projection-observation freeze and development handoff

- The outcome-blind observation contract was frozen before any complete SWC
  body was fetched. It uses a 56-coordinate Allen atlas/QC map, replacing the
  coarse `MO` anchor with `MOs` so local MOp axon is not treated as a target.
- The primary exact model is prospectively limited to 16 coordinates:
  MOs, SS, ACA, STR, TH, MB, P, and MY crossed with soma-relative
  ipsilateral/contralateral side. This anatomy-defined motor-routing panel has
  at most 12,870 fixed-K subsets; it was not chosen from observed target
  prevalence or candidate effects.
- A target is detected only when a positive-length target component contains a
  rooted axon leaf/final leaf edge. Cable without a terminal-bearing component
  is recorded as `passing_only`. A zero means only `not observed in the eligible
  released reconstruction`, never biological or synaptic absence. A tree,
  registration, or complete-observation failure makes all coordinates unknown
  and excludes the cell; unknown is never zero.
- Primary modeling is restricted to the frozen 200-cell development
  common-support domain. Structural QC must retain at least 190 overall and the
  prespecified 90%-ceiling minimum in each of all 12 groups; failure means
  revise or stop without replacement, promotion, or threshold relaxation.
- The four `Axon_only` cells cannot support a sensitivity under the frozen roles
  (0 development, 1 final, 3 unassigned), so none will be opened for
  development and that sensitivity is omitted.
- The contract pins the positive-diagonal AP/DV/ML NRRD affine, Fortran-order
  loading, the pre-laterality six-neighbor erosion rule, and a complete
  node-transition rule that cannot turn an axon-to-dendrite/marker transition
  into a terminal.
- The role-safe manifest contains exactly the 200 common-support primary
  development SWCs from 12 frozen fMOST/sample units. It contains none of the
  nine unsupported development-role cells and no final, unassigned,
  `Axon_only`, or EP11-shared row. The final 55 and unassigned 573 files remain
  absent.
- A header-only provider check confirmed direct file length/type support. No
  SWC body was fetched. The current agent session cannot authenticate
  noninteractively to the Sherlock DTN, so the attended DTN transfer remains
  pending rather than being rerouted through the login node.
- Evidence: `outputs/stage0/projection_observation_contract.yaml`,
  `outputs/stage0/allen_target_vocabulary.csv`,
  `outputs/stage0/gao_mop_development_manifest.csv`, and
  `outputs/stage0/gao_mop_development_acquisition.md`.

## 2026-09-26 — Stage-0 reproducibility and exact-normalization preflight

- Slurm job `45412269` completed in 16:57 with exit 0 on a compute node. No
  SWC body, projection outcome, final cell, or network resource was opened.
- The safe soma-geometry derivation reproduced 209 development-role rows with
  exactly 200 common-support cells and nine excluded cells. The generated
  200-row acquisition manifest matched the frozen manifest exactly, contained
  all 12 development groups, and contained zero EP11-shared files.
- The metadata/geometry-only role solver reproduced the frozen 12 development,
  8 final, and 56 unassigned allocation exactly; all 15 EP11-overlap units
  remained unassigned.
- Synthetic terminal-constructor checks passed all 13 frozen tree, transition,
  laterality, terminal-versus-passing, unknown, and sealed-sensitivity cases.
- Ninety-nine outcome-blind exact-normalization suites took 650.99 seconds
  total (median 6.58 seconds, range 6.43--6.65). Each suite exhaustively scored
  all 65,536 states across the 16-coordinate panel and the bounded
  56-coordinate, `K=5` case of 3,819,816 states. This is a throughput benchmark,
  not a complete 99-null search benchmark.
- Evidence: `outputs/stage0/exact_normalization_benchmark.json` and Scratch
  run directory `stage0_preflight_v1/job_45412269`.

## Remaining uncertainty

- Gao MOp provides 76 joined released brain/specimen units and therefore passes
  the 12/8 count arithmetic, but distinct-animal identity is not certified by
  the public metadata. Exact soma-position/common-support geometry now passes;
  the complete-target observation rule is frozen but remains untested on full
  trees, and full-tree structural completeness remains unknown.
- SEU-to-Allen transform compatibility and target-calling reliability remain
  unverified because Stage 0 deliberately did not decode morphology or voxel
  payloads.
