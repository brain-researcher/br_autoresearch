# Verification

- Goal, dataset description, and search policy: revised to the v2 question of
  combination stability, applicability boundaries, and branching anatomy.
- Shared-target terminal organization: added as a conditional secondary
  endpoint; A+B/A+C definitions, within-A metric, direction, support, and
  multiplicity remain unset.
- Novelty boundary: generic co-target/within-target laminar association is
  treated as prior art from the 2024 axonal BARseq study.
- Paper plan: added under `outputs/`; all claims and figures are conditional,
  and no result or new data access is asserted.
- Evidence-chain figure: synthetic design illustration; it contains no
  observed data.
- Candidate resource roles: Gao MOp passes the released brain/specimen-count
  and soma common-support screens; 12 development and 8 final units are frozen,
  while all 15 units shared with EP11 remain unassigned.
- First data pass: SEU metadata/identity and Gao metadata/soma-geometry
  qualification are complete. The Gao target-observation rule and
  development-only manifest are frozen; full-tree reconstruction/observation
  QC has not started.
- Inferential split: the 99 reruns calibrate target-set search and combination
  selection; frozen full-axon metrics use separate biological-group inference.
- Model extensions: activated only when grouped-development evidence supports
  them; no separate shared-policy reconciliation is required.
- Model-search coverage: at least 16 valid anchors cover the sparse core and
  every development-triggered extension; this is distinct from axonal branching.
- Shared-target timing: exact A+B/A+C eligibility is assessed in development
  groups after the split, while final groups remain unopened; a later final
  support shortfall is inconclusive rather than a reason to reselect.
- Pooling discrepancy: its estimator, null, useful margin, and group-level
  uncertainty are frozen before final evaluation.
- Outcome computation: not run.
- Final evaluation: not opened.
- Task startup: allowed by explicit scientist instruction. Data qualification
  is the first stage, and scientific outcome analysis has not started.

## Stage-0 scientific readiness — 2026-09-26

| Check | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| Workbook-to-CCFv3 ZIP identity join | pass, 1,876/1,876 | `stage0/qualification_run005/qualification.json` | Member names only; SWC contents were not opened. |
| Identity issues after SpreadsheetML repair | pass, zero | `stage0/qualification_run005/identity_issues.csv` | Filename identity is not animal lineage. |
| Forbidden data-cell dereferencing | pass, zero lookups | `stage0/qualification_run005/qualification.json` | The XML transport is parsed, but forbidden shared-string indexes are not resolved or retained; occupancy is not reported. |
| Annotation/template NRRD header geometry | pass | `stage0/qualification_run005/qualification.json` | Tests fail on any post-header read; SEU transform compatibility was not tested. |
| MOp 12/8 group gate | fail, 17 total groups | `stage0/qualification_run005/source_group_support.csv` | Feasibility failure, not a biological null. |
| Shared SEU EP09--EP11 role ledger | unresolved but not used | `stage0/readiness_status.yaml` | The selected EP10 resource is Gao rather than SEU. |
| Final outcome access | pass, zero opens | `stage0/readiness_status.yaml`; `experiment_log.md` | Final-role morphology remains physically absent and may be opened only once after development choices are frozen. |

An initial pre-artifact trial produced 596 false identity mismatches because
SpreadsheetML `_x005F_` escapes were treated literally. The code-only repair
preserved the same data and scientific configuration, and run 001 reduced the
issue count to zero. Superseded implementation details remain in
`experiment_log.md` and the stage-0 artifacts as historical records; they are
not active readiness gates.

Scientific validity, novelty, causal interpretation, independent replication,
and target-observation reliability are not mechanically verified.

## Lean Gao qualification — 2026-09-26

| Check | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| Authoritative source | pass | Gao et al., *Neuron*, DOI `10.1016/j.neuron.2025.10.019`; provider portal; Zenodo `16909126` | Publication/repository identity does not establish EP10 measurement validity. |
| Outcome-blind MOp rule | pass | Exact provider soma acronym begins with `MOp` | No target or projection field used. |
| Cell and released-unit support | pass, 837 cells / 76 fMOST-sample units | `stage0/gao_source_group_support.csv` | Necessary count screen only. |
| Sample join | pass, 76/76 unique | provider filename prefix to sample `fMOST_id` | Sample identity is not certified animal identity. |
| Independent-animal identity | unresolved | no public animal/mouse/donor key | Current claim must remain cross-brain/specimen-unit. |
| Reconstruction scope | 833 axon-and-dendrite; 4 axon-only | safe provider metadata fields | Full-tree completeness remains unqualified; Axon_only sensitivity is unavailable under frozen roles (0 development, 1 final, 3 unassigned) and will not be opened for development. |
| Exact soma geometry | pass, 837/837 cells and 76/76 units | `stage0/mop_geometry_summary.json`; job `45398661` | Soma roots only; no axon-tree row used. |
| Finite common support | pass, 500 um / 0.20 | `stage0/mop_geometry_support.csv` | Primary development domain is the fixed 200 supported cells among 209 development-role cells; QC must retain at least 190 and every frozen per-group minimum across all 12 groups, or revise/stop without replacement. |
| Frozen roles | pass, 12 development / 8 final / 56 unassigned | `stage0/gao_mop_frozen_roles.csv`; job `45401314` | Unit identity is not certified animal identity. |
| EP11 overlap | pass, 15/15 forced EP10-unassigned | frozen-role ledger | Preserves EP11 options; EP11 roles remain unfrozen. |
| Gao access boundary | soma roots only | job `45396282`; `stage0/gao_mop_soma_inventory.csv` | No full morphology, target, branching, terminal, voxel, or final outcome opened. |
| Target observation contract | definition frozen before axon-tree/target access; empirical QC pending | `stage0/projection_observation_contract.yaml`; `stage0/allen_target_vocabulary.csv` | Terminal-bearing binary calls; passing cable is not a target; zero means not observed in the eligible released reconstruction, not biological absence. |
| Development acquisition boundary | pass, 200 common-support cells / 12 groups / 0 shared | `stage0/gao_mop_development_manifest.csv` | The nine unsupported development-role cells, final 55, and unassigned 573 files remain absent, and DTN transfer has not occurred. |
| Reproducibility preflight | pass, exact 200/9 support and 12/8/56 roles reproduced | job `45412269`; Scratch `stage0_preflight_v1/job_45412269` | Safe soma geometry and metadata only; no SWC body or outcome opened. |
| Exact-normalizer throughput | pass, 99 synthetic suites / 650.99 s total / 6.58 s median | `stage0/exact_normalization_benchmark.json`; job `45412269` | Exhaustive normalizer workload only, not a complete 99-null model-search runtime. |
| Decision | continue pre-outcome qualification | `stage0/gao_mop_qualification.md` | Outcome search remains closed. |
