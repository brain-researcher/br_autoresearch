# Experiment log

## 2026-10-20 — matched_donor_qualification: graph-kernel tests passed

- Coordinator allocation **48756235**, `sh04-18n32.int`: all **8** graph/oracle/census-integration tests passed in **0.709 s**, exit **0**.
- Evidence: [attempt002 test output](runs/matched_donor_qualification/attempt002/graph_kernel_tests.txt); no real scientific input was read.
- Attempt001 preserves an interpreter mismatch (Python 3.13 with Python 3.12 NumPy); the coordinator corrected invocation to `python3` with native modules, with no graph-code change.
- This validates feasible-edge pruning and opt-in census integration, not real donor support, power or biological prediction; historical census artifacts remain unchanged.
- Reconciled the GOAL entrypoint with already adopted CEBSIT/Cux2/Wfs1/27–16 records, retaining SEU feasibility as history and the explicit animal-identity assumption.
- Next: resolve the pending scientific population/matching and claim gates, then advance authorized donor/measurement stages; no axon or audit access was opened here.

## 2026-10-20 — measurement_and_roles: adopted source entrypoint reconciled

- Corrected DATASETS to the adopted CEBSIT source, Cux2/Wfs1 L2/3–L4 population, registered dendrites and 27/16 whole-brain roles; marked prior CEBSIT proposal/SEU inventory explicitly historical.
- Preserved frozen artifacts and roles. Current dependencies now refer to stage A and the remaining companion decisions rather than historical SEU provisioning.
- The catalog review supports unique IT-line sample identifiers, but the retained records cite no explicit provider animal/acquisition crosswalk; one fMOST brain per mouse is reported as the adopted identity assumption, not a new certificate.
- No payload read, test, job or additional outcome access accompanied this documentation correction.

## 2026-10-20 — measurement_and_roles: prospective implementation choices adopted

- Under the scientist's animal-study continuation and repository standing authority, the lead adopted D03 whitelist, D04 genuine bijective plans and D06 registered nested-animal models/residualization for implementation.
- Preserve the current 40/96 search bounds and patience 18 (D07); use conservative critical-value calibration with unchanged nominal 95% inference (D08).
- Implement frozen 4×3 anatomical blocks/full-target matrices (D10) and preserve original roles/development pseudo-audits (D12); this adopts no scientific sign gate or population-dependent change.
- The prospective 512-plan count still requires outcome-free stability/workload qualification; no real donor bank or scientific result exists.
- The original companion draft stays unchanged; [current navigation](runs/measurement_and_roles/attempt006/README.md#october-20-prospective-implementation-defaults-under-standing-authority) distinguishes adopted defaults from pending D01/D02, confirmatory gates, margins and secondary hierarchy choice.
- No data read, new job or axon/audit access accompanies this record; the graph-kernel compute check remains the selected immediate implementation test.

## 2026-10-20 — matched_donor_qualification: exact donor-graph kernel prepared

- Activated a contract-preserving implementation step under the scientist's animal-study continuation; no population, radius, threshold, role or outcome-access amendment is adopted.
- The prior support census degree-prunes and checks for a perfect matching but does not remove edges absent from every complete plan; its historical support counts remain preliminary.
- Added `tools/ep09_donor_graph.py` and opt-in `--qualify-feasible-edges` integration in the existing census; the original census mode and frozen artifacts are preserved.
- Selected compute-only finite-graph oracle tests in `tools/test_ep09_donor_graph.py`; proposed resource use is one CPU, under 1 GiB and two minutes, coordinated by the lead.
- No real data run is selected. Pending artifact: `runs/matched_donor_qualification/attempt001/graph_kernel_tests.txt`; a pass would validate the graph algorithm, not donor support, scientific power or model performance.
- Next scientific dependency remains the unadopted donor-population/matching amendment and unresolved companion specifications before real donor qualification or axon access.

## 2026-10-18 — donor-population amendment preparation activated; no execution

- Scoped document work: turn attempt006's supported-animal recommendation into one reviewable, unadopted amendment; no job, test, scientific code or data access.
- Preserve CEBSIT, Cux2/Wfs1, the 27/16 roles, measurement v1.2 and all stage-A freezes; original companion draft remains historical and unadopted.
- Define primary/benchmark sets, prospective support filtering, effective animal counts and pseudo-audits; explain the 1,000/500 µm locality tradeoff and narrower-population calibration limit.
- Evidence: existing attempt006 census and stage-A manifest only. No new numeric result or threshold is selected.
- Handoff: [proposed amendment](runs/measurement_and_roles/attempt006/companion/donor_population_amendment_proposed.md); scientist decision and remaining companion qualification precede any claimed adoption or readiness.

## 2026-10-17 22:00 PDT — stage A frozen: n_coverage = 6; boundaries; cortical axis v2; companion draft needs decisions

- **n_coverage = 6** (job 48311775). n = 3–5 fail only on the predicted binding scenario; at n = 6 the worst SD is 0.0474. The analytic guard passes on all 648 checks. V_6 = 666.
- **Dendrite boundaries** (job 48311776, all 1,114 development cells): R_cap 600 µm; radial edges 50/100/150 µm; branch-order ranges 1–3, 4–5, > 5.
- **Cortical axis:**
  - v1 (job 48311786) left the medial and orbital pial surfaces no-flux, giving ACAv L2/3 a depth of 0.72;
  - v2, a reviewer correction (job 48312305): the whole outer L1 surface is pial (except sheet continuation and fibre), and the hemispheres are split;
  - result: ACAv 0.225, PL 0.135; lateral areas unchanged.
- **Donor support:**
  - recount (job 48311488): 853 development and 406 audit cells in exact pools of ≥ 5;
  - caliper census (jobs 48312584, 48314384): the companion draft's every-animal floor leaves no V6 target, while an animal-level rule keeps 21–27 development and 13–16 audit animals.
- **Freeze manifest:** `attempt006/stage_a_freeze_manifest.json` (SHA-256 `a12f4bf8…`). [Record](runs/measurement_and_roles/attempt006/README.md).
- **Next:** the scientist rules on companion decisions D01–D12 and gives a stage-B go-ahead. No axon read.

## 2026-10-17 17:45 PDT — R1–R5 confirmed; remaining stage-A outputs started

- **Decision:** the scientist confirmed readings R1–R5 of contract v1.2 and asked for the remaining stage-A outputs, with n_coverage reported before stage B.
- **Drafting:** Codex (gpt-6-astra, max effort) is drafting four pieces in parallel:
  - the n_coverage simulation tool;
  - the development-dendrite boundary tool;
  - the cortical Laplace tool;
  - the companion model/donor contract.
- **Recipient-set issue found:**
  - only 865 of 1,114 development cells (407 of 589 audit) sit in pools of ≥ 5, so a donor swap is possible only for them;
  - per animal, as few as 6 cells are donor-supported;
  - the contract's coverage counts and n range (3–14) use all eligible cells.
  The companion contract carries this as a decision for the scientist.
- **Recount:** `ep09_recipient_support.py` (job 48311488) recounts V_n over donor-supported cells after the contract's 24-cell coordinate relabel. Somata and atlas only.

## 2026-10-17 17:10 PDT — contract v1.2 adopted; stage A precheck rerun PASSES (somata and atlas only)

- **Contract:** v1.2 (read-only, SHA-256 `b01b593e…69b3`), recorded as `ep09_amend_2026_10_17_contract_v1_2`. Codex drafted it and the tool `ep09_feasibility_precheck_v1_2.py`; I reviewed both diffs. Codex raised no objection to R1–R5. [Record](runs/measurement_and_roles/attempt005/README.md).
- **Job 48295447:**
  - vocabulary of 355 anatomical / 710 sided targets, matching an independent atlas-only rebuild (job 48294937);
  - 15 parent residues unassigned, 36.7 mm³ in total (PAG, ZI and PB lose their main bodies; a diffuse 2.7 mm³ striatal residue).
- **Gates:**
  - **V3 = 689**; all 12 divisions are supported (minimum PAL 18); the primary, secondary and strict readings all pass;
  - **V_n** falls from 689 to 603 over n = 3–14, with all 12 divisions supported at every n;
  - the 21 failing targets are all ipsilateral and fail in only 1–3 animals.
- **Limit:** V3 survival is distance-driven, and the 498 non-telencephalic targets should fail stage C's detection rule.
- **Next:** the scientist confirms R1–R5; then the n_coverage simulation, dendrite boundaries and the companion contract. No axon read.

## 2026-10-17 16:53 PDT — scientist amends the vocabulary and families (contract v1.2); precheck rerun selected

- **Amendment:**
  - a parent with direct voxels hands its target role to its nonempty children, and its own voxels stay unassigned;
  - families are the 12 Allen major divisions;
  - a family-gate failure blocks only the secondary family analyses.
- **Readings I add, reported with the result:**
  - laminar pooling covers all cerebral-cortex areas (CTX 688), not just Isocortex;
  - a division is supported when it holds ≥ 3 sided targets, and the secondary analyses need ≥ 3 supported divisions;
  - the primary gates only require nonempty sets (V3, V_n, T).
- **Plan:** Codex drafts contract v1.2 and `ep09_feasibility_precheck_v1_2.py`; I review, then rerun from somata and the atlas only into `measurement_and_roles/attempt005`.

## 2026-10-17 14:05 PDT — stage A feasibility gate FAILS; amendment needed before any axon access

- **Runs:**
  - Job 48277641 failed at write time (missing `import json`; outputs rolled back).
  - Job 48278105 used the original laminar test: 58/116 targets, V3 = 99, 1 family, FAIL.
  - The corrected v2 run, after the laminar-name fix (ACAv6a/6b): **54/108 targets, V3 = 91, 1 family (grey), gate FAIL** ([record](runs/measurement_and_roles/attempt004/README.md)).
- **Causes:**
  - The family split is blocked because CB has only 2 sided targets.
  - Subcortical targets collapse because parent nodes carry direct voxels (STR, TH, MB, P, MY, HY, PAL, CB, OLF, HPF, CTXsp).
  - Coverage itself is adequate: 77 targets at n = 5.
- **Somata only;** no axon or dendrite geometry read. Next: the scientist's amendment decision.

## 2026-10-17 13:55 PDT — stage A precheck tool reviewed; run submitted (somata and atlas only)

- **Tool:** `ep09_feasibility_precheck.py`, drafted by Codex (gpt-6-astra max) and reviewed by me. Checked:
  - vocabulary partition recursion, antichain and disjointness;
  - exact point-to-box distances against 1,200 µm, with a 1,222 µm search radius, outward bounds and a rational tie check;
  - ipsilateral source exclusion over manifest and coordinate areas;
  - the family split rule and gates;
  - write-once read-only outputs.
- **Edit:** a soma in a background voxel is no longer fatal (its coordinate area becomes None).
- **Format check:** all 1,703 cohort SWCs have the soma as row 1 (7 fields, type 1, parent −1); soma rows only.
- **Run:** `attempt004/precheck.sbatch` (1 CPU, 12 GB, 2 h). No axon or dendrite geometry is read.

## 2026-10-17 11:00 PDT — contract v1.1 adopted with two tweaks; stage A feasibility precheck selected

- **Adopted:** `measurement_contract_v1_1_adopted.yaml` (read-only; SHA-256 `9f5bb299…881e`), recorded as `ep09_amend_2026_10_17_measurement_contract` in `SEARCH_POLICY.yaml`. The two tweaks:
  - nodes up to 100 µm outside the brain mask are tolerated;
  - at least one axon root is required, with no disconnected axonal components.
- **Stage A, first output:** the precheck from somata and the atlas only. It counts A_pre cells per animal for every sided target and applies the every-animal rule and the family gate.
- **Plan:** Codex drafts the tool; I review it and self-test it; it runs in Slurm. The gate result goes to the scientist before any other stage-A output.

## 2026-10-17 10:45 PDT — measurement contract v1.1 proposed (unadopted); awaiting the scientist

- **v1.1:** Codex revision under amendments A1–A5, reviewed by me.
  - Saved as [measurement_contract_v1_1_proposed.yaml](runs/measurement_and_roles/attempt003/README.md) (SHA-256 `0e30f7ae…`), with both Codex drafts and their rationale and change log.
  - It is now executable without unavailable certificates: assumptions are explicit, there are five non-selectable sensitivities, and a feasibility gate from somata and the atlas comes first.
- **My proposed tweaks:**
  - Containment tolerance of 100 µm outside the brain mask.
  - Require at least one axon root rather than exactly one (about 43 cells have 2 axonal soma children).
  - The feasibility risk from the 1,200 µm rule combined with every-animal coverage.
- **No axon or outcome has been read.**
- **Next, on approval:** stage A, beginning with the feasibility precheck.

## 2026-10-17 10:20 PDT — contract draft v1 reviewed: infeasible as written; Codex revising

- **Draft v1** (Codex) is a thorough 716-line YAML: terminal-subtree arbor rule (>1,000 µm and ≥ 2 tips), 1,200 µm whole-target distal rule, 100 µm erosion/dilation envelopes, an anatomy-derived area-level vocabulary split by side, a deterministic ontology family cut, the `n_coverage` simulation, CCF dendrite blocks, and staged access.
- **Review finding:** it requires per-cell QC certificates, image-coverage masks, local registration certificates and certified terminals. The release supplies none of these, so every pair would be uncertain and the contract is infeasible by construction.
- **Amendments sent to Codex:**
  - A1: cell-level observation from release inclusion plus the `Axon_and_dendrite` flag plus stage-B technical validity; a clipping sensitivity.
  - A2: erosion/dilation envelopes plus the cohort soma check (98.6%) replace certificates; a 200 µm sensitivity.
  - A3: terminals are the validated tree's tips.
  - A4: an outcome-blind feasibility gate first. The distal rule combined with every-animal coverage may remove nearby cortical targets.
  - A5: tool-enforced procedural separation, disclosed.

## 2026-10-17 09:55 PDT — measurement contract: SWC convention verified; Codex drafting the contract

- **Scientist:** have Codex draft the measurement contract, review it, and bring it back before any development outcome is opened.
- **Convention check** (job 48255686; [json](runs/measurement_and_roles/attempt003/swc_soma_convention.json); soma rows only):
  - SWC coordinates are µm, with axes (AP, DV, ML) matching the volume order.
  - Agreement with the catalog: area 1,679/1,703; area and layer 1,630/1,703.
- **Delegation:** `codex exec -m gpt-6-astra` at max effort, with web search, received the contract prompt: requirements, verified data facts, prior practice to cite. The prompt is in EP09 scratch `codex/`. Its output is reviewed before anything is recorded as a draft.

## 2026-10-17 09:46 PDT — measurement_and_roles attempt002: roles frozen (16 audit / 27 development)

- **Freeze:** job 48254820 at 16:44:00Z ([record](runs/measurement_and_roles/attempt002/README.md)).
  - Audit: Cux2 3 + 2, Wfs1 3 + 8 (pools 5–9 + 10+) = 16 animals, 589 cells.
  - Development: 27 animals, 1,114 cells.
  - The two leftover slots, tie-broken by seed, went to the 5–9 strata.
  - Manifests are write-once and read-only, with SHA-256 recorded.
- **Ledger v2** (job 48254918) adds EP09 roles. `EPISODE_REGISTRY.yaml` `seu_a1876_ep09_ep11.evidence_ref` now points to it.
- **Tool provenance:** Codex drafted the freeze tool (gpt-6-astra max). Claude reviewed it and self-tested allocation, determinism and write-once behaviour on synthetic strata.
- **No outcome opened.**
- **Next:** freeze the measurement contract (arbor and observability rules, target vocabulary, `n_coverage`, dendrite blocks). Then open development outcomes only.

## 2026-10-17 01:05 PDT — role rule approved; measurement_and_roles attempt002 (role freeze) selected

- **Decision (scientist):** 16 audit animals come from the 32 unclaimed brains, stratified by line (Cux2/Wfs1) × largest pool (5–9 vs ≥ 10). The other 27 usable animals are development, including the 10 EP10-shared and the 1 EP10-development brain. The manifest freezes before any outcome is opened.
- **Fixed rule:**
  - Seed 20261017.
  - Largest-remainder proportional allocation of the 16 slots over the four unclaimed strata (5, 5, 5, 17), ties broken by the seed.
  - Within each stratum, a seeded permutation of brains sorted by fMOST id.
  - Write-once outputs, made read-only, with SHA-256 in the summary.
- **Plan:** Codex drafts `freeze_ep09_roles.py`; Claude reviews and self-tests it; the freeze runs in Slurm.

## 2026-10-17 00:55 PDT — measurement_and_roles attempt001 complete: ledger built; SWC type codes resolved

- **Job 48222047** completed in 30 s ([record](runs/measurement_and_roles/attempt001/README.md)).
- **Ledger,** for the 52 Cux2 + Wfs1 brains with donor pools ≥ 5:
  - 32 unclaimed and 10 shared with EP10 unassigned units;
  - 1 EP10 development brain (EP09 development only);
  - excluded: 2 EP10 final and 7 EP11-reserved.
  - EP09 can use 43 brains, 42 of them audit-eligible.
- **Type codes** (3,516 files, pooled topology only): 0 and 2 are alternative axon codings, one per file, never joined to dendrite; 3 is dendrite; 4 (7 files) is apical; 10–12 are inline markers, assigned the compartment of their typed ancestor.
- **No outcome has been opened.**
- **Next:** the scientist approves a whole-animal role split. Roles come from identity, provenance and coverage only, and freeze before any outcome.

## 2026-10-17 00:45 PDT — measurement_and_roles attempt001: shared ledger and SWC type-code job submitted

- **Codex drafts reviewed** (gpt-6-astra max; prompt in EP09 scratch `codex/`):
  - `build_gao_exposure_ledger.py`: one edit, to tolerate blank descriptive IDs.
  - `resolve_swc_type_codes.py`: row parsing changed to split-based (only id, type and parent converted); duplicate and short rows are counted rather than fatal.
  - Both passed synthetic self-tests: eligibility categories and pooled parent-type counts checked by hand.
- **Ledger rules:**
  - EP10 development brains: EP09 development only.
  - EP10 final brains and EP11-reserved brains: excluded from EP09.
  - EP10 unassigned brains: shared-eligible.
  - Everything else: unclaimed.
  - Inputs: the catalog, EP09 QC, and the EP10/EP11 stage0 role and inventory tables, read-only and hashed.
- **Type codes:** pooled child-type × parent-type counts over Cux2/Wfs1/Scnn1a SWCs only. No coordinates and no per-cell values.
- **Next:** read the job's results, then propose whole-animal roles.

## 2026-10-17 00:25 PDT — IT definition amended to Cux2 + Wfs1 (L2/3, L4; 52 animals); ledger and type codes next

- **Decision (scientist):**
  - IT core = Cux2-Cre (L2/3, L4) + Wfs1-Tg2-CreERT2 (L2/3).
  - PlexinD1, Tlx3 and Rorb stay out; Scnn1a remains sensitivity-only.
  - The population narrows to L2/3–L4 IT; no L5 claim.
  - Recorded as `ep09_amend_2026_10_17_it_line_set` in `SEARCH_POLICY.yaml`.
- **Support:** 52 animals with within-animal pools ≥ 5 (38 with ≥ 10), 1,756 pooled cells ([recount](runs/metadata_feasibility/attempt005/pool_recount.json)).
- **Other users of the same release (read-only from their stage0 records):**
  - EP10 froze roles over 76 Gao MOp sample units (12 development, 8 final, 56 unassigned). Its M1 run opened outcomes for the 12 development groups; final and unassigned stay sealed.
  - EP11 holds a 36-unit SSp-tr inventory with outcomes closed.
- **Next (before roles):**
  - Build the shared exposure ledger.
  - Resolve SWC type codes 0 and 10–12 from pooled parent-type topology only.
  - Both tools to be drafted by Codex, then reviewed and run in Slurm.

## 2026-10-17 00:10 PDT — attempt005 complete: dendrites pass; the Cux2-only core gives 17 animals (below 20)

- **Codex fix reviewed and applied** (minimal join change; synthetic self-test on the real layout passed). Dendrite QC (job 48218964, 1:18) found all 6,469 IT-line cells. 6,302 pass (soma, ≥ 2 stems, ≥ 500 µm of dendrite), including 99.7% of the cells flagged `Axon_and_dendrite`. [Results](runs/metadata_feasibility/attempt005/README.md).
- **Recount** (job 48218966), animals with pools ≥ 5: Cux2 alone 17 (10 with ≥ 10); Cux2 + Wfs1 52 (38); with Scnn1a 54; all six lines 94 (post hoc, descriptive).
- **Codex line review**, with Wfs1 citations spot-checked: Wfs1-Tg2-CreERT2 is an L2/3 excitatory line with an IT projection pattern (Zariwala 2011; Shrestha 2015).
- **Open:** SWC type codes 0 and 10–12 are unidentified (needed before feature extraction), and the shared EP09/10/11 ledger is not yet built. No outcome has been opened.
- **Next:** the scientist chooses the line set: keep Cux2 only (infeasible), promote Wfs1 (52 animals), or run a new prospective test to restore PlexinD1, Tlx3 and Rorb.

## 2026-10-16 23:55 PDT — archive verified; dendrite QC failed on a join defect; Codex delegations launched

- **Download:** job 48187737 completed in 1:22:49. MD5 `db2da0aa83e6967b40f6acbe3c991176`, 13,605,777,073 bytes. The archive is now read-only, with `SHA256SUMS` and `ACQUISITION_COMPLETE_UTC` written in the steward directory.
- **Dendrite QC:** job 48188197 failed in 12 s with `IndexError`, because zero catalog cells matched. Archive members are `data_v3/<brain>/<cell>.swc`; catalog files are `<brain>_<cell>.swc` (job 48218037 listed member names only). The recount job 48188318 was cancelled with its dependency.
- **Delegation:** at the scientist's instruction, `codex exec -m gpt-6-astra` at max effort (read-only, text only) received two prompts: the join fix, and a cited line-characterization review with scope options. Prompts and outputs are in EP09 scratch `codex/`. Both are to be reviewed before use.
- **Not delegated:** no Codex prompt asks about the labelling vector, because the earlier search on it was blocked by a safety filter.

## 2026-10-16 18:55 PDT — attempt005: Cre-dependence confirmed; three lines fail the pre-set specificity rule

- **Download:** job 48187737 is running at about 2.5 MB/s (ETA about 19:50). Dendrite QC (job 48188197) and the pool recount are chained on it with `afterok`. The QC reader passed a synthetic two-cell self-test.
- **Cre test** (steps of 48187737; [results](runs/metadata_feasibility/attempt005/README.md#results)): the Ntsr1 control has an L6 odds ratio of 32.4 against wild type. Cux2 5.7, Wfs1 4.3 and Scnn1a 19.8 pass. PlexinD1 1.5, Tlx3 2.4 and Rorb 2.0 fall below the pre-set odds ratio of 3 (each interval excludes 1).
- **Consequence:** the approved core shrinks to Cux2, which had 17 animals with pools ≥ 5 before dendrite QC, below 20. Wfs1 passes but was approved only as a sensitivity line. Restoring PlexinD1, Tlx3 or Rorb would be a post hoc amendment.
- **Next:** dendrite QC and the recount, then a scope question for the scientist on the line set.

## 2026-10-16 18:30 PDT — CEBSIT proposal approved; metadata_feasibility attempt005 selected

- **Decision (scientist, Oct 16):** approve the CEBSIT proposal. Acquire Zenodo 16909126 into the Oak steward area, then run dendrite-only checks and confirm Cre-dependence before any outcome is opened. Recorded as an approved amendment in `SEARCH_POLICY.yaml`.
- **attempt005 plan:**
  1. Slurm download job into `steward_acquisition/cebsit_whole_cortex_zenodo_16909126/`, checked against Zenodo's MD5.
  2. Archive member names only, mapped to the catalog.
  3. A dendrite-only SWC reader on IT-line cells: soma and dendrite nodes; axon lines are skipped unparsed.
  4. Laminar specificity of somata by line, checked from the catalog.
- **Boundary:** no axon geometry, target, projection class or prevalence is computed. Roles and the shared EP09/10/11 ledger come before any outcome.

## 2026-10-16 18:10 PDT — attempt004: CEBSIT passes on group count; proposal written, unadopted

- **Survey:** the delegated literature survey finished. Only the CEBSIT whole-cortex projectome passes (Gao et al., *Neuron* 2025; Zenodo 16909126, CC BY 4.0, 13.6 GB of SWC files in CCFv3). Sorensen 2026, MouseLight and the Gao prefrontal set fail on projection-defined or wild-type labels.
- **My recount** (job 48185037, catalog only; `class` and `type_array` dropped unread): within-animal pools of ≥ 5 cells (line × CCF area × layer, `Axon_and_dendrite` cells) in Cux2 17, PlexinD1 23, Tlx3 12 and Rorb 5 animals. That is **57 core-IT animals** (35 with pools ≥ 10), against 20 needed. Wfs1 adds 35 as a sensitivity.
- **Identity:** IT-line brains have unique sample IDs. The shared sample IDs are wild-type prefrontal lab numbers.
- **Open before adoption:** dendrite completeness, the CSSP vector's Cre-dependence (one search was blocked by a safety filter and not retried), CCF-only coordinates, layers assigned by registration (592 cells register to L1), a ledger shared with EP10/EP11, and novelty against the dataset's own paper.
- **Next:** the scientist approves or rejects the [proposal](../DATASETS.md#proposed-primary-source-cebsit-whole-cortex-projectome--unadopted). Nothing has been downloaded or adopted.

## 2026-10-16 17:25 PDT — new-dataset route chosen; attempt004 BIL catalog survey (logged after the run)

- **Decision:** the scientist chose "find a new dataset" for EP09, to come back as a design proposal. No SEU-A1876 outcome is opened.
- **attempt004:** catalog-only survey through the BIL metadata API; no data files. Job 48169442 failed (the API answers 404 for an empty result); fixed; job 48169548 completed on the dev partition. Output: [bil_it_line_survey.json](runs/metadata_feasibility/attempt004/bil_it_line_survey.json).
- **Catalog counts (animals with any BIL dataset tagged "cell morphology"):** Cux2 22 (10 not in SEU-A1876), Plxnd1 73 (64 new), Tlx3 5 (3 new), Rorb 3, Nr5a1 11, Scnn1a 10.
- **Caveat:** most new-animal entries look like whole-brain image sets (dense Ai14 labelling, or fMOST stacks not yet reconstructed), not reconstructions. Only 1 new Cux2 and 4 new Plxnd1 animals have per-cell SWC entries.
- **Literature survey:** the delegated survey stopped early on an API rate limit; none of its output is used.
- **Next:** separate reconstructions from images among the new animals; finish the literature survey; draft the proposal.

## 2026-10-16 15:30 PDT — RAW→CCF transform search: not released; skipped as instructed

- **Instruction:** the scientist said to search hard for the transform data and skip if it does not exist.
- **Searched:** Zenodo 13944322 and Zenodo-wide; the Liu 2024 Google Drive mirror (7 folders); both papers' data statements; the BIL catalog via its metadata API (all datasets for the 12 IT brains); the mBrainAligner repository.
- **Result:** no RPM/FFD/landmark transform files for any IT brain. The only public example is the mBrainAligner demo for brain 18458 (non-IT).
- **Gap:** BIL storage is offline for a data-centre move into the week of Oct 18, so folder contents were not listed.
- **Orientation, resolved:** BIL image axes for 15 brains agree with the soma affine fit in all 11 overlaps. Raw z = AP; x/y order is per brain; 18453 is reflected. [Register, item 6](runs/metadata_feasibility/attempt003/README.md).
- **Effect:** none on the census or the feasibility verdict. Light API calls only; no job.

## 2026-10-16 14:35 PDT — attempt003 complete: at most 3 donor-supported animals

- Job 48155880 (2 CPUs, 1:02, ≈2.1 CPU-h; released). [Register](runs/metadata_feasibility/attempt003/README.md) and [figure](figures/ep09_metadata_feasibility_attempt003.png).
- **Lineage, resolved:** 1,736 Peng 2021 cells (1,727 exact names + 9 renamed 210254→15257) and 140 new cells from 19 brains.
- **`isManuallyChecked`, resolved:** equals that split exactly; it marks lineage, not QC.
- **Layers:** 74 blanks recovered from Peng 2021 curated layers, if the blank is an omission (post hoc D25 agreement 0.38). The 59 new Cux2 cortical cells in 18864 stay unknown.
- **IT:** line-based provenance, resolved; no finer non-axon class exists. **RAW→CCF:** exact transform not released; affine residual median 132 µm.
- **Census (animals with a ≥5-cell pool):** S0 2 (= Oct 6), P 3 (85 cells, 12 pools), S1 2, S2 3, S-IT 3. All below 9 and 20: the design is infeasible on SEU-A1876, a data limitation.
- **Next:** scientist scope decision in the daily batch. Recommended: headline primary unavailable; close EP09 or authorize a new labelled dataset.

## 2026-10-16 13:55 PDT — attempt003 Stage A: D25 calibration fails the fill condition

- Allocation 48155880 on sh04-04n22: [schemas.json](runs/metadata_feasibility/attempt003/schemas.json) and [stageA_layer_calibration.json](runs/metadata_feasibility/attempt003/stageA_layer_calibration.json). Column K not decoded.
- 1,876 cells: 511 recorded layers, 1,365 blank. Axis check: region agreement 0.649 (≥ 0.5, so no permutation search).
- Agreement with recorded layers (22 animals): 0.575 (95% CI 0.479–0.638), κ 0.406 (0.182–0.487). Pre-set condition (≥ 0.80, κ ≥ 0.60) **fails**. IT-driver cells: 0.568, κ 0.341.
- Main confusions: recorded L2/3 → L1 23/117; L5 → L6 57/260; L6 → L5 30/81.
- **Consequence:** no D25 fill in the primary census; S1 and S2 stay unvalidated sensitivities.
- **Next:** post hoc diagnostic: does D25 reproduce Peng 2021 registered regions? This separates an implementation defect from curation. The fill rule is unchanged.

## 2026-10-16 13:45 PDT — metadata_feasibility attempt003 selected: source-fact recovery

- **Authority:** the scientist's EP09 request and the 2026-10-16 `AGENTS.md` clause. No provider contact.
- Rules for layer, IT, lineage, QC and census fixed first in [ASSUMPTIONS.md](runs/metadata_feasibility/attempt003/ASSUMPTIONS.md), before any new cell-row decoding.
- **Sources:** Liu 2024 (PMC11599929), Peng 2021 (PMC8494643), provider code; Peng Supplementary Tables 1–2 downloaded to EP09 scratch.
- **Next:** bounded Slurm allocation for headers, D25 calibration, lineage and census. Projection class, Peng `Subclass_or_type`, matrices and payloads stay closed.

## 2026-10-06 Pacific — metadata_feasibility: provider alias confirmed

The selected bounded metadata check exited 0 in coordinator-owned job
46842156 on `sh04-04n22.int`. Its
[direct observation](runs/metadata_feasibility/attempt002/provider_alias236174_observation_job_46842156.json)
returns exactly one matching row from the retained public supplement, row 81:
Animal ID `236174`, Image ID `16124`, fMOST,
`Gnb4-IRES2-CreERT2;Ai139`, provenance `BIL meta (20231223)`, and
`Claustral and cortical Car3+ IT`, referencing Peng et al. 2021. The provider
locator is `https://download.brainimagelibrary.org/7f/53/7f537a62e521a26a/mouseID_236174/`.
The animal ID does not already occur in the retained 38-animal join. Its
original unmatched record remains brain `236174` / 43 cells.

This resolves the factual alias/genotype gap: the original implementation
searched the Image ID column, while this source group uses the animal ID.
The observation agrees with the independently checked BrainTell string and
TeraVR Methods. No EP11 crosswalk, source-cell workbook, morphology or axonal
outcome was read. The shared parent allocation is charged once to EP18;
no separate EP09 allocation or additional charge is introduced.

Updated current GOAL, DATASETS, paper plan and README so they no longer ask
for this resolved identity/genotype. Preserved `brain_genotype_join.json`,
`genotype_support_summary.json` and all historical entries. Reuse the direct
observation as correction evidence; any future integrated join must use a
separate output and retain original brain, animal and image IDs explicitly.
No blanket cross-namespace fallback, primary IT eligibility, layer recovery,
donor support or role assignment follows. The 12-driver-animal and
two-potential-donor-animal feasibility limits, original 12/8 design and
unadopted nine-animal proposal remain unchanged. Remaining source questions
concern authentic layer/class and matched-donor support, not this alias.

## 2026-10-06 Pacific — metadata_feasibility: exact provider alias check prepared

Rechecked the original BrainTell genetic table and retrieved the primary
[TeraVR Methods through PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6677772/).
They retain the recorded `236174`/`16124` Gnb4 lead. The named join implementation
matches source brain IDs only to the supplement's Image ID column B; the saved
schema separately identifies Animal ID in column A. Selected a bounded check
of those two exact keys in the already retained public brain-metadata workbook.
No neuron/source-cell workbook, morphology, axon, projection class or sibling
output is needed. BrainTell's 45-neuron gallery count is distinct from the 43
retained SEU-A1876 source cells and will not substitute for cell lineage.

Prepared `tools/reconcile_brain236174_provider_alias.py`: actual non-login Slurm
shell, one CPU and explicit single numerical-thread caps required. It reads
only `runs/metadata_feasibility/attempt002/provider_brain_metadata.xlsx` and
the existing non-outcome `brain_genotype_join.json`; stdout contains selected
identity/genotype/provenance fields and whether the animal ID is already in
that join. The coordinator will run it for at most five seconds inside the
substantive EP18 provisioning allocation and redirect stdout to
`runs/metadata_feasibility/attempt002/provider_alias236174_observation_job_JOBID.json`.
No separate metadata job is requested. This is the existing
`metadata_feasibility` analysis, not a new scientific experiment.

Code was prepared and read back on sh04-ln04 without an allocation; no test,
reader execution, scheduler call, outcome or frozen-artifact mutation occurred.
Actual observation and reconciliation remain pending. The current 12/8 roles,
IT-driver support, donor bound and unadopted nine-animal proposal are unchanged.

## 2026-10-06 Pacific — public brain identity/genotype lead recorded

At the user's request to add the verified online findings, recorded the
original TeraVR Methods and BrainTell genetic table in the
[existing paper plan](paper_plan.md#october-6-public-source-lead--brain-236174-and-provider-key-16124)
and DATASETS. The paper names brain 236174 and Gnb4-IRES2-CreERT2;Ai139;
the provider table links its key 16124 to the genetic-background string
containing 236174-16124. This is a concrete reconciliation lead, not a
completed EP09 join, extra authenticated IT animal or qualified layer/donor
pool. Existing saved joins, feasibility counts, roles and policy are unchanged;
no EP11 crosswalk or live sibling output was consumed.

Documentation-only work on verified login host sh04-ln04 without allocation.
The preceding public-source search returned incidental published-result text;
none selected a model or criterion. No morphology/axon payload, closed outcome,
test, import, scheduler call, contact, job or scientific amendment occurred.

## 2026-10-06 Pacific — retained source/layer counts limit donor support to two animals

Under the instruction to continue, inspected only the already permitted
October 3 `brain_genotype_join.json` and its recorded per-animal
`source_layer_counts`. The original metadata authority covers this bounded
refinement; the earlier proposal's extra count-approval step was unnecessary.
No new workbook/payload access, code execution, allocation or count job was
needed. Actual shell: sh04-ln04.stanford.edu, no Slurm allocation; work was
limited to reading the saved small tables and editing documentation.

The [living plan](paper_plan.md#october-6-metadata-feasibility-and-unadopted-exploratory-route)
now transcribes all 12 authenticated IT-driver animals and the ten exact
source/layer strata with at least five recorded cells. The active donor floor
of four nonself donors requires at least five cells, without selecting the
still-unset `minimum_donors_per_pool`. Only Cux2 animals 374712 and 382148 have
such driver-compatible recorded strata: seven strata / 53 cells and three
strata / 19 cells respectively. The other ten animals have maximum compatible
stratum size at most three, or no compatible recorded layer. Thus the current
documented labeling route supplies at most two potentially donor-supported
animals and 72 cells before finer class, calipers, duplicate/paired identity,
quality and target support. This is not qualified eligibility or a finding
about dendrite–axon coupling.

Missing layers and brain 236174 remain unresolved. Authentic corrections or
additional independent annotations could change the labeling route and this
bound; they were not inferred from projection class, axonal features or EP11.
The nine-animal exploratory draft remains unadopted and unsupported by current
metadata. Original roles, thresholds, budgets and access are unchanged. No
code, test, simulation, scheduler action, empirical fit or new authorization
was produced. Historical attempts and the earlier proposal entry below are
preserved; the unnecessary metadata-count job is superseded by this result.

## 2026-10-06 Pacific — authenticated label count corrected; fallback unadopted

Read the existing EP09-only genotype support metadata: 38 matched brains
map to 38 distinct animals; 6 Cux2 + 5 Plxnd1 + 1 Tlx3 give at most 12
authenticated cortical IT-driver animals before restrictions. The historical
parenthetical below allowing a hypothetical thirteenth is not an eligible
count: brain 236174 remains unmatched. The external critique reports a
different EP11 crosswalk; no EP11 data or assignments were consumed, and an
original-provider identity reconciliation remains necessary.

Updated GOAL, DATASETS and the existing paper plan with those facts and a
clearly unadopted metadata-count/at-least-nine-animal exploratory LOAO proposal.
The metadata count cannot settle morphology/donor quality or outcome support,
and neither 20 nor nine is represented as a demonstrated power result. Active
12/8 roles, thresholds, resource ceilings and access remain unchanged. No
metadata job, source workbook/payload read, test, simulation, scheduler action
or empirical model was run. Original preparation records remain preserved.

## 2026-10-03 — metadata_feasibility selected

The scientist authorized feasible work across EP01–EP21. The official SEU-A1876
source is already preserved in the steward quarantine; it remains read-only
and will not be mounted in a candidate workspace. First qualify the spreadsheet
schema with `outputs/tools/inspect_metadata_schema.py` in Slurm job 46425841,
one CPU, 1 GiB and at most five minutes. This steward operation reads worksheet
headers and the exact shared-string entries they reference. It does not decode
cell-row values, inspect axonal morphology, assign roles or infer IT membership
from projection class. Preserve the result in
`runs/metadata_feasibility/attempt001/metadata_schema.json`.

The purpose is to identify a potential non-axon IT and biological-group source
before any scientific pipeline can use these mixed-role inputs. Acquisition
alone is not a qualified handoff. Shared EP09/10/11 exposure and audit rules
remain binding; existing EP10/11 jobs and results are untouched.

Schema inspection completed on sh03-06n50, job 46425841. The 12 columns include
cell and brain identities, soma coordinates, manual-check status, source region,
projection class and cortical layer. No independent genetic label is present.
No cell-row values were decoded. This is not proof that a usable genetic
provenance join is unavailable elsewhere.

The [provider paper, Methods](https://www.nature.com/articles/s41467-024-54745-6)
identifies Supplementary Data 1 as brain-level acquisition, transgenic-label and
targeted-neuron metadata. Selected next step: acquire that exact public XLSX
into `runs/metadata_feasibility/attempt002/` and inspect its schema, using one
CPU/1 GiB/<5 minutes in job 46425841. A potential join will use brain identity,
genotype and acquisition metadata only; provider projection classes and axonal
values remain excluded. The earlier [molecular-lineage source](https://www.nature.com/articles/s41586-021-03941-1)
provides a lead for Cux2 and Plxnd1 labels, but neither source substitutes for
an authenticated cell-to-brain-to-genotype join or the minimum group support.

The public supplement was acquired successfully (93,468 bytes; SHA256
`293852b3c524de509886d68096606716c53c39338ca2c1a92b0569a7afc439f7`).
Its schema supplies animal ID, image ID, transgenic line, provenance and
targeted-neuron references. Selected `tools/join_brain_genotype_metadata.py`
next, one CPU/2 GiB/<5 minutes in job 46425841. It decodes only Full_morphometry
columns A/B/J/L (cell ID, brain ID, soma region and layer), excluding projection
class and every morphology/outcome value. It performs an exact brain-ID/image-ID
join and reports non-outcome support; no role or eligibility rule is selected.

The first join stopped on conflicting public supplement image identities
(`ValueError: Conflicting provider metadata for image identity`), before any
joined report or eligibility decision. The diagnosed repair retains all distinct
provider records and labels a source brain ambiguous if its exact key has
multiple records. Unrelated ambiguous public keys no longer abort all joins;
no ambiguous source brain is silently assigned a genotype. Retry selected
under the same one-CPU, five-minute bounds; no outcome or science rule changed.

The repaired join completed: 1,876 cell identities in 39 brain IDs; 38 brains
match the public genotype supplement exactly, with no ambiguous matched brain.
Brain 236174 (43 cells) lacks a matching image ID. Projection class, morphology
and target values were not decoded. Selected a one-CPU summary of animal/group
and layer support by genotype before judging whether 12 development plus 8
audit biological groups could exist; this is descriptive metadata support,
not cohort selection or an interpretation of axonal outcomes.

Support summary completed in job 46425841. The 38 matched brain IDs map to
38 distinct animal IDs. Explicit cortical L2/3/4/5 IT-targeting genotypes
span 6 Cux2, 5 Plxnd1 and 1 Tlx3 animals before any layer/donor/QC restriction.
This route therefore supplies at most 12 groups (13 even if the unmatched
brain were eligible), below the frozen 12 development plus 8 audit groups.
Gnb4's documented L6/claustral IT population does not meet the primary layer
scope. Other lines cannot acquire an IT label from forbidden projection-class
values. The result is a source-support limitation for this authenticated
labeling route, not a dendrite–axon association result or a proof that no
additional non-axon annotation could exist. Further primary execution needs
such an annotation or an explicitly approved downgrade. No roles, dendritic
features, targets, model fits or audit opening were produced.

## 2026-10-02 — Concept figure v3

Built-in imagegen produced the lean [own-versus-donor schematic](ep09_conceptual_question-v3.png);
targeted corrections removed ambiguous axon joins and a cross-species icon.
The [exact prompt set](ep09_conceptual_question-v3-prompt.md) is saved.
Reader links/captions now match the three-panel design and no longer describe
equivalence as a shared-class cause. Earlier assets remain. No scientific
contract, endpoint, data access or execution state changed.

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md) completed from existing design and primary literature. Correct-versus-donor equivalence now means no meaningful own-dendrite advantage at the tested scale, not identification of a shared-class cause. Prior read-only BR retrieval on the related projection question returned no relevant evidence; no outcome or execution state was newly inspected.

## 2026-09-27 — scientific design revision

- Reframed the question as whether a neuron’s own native dendrites add
  information about distal target families beyond source anatomy and
  axon-independent acquisition or dendrite quality.
- Separated an overall prediction gain from the stronger scale-by-target-family
  explanation needed for an in-depth paper.
- Defined the exact equal-target, equal-group Brier estimand and kept
  biological groups as the inference unit.
- Added 20 outcome-free, independent-block residual derangements as
  locked-pipeline capacity controls. Kept the cell-identity test distinct by
  swapping an intact dendrite within a frozen dendrite-only morphology class.
- Changed the target-hierarchy analysis to compare the dendritic increment under
  true versus matched shuffled families, so context-only base rates cannot
  create the explanation.
- Defined train-and-test conditional block replacement with 10 full refits
  rather than a post-fit feature permutation.
- Added fixed E0/E1 explanatory models containing every scale block, so an
  adaptively omitted feature cannot be mistaken for biological absence.
- Defined matrix replication through development-scaled cosine similarity,
  minimum norms, required signs, and nested pseudo-audits that relearn the
  entire template.
- Split small-group and max-`t` simulations into frozen 50,000-replicate
  calibration and validation sets.
- Split lower and upper small-group tails, including the cell-swap equivalence
  interval, so skewed group contrasts cannot silently miscalibrate one side.
- Made the development-frozen block-by-family matrix a prospective audit
  prediction rather than a post-hoc heat map.
- Added an outcome-dependent decision tree that distinguishes cell-specific
  organization, shared morphology-class organization, unresolved cell identity,
  diffuse information, unavailable explanation, invalid evidence, informative
  null, and unresolved data.
- No source morphology, development score, or audit outcome was opened during
  this revision.
- Generated `ep09_conceptual_question.png` with the built-in image-generation
  tool from a synthetic prompt; it contains no source neuron or outcome value.

Planning validation passed with no warnings. The final independent rereview
confirmed the tail calibration, nuisance logic, cell-identity branches, matrix
replication language, and revised decision figure; no protected outcome was
opened.

## 2026-09-29 — novelty-driven primary amendment

- Reclassified M1-correct versus M0 as a required benchmark rather than the
  paper's novelty claim after reviewing Peng 2021, Gao 2023, Liu 2024/2026,
  and Sorensen 2026.
- Promoted correct-versus-coherent-matched-donor prediction to the primary
  biological estimand, averaged across frozen donor plans and inferred at the
  biological-group level.
- Restricted the proposed headline population to cortical IT neurons whose IT
  membership is independent of axonal target outcomes.
- Made exact CCFv3 targets the decisive outcome resolution and changed the
  explanatory template to correct-minus-donor scale × exact target. Broad
  families remain secondary summaries and falsifiers.
- Corrected the source citations and recorded that SEU-A1876 is predominantly
  a refinement of the Peng-2021 data lineage, so held-out groups support
  internal grouped reproducibility rather than independent replication.
- Added a metadata-only feasibility gate for IT-label lineage, original versus
  newly annotated source lineage, exact-target support, and donor support under
  the frozen matching strata.
- Regenerated the conceptual figure to show the new primary hierarchy.
- No source morphology, development association, target prevalence, or audit
  outcome was opened during this amendment.

## 2026-09-30 — amended design completion

- Preserved the earlier revision records above; their prior implementation
  descriptions do not override the amended GOAL and search policy.
- Synchronized DATASETS with the fixed-model held-out donor substitution used
  by both the primary and fixed all-block explanation. The old ten full
  training-and-test replacement refits are superseded, not performed results.
- Added non-axon class to the M0 prose and aligned the positive primary,
  benchmark, E1 increment, norm and narrower-result interpretation across docs.
- Visually inspected the previously generated built-in imagegen output and
  copied it non-destructively to `ep09_conceptual_question-v2.png`. The old
  `ep09_conceptual_question.png` remains historical. The new schematic has no
  source neuron or observed heat-map/bar value. No image was generated or
  edited in this completion task.
- Replaced the obsolete single-analysis ASTRA snapshot with an experiment
  program for the amended contract. Pinned schema/official semantic tooling
  has not been located; official conformance is not claimed.
- No payload acquisition, source morphology or outcome read, job submission,
  model fitting, Git mutation, or canonical scientific-state mutation occurred.

### Current execution handoff, keyed to ASTRA analyses

| Analysis ID | State | Actual blocker / next action |
| --- | --- | --- |
| `metadata_feasibility` | Not started; no attempt or process/job ID | Source metadata not provisioned. On a separately authorized data task, obtain metadata and test independent IT labels, biological groups, lineage and donor support without outcomes. |
| `measurement_and_roles` | Not started | Requires a usable primary cohort; then establish cell/coordinate/measurement rules and whole-group shared roles. |
| `grouped_development` | Not started | Requires instantiated development measurements and roles; no fitted model exists. |
| `matched_donor_qualification` | Not started | Requires morphology, outcome-independent matching support and the development pipeline; no donor bank exists. |
| `nuisance_qualification` | Not started | Requires morphology and the development pipeline; no nuisance generator or fitted controls exist. |
| `explanation_and_inference_freeze` | Not started | Requires development and qualified controls; margins, exact-target template, simulations and resource feasibility remain unresolved. |
| `one_shot_group_audit` | Closed, not started | Requires every EP09/10/11 shared contract to be frozen and all primary dependencies satisfied; no audit has opened. |
| `paper_synthesis` | Design/paper plan only; evidence synthesis not started | Requires the locked audit report; no scientific finding exists. |

Completed durable deliverables are the revised design documents and v2
conceptual schematic only. No empirical artifact, active job, or completed
experiment is reported.

Focused completion checks passed: whitespace check, all seven current local
Markdown links, retention of the old PNG plus integration of v2, and search-
policy/ASTRA YAML parsing with duplicate-key detection in the existing
`encoding_model` environment (PyYAML 6.0.2). All eight ASTRA analysis IDs, 24
input references, output-input declarations and the acyclic dependency graph
pass the focused local check. Official ASTRA schema and semantic conformance
remain unverified. No package was installed.
