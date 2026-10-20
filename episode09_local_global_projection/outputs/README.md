# EP09 outputs

**October 18 — donor-population amendment prepared, unadopted.** The
[scoped proposal](runs/measurement_and_roles/attempt006/companion/donor_population_amendment_proposed.md)
defines a donor-supported primary, separate all-eligible benchmark, prospective
animal/target support rules and 1,000 µm matching with a report-only 500 µm
sensitivity. It propagates effective animal counts and pseudo-audits and links
the completed stage-A bindings. No scientific choice or outcome access was
adopted; [activation record](experiment_log.md#2026-10-18--donor-population-amendment-preparation-activated-no-execution).

**October 17, 22:00 — stage A frozen: n_coverage = 6. The companion contract
awaits decisions.** See [attempt006](runs/measurement_and_roles/attempt006/README.md).

- **Boundaries:** R_cap 600 µm; radial shells 50/100/150 µm; branch-order
  ranges 1–3, 4–5, > 5.
- **Cortical axis:** the depth field was corrected in v2.
- **Donor support:** in exact pools, 853 development and 406 audit cells have
  ≥ 4 donors. The draft companion's every-animal support floor is infeasible;
  an animal-level rule is recommended.
- No axon has been read, and stage B has not started.

**October 17, 17:10 — contract v1.2 adopted; the stage-A precheck passes.**
See [attempt005](runs/measurement_and_roles/attempt005/README.md).

- **Amendment:** a parent with direct voxels hands its target role to its
  children; families are the 12 Allen major divisions; a family-gate failure
  blocks only the secondary family analyses.
- **Vocabulary:** 355 anatomical / 710 sided targets, with 36.7 mm³ of parent
  residue unassigned.
- **Gates:** V3 = 689, and all 12 divisions hold ≥ 3 targets. V_n falls from
  689 to 603 over n = 3–14.
- **Next:** the n_coverage simulation, dendrite boundaries and the companion
  contract. No axon has been read.

**October 17, 14:05 — the v1.1 stage-A gate failed (1 family).** See the
[v1.1 precheck record](runs/measurement_and_roles/attempt004/README.md).

**October 17, 10:45 — measurement contract v1.1 proposed; awaiting the
scientist's review.** See the
[contract and review notes](runs/measurement_and_roles/attempt003/README.md).

- The arbor rule is a terminal subtree of more than 1 mm with at least 2 tips,
  with 100 µm envelopes.
- The distal rule is 1,200 µm to the whole target.
- Targets are area-level atlas nodes split by side, with a deterministic
  family cut.
- `n_coverage` comes from a simulation, and dendrite blocks are
  rotation-invariant.
- A feasibility gate from somata and the atlas runs first.
- No axon has been read.

**October 17, 09:46 — whole-animal roles frozen: 16 audit, 27 development.**

- [Role manifests](runs/measurement_and_roles/attempt002/README.md): read-only,
  with SHA-256 recorded.
- Shared ledger v2 records EP09's claims for EP10 and EP11.
- No outcome has been opened.
- Next: freeze the measurement contract, then open development outcomes only.

**October 17, 00:55 — IT core amended to Cux2 + Wfs1 (L2/3, L4). The shared
ledger is built and SWC type codes are resolved; a role split is pending.**
See [measurement_and_roles attempt001](runs/measurement_and_roles/attempt001/README.md).

- 43 animals with donor pools are usable by EP09. 32 of them are claimed by
  no other episode; 10 are shared with EP10 unassigned units; 1 is EP10
  development (EP09 development only).
- EP10's final brains and EP11's reserved brains are excluded.
- Axon is coded 0 or 2, dendrite 3, and codes 10–12 are inline markers.
- No outcome has been opened.

**October 17, 00:10 — CEBSIT qualification done; a line-set decision is
pending.** In
[attempt005](runs/metadata_feasibility/attempt005/README.md):

- the archive is acquired and checksum-verified;
- Cre-dependent labelling is confirmed, with Ntsr1 as positive control
  (odds ratio 32);
- 6,302 of 6,469 IT-line cells pass dendrite QC.

Under the pre-set line rule only Cux2 remains, giving 17 animals with donor
pools, below the 20 needed. Promoting Wfs1, a literature-supported L2/3 IT
line that passed the test, gives 52. No outcome has been opened.

**October 16, evening — a new-dataset proposal awaits approval.** The
[attempt004 search](runs/metadata_feasibility/attempt004/README.md) found the
CEBSIT whole-cortex projectome: Zenodo 16909126, CC BY 4.0, SWC files in
CCFv3. 57 core IT-line animals have within-animal pools of ≥ 5 cells, and
none of its brains comes from Peng 2021. The proposal and its open checks are
in [DATASETS](../DATASETS.md#proposed-primary-source-cebsit-whole-cortex-projectome--unadopted).
Nothing has been downloaded or adopted.

**October 16, Pacific — source facts recovered without provider contact; the
design is infeasible on SEU-A1876.**
[Attempt003](runs/metadata_feasibility/attempt003/README.md) classifies every
open item and recomputes donor support from metadata alone. Its
[figure](figures/ep09_metadata_feasibility_attempt003.png) shows layer
calibration and per-animal pools; its rules were
[fixed beforehand](runs/metadata_feasibility/attempt003/ASSUMPTIONS.md).

- **Lineage, resolved:** 1,736 cells are of Peng 2021 lineage and 140 are
  new, from 19 brains. `isManuallyChecked` is exactly this lineage flag.
- **Layers:** 74 blank layers were recovered from Peng 2021 curated layers,
  assuming each blank is an omission. The 59 new cortical Cux2 cells stay
  unknown, because the coordinate-derived layer failed calibration (κ 0.41).
- **Donor support:** at most **3 of 12** IT-driver animals have a matched
  pool of ≥5 cells in any scenario, against 20 required (9 for the unadopted
  route).
- **Next:** a scope decision by the scientist; see the
  [log](experiment_log.md).

**October 6, Pacific — identity/genotype gap resolved:**
[the completed metadata check](runs/metadata_feasibility/attempt002/provider_alias236174_observation_job_46842156.json)
confirms supplement row 81: Animal ID `236174`, Image ID `16124`,
`Gnb4-IRES2-CreERT2;Ai139`, with `BIL meta (20231223)` provenance.
The original join searched Image ID alone; its 38-match result and unmatched
43-cell record remain preserved. This correction adds no qualified L2/3–L5
IT animal, layer or donor support. No provider answer about this alias is now
needed. [Completion and next action](experiment_log.md#2026-10-06-pacific--metadata_feasibility-provider-alias-confirmed).

[Public online resources, checked October 6](../DATASETS.md#public-online-resources--checked-october-6-2026): exact original morphology archive inventory and provider genetic-identity table, with eligibility limits.

**Earlier October 6, Pacific — original-provider identity lead added:** the
[236174/16124 source note](paper_plan.md#october-6-public-source-lead--brain-236174-and-provider-key-16124)
links the TeraVR Methods and BrainTell genetic table. They identify a Gnb4
genotype/alias lead, now resolved by the direct supplement observation above.
They do not add an eligible IT animal. See the historical
[source-update record](experiment_log.md#2026-10-06-pacific--public-brain-identitygenotype-lead-recorded).

**October 6, Pacific — saved metadata resolves the next feasibility question:**
Of the 12 documented cortical IT-driver animals, only two have currently
recorded driver-compatible exact source/layer strata with at least five cells,
necessary for four nonself donors. Their ten strata contain 72 recorded cells
before finer class, calipers, quality and target restrictions. The
[12-animal table and limitations](paper_plan.md#october-6-metadata-feasibility-and-unadopted-exploratory-route)
come directly from the existing permitted join; no new count job or access
approval is needed. This current-metadata upper bound supports neither 12/8
nor the unadopted nine-animal route. Authentic missing-layer/identity/label
corrections could change support; none are inferred here. Active roles,
thresholds, budgets and access remain unchanged; no fit or job is selected.

2026-10-03: the [non-axon genotype support check](runs/metadata_feasibility/attempt002/genotype_support_summary.json)
recovers animal/genotype metadata for 38 of 39 source brains. Its documented
IT driver route supplies too few groups for the frozen 12/8 design before
further restrictions. Additional independent labels or an approved downgrade
are needed; [the execution log](experiment_log.md) records the exact scope.

The current design asks whether the correct recipient dendrite predicts exact
distal targets better than a coherent dendritic residual from a tightly
matched donor. M1-correct versus context is a supporting benchmark, not the
novelty claim.

See the [paper plan](paper_plan.md) and the current image-generated
[conceptual schematic](ep09_conceptual_question-v3.png), with its
[exact imagegen prompt set](ep09_conceptual_question-v3-prompt.md). Earlier
versions remain historical. Panel B holds the recipient and its target vector
fixed; only its dendritic representation changes. Colors, target slots and
scale bands are illustrative. No heat map, measured score or biological cause
of equivalence is depicted.

The [ASTRA experiment program](astra/v0.0.14/astra.yaml) records the required
experiments and artifact dependencies. Their execution state is recorded in
the [experiment log](experiment_log.md); the program is not a scientific result
or execution authorization.

No model run has started, no biological roles have been instantiated, and no
audit axonal outcome has been opened. Transient archive expansion and feature
generation belong in episode-specific scratch, not here.
