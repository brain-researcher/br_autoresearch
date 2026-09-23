# Autoresearch data locations

This document is the human-readable companion to
[`DATA_LOCATION_MANIFEST.json`](DATA_LOCATION_MANIFEST.json). It records the
filesystem layout observed on 2026-09-23 and the currently selected routing
for a later migration phase.

This is **inventory only for scientific payloads**. Phase 1 created only the
empty private namespace, its empty routing subdirectories, and an explanatory
README. No scientific source or payload was moved, copied, linked, deleted,
renamed, mounted, or permission-modified. A path appearing here does not grant
data access, assign a scientific role, make an episode launch-ready, or turn a
local artifact into canonical Brain Researcher state.

## Storage model

| Logical ID | Role | Current path | Planned path | State |
| --- | --- | --- | --- | --- |
| `canonical_workspace` | Lightweight Git contracts and code | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch` | same | `canonical_in_place` |
| `canonical_historical_prior_records` | Curated, Git-tracked frozen-prior snapshot | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/_examples/historical_prior_records` | same | `canonical_in_place` |
| `public_shared_root` | Durable public source releases | `/oak/stanford/groups/russpold/data/br_autoresearch_data` | same | `canonical_in_place` |
| `planned_private_data_root` | Empty private destination namespace | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data` | same | `destination_namespace_created` |
| `legacy_worktree` | Mixed legacy code, inputs, outputs, and runtime history | `/oak/stanford/groups/russpold/users/zijiao/autoresearch` | none as a whole | `destination_unassigned` |
| `legacy_steward_acquisition` | Private steward/quarantine tree | `/oak/stanford/groups/russpold/users/zijiao/autoresearch/.steward_acquisition` | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition` | `staged_not_moved` |
| `legacy_ep17_restricted_raw` | Restricted CNeuroMod-THINGS raw source | `/oak/stanford/groups/russpold/users/zijiao/autoresearch/episode17_cneuromod_model_ranking_stability/inputs/cneuromod-things-1.0.1-restricted-raw` | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw` | `staged_not_moved` |
| `recovery_ep02_boundary_semantics` | Protected EP02 recovery material | `/oak/stanford/groups/russpold/users/zijiao/.autoresearch_recovery/20260922_ep02_boundary_semantics` | none approved | `protected_hold` |
| `scratch_autoresearch_runtime` | Legacy transient episode runtime | `/scratch/users/zijiao/autoresearch` | no durable destination | `transient_retained` |
| `scratch_canonical_runtime` | Current launcher runtime convention | absent until needed | `/scratch/users/zijiao/br_autoresearch` | `absent_planned_target` |

The private namespace
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data` now exists in
mode `2750`, with empty `manifests/` (`2750`), `restricted/` (`2700`), and
`recovery/` (`2700`) directories plus `README.md` (`0640`). The final
`steward_acquisition/` and restricted CNeuroMod payload directories remain
absent. No scientific payload was materialized there.

The canonical Git workspace is about 12 MiB and contains no data symlinks.
Episode `inputs/` directories contain only their tracked documentation, plus
the EP01 prior-lineage manifest. The manifest addresses a separate canonical
tracked prior subset under `_examples/historical_prior_records`: 71 regular
files totaling 3,114,947 logical bytes, with no symlinks. Large inputs remain
outside the canonical worktree until a separately authorized, role-aware
provisioning step.

## Durable public shared releases

The public shared root currently contains 28 release directories. Their full
release trees contain 11,086 regular files totaling 396,190,786,822 bytes;
directory-entry sizes are excluded. The Phase 1 `DATA_CATALOG.md` now lists all
28 and declares 10,817 source files and 396,139,712,889 source bytes. The
difference is release-local metadata and control files; the two totals use
different, intentional counting scopes.

### Releases currently bound to campaign episodes

| Episode | Durable release | Logical size | Observed mode |
| --- | --- | ---: | ---: |
| EP02 | `dudman_learning_rate_2023/figshare-21816054-v1` | 0.27 GB | `2550` |
| EP05–EP08 | `sensorimotor_lfp/dryad-xd2547dkt-v5-payload` | 9.57 GB | `2750` |
| EP12 | `flyem_male_cns/gcs-male-cns-v1.0-flat-connectome` | 31.32 GB | `2550` |
| EP15 | `functional_fusion_mdtb/zenodo-16788784-v1.0` | 14.59 GB | `2550` |
| EP16 | `braingate_long_term_array_performance/dryad-x0k6djj1h-v6` | 84.73 GB | `2770` |
| EP18 | `things_eeg1/openneuro-ds003825-v1.2.0` | 59.44 GB | `2550` |

EP16 needs attention: Phase 1 corrected its documentation to distinguish
integrity verification from write sealing, but the release root and sampled
payload remain writable at mode `2770`. The permission seal is still a
readiness blocker, and Phase 1 made no source permission change.

The 25 scientific payload files in the sensorimotor release are hard-linked
to the legacy EP05 input store. They are one OAK allocation, not two
independent copies. The EP05 scratch source is a separate copy.

### Other public shared releases

| Release | Logical size | Catalog state | Mode |
| --- | ---: | --- | ---: |
| `brochier_multielectrode_grasp/g-node-f83565` | 15.50 GB | listed | `2750` |
| `epic_ptir_coating/mendeley-7p3cxn7jtn-v1` | 0.27 MB | listed | `2750` |
| `falcon_h2/dandi-000950-0.241029.1403` | 1.23 GB | listed | `2750` |
| `falcon_m1a/dandi-000941-0.241029.1405` | 0.31 GB | listed | `2750` |
| `falcon_m1b/dandi-001209-draft-snapshot-2026-09-16` | 0.23 GB | listed | `2750` |
| `foley_reversal_learning/zenodo-5010248` | 0.59 MB | listed | `2550` |
| `hennig_neural_engagement/github-ab2f1ec3f425` | 11.94 MB | listed | `2750` |
| `ibc_neurovault/neurovault-collection-6618-snapshot-2026-08-21` | 29.20 GB | listed | `2550` |
| `kathman_working_memory_2026/zenodo-20053990` | 1.27 GB | listed | `2550` |
| `kc_odor_imaging/zenodo-8166598-v1.0` | 15.57 MB | listed | `2550` |
| `link_long_term_intracortical/dandi-001201-0.251023.2336` | 12.56 GB | listed | `2750` |
| `long_term_unsupervised_recalibration/dryad-1jwstqk6g-v4` | 1.69 GB | listed | `2770` |
| `matheson_wind_circuit_2022/zenodo-6863832-v1` | 23.85 GB | listed | `2550` |
| `mindful_ibci_stability/dryad-n2z34tn5s-v6` | 0.41 GB | listed | `2770` |
| `narps_results/zenodo-3634120-v2.0.1` | 52.10 MB | listed | `2550` |
| `neuromaps/neuromaps-data-v0.0.7-public` | 0.32 GB | listed | `2550` |
| `neurosynth/neurosynth-data-v0.7` | 51.46 MB | listed | `2550` |
| `neuroxiv_portal/portal-snapshot-2026-08-21` | 20.77 GB | listed | `2550` |
| `odoherty_nhp_reaching/zenodo-3854034-plus-broadband-2026-09-16` | 81.33 GB | listed | `2770` |
| `odor_experience_plasticity/zenodo-5781484` | 0.21 GB | listed | `2550` |
| `rajagopalan_operant_matching_2023/zenodo-7449214-v1.0.0` | 6.49 GB | listed | `2550` |
| `siliciano_edge_vector_2026/zenodo-20751812` | 0.76 GB | listed | `2550` |

Exact byte and file counts are in the JSON manifest. Unassigned releases are
available inventory, not implicit episode inputs.

## Legacy steward tree and planned destination

`legacy_steward_acquisition` is mode `2700`, approximately 1.107 TB logical,
and currently lives inside the old worktree. Every listed subtree has the same
planned relative name under `planned_steward_acquisition`; none has moved.

| Logical ID | Episode | Relative subtree | Logical size | Mode |
| --- | --- | --- | ---: | ---: |
| `asset_ep03_neuroeffect_spatial_sources` | EP03 | `neuroeffect_spatial_sources_20260917T200750Z` | 44.29 GB | `2700` |
| `asset_ep03_neuroeffect_promotion_control` | EP03 | `neuroeffect_spatial_promotion_control` | 8.9 KB | `2700` |
| `asset_ep04_laion_mixed_role` | EP04 | `laion_fmri_mixed_role_20260922` | 177.15 GB | `2500` |
| `asset_ep04_laion_raw_stimuli_dua` | EP04 | `laion_fmri_raw_stimuli_dua_20260922` | 3.42 GB | `0700` |
| `asset_ep09_ep10_ep11_seu_a1876` | EP09–EP11 | `seu_a1876_zenodo_13944322` | 3.41 GB | `2500` |
| `asset_ep10_seu_optional_bouton` | EP10 | `seu_a1876_optional_bouton_zenodo_13944322` | 1.42 GB | `2550` |
| `asset_ep09_ep10_ep11_allen_ccfv3` | EP09–EP11 | `allen_ccfv3_25um_2017_20260922` | 37.69 MB | `2700` |
| `asset_ep09_ccf_me` | EP09 | `ccf_me_zenodo_13801372_v2` | 0.74 GB | `2550` |
| `asset_ep17_cneuromod_images` | EP17 | `cneuromod_images_fmri_pinned_f8f5162_20260922` | 0.92 GB | `2550` |
| `asset_ep18_things_images` | EP18 | `things_images_osf_jum2f_snapshot_20260922` | 5.02 GB | `2550` |
| `asset_ep18_things_metadata` | EP18 | `things_metadata_osf_jum2f_20260922` | 53.31 MB | `2700` |
| `asset_ep18_thingsplus_cc0` | EP18 | `thingsplus_cc0_osf_jum2f_20260922` | 1.18 GB | `2550` |
| `asset_ep18_thingsplus_control` | EP18 | `thingsplus_cc0_control_20260922` | 9.9 KB | `2770` |
| `asset_ep19_public_sources` | EP19 | `ep19_public_sources` | 23.93 GB | `2500` |
| `asset_ep19_safe_commit` | EP19 | `ep19_safe1_commit_36fe555_20260922` | 0.30 MB | `2700` |
| `asset_ep19_ajile12` | EP19 | `ajile12_dandi_000055_0.220127.0436` | 845.87 GB | `2700` |
| `asset_ep20_neurocam_documentary` | EP20 | `neurocam_documentary_20260922` | 3.00 MB | `2700` |
| `asset_steward_logs` | support | `logs` | 0.10 MB | `2700` |

AJILE12 dominates this tree. Its existing `ACQUISITION_COMPLETE_UTC`,
`VERIFIED_COUNT_BYTES`, and `ASSET_MANIFEST.json` records cover 55 assets and
845,869,698,341 payload bytes. That establishes acquisition inventory
completion; it does **not** establish the still-incomplete EP19 role-filtered
handoff, episode-level readiness, or launch authorization.

## Legacy-contained and adjacent locations

These locations are children of a larger root already listed above, or belong
to an adjacent legacy project. They are explicit here because they carry
different provenance, access, or cleanup semantics:

| Logical ID | Path | Observed size | Treatment |
| --- | --- | ---: | --- |
| `legacy_historical_prior_records` | `/oak/stanford/groups/russpold/users/zijiao/autoresearch/_examples/historical_prior_records` | 3,551,956,575 bytes | Immutable legacy runtime archive with unique artifacts; not an ordinary duplicate; `protected_hold` |
| `legacy_ep02_materialized_runtime` | `/oak/stanford/groups/russpold/users/zijiao/autoresearch/episode02_dopamine_learning_rate_causality` | 164,792,628 bytes | Current Dudman EP02 legacy materialization/runtime; destination unassigned |
| `legacy_ep02_evaluator_firewall` | `/oak/stanford/groups/russpold/users/zijiao/autoresearch/.evaluator_vault/episode02_dopamine_learning_rate_causality` | unknown | Mode `0100`; unreadable and unenumerated; `protected_hold` |
| `legacy_ep05_public_shared_hardlink_alias` | `/oak/stanford/groups/russpold/users/zijiao/autoresearch/episode05_sensorimotor_lfp/inputs/dryad_xd2547dkt_files_v5` | 9,571,575,300 bytes | 25 scientific files hard-linked to the public release; same allocation, not another OAK payload copy |
| `legacy_brain_researcher_data` | `/oak/stanford/groups/russpold/users/zijiao/brain_researcher/data` | 46,891,683 bytes | Adjacent legacy project data; no current episode assignment or destination |

The legacy worktree's 1,259,022,510,824-byte figure is an **accessible lower
bound**, not a complete total: it includes several children listed separately
and excludes contents that cannot be enumerated through the execute-only
evaluator firewall.

The canonical and legacy historical-prior paths are not interchangeable. The
canonical path contains 71 tracked files. All 71 have the same relative path
and identical content in the legacy archive, but the legacy path contains 714
regular files and three symlinks in total: 643 regular runtime artifacts are
absent from the canonical subset. The canonical EP01 lineage manifest resolves
the canonical path, not the live legacy path. Consequently, the legacy archive
cannot enter ordinary duplicate cleanup. Retiring it would require an explicit
governance decision and a content-addressed cold archive for its unique
artifacts.

## Restricted and recovery material

EP17 restricted raw data are a separate migration unit from the steward tree:

```text
current: /oak/stanford/groups/russpold/users/zijiao/autoresearch/episode17_cneuromod_model_ranking_stability/inputs/cneuromod-things-1.0.1-restricted-raw
planned: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw
```

The current directory is mode `2550` and approximately 138.24 GB logical. A
later migration must preserve or tighten its access boundary. It must not be
folded into the public shared root or exposed to a candidate worker.

The EP02 recovery quarantine remains at:

```text
/oak/stanford/groups/russpold/users/zijiao/.autoresearch_recovery/20260922_ep02_boundary_semantics
```

Its accessible content is at least 163,987,918 bytes. That is a lower bound,
because its `evaluator_vault` child is execute-only mode `0100` and was not
inspected. The entire recovery tree is `protected_hold`, not a canonical
source and not a cleanup candidate.

The separate restricted FALCON H1 snapshot remains at:

```text
/oak/stanford/groups/russpold/users/zijiao/restricted_data/falcon_h1/dandi-000954-draft-snapshot-2026-09-15
```

It is about 102 MB logical, mode `2700`, and has no current episode assignment.

## Scratch and cleanup candidates

| Logical ID | Path | Logical size | Treatment |
| --- | --- | ---: | --- |
| `scratch_ep04` | `/scratch/users/zijiao/autoresearch/episode04_shared_scene_geometry` | 179.41 GB | Legacy runtime copy; **not** a planned final destination |
| `scratch_ep05` | `/scratch/users/zijiao/autoresearch/episode05_sensorimotor_lfp` | 10.36 GB | Includes an independent 9.60 GB source copy |
| `scratch_things_eeg1_ingest` | `/scratch/users/zijiao/autoresearch_ingest/things_eeg1` | 59.76 GB | Ingest staging duplicate; temporary DataLad tree has 302 broken annex symlinks |
| `scratch_narps_ep02_historical` | `/scratch/users/zijiao/episode02_narps_analysis` | 3.85 GB | Frozen-prior historical checkpoints/logs |

No scratch deletion is approved. Scratch and OAK are on different devices, so
promotion or cleanup cannot be implemented as an atomic rename. A future
cleanup must first prove that the durable copy is complete, byte-verified,
role-correct, and independently readable.

The canonical launcher and current episode documentation now use
`$SCRATCH/br_autoresearch/<episode-name>/`. Existing
`$SCRATCH/autoresearch/` content is legacy runtime, including the large EP04
and EP05 trees above, a small EP02 tree, and an empty EP14 directory. EP08 now
uses its current `episode08_lfp_electrode_trial_policy` name.

## External shared pools

These data are managed outside this campaign and remain referenced in place.
The Phase 1 catalog now lists both NARPS pools as broad collections; they do
not count toward the 28 curated-release total.

| Logical ID | Path | Observed inventory |
| --- | --- | --- |
| `external_openneuro_fitlins_analyses` | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses` | 55 `ds*` directories plus two auxiliary directories |
| `external_openneuro_fitlins_fmriprep` | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep` | 9 dataset directories |
| `external_openneuro_ds001734_raw` | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/input/ds001734` | EP01 raw source; approximately 221.44 GB logical |
| `external_openneuro_ds001734_fmriprep` | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep/ds001734/derivatives` | EP01 derivatives |
| `external_openneuro_ds001734_fitlins` | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds001734` | EP01 development products |
| `external_openneuro_ds000005_fitlins` | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds000005` | EP01 audit checkout, including `sourcedata/` |
| `external_hcp_ya_bids` | `/oak/stanford/groups/russpold/data/HCP_YA/HCP-YA-BIDS` | 1,098 first-level directories |
| `external_hcp_ya_connectivity` | `/oak/stanford/groups/russpold/data/HCP_YA/HCP1200_PTN` | HCP connectivity pool |
| `external_narps_shared` | `/oak/stanford/groups/russpold/data/NARPS` | 4,646,421 bytes; small shared NARPS support pool |
| `external_narps_pupillometry` | `/oak/stanford/groups/russpold/data/NARPS_pupillometry` | 9,382,338,780 bytes; not assigned to a current episode |

The OpenNeuro, HCP, and small NARPS roots were mode `2770`; the NARPS
pupillometry root was mode `0755`. They are not immutable merely because they
are shared. Episode-level manifests must still bind exact files, hashes,
versions, and exposure roles.

## Personal duplicates and precursors

The following remain in place pending equivalence checks and explicit cleanup
authority:

| Logical ID | Current path | Candidate authoritative counterpart |
| --- | --- | --- |
| `personal_link_001201_full` | `/oak/stanford/groups/russpold/users/zijiao/data/bci_poc/link_001201_full` | public shared `link_long_term_intracortical/dandi-001201-0.251023.2336` |
| `personal_link_001201_sentinels` | `/oak/stanford/groups/russpold/users/zijiao/data/bci_poc/link_001201_sentinels` | same shared release |
| `personal_epic_ptir_coating` | `/oak/stanford/groups/russpold/users/zijiao/data/bci_poc/epic_ptir_coating` | public shared `epic_ptir_coating/mendeley-7p3cxn7jtn-v1` |
| `personal_braingate_20y_t3_empty` | `/oak/stanford/groups/russpold/users/zijiao/data/bci_poc/braingate_20y_t3` | observed empty; EP16 public shared source exists |
| `personal_dandi_001201_empty_scaffold` | `/oak/stanford/groups/russpold/users/zijiao/data/dandi/001201` | empty version-directory scaffold; LINK public shared release exists |
| `personal_ingest_controls` | `/oak/stanford/groups/russpold/users/zijiao/.codex_ingest_staging` | small EPIC/LFP verification metadata only |

The legacy OpenBHB prepared input and EP20 example bundles also remain in the
old worktree with no approved destination:

```text
/oak/stanford/groups/russpold/users/zijiao/autoresearch/episode14_openbhb_roi_site_generalization/inputs/openbhb
/oak/stanford/groups/russpold/users/zijiao/autoresearch/episode20_neurocam_prior_guided_codesign/inputs/reference_bundles
```

## Same-filesystem preflight

The legacy steward tree, EP17 restricted source, public shared root, canonical
workspace, and newly created private namespace were all on OAK device
`3626059016`. Sherlock scratch was on device `3863872424`. The final
`steward_acquisition/` and restricted CNeuroMod payload leaves were still
absent; their nearest existing ancestors were re-statted under that namespace.

Matching source and ancestor device IDs suggest that an atomic rename may be
technically possible for a later OAK-to-OAK migration. It is not permission to
perform one, and the conclusion must be rechecked against the actual final
destination immediately before any operation.

Before any future move:

1. Re-stat the source, destination, and destination parent.
2. Check ACLs, modes, ownership, quotas, free space, jobs, writers, and open
   handles.
3. Freeze a complete path/type/size/ownership/mode/mtime/inode/link-count and
   symlink-target inventory; hash control files and existing checksum
   manifests.
4. For a same-filesystem rename, require the directory inode and Lustre FID to
   remain unchanged. Full payload rehashing is a separate scientific-integrity
   or duplicate-deletion gate, not transport proof for the rename itself.
5. Deliberately preserve symlink, hardlink, sparse-file, and permission
   semantics.
6. Use a reversible staging name and write a rollback procedure.
7. Independently verify the destination.
8. Obtain explicit move authority, followed by separate deletion authority if
   cleanup is desired.

## Open inventory exceptions

EP16's observed mode-`2770` public BrainGate source still needs a write seal
before readiness. Phase 1 corrected the contract wording but did not change
source permissions.

The earlier 14-versus-28 release-catalog gap, 54-versus-55 OpenNeuro count,
scratch-root mismatch, stale EP08 scratch name, and EP19/EP20 legacy-relative
path ambiguity were resolved in Phase 1. The JSON manifest keeps those as
historical findings with explicit resolution status rather than presenting
them as current defects.

## Migration state

Phase 1 ends at documentation:

- public shared releases and external pools stay in place;
- legacy steward and EP17 restricted roots are `staged_not_moved`;
- the empty private destination namespace exists, while both final planned
  payload leaves remain absent;
- recovery and separately restricted data are on `protected_hold`;
- scratch and personal duplicates remain untouched pending verification;
- no episode input has been provisioned merely by adding this manifest; and
- no canonical MCP state, evidence role, launch gate, or scientific claim has
  changed.
