# EP22 data contract

## Source

- **Dataset:** LAION-fMRI, a 7T image-viewing dataset with five participants
  and 30 main `task-images` sessions each. Provider documentation:
  - [overview](https://laion-fmri.hebartlab.com/index.html)
  - [stimulus set](https://laion-fmri.hebartlab.com/stimulus_data.html)
  - [stimulus selection](https://laion-fmri.hebartlab.com/stimulus_selection.html)
  - [stimulus derivatives](https://laion-fmri.hebartlab.com/stimulus_derivatives.html)
- **Licensing and risk:**
  - fMRI data, metadata and derivatives are CC0.
  - Packed raw images fall under the provider DUA (Terms v2026-05-12), already
    accepted for the 2026-09-22 acquisition.
  - The scientist-supplied provider confirmation of Low Risk is recorded in
    [EP21's DATASETS](../episode21_wang_neural_code_conversion_reproduction/DATASETS.md).
  - EP22 needs no raw images.
- **Physical location:** resolve paths only through the root
  [DATA_LOCATION_MANIFEST.json](../DATA_LOCATION_MANIFEST.json), logical ID
  `asset_ep04_laion_mixed_role`. This is the mixed-role steward acquisition
  of 2026-09-22.
  - EP04 and EP21 use the same upstream bytes, so this is not independent
    evidence.
  - The location record does not authorize reading OOD response rows.

## Data roles

| Slice | EP22 role | Exposure |
| --- | --- | --- |
| 4,712 subject-unique regular images × 5 participants (23,560) | Development: D0–D3 | Exposed in EP04 as selection-exposed development; never used for confirmation |
| 1,121 regular shared images (LAION 641, THINGSplus 240, NSD 240) | Excluded | Responses opened in EP04's earlier round; never used for an EP22 score |
| 371 shared OOD images | One-shot confirmation (C1–C3) | Believed unopened; usable only after the review below |

Units:

- The five participants (sub-01, sub-03, sub-05, sub-06, sub-07) are the
  biological units.
- Images, RDM edges, voxels, folds and repetitions are not independent
  observations.
- Development image pools differ between participants, while all five share
  the OOD images.

## Selection facts relevant to the accounts

- **LAION-natural images:** per the provider, a large sample was embedded with
  CLIP, reduced with PCA and clustered, and candidates were chosen to maximize
  the effective dimensionality of the selected set. Visually similar
  alternatives around each prototype came from approximate nearest-neighbour
  search.
  - Shared images fill the selected feature-space slots.
  - Each participant's unique images come one per prototype-centred pool.
- **NSD and THINGSplus shared images:** these were assigned to nearby LAION
  feature-space slots.
- **Not documented:** the CLIP variant, the cluster count, and how the
  subject-unique THINGS images were chosen.
- **OOD images:** the documented types are classical illusions, Gabor patches,
  simple shapes, cropped object cutouts with transparent backgrounds, unusual
  spatial configurations, gaudy chromatic patterns, and self-made images. The
  provider metadata use nine type labels:
  - `cropped`
  - `gabor`
  - `gaudy`
  - `illusion-classic`
  - `illusion-natural`
  - `relations`
  - `selfmade`
  - `shape`
  - `unusual`

  The documentation gives no CLIP-space sampling for OOD types. Any such
  indication found before the lock moves that category out of the
  confirmatory subset.
- **Display:** RGBA PNGs (131 in total, all OOD) were alpha-composited over
  the PsychoPy mid-grey background.

## Reused frozen EP04 definitions (read-only)

EP04 is terminal. Reusing its frozen development definitions keeps EP22's
diagnostics directly comparable with the result they explain. EP22 never
writes to EP04, and it reads these items as fixed definitions rather than live
outputs:

- `../episode04_shared_scene_geometry/outputs/runs/source_readiness/20261003-metadata_folds04/`:
  development image, trial and group tables, source strata, and the frozen
  image-disjoint, near-neighbour-grouped five-fold assignment.
- `../episode04_shared_scene_geometry/outputs/runs/source_readiness/20261003-development_embeddings01/`:
  development rows of the four released embeddings, in a common order.
- `../episode04_shared_scene_geometry/outputs/neural_preprocessing_proposal.md`
  and `../episode04_shared_scene_geometry/outputs/config/roi_mapping.json`:
  - training-image-only session standardization;
  - the rule of at least two valid presentations;
  - finite support;
  - same-index ROI mapping.

EP04's transient beta caches under
`$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/` may be read only
if they are still present and match these frozen exports. Otherwise EP22
re-extracts development betas from the canonical source into its own scratch
space.

## Model embeddings

All four embeddings come from the provider release:

| Row | Released model | Released representation |
| --- | --- | --- |
| CLIP | OpenCLIP LAION ViT-H/14 | Final 1,024-d projected embedding, L2-normalized |
| PEcore | PE-Core L/14, 336 px | Final 1,024-d embedding, L2-normalized |
| SigLIP2 | SigLIP2 SO400M/14, 384 px | Final 1,152-d embedding, L2-normalized |
| DINOv2 | DINOv2 ViT-L/14 | 1,024-d mean of layer-23 patch tokens, unnormalized |

- EP22 L2-normalizes every row before computing geometry, as EP04 did.
- The DINOv2 row is an intermediate patch-token summary, while the VLM rows
  are projection-head outputs. This stage difference is part of what D1–D2
  attribute to "spectrum" or "eigenvectors".
- New embeddings, layers and models are out of scope.

On 2026-09-28 the provider corrected all four embedding files for
transparency handling in 116 OOD stimuli, and states that the other 24,936
rows are unchanged.

- The development rows come from the 2026-09-22 acquisition and are
  unaffected.
- Confirmation must bind the corrected OOD rows. Acquire the corrected public
  files, record their release identity, and check that the 255 unchanged OOD
  rows match the local copy.
- Using uncorrected OOD rows is a technical failure.

## Sealed confirmation requirements

All four steps below must be complete before the lock.

1. **Outcome-blind exposure review.** Establish that no OOD response values,
   reliabilities, normalization statistics, RDMs or comparison summaries have
   ever been opened.

   Evidence observed on 2026-10-12, gathered without opening any neural value:
   - EP04's terminal decision records `OOD_outcomes_opened: false`.
   - EP04's logs state that conditional OOD outcomes remained unopened.
   - The legacy EP04 validator
     (`/scratch/users/zijiao/autoresearch/episode04_shared_scene_geometry/outputs/code/validate_neural_inputs.py`)
     explicitly excluded the 371 OOD images and read only metadata, split
     JSON, trial TSVs and NIfTI headers.
   - That legacy tree's `outputs/` directory holds only the validator and its
     test.
   - EP21's contract closes every OOD derivative.

   Still to be reviewed:
   - EP04's earlier-round canonical Brain Researcher history (read-only).
   - Any other process that read the mixed-role beta or HDF5 files. EP04's
     role-filtered readers returned only development elements, but chunked
     reads can touch excluded bytes internally. That is a trusted-reader
     boundary, not a byte seal.

   If any summary derived from OOD outcomes turns up, the OOD route closes and
   EP22 ends as `development_only`.
2. **Outcome-blind coverage check.** The following must be complete:
   - corrected embeddings for all four models;
   - the provider OOD type label for each image;
   - valid image IDs;
   - trial rows for all 371 images and all five participants.

   Missing coverage makes the confirmation technically unevaluable; no
   substitute data may be used.
3. **OOD preprocessing defined from development only.** Session location and
   scale come from each session's development trials. EP04's repeat rule and
   ROIs apply unchanged.
4. **Write-once lock.** A lock file in `outputs/` records:
   - the evaluator code, by content checksum;
   - every setting;
   - the final C1–C3 statements;
   - the outcomes of steps 1–3.

**Separation:** procedural, adopted by the scientist on 2026-10-12. The same
account runs everything, with no barrier enforced by the operating system.

- Only the locked evaluator reads OOD response rows.
- It runs once and releases nothing until it finishes.
- An exact software retry is allowed only if no score was produced.

The root README treats this as weaker than a restricted runtime or an
independent human steward, so the claim wording must say "procedurally
sealed".

## Storage

- **Durable:** code, reports, the lock and compact results go under
  `outputs/`.
- **Transient:** beta matrices and caches go under
  `$SCRATCH/br_autoresearch/episode22_geometry_gap_decomposition/`.
- **Inputs:** once acquired, the corrected OOD embedding files are recorded
  under `inputs/` and are read-only from then on.
- **Not inputs:** legacy `$SCRATCH/autoresearch/` trees.
