# EP04 source and implementation handoff

## Completed development evidence — October 6, 2026

The source/readiness work described below progressed to a complete fixed-panel
analysis and the valid terminal result **`closed_no_candidate`**. The
[terminal record](experiment_log.md#2026-10-06-0121-utc--candidate_decision--closed_no_candidate-reporting-complete)
and [saved decision](runs/panel_decision/registered_panel_decision.json) are the
current disposition. The October 3 pilot paragraphs below are historical,
not a statement that the later selected handoffs are still unprovisioned.

The frozen metadata cover 23,560 unique development images, 94,216
presentations and 14,285 grouping units. All four release-supplied embeddings
have complete development coverage. Selected beta/ROI exports and training-only
preprocessing supplied all five participants and 25 folds. The approved RGB
control completed finite 3,150-coordinate coverage for every development image;
the [coverage closeout](experiment_log.md#2026-10-05-2331-utc--full-low-level-coverage-complete-observed-panel-running)
preserves the exact source/settings and execution evidence. No shared/OOD
image or response was admitted into that branch.

The final decision admitted all 400 native/low-level comparisons, 400 encoding
rows, matched training/reliability/EVC context, all mandatory sensitivities and
the complete 25-cell null panel (10,000 permutations and 1,000 geometries per
VLM). Mean native VLM-minus-DINOv2 gains were CLIP −0.0347765675, PEcore
+0.0010264769 and SigLIP2 −0.0392206391 Fisher z, with positive participant
effects in 0/5, 3/5 and 0/5. All three show unsubtracted alignment above both
familywise null thresholds, but none passes the +0.02 comparative margin and
required participant/influence/low-level/near-neighbour gates. Alignment is
therefore distinct from an eligible advantage. The [complete result figure](figures/registered_panel_decision/20261005-full01/registered_panel_decision.png)
retains the full fixed-panel comparison.

No selected-row rerun, lock or OOD audit follows. This is selection-exposed
development in five participants, not equivalence, population confirmation,
universal DINOv2 superiority or a causal language-pretraining conclusion.
Required OOD isolation/version evidence remains unresolved without being a
reason to reopen a closed candidate search. All jobs and reservations are
closed; the log records 24.8327777778 explicit allocated CPU-hours and retains
the larger conservative historical exposure separately. The optional ASTRA
semantic-validation parser issue and unvalidated documentation correction are
also disclosed there; neither changes the admitted scientific result.

## Historical October 3 source-readiness snapshot

This October3,2026 record implements `source_readiness`. The adopted
metadata-only join and fold preparation completed in
`runs/source_readiness/20261003-metadata_folds04`. The
[development embedding handoff](runs/source_readiness/20261003-development_embeddings01/embedding_handoff.json)
then completed in job46433944.3 at22:19UTC: all23,560 development rows for all
four fixed models, original float16 values and common order, zero errors.
Development beta and fixed ROI values have since opened only for the first-session
pilot below. No raw-image values, historical comparison or conditional OOD
result was opened. Full source qualification and the remaining neural numeric
handoffs remain pending. Selected commands and actuals are in the execution log.
The coordinator adopted the [neural preprocessing recipe](neural_preprocessing_proposal.md)
before beta/ROI values: fixed-mask training-only standardization, finite/repeat
support, training-repeat reliability/provider-formula NC and fixed ridge.
The matched training-image context table is mandatory and distinct from
held-fold raw high-level/EVC RSA. The fixed sub-01/ses-01 selected-row NIfTI/ROI
pilot completed in26.02s, exporting575 development trials and12,833 union
voxels, with no excluded numerical trial decoding. Original-value beta/mask
access is now established only for that pilot. No neural standardization,
reliability/NC, scoring or raw-image read has executed. The coordinator selected
the same source-only route for the remaining149 sessions in five bounded
participant groups, reusing the completed pilot. Root submitted array46436056
once at22:34UTC after eight focused tests and ASTRA validation passed; root
owns monitoring, and full completion remains pending.

## Resolved from current provider documentation

The provider documents these default arrays:

| Episode family | Released file label | Documented geometry |
| --- | --- | --- |
| OpenCLIP | CLIP | LAION ViT-H/14,1024 features, released L2 normalized |
| DINOv2 | DINOv2 | ViT-L/14, mean patch tokens from layer23,1024 features, not normalized |
| PEcore | PEcore | L/14 at336px,1024 features, released L2 normalized |
| SigLIP2 | SigLIP2 | SO400M Patch14 at384px,1152 features, released L2 normalized |

Files are `stimuli/task-images_desc-<label>_embeddings.h5` with `embedding`,
`image_ids` and `valid` datasets. The completed
[structural pass](runs/source_readiness/20261003-array_metadata02/array_metadata.json)
directly binds25,052-row float16 arrays and local model_id attributes:
`hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `vit_large_patch14_dinov2`,
`PE-Core-L14-336`, and `google/siglip2-so400m-patch14-384` respectively.
The normalize attributes match the declarations above. The completed
`code/prepare_development_embeddings.py` handoff joined all four complete ID
arrays before decoding only the23,560 matched development rows in512-row
chunks. Full valid/finite/nonzero development coverage passed without fold
shrinkage; original float16 values and common row order are preserved. Fixed
downstream numerical L2 normalization remains unperformed. The handoff made
no RDM or neural comparison. HDF5 storage chunks can include excluded rows internally;
only selected development value elements are returned/exported. This is a
trusted reader boundary, not an OS permission or physical-byte seal.
The provider corrected all four embedding files on September28 for transparency
handling in116 OOD images. The steward relocation preceded this correction,
so local version identity must be bound explicitly. The provider states that
all other24,936 rows are unchanged: preserved originals may supply regular-
unique development without replacing the whole mixed-role source. Subject-
unique object masks are explicitly absent; their optional anchor cannot run.
[Provider stimulus derivatives](https://laion-fmri.hebartlab.com/stimulus_derivatives.html)

Betas, metadata and stimulus derivatives are publicly CC0. The raw-image DUA
applies to packed source images, which are unnecessary for the native embedding
panel. No DUA is accepted or raw payload acquired by this record. The existing
September22 raw-stimulus acquisition receipt records acceptance of official
DUA version2026-05-12; duplicate acceptance is unnecessary. New raw-derived
low-level settings remain subject to the explicit extension clause.
[Provider access terms](https://laion-fmri.hebartlab.com/data_access.html)

The beta-to-trial TSV uses `session`, `run`, zero-based run-local `beta_index`
and `label`; `label` joins stimulus identity. The published prose mistakenly
calls the index session-global: the root's bounded source metadata diagnosis
shows it restarts per run. Preserve that source index, and separately retain
the original zero-based TSV row position as `session_beta_axis_index` before
role filtering. The documented row-to-volume/column alignment and pinned
loader's direct trial-mask indexing support this array coordinate; actual
array shape/finite-value qualification remains later. TYPED is the supplied beta version, in native
T1w space. The provider's whole-source noise ceilings use response repeats and
therefore cannot silently become training-fold voxel-selection criteria.
[Provider beta documentation](https://laion-fmri.hebartlab.com/glmsingle_betas.html)

The source-only loader at the same pinned provider commit uses common array
indices: `subject.py::_load_roi_volume_mask` indexes a flattened ROI by the
flattened brain mask, and [`io.py`](https://github.com/ViCCo-Group/LAION-fMRI/blob/cf82765ced029de81c3bec7254b8cbd6a65e3686/laion_fmri/io.py#L9)
uses the same C-order flattening for masks and each beta volume. This path
does not resample by affine. It supports intended index alignment of the T1w
exports. The completed no-threshold corner diagnostic found a maximum
2.8875447071641247e-05 beta-voxel coordinate discrepancy. Root then adopted
[same-index mapping](config/roi_mapping.json) for this fixed panel, with the
discrepancy retained and no interpolation/resampling. No provider loader was
executed; those structural mapping checks opened no numeric arrays.

Provider subject pools include1121 regular-shared images and4712 unique images;
using those pools directly would import excluded exposed responses. The
provider's random and cluster splits are distinct choices. Existing duplicate
grouping is useful structural evidence but does not by itself bind the current
near-neighbour definition or source-balanced unique-image folds.
[Provider splits](https://laion-fmri.hebartlab.com/train_test_splits.html)

## Concrete handoff before any neural scoring

An authorized steward must provide the following source-selected material;
the candidate evaluator must not receive a mixed source and filter it after
opening outcomes.

| Needed item | Exact use and unresolved decision |
| --- | --- |
| Release/version and feature sidecars | Original-file dimensions, float16 storage and model_id/normalize declarations are bound by array_metadata02. Development_embeddings01 completed full23,560-ID valid/finite/nonzero coverage and the four selected-row handoffs. Corrected OOD rows are a separate later custodian dependency. Keep storage precision distinct from float64 computation. |
| Subject-unique trial export | Include participant/session/run/beta index/image ID/source stratum/repeat validity. Exclude regular-shared and OOD response bytes physically before candidate access. Preserve image identities without filename heuristics. |
| ROI and beta representation | All150 beta axes/TYPED sidecars,15 high-level masks and30 V1–V3 components are structurally bound. Retinotopic EVC is the six-component union, not laion-EVC. Root adopted the documented same-index mapping after active-sform measurement. Freeze finite-voxel and reliability treatment using training images; whole-source ceilings are descriptive. |
| Session preprocessing | Freeze train-image-only within-session location/scale estimation and how sessions lacking usable training observations are handled. A global-session z-score using held-out images would violate the current contract. |
| Low-level geometry | Identify an existing lawful release-supplied or previously registered covariate with complete unique-image coverage. No such feature is established by the four embedding files. Do not silently substitute DINOv2, captions or object detections. Raw-image extraction requires the separate approved source/settings route. |
| Folds and neighbour groups | Supply source-balanced image folds with all repetitions and frozen neighbour groups together. Document fold count, grouping source and threshold, and singleton/missing-group handling before responses are opened. |
| Structural missingness | Determine modal planned repeats from metadata and register unavailable ROI/image/session handling before scoring. Any missing required panel or sensitivity row blocks lock. |
| Finite-score and null conventions | Bind the empirical95th-percentile convention and undefined geometry treatment before scoring. Current component code treats constant RDMs/perfect Fisher correlations as not evaluable; it does not invent an epsilon-clipped scientific score. |
| Identical-fold encoding | Select one permitted predictive diagnostic and its training-only fitting procedure. It never selects/rescues the primary native geometry. |
| OOD exposure and permission separation | Provide an outcome-blind exposure review covering response bytes and candidate summaries; establish an actual separate custodian/evaluator path for eligible OOD data. Ownership by the same Unix user or a different subagent is not sufficient evidence of separation. |

No new independent participant/dataset exists. If OOD sealing cannot be
established, the current contract permits a development-only terminal after a
valid panel; it does not permit reuse of regular-shared responses as audit.
The neural role handoff remains unprovisioned. The metadata above is a
concrete input contract, not a signature/hash/receipt requirement.

## Adopted metadata-only preparation

Completed in job46430982.8 on October3 at21:31:11UTC:150 files,23,560
development images,94,216 presentations and14,285 grouping units. Every
participant/source cell is balanced across five folds within one image.
The [result](runs/source_readiness/20261003-metadata_folds04/source_metadata_report.json)
retains intended four repeats separately from the observed23,536 four-repeat
and24 three-repeat images; neural validity remains unknown.

On October3 before neural scoring, the coordinator adopted
[`config/development_metadata.json`](config/development_metadata.json).
The adapter reads only the original stimulus CSV, acquisition file-list receipt
and exact150 trial TSVs for sub01/03/05/06/07 sessions01–30. Metadata role fields
define the joins; filename suffixes do not infer participant identity. Derived
row exports contain unique-development images/trials only; other roles produce
counts. Trial presence does not authenticate finite beta values or valid repeats.
The source CSV leaves all unique-image `n_reps` fields blank. These blanks stay
explicit in row exports. The provider's separately documented four-repeat
design supplies intended counts; actual trial-table occurrences and unknown
valid-neural counts remain distinct. The ordinary minimum-two primary and
complete-planned-repeat sensitivity are not conflated.

The same bounded compute step resolves one official GitHub commit and preserves
the four small group/correction JSONs and README at that immutable revision.
It verifies the provider's seed42 random-assignment corrections and known
duplicates, then uses published pooled similarity components rather than the
provider random or feature-cluster folds. These components include known
duplicates and documented CLIPA/DreamSim/DINOv2/SSCD distance cutoffs; they are
not exact-duplicate claims. Removing shared members preserves the remaining
component identity. No feature is read or extracted.
[Provider split method](https://github.com/ViCCo-Group/LAION-fMRI/blob/main/scripts/splits/README.md).

Five folds use one deterministic global component assignment balancing the
15 participant/source cells. Components are sorted by descending maximum cell
fraction, descending retained size, then lowest retained image ID. Each is
assigned to minimize the increase in summed squared deviations of fold cell
proportions from1/5; ties choose the smaller whole fold then lower fold index.
There is no restart, refinement, component splitting or outcome-based change.
All repeats inherit their image's fold. At least three images in each
fold/source cell are needed to construct a nontrivial upper-triangle rank
association; failure is reported without retuning. Exact balance is emitted.

Empirical95th percentiles use NumPy `method="higher"`, observed scores strictly
exceeding the threshold. All frozen null counts/seeds remain unchanged.

Before the first primary score, the 2026-10-05 coordinator binds random-geometry
`X` to each participant/frozen fold's exact eligible primary evaluation-image
rows of the raw released embedding, before analysis L2 normalization. The null
therefore preserves the mean and centered covariance of the feature matrix
whose geometry is scored in that cell. Use the existing model/draw seed formula
unchanged in every cell; no full-participant randomization followed by slicing.
Caching that fixed mean/SVD and whole-image rank permutations changes only
execution cost, not the registered generator or image-level null. All 25
participant/fold contributions are required before any family-wise threshold;
shards never select their own winner. Signed finite null correlations are
retained, including negative values; "positive alignment" does not truncate
the null distribution. The already fixed percentile and strict exceedance
convention above remains unchanged.

Undefined correlations remain not evaluable. These prospective choices do not
select a new model, alter the0.02 margin or authorize scores. Session
preprocessing, ROI lineage, the required low-level control, identical-fold
encoding and actual audit permissions still need corresponding work. Metadata
folds are not a completed source handoff.

## Implemented and executable now

`code/native_rsa.py` implements fixed native cosine/correlation RDMs,
average-tie Spearman and partial-rank regression, finite Fisher-z,
whole-image stratum-preserving permutations, the registered centered-SVD/Haar
random geometry, valid-repeat averaging, five-participant/three-sector
aggregation, unconditional three-row null argmax and screen-before-selection.
When feature dimension reaches the number of images, centered geometry has
rank at most n-1; the generator uses that mathematical rank bound rather than
trying to draw n orthogonal vectors in the n-1-dimensional centered subspace.
This preserves pre-L2 mean/covariance up to floating-point roundoff without a
data-driven rank search. The supplied matrix is the released embedding,
including any provider normalization; unavailable pre-provider features are
not reconstructed.

`code/qualify_native_rsa.py` runs ten meaningful mathematical checks, accepts
no empirical input path, and requires Slurm with single numeric threads. It
checks closed-form identities, covariance preservation, image-level permutation
structure, equal biological weighting, unconditional signed null statistics,
and a counterexample in which the top-scoring row fails a mandatory control.
Tests are authored with the implementation; no independent evaluator claim is
made. This qualifies only components, not production preprocessing, full
10000/1000-draw loops, encoding, sensitivity orchestration or one-shot audit.
Those remain necessary work after the source handoff and pre-score bindings.

The frozen total remains2500 CPU-core-hours,120 hours from first panel
evaluation,48 concurrent CPU cores,256GB per evaluation,1500GB scratch and
zero GPUs. The current one-CPU qualification consumes seconds to minutes,
does not count as a primary row and does not start the panel evaluation clock.
