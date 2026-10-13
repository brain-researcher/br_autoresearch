# EP04 low-level control: approved specification

**Approved on 2026-10-05:** the scientist replied “agree, please go ahead” to
the linked EP04 and EP21 proposals. This activates this exact control, the
fixed 60-image representative and the complete 23,560-image development pass.
At activation no source pixels have yet been read. Existing DUA acceptance is
reused. Root records execution and selects the full bound from measured cost.
The specification leaves the four native model embeddings, primary
RSA geometry, model-selection rule and all gain/null thresholds unchanged.

## Approved source and permitted scope

Use only the official LAION-fMRI v1 raw-image acquisition already completed
September22 at
`$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_raw_stimuli_dua_20260922/`.
Its existing `ACQUISITION_RECEIPT.json` identifies
`task-images_stimuli.h5` (3,422,259,504 bytes) and its row-aligned stimulus CSV
(1,661,052 bytes), both under `payload/` (root verified file metadata only).
Its existing `QUARANTINE.md` records accepted DUA
version2026-05-12 and controlled noncommercial academic research use.
No new acceptance, credentials, signed URLs, download or redistribution is
requested. This is analysis of the LAION-fMRI experiment, not training or
evaluation of a new general-purpose model.

A trusted steward joins the raw pack's CSV to the frozen EP04 development
image IDs and decode **only those23,560 regular participant-unique images**.
An exact CSV identity/role mismatch, duplicate, missing image or unsupported
format stops extraction; there is no alternate source or post-score omission.
The source `images` HDF5 dataset stores one encoded byte array per metadata row,
according to the [official image accessor](https://github.com/ViCCo-Group/LAION-fMRI/blob/main/laion_fmri/stimuli.py).
The builder must resolve each approved image to its row before reading that
one element; it cannot materialize the complete dataset or decode excluded
regular-shared/OOD rows. Only unique-ID-aligned numeric control features and
coverage metadata go to the development evaluator. No pixels or thumbnails
are exported or displayed, and nothing uses sibling feature caches.

The same fixed recipe can later be supplied to an independent OOD custodian,
but this proposal does **not** open OOD images or responses, establish audit
isolation, or authenticate all371-image audit coverage. Audit permission and
complete feature coverage remain separate later requirements.

## Fixed transform and geometry

Control identifier: `ep04_rgb_block_moments_5_10_20_v1`.
The complete pure component is
[`code/low_level_pixels_proposal.py`](code/low_level_pixels_proposal.py).
It accepts one authorized byte string, not a filesystem path, and is not
imported by the active metadata adapter. The separate proposal-stage trusted
reader is [`code/prepare_low_level_proposal.py`](code/prepare_low_level_proposal.py).
It checks complete23560-image metadata coverage before opening the HDF5,
constructs an allowlist of selected unique image IDs and their exact raw CSV
rows, and reads only one selected `images[row]` element at a time. The
representative is exactly four lexicographically lowest IDs in each cell;
full mode covers every frozen development ID. Source-read execution is now
approved for these two modes only; no automatic follow-on is dispatched.

1. Decode the stored1000×1000 RGB JPEG or RGBA PNG without cropping, resizing,
   EXIF rotation, color-profile conversion or augmentation. Reject another
   size/mode/format. Composite alpha onto RGB(128,128,128), with nearest-byte
   rounding. This is the documented display-background convention; regular
   development images are documented as RGB JPEGs. The [provider format and
   display account](https://laion-fmri.hebartlab.com/stimulus_data.html) is the
   source of these choices, not inspected pixels.
2. Represent stored RGB bytes as float64 values divided by255. These are
   display-code intensities, not calibrated linear-light measurements.
3. Partition the square into5×5,10×10 and20×20 equal nonoverlapping grids.
   In every block compute each RGB channel's mean and population SD (`ddof=0`).
   Concatenate row-major means then SDs in each grid. Divide the grid vector
   by `sqrt(3 * 6 * grid_size²)`, then concatenate the three grids. The result
   has3150 fixed features. Every scale receives equal squared-distance weight;
   block/channel/statistic coordinates have fixed units. There is no image-
   sample centering, learned scaling, PCA, detector, L2 normalization or fit.
4. The low-level RDM is Euclidean distance between these vectors, computed in
   float64. Thus squared distance is the equal-scale average of the mean
   squared difference in block RGB means/SDs. Use this one RDM in the existing
   intercept-plus-ranked-low-level partial-Spearman control in each fixed
   participant/fold/ROI. Native VLM and DINOv2 RDMs remain L2/cosine.

This is a transparent control for coarse color, luminance layout and local
contrast at three fixed spatial resolutions. The scales divide the documented
stored image size exactly and do not depend on observed neural fit, candidate
ranking or audit categories. It is **not** an exhaustive model of low-level
vision: orientation, precise spatial-frequency structure, shape and all
higher-order texture differences are not controlled. A surviving partial RSA
must therefore be described as surviving this specified pixel-moment control,
not proving exclusively semantic or language-derived alignment.

## Complete coverage, missingness and resources

The steward first times a deterministic60-image representative: the four
lexicographically lowest frozen development IDs in each of the15 participant/
source cells. This is a cost/format qualification, not a new neural score or
scientific sample. Its internal bound is 300 seconds, with one CPU, 2 GiB and no image display.
Use the already available Pillow/NumPy/h5py modules and record their actual
versions before extraction; no environment install is requested.

After the representative succeeds, root sizes one complete23,560-image pass
from measured decode/transform time. The source is about3.42GB compressed;
the transform processes about70.7 billion channel values across the complete
development set. Expected CPU cost is tens of minutes, but this is unmeasured
and is not a launch request for a padded job. One image is decoded at a time;
temporary float64 image/moment arrays fit comfortably below2GiB. Store all
features in float64 (593,712,000 bytes, about566MiB), plus image identities;
RDM arithmetic also remains float64, with no storage quantization.
Features remain controlled derived research inputs, not a publicly released
image substitute. Use a role-filtered durable destination selected by root,
with transient files in the episode's scratch directory.

No model score is computed until every development ID has one finite feature
row. Missing/invalid coverage blocks scoring and does not shrink the primary
image set after folds. A constant/degenerate nuisance RDM or rank-residual
correlation is `not_evaluable` under the existing rule, never an epsilon repair.

## Reviewable execution recipes

Three already-passed analytic tests used generated1000×1000 arrays only: constant-color norms
and RDM distances, identical-global-statistics images with analytically known
spatial/block-contrast distance, and alpha0/1/128/254/255 background/rounding
boundaries. Their unchanged numerical transform is reused without a new test
program or repeated execution. The original reproduction command is:

```bash
python3.12 -m unittest discover -s outputs/tests -p test_low_level_pixels_proposal.py -v
```

The following representative command is now approved. It consumes completed
metadata-fold attempt04 and has no additional approval-token mechanism.
Root uses one bounded attended allocation (one CPU, 2 GiB, 90 minutes) for
the representative and, only if its measured cost fits, the complete pass.
No artificial padding, automatic retry or silent deadline extension is used.

```bash
python3.12 outputs/code/prepare_low_level_proposal.py \
  --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
  --raw-metadata "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_raw_stimuli_dua_20260922/payload/task-images_metadata.csv" \
  --raw-h5 "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_raw_stimuli_dua_20260922/payload/task-images_stimuli.h5" \
  --mode representative --max-seconds 300 \
  --output-dir outputs/runs/source_readiness/low_level_representative01
```

The full command uses the same exact metadata/source/code, `--mode full`,
a fresh `low_level_full01` output directory, and a root-selected time bound
based on the representative's actual cost. It reads no other source elements.
Each completed output includes unique IDs, float64 features, coverage/counts,
software versions and elapsed time. Partial attempts retain explicitly partial
features and completed-row counts; they are not complete coverage. No pixel
or feature-value summary is printed. All files remain controlled episode
research material; the reader's logical boundary is not an OS-level audit seal.

## Approved extension and unchanged boundaries

[GOAL.md lines152–154](../GOAL.md#bounded-scientific-analysis-grammar) states:
“Raw stimulus access, new detector execution, or new embeddings are outside
this grammar unless their DUA, source/model identity, frozen settings, and
complete coverage are separately approved before fixed-panel scoring.”
[DATASETS.md lines120–124](../DATASETS.md#required-frozen-development-inputs)
also requires the approved extension and revised policy before neural scoring.

The approval covers this existing accepted source, exact analytic control,
unique-development representative and complete extraction. GOAL, DATASETS and
SEARCH_POLICY now record that exception before fixed-panel scores. It opens
no new neural outcomes, replaces no native embeddings, adds no trained model
and does not open the OOD audit. Root records launch/results separately.

The selected module recipe is `module purge; module load devel python/3.12.1
py-h5py/3.10.0_py312 math py-numpy/1.26.3 viz py-pillow/10.2.0_py312`.
Use `python3.12`, one allocated CPU and all six numerical thread limits set to
one. Reports retain actual NumPy/Pillow/h5py versions and separate image-read
and decode/transform seconds. External representative timeout is 310 seconds
with a 10-second termination grace. Root chooses the full internal timeout
and a corresponding internal+10-second external timeout with the same grace
from observed cost and remaining allocation time.
