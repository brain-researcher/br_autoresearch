# Exact-source stimulus adapter and private development-feature pilot

## October 9 runtime repair proposal — no installation or retry selected

The closed 47101165 stimulus-test attempt failed at `import h5py`; none of the
five fixture bodies ran. ASTRA validation passed. Bounded metadata inspection of
`$GROUP_HOME/zijiao/miniforge3/envs/encoding_model` confirms CPython **3.11.15**,
NumPy **1.26.4**, Torch **2.4.1+cu121**, Torchvision **0.19.1+cu121**, Pillow
**12.2.0** and pip **26.1.2**, with no `h5py` distribution metadata. This read did
not import any scientific library. `ml spider py-h5py` lists builds for Python
2.7/3.6/3.9/3.12, including `py-h5py/3.10.0_py312`, but none for the retained
CPython3.11 runtime; adding that Python3.12 binary path would not repair it.

Propose the official **h5py3.10.0 CPython3.11 manylinux2014 x86_64 wheel** as one
additive EP17-only package target, leaving the qualified environment files and
all existing package versions untouched. [Official release metadata](https://pypi.org/project/h5py/3.10.0/)
lists this approximately4.8-MB wheel and Python>=3.8. Its
[tagged dependency declaration](https://raw.githubusercontent.com/h5py/h5py/3.10.0/setup.py)
requires NumPy>=1.17.3 and specifically identifies NumPy>=1.23.2 for Python3.11;
the existing1.26.4 meets that requirement. The Linux wheel
[bundles HDF5 1.14.2](https://docs.h5py.org/en/3.10.0/whatsnew/3.10.html).
These are compatibility facts, not an observed successful import or adapter run.

The following **unexecuted** commands are for a coordinator-selected existing
substantive compute allocation, one CPU and at least2GiB, with no new standalone
job. They cap installation at180seconds plus5seconds and the five checks at
120seconds plus5seconds. The fixed wheel hash comes from its official PyPI
file metadata. `--no-deps` preserves NumPy/Torch/Pillow; binary-only installation
refuses a source build. A fresh target is required; a partial installation is
preserved on failure, without automatic overwrite, retry or alternative runtime.
Installation memory and elapsed time remain unmeasured.

```sh
# Execute only after the coordinator verifies this shell is on its compute node.
set -euo pipefail
: "${SLURM_JOB_ID:?coordinator-owned compute allocation required}"
: "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
case "$(hostname)" in *-ln*) exit 2 ;; esac
ep17_python="$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python"
ep17_h5py_site="$GROUP_HOME/users/zijiao/ep17_h5py_3_10_0_cp311"
ep17_runtime_logs="outputs/runs/stimulus_runtime_repair_job_${SLURM_JOB_ID}"
ep17_install_tmp="$L_SCRATCH_JOB/ep17_h5py_repair_${SLURM_JOB_ID}"
test ! -e "$ep17_h5py_site"
test ! -e "$ep17_runtime_logs"
umask 077
mkdir -p "$ep17_runtime_logs" "$ep17_install_tmp"
env -u PYTHONPATH PYTHONNOUSERSITE=1 TMPDIR="$ep17_install_tmp" PIP_CONFIG_FILE=/dev/null \
  timeout --signal=TERM --kill-after=5s 180s "$ep17_python" -B -m pip --isolated install \
  --no-deps --only-binary=:all: --no-compile --no-cache-dir --disable-pip-version-check \
  --no-input --timeout 30 --retries 0 --index-url https://pypi.org/simple \
  --target "$ep17_h5py_site" --report "$ep17_runtime_logs/install.json" \
  --require-hashes --requirement /dev/stdin >"$ep17_runtime_logs/install.txt" 2>&1 <<'REQUIREMENT'
h5py==3.10.0 --hash=sha256:6c013d2e79c00f28ffd0cc24e68665ea03ae9069e167087b2adb5727d2736a52
REQUIREMENT
env -u PYTHONPATH PYTHONPATH="$ep17_h5py_site" PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' TMPDIR="$ep17_install_tmp" \
  timeout --signal=TERM --kill-after=5s 120s "$ep17_python" -B - \
  >"$ep17_runtime_logs/stimulus_adapter_tests.txt" 2>&1 <<'PY'
from importlib.metadata import version
import sys
import unittest
assert sys.version_info[:3] == (3, 11, 15)
expected = {"numpy": "1.26.4", "torch": "2.4.1+cu121", "torchvision": "0.19.1+cu121",
            "pillow": "12.2.0", "h5py": "3.10.0"}
actual = {name: version(name) for name in expected}
assert actual == expected, actual
print(actual, flush=True)
suite = unittest.defaultTestLoader.discover("outputs/tests", pattern="test_private_stimulus_feature_pilot.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
```

The unchanged five source-free fixtures exercise HDF5 writes and reads with
generated inputs and fake backbones, so their actual execution also checks the
new wheel against the retained NumPy runtime. No acquired pixels, secrets,
checkpoints, private operators or neural payloads enter this repair. It neither
selects the actual image pilot nor changes the enforced reader boundary.

## October 9 implementation — execution remains unselected

`tools/private_stimulus_feature_pilot.py` now provides the callable
`run_private_development_feature_pilot(password_fd=..., private_output_dir=...,
operational_report_path=..., random_checkpoint_path=...)`. The three output
paths are fixed to this document's private pilot001 staging/report and
`outputs/cache/model_checkpoints/random_resnet50_seed17.pth`; there is no CLI
or credential path discovery. The inherited descriptor is caller-owned and
duplicated for reading. Delivery is bounded to 4,096 bytes including an optional
single terminal LF/CRLF; its removal preserves other password bytes. No Python
or ZIP-library secure-erasure guarantee is claimed.

The existing exact 32-image selection, archive identity, member/image limits,
four models and fixed preprocessing below remain unchanged. The implementation
reuses completed directory-inspection code inside the actual adapter, without
selecting another standalone metadata run. It streams the acquired archive's
recorded MD5 and SHA-256 before decoding because the earlier directory-only
inspection did not bind a fresh content check to this open object. It preserves
one seed-17 random artifact, private partial files and first-access flags; only
operational metadata are reported. Implementation storage refusals are 4 MiB
for the four feature containers plus index, 128 MiB for the random checkpoint
and 256 KiB for the operational report; no automatic cap expansion is permitted.

Independent source review is complete. Newly created owned directories have
inherited setgid normalized to 0700 before content is written; pre-existing
directories are never changed. The cost projection below now includes all
measured preprojection work, conservatively scaling otherwise unclassified
selection/index/container/flush overhead with development image count. Only
final operational-report serialization/flush is outside that stated timing
coverage. The five focused source-free test bodies remain unrun after the
missing-h5py import failure recorded above; ASTRA validation passed. Their
original optional post-assembly selection is recorded in the
[execution log](experiment_log.md#2026-10-09-0433-pacific--stimulus_features-private-callable-implemented-source-free-checks-selected).
This source implementation does not activate the pixel pilot, read a credential,
establish isolation or release a cache. Existing research-term acceptance stands;
authorized private descriptor delivery and enforced reader separation remain
unresolved. The previously stopped EP18 namespace route stays stopped; this
does not document an EP17 namespace failure. See the
[current reader and delivery handoff](source_roles_proposal.md#october-9-current-reader-and-delivery-handoff).

## October 6 source-interface update

The [pinned provider README](https://raw.githubusercontent.com/courtois-neuromod/things.stimuli/f8f5162f79fb2e9fb0d8cbb8bac610db11a54180/README.md)
names `password_images.txt` as the password source for `images_fmri.zip`.
That resolves the provider-interface identity; availability through EP17's
authorized private delivery path and successful use are still unverified.
No password/key file was opened, copied or displayed. Reuse existing accepted
source terms, retain private secret handling and the required reader boundary,
and bind this documented file interface when the bounded adapter is selected.
No generic credential-identification or repeated acceptance question is added.
See the [source-role handoff](source_roles_proposal.md#october-6-public-source-update--documented-stimulus-password-interface).
Historical metadata/pilot descriptions below retain their original scope;
the completed directory inspection does not itself authorize pixel decoding.

The bounded archive metadata prefix is adopted on October 4 at 23:15 UTC,
after independent review and all 19 focused fixtures passed. Execute
`inspect_stimulus_archive.py --out outputs/runs/stimulus_features/archive_metadata001`
under 60 seconds internal / 90 seconds external plus 10 seconds termination,
one CPU and 2 GiB. Read only the bounded classic-ZIP tail and central directory;
ZipFile receives an in-memory directory-only wrapper. Combined output is capped
at 256 KiB. Exact member mapping and completed archive inspection are separate
flags. The broader pixel/feature pilot described below remains prospective.

Original status: prospective specification, 2026-10-04; no archive, pixel or feature
access adopted by this document. The next useful stage is a fixed 32-image
development-only pilot using the already adopted
[model representation definition](model_pair_registry_specification.md).
The exact acquired archive is now located. Remaining adapter facts are the
member namespace/encryption method and the approved decryption interface.
No extraction, download, source
pixel read, credential inspection or feature computation occurred here.

## Source and authority already documented

Reuse the September 22 acceptance recorded in the
[EP17 execution log](experiment_log.md#2026-10-03--source_metadata--metadata-preparation-selected-attempt-001)
and the named acquisition `ACCEPTANCE_RECEIPT.txt`: the source images are for
this project's academic, noncommercial scientific use, without distribution
or alteration of the originals. A credential-related field followed the
permitted acceptance text; that line and the rest of the receipt were
withheld and the receipt was closed. No credential value was obtained.
There is no new acceptance action in this proposal. Preserve the archive
and source-image bytes; the prospective in-memory model preprocessing comes
from the adopted research design. Do not infer broader image, checkpoint or
redistribution rights from the participant release's CC0 license.

The separately read, plain `VERSION_CHAIN.txt` establishes:

| Item | Recorded identity |
| --- | --- |
| Release | CNeuroMod-THINGS 1.0.1; root `0f22b6c5eca4001e86fa936f3b6c2569a0179970` |
| fMRIPrep | `5155ba0d43836a4451141b90b5a70507f41205cf` |
| Task source | `b17d3697171824c198ebc6f15fe7ceee9a1f526f` |
| Stimulus repository | `courtois-neuromod/things.stimuli`, `f8f5162f79fb2e9fb0d8cbb8bac610db11a54180` |
| Logical archive | `images_fmri.zip`, still encrypted and unextracted |
| Existing annex identity | `MD5E-s917286854--bb6aa66c55247e4660cb66cb133e5bd6.zip` |
| Declared bytes / MD5 | 917,286,854 / `bb6aa66c55247e4660cb66cb133e5bd6` |

The acquisition record directory is
`$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/cneuromod_images_fmri_pinned_f8f5162_20260922/`.
Bounded nonrecursive filename/type/size inspection located its ordinary file
**`archives/images_fmri.zip`**, with the declared 917,286,854 bytes.
`PROVIDER_MANIFEST.tsv` binds that exact relative path to the MD5 above and
recorded local SHA-256
`f3c41fa59485faab5debb882b15f046253acfc144564abfdcd0782fba7d90912`.
No archive bytes were opened for this specification.

The plain `CENTRAL_DIRECTORY_RECONSTRUCTABILITY_RECEIPT.md` records 8,640
image entries matching the pinned experimental-path set. All 8,640 differ
in declared size and/or CRC32 from their translated full-THINGS counterparts.
It therefore rejects reconstructing this source by selecting/renaming
full-THINGS images. This is existing metadata evidence, not a new pixel
comparison or a cryptographic identity proof for individual images. The
receipt gives no encryption method, exact member-prefix translation for this
adapter, or approved local secret-delivery interface. Those were unresolved
by that receipt; the later completed directory inspection and October 6 public
interface note above must be considered separately.

Before pixel execution, bind the documented provider file interface to an
authorized local delivery mechanism. The exact-member and
encryption-header checks below can run as the adapter's bounded metadata-only
prefix once selected, without a secret or any decode, and stop there if the
remaining interface is unavailable. This is ordinary source-adapter work,
not a new eligibility stage. Do not download another archive, substitute a
full-THINGS release, inspect a password/key file, expose a secret in command
arguments/logs, or search unrelated steward directories. Existing acceptance
is sufficient; no repeated terms confirmation is requested.

## Reuse the completed image and role identities

The pinned provider metadata is already named in
`runs/source_metadata/attempt003/metadata_report.json`:

`$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/cneuromod_images_fmri_pinned_f8f5162_20260922/provider_git/image_paths_fmri.csv`

Its exact image key is `image_path`, for example
`images_fmri/acorn/acorn_01b.jpg`; the schema separately records condition,
provider category/exemplar and THINGS category identifiers. Reuse the
completed event reconciliation, eligibility and chronology through
`runs/concept_role_assignment/full001/assignment_report.json`,
`image_assignments.csv` and `concept_assignments.csv`. The terminal report is
completed with `metadata_assignments_complete: true`. It preserves 720
concepts, 4,320 observed image identities and their inherited 480/120/120 roles;
3,634 images are already repeat-qualified. No new universe, exclusion,
balancing optimization, chronology or fold assignment is needed.

For the feature adapter, exactly join `provider_image_path` to provider
`image_path`, retaining the original string and the known concept/category/
exemplar/condition fields. Require consistency with the completed map and
one immutable concept role. Use `eligible_image == True` and
`role == development` for the primary development feature pool. Ineligible
images retain their existing metadata/reasons; this adapter does not make
them eligible or silently remove their metadata. Calibration images need no
candidate features. Audit images and features are outside this pilot.

At an adopted archive-adapter stage, identify members by the full provider
path, with no basename-only, substring, case-folded or fuzzy matching. The
default is an identical archive member path. A different fixed archive root
would need explicit provider/central-directory evidence and a recorded
one-to-one prefix mapping before decoding; it cannot be chosen from pixels.
Missing or duplicate selected members, conflicting IDs, absolute/traversal
paths or unsupported image/encryption formats stop the route. Do not fill
missing images from an external THINGS collection. Directory entries and
unselected catch/test/other members are not extracted.

## Fixed pilot and preprocessing

Select the first 32 development concepts in increasing numeric
`things_category_nr`. Within each, select the eligible image with the
lexically first exact UTF-8 provider path. Every qualified concept has at
least one eligible image; a missing selection is a metadata failure, not a
reason to choose another concept. This fixes 32 distinct concepts/images
without inspecting pixels, neural responses or embeddings. Publish these
metadata-only selected IDs before the first decode. Do not choose a favorable
image subset or increase the pilot after observing model-derived values.

The trusted builder first binds the selected physical archive to its declared
length and existing annex MD5, reusing a completed content-identity check if
the records bind it to that same unchanged object. Otherwise the pilot makes
that bounded identity check before decoding. It then uses the approved
secret interface privately.
The implemented interface takes an explicitly inherited descriptor, never
command-line secret text, a public environment dump or an exported exception.
Its authorized private delivery has not been established or selected for an
actual run; this document does not read, create or relocate any secret.

Decode only the 32 selected member streams into memory. No bulk extraction,
persistent original-image copies or thumbnails are needed. Before decoding,
bound the central directory to 16 MiB and 16,384 entries and each selected
compressed/uncompressed member to 16 MiB. Require one image frame and at most
16 million decoded pixels. These are prospective resource refusals, not
eligibility exclusions; a violation stops the pilot without replacement or
automatic limit increases. Record encryption/decoder compatibility without
logging raw archive data or secret-bearing errors.

Use all four adopted backbones, sequentially, with batch size one and no
loader subprocesses: DINO_R50, SWAV_R50_800, SUP_R50_V1 and RANDOM_R50_S17.
Reuse the acquired checkpoints and completed compatibility work. Reuse the
single persisted random backbone if already instantiated under its adopted
CPU-seed-17 recipe; otherwise its one-time creation must precede source
features. Never initialize a second seed or reload alternative weights.
Decode/preprocess each selected image once and release its original decoded
array immediately. Retain only the 32 transformed input tensors in builder
RAM (19,267,584 float32 bytes) for the four sequential backbones; do not
persist that image-derived input bank.

The registry fixes Torch 2.4.1/Torchvision 0.19.1, RGB conversion without EXIF
rotation, shorter-edge 256 PIL bilinear resizing, 224 central crop, fixed
ImageNet normalization, evaluation/inference mode, float32 and arithmetic
spatial pooling at layer1–4. Use that exact implementation and its source-free
checks. Freeze the observed Pillow version before creating the first cache.
No PCA, feature centering, unit-length normalization, fine-tuning, BatchNorm
adaptation, layer choice or neural fitting occurs in this pilot. A later
development-training transform is learned within its registered folds.

Proposed initial resource envelope: **one CPU, 4 GiB, no GPU**, 600-second
internal deadline, 660-second external timeout plus 10-second kill grace;
aggregate with useful preparation in an attended allocation. One backbone
and one decoded image are resident at a time. This measures the CPU route
directly; a GPU route would need an explicit later resource selection/pilot,
not an unmeasured backend substitution. No package install or hardware search
is part of the source pilot.

## Private artifacts and truthful reporting

Proposed fresh staging root:
`outputs/cache/private/stimulus_features/pilot001/development/`.
Create ordinary private files under the existing ignored cache tree, with
0700 directories and 0600 files from creation. Keep one ordered
`image_index.csv` and one HDF5 file per registry model containing ordinary
float32 datasets `layer1`, `layer2`, `layer3`, `layer4`, with shapes
`(32,256)`, `(32,512)`, `(32,1024)`, `(32,2048)`. Every model uses the identical
row index. No soft/external links, virtual datasets, references to mixed
source arrays or mixed-role feature files are allowed. Verify exact shapes,
dtype and finite values inside the builder; export only pass/fail booleans,
never feature values, norms, moments, sample vectors or model comparisons.
The index fields are `feature_row_zero_based`, `provider_image_path`,
`image_name`, `things_category_nr`, `uniqueID`, `role`, `outer_fold`,
`development_block`, and `eligible_image`, copied through the exact completed
image/concept joins. Rows follow the fixed 32-concept selection order.

The four models together produce 61,440 float32 payload bytes per image:
1,966,080 bytes for this pilot, excluding small container/index overhead.
The completed metadata will supply the exact eligible development-image
count at pilot execution; do not compute a new role assignment to obtain it.
The completed eligibility report's maximum is six qualified images per
concept; 480 development concepts × 6 therefore gives at most
176,947,200 feature payload bytes for a later development-only four-model
cache. Calibration and sealed-audit images are excluded from that bound and
from the proposed full development stage.

Only a bounded operational report is controller-visible at
`outputs/runs/stimulus_features/pilot001/pilot_report.json`: selected public
IDs, fixed model/layer/preprocessing identities, source-object verification,
archive/member compatibility, shapes/dtypes, bytes, timings, peak RSS,
finite-check booleans, error/completion flags and first-access timestamps.
Set `image_pixels_read` at the first selected decode and
`model_features_computed` when inference begins. Audit pixel/feature access
remains false for this development-only pilot. Preserve source/model
attribution and the already recorded unknown pretraining-image exposure;
runtime success establishes neither nonoverlap nor promotion eligibility.

Set `pilot_feature_extraction_complete` only after every selected image/model/
layer and index map succeeds; any missing output, refusal or deadline leaves
all partial files ineligible. No partial cache is released as complete.
Record archive-identity/decryption, decoder, per-model load/inference and
write time separately so fixed overhead is not multiplied by image count.
For fixed overhead `I` (archive identity/open, metadata joins and model loads)
and measured 32-image work `V` (decryption/decode/transform, all four forward
passes and feature writes), report `2 * (I + M_development / 32 * V) + 60`
seconds as the buffered full-development projection. Explicitly label this
an estimate, not a guarantee for every unopened image. Full cache
execution, resources and deadline are selected only after reviewing the
measured pilot, with no automatic scale-up or retry.

These are **builder-private staging files**. Mode 0700/0600 and Git exclusion
do not isolate agents running under the same UID or remove their access to
the mixed source. Keep `physical_role_isolation_verified`,
`reader_handoffs_activated`, `source_roles_complete` and
`candidate_scoring_authorized` false. Only separately qualified development
features may later reach the search worker. Sealed-audit features remain
evaluator-only after the write-once predictor/contract/forecast lock and are
not generated speculatively by this pilot. Do not work around an unresolved
reader boundary by relying on directory names or voluntary filtering.

Record this episode's image/feature first-access history for the later
required EP17/EP18 overlap decision. No EP18 outputs are inputs here, and no
cross-episode reuse or independence claim is authorized. Neural qualification,
calibration/support construction, linking grids, scientific margins and the
one-shot audit prerequisites remain separate requirements before scoring.
