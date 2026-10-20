# EP03 grouped comparison: selected methods and pending decision

Selected 2026-10-18 under standing methods/source-fact authority, before baseline
scores. This implements the existing question: does signed condition semantics
recover map geometry beyond recognition of the task family? A gain surviving the
registered lexical, metadata, within-family permutation and semantic-neighbor
controls weakens the prototype-only explanation. A gain disappearing in one of
those controls weakens the semantic interpretation. An imprecise zero gain or
failed known-positive check is underidentified, not equivalence.

## What is selected and executable

`config/methods_preparation.json` records an explicit route-to-family dictionary
from the frozen route and condition-role metadata. Families describe paradigm
context; assumptions and unresolved routes remain visible. No labels are chosen
from map values, correlation, prediction loss or desired family size. Retain
families with only one accession in the inventory and report unsupported transfer;
do not merge them merely to obtain a score. A documented methods refinement can
be recorded prospectively, before affected outcomes, under standing authority.

The previously adopted independence assumption remains one OpenNeuro accession
per group, pooling every task, session and derivative. It is **not authenticated
cross-accession independence**. Explicit evidence of shared participants, input
images or derivative lineage merges the connected accessions before any fold.
Equal local labels such as `sub-01`, common software and common repository commits
do not establish shared participants. With no cross-accession evidence in the
three permitted metadata inputs, preparation reports the assumption rather than
inventing clean lineage. The duplicate-lineage falsifier remains required.

The old NeuroEffect `partition_status` is retained as provenance, not silently
turned into a new EP03 development exclusion. Every route/map in the current
51-candidate-accession source remains accounted for; the adopted orientation,
duplicate, never-convolved, N=2 and ds003545 timing screens still apply. Missing
source files and remaining sign/license/space eligibility stay explicit. Fitted
N uses the completed October 17 lineage conclusion, correcting ds003545 compL1
to 20; this does not release the timing hold. The frozen original TSVs stay intact.

Outer evaluation leaves out one whole lineage group. All its contrast/session
copies and trial-split controls remain excluded together. Inner selection repeats
whole-group exclusion using **outer training groups only**; recompute family
representation in each inner fold. A family must be represented in that fold's
training groups to contribute a matched prototype/residual comparison. Unsupported
rows remain in the report and the prescribed leave-one-family-out diagnostic;
they never receive a zero prototype or contribute to a different denominator.
Metadata support is preliminary: it does not certify scientific map eligibility.

`code/prepare_grouped_methods.py` reads only the exact frozen route manifest,
orientation TSV and role JSON plus this methods configuration. It writes a new
metadata/fold preparation file under `runs/grouped_evaluator_qualification/`,
refuses to overwrite it, and leaves baseline access/scoring false. It cannot
read image paths from a manifest. The build and all eight focused tests passed
in step 47586225.6 on October 18; [execution evidence](experiment_log.md#2026-10-18-1803-pdt--grouped_evaluator_qualification--methods001-verified-and-built)
records the provisional support counts and remaining eligibility limits.

## Normalization, comparator and score

`code/grouped_evaluator.py` has pure array functions, no data reader or model fit.
The common anatomical support is now prospectively bound to the **existing**
TemplateFlow `MNI152NLin2009cAsym` res-02 `desc-brain_mask.nii.gz` in the current
account's cache, as recorded in `config/methods_preparation.json`. Its template
metadata gives shape 97 × 115 × 97, 2-mm voxels and origin
(-96.5, -132.5, -78.5). The copied derived mask retains the source LICENSE.
The reference array and all 628 target headers still require the selected compute
check below; this binding does not assert that they already match. Source Z maps keep their
original sign. Within that exact mask, subtract each map's spatial mean and divide
by its Euclidean norm in float64, then multiply by the adopted polarity `s` for
canonical fitting. No epsilon floor, mapwise voxel dropping, imputation, zero-based
mask selection or score-driven common support is allowed. Constant/nonfinite data
trigger diagnosis; the runner cannot silently prune an unfavorable group or map.

For each family, first average the canonical training contrasts **within each
training group**, then average those group means equally. Learn residuals from
the same canonical training objects. Both source-sign predictions are restored
once: `s*p_family` and `s*(p_family+r)`. Predictions are not re-normalized; a fitted
prototype's norm carries its dispersion. The global-map control uses the same
group weighting and final sign. Metadata/model vocabularies, bases, residualizers
and hyperparameters are fitted inside their training folds.

For each excluded group, average voxel MSE across that group's identical eligible
contrasts, then compute `G_d = 1 - L_model,d / L_family,d`. Average G equally across
groups. Do not average per-map ratios, pool group denominators or clip gains. A
zero/nonfinite prototype denominator is an invalid comparison requiring diagnosis,
not an epsilon substitution. The same rows, mask, folds and contrast weights apply
to every comparator and falsifier. The initial implementation uses the registered
uniform within-group option; reliability-capped variants remain separate trials.

## Independent-unit uncertainty and positive sensitivity

Ordinary confidence intervals over out-of-fold group scores ignore overlapping
training. The proposed uncertainty method resamples **whole independent groups
with refitting of the complete nested procedure**, preserving each source group's
duplicate identity within a draw. Multiplicity in a bootstrap draw changes the
group weight; copies of one source group must never enter both training and
validation. Leave-one-group influence must also refit the affected procedure.
These intervals describe the exposed development procedure, conditional on the
lineage assumptions; they are not independent prospective confirmation. Exact
resampling count, inferential thresholds and control acceptance are in the packet
below and are not silently implemented as accepted scientific rules.

Before interpreting a null/equivalence, the same analysis must detect an injected
condition-linked residual of the accepted practical gain on the same group/family
structure. For this known-positive check, detection means rejecting zero gain
under the proposed simultaneous error rule (lower uncertainty bound above zero).
At an injected true gain of delta, the proposed 90% sensitivity concerns this
zero-null rejection; it does not require the lower bound to exceed delta. The
separate promotion rule still requires a lower bound above delta. The control
must run through preprocessing, grouping, selection and
scoring, not merely test the scoring formula. Fix its construction and acceptance
before opening the affected evaluation; failed sensitivity yields underidentified.
No synthetic sweep, new model family or target opening is selected here.

## One scientist decision packet — recommendations, NOT adopted

The following is a concrete package for the still-empty inference fields. These
are scientific conventions proposed before baseline scores, not estimates justified
by the old 2.2% gain over a different global-mean comparator. Alternative values
can be supplied in one response. The methods build continues while this is pending.

| Field | Recommended rule and reason |
| --- | --- |
| Meaningful gain | `delta = 0.01` absolute group-mean relative MSE gain: at least a 1% reduction against the **family** prototype. This is an explicit operational bar, not a known biological minimum. Promotion needs its lower uncertainty bound above delta; a positive point estimate alone does not qualify. |
| Distribution over groups | Strictly more than half of represented independent groups have `G_d > 0`, and the mean direction stays positive on every full leave-one-group refit. This implements the existing non-dominance and minimum-fraction requirements. |
| Negative-control pass | Simultaneous one-sided upper bound below delta for the full-procedure trial-split/replicate-order control. A nonsignificant gain is insufficient. This interprets the existing “no positive gain” requirement as excluding a practically meaningful spurious gain and therefore needs the scientist's explicit adoption. |
| Error and selection | One-sided familywise alpha 0.05 for promotion and falsifier families, using a maximum-statistic complete-procedure permutation/null analysis over all actually compared candidates. Use 1,999 permutations and 2,000 whole-group refit draws initially; report Monte Carlo uncertainty and extend if it changes a threshold decision. Earlier adaptive exposures remain disclosed. |
| Null/precision and reliability | No equivalence claim unless the two-sided 95% simultaneous interval lies inside ±delta **and** the full procedure rejects zero gain for an injected true delta gain with at least 90% sensitivity under the proposed simultaneous error rule. Detection here means a lower bound above zero, not above the separate promotion margin delta. Require the group-level lower uncertainty bound for authenticated residual-map repeatability above zero as a necessary reliability check; repeatability alone does not establish enough sensitivity. Missing repeatability or failed sensitivity is underidentified. These are proposed development rules, never a substitute for a prospective audit. |
| Search patience | Material improvement tolerance `0.002` relative-gain units (one fifth of delta), keeping the existing minimum/maximum trial counts, patience length, full grammar, two successor cycles and all falsifiers. Its operational purpose is avoiding score-noise-driven cycling; it is not a new terminal evidence threshold. |

The package does not authorize a fresh dataset, audit opening, model-family change
or confirmation claim. Those remain separate consequential decisions if needed.
No baseline image may be opened until the policy's taxonomy, normalization and
negative-control threshold prerequisites are actually satisfied. This preparation
selects the first two methods choices; it leaves the third explicitly unadopted.
Verification of the selected common mask/space, remaining source eligibility,
redacted-text completeness/cache, full predictor composition and compute checks
also precede scientific scoring.

## Selected anatomical reference and signed-text preparation — October 18

`code/prepare_mask_and_signed_text.py` consumes the completed `methods001` metadata
without altering its 628 identities, exclusions, grouping assumptions or folds.
The reference-only phase loads the one explicitly bound anatomical mask; it checks
binary/nonempty support, coded spatial forms and the selected grid, then stores a
boolean mask without target-dependent voxel selection. It does not read any target
array. A mismatch stops this phase for diagnosis, with no alternate mask or silent
resampling. This is a common anatomical support assumption, not a claim that every
participant or source map provides reliable signal throughout that support.

The first reference build in step **47586225.7** exposed an overly exact binary
comparison after all six focused checks passed. Reference-only diagnosis in
**47586225.8** found uint8 labels 0 (846,195 voxels) and 255 (235,840 voxels), with
float32 slope `0.003921568859368563`, intercept zero and scaled foreground
`1.0000000591389835`; geometry matches the selected grid and all values are finite.
The [diagnostic](runs/grouped_evaluator_qualification/mask_text001/reference_diagnosis.json)
supports a numerical repair: require exactly two finite discrete levels including
zero, accept the foreground only within one float32 epsilon of one, and convert
that exact label to bool. Preserve all 235,840 foreground locations. This does not
threshold a probability image; two-level 0/0.5 and multilevel masks remain invalid.
The original `mask_text001` attempt stays intact. The repaired `mask_text002`
build and extended focused test passed in step 47586225.9, preserving all 235,840
foreground locations. The separate `headers001` check passed all 628 exact paths
in step 47586225.8. Neither operation read target voxel arrays.

Canonical text uses only the adopted structured factor names/levels, regressor
kind/direction and an exact reviewed allowlist of redacted quantity phrases. Exact
per-map executed coefficients come from `dataset_maps.json` and must uniquely match
the full ordered frozen orientation definition. Within that exact-map scope,
contrast names reuse the producer census's lowercase-alphanumeric normalization
(for example, `cue_LvNeut` → `cueLvNeut`); no fuzzy or cross-map lookup is allowed.
The rounded TSV is a join check,
never a replacement for numeric precision. The embedded route models supply
regressor transformations, and the existing adopted MID role parser is reused.
Each canonical coefficient is its source coefficient times the already adopted
polarity, with no rounding, renormalization or extra sign change. Original terms,
weights, signs and identities remain provenance. Only `model_text` is a text-model
input: paper/dataset titles, identifiers, raw methods JSON, notes, sample sizes,
paths and evidence codes are excluded. This is one input construction for the
registered comparison, not a replacement for its full grammar or controls.

Unknown directions, missing roles, undocumented signed quantities and ambiguous
definitions remain explicit unresolved records with no canonical model text.
Distinct source conditions can share an adopted factor description; preserve all
terms and disclose this compression rather than asserting complete semantic
coverage. Neither unresolved text nor successful text preparation changes map
eligibility. The gas/eye family entries, opaque CND meanings and all original
metadata screens remain as selected in `methods001`; no outcome-driven recoding.

Exact image paths already exist in the permitted census: join
`dataset_maps.json` on `(route, session null→empty, contrast)` to every methods001
map and form `route_provenance.inputs.analysis_root / dataset_id / path`.
The preparer rejects incomplete/duplicate joins or unsafe relative paths and
does not glob the source tree. Its independent `--headers-only` phase reads only
352 decompressed NIfTI header bytes per enumerated image; it reports compatibility
or errors for **all 628**, retaining existing exclusions. Grid agreement does not
resolve remaining source/sign/license eligibility or establish target signal.
No target voxel array, scientific score, model fit or new dataset is opened here.

## Coordinator compute handoff

From the episode directory, in a verified task-owned Slurm shell, use the existing
Python with NumPy; there is no install requirement:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3.12 -B -m unittest discover -s outputs/code -p test_grouped_methods.py -v
python3.12 -B outputs/code/prepare_grouped_methods.py \
  --output outputs/runs/grouped_evaluator_qualification/methods001/preparation.json
```

These eight checks target unequal-group weighting, held-out leakage, sign
restoration, ratio aggregation, fixed support, adopted screens and lineage folds.
The build uses 628 metadata rows and 103 routes; no NIfTI, Torch, encoder or fit is
loaded. The completed handoff used one CPU, 512 MiB, a 5-minute attended companion allowance within
task-owned substantive allocation 47586225 (step 6), selected by the coordinator.
Measured process peaks were 39,872 KiB for tests, 17,824 KiB for the build and
73,340 KiB for pinned ASTRA validation; all passed. These commands document that
completed execution, not a request to repeat it. The coordinator owns any further
allocation choice and combines short work as site rules require. Verification
does not authorize target access; preserve any failed output attempt.

The mask/text and header phases below have completed in coordinator-owned compute.
These commands document the runs; no repeat of passing checks or artifacts is
selected. The original six-test run passed before the reference-encoding repair;
only the affected mask test was repeated afterward.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3.12 -B -m unittest discover -s outputs/code -p test_mask_and_signed_text.py -v
python3.12 -B outputs/code/prepare_mask_and_signed_text.py \
  --output-dir outputs/runs/grouped_evaluator_qualification/mask_text002
python3.12 -B outputs/code/prepare_mask_and_signed_text.py --headers-only \
  --output-dir outputs/runs/grouped_evaluator_qualification/headers001
```

The corrected mask regression was invoked directly as
`python3.12 -B outputs/code/test_mask_and_signed_text.py MaskAndSignedTextTests.test_mask_order_binary_support_and_conflicting_geometry -v`,
preserving the module-provided import path. A preceding coordinator invocation
had replaced that path and could not import NumPy; it ran no test or build.

Six checks address predictor leakage, exact weights/polarity, unresolved meanings,
executed-definition binding, mask binary support/spatial order and header-only
exact-path access. They do not repeat the eight already passing evaluator checks.
Execution used one CPU/512 MiB in substantive allocation 47586225. The corrected
build took 0.82 s/71,928 KiB; the affected test 0.75 s/40,620 KiB; 628 header opens
34.73 s/54,172 KiB; changed ASTRA validation 3.75 s/73,488 KiB. The mask/text and
header phases are separate so I/O failure cannot erase a completed preparation.
The successful input build describes 574/628 maps; among 500 metadata-supported
maps, 477 have text and 23 need meaning resolution. No map is removed for missing
text. New attempts refuse overwrite; all threshold fields stay unadopted.
