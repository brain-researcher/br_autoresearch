# EP22 experiment log

Newest entries first. Entries are keyed by ASTRA analysis ID
([program](astra/v0.0.14/astra.yaml)).

## 2026-10-13 04:13 UTC — reporting — summary and confirmation figure complete

**Codex output review.** Codex `gpt-6-astra` (`max`, read-only, no commands)
returned two deliverables, which the orchestrator reviewed before applying.

- **[summary.md](summary.md):** every number checked against the facts
  extracted from the reports.
- **[plot_ood_confirmation.py](code/plot_ood_confirmation.py):** key names
  match the report schema; the panel computations are correct; the "all four"
  group equals the C1 participant gap; the validated palette is used.

Both were applied without edits.

**Rendering.**
[plot_ood_confirmation.sbatch](code/plot_ood_confirmation.sbatch) ran as job
47608692 (`COMPLETED`, 16 s, 1 CPU, below 0.01 CPU-hours). It read only the
read-only confirmation report. The orchestrator inspected the output,
[ood_confirmation.png](figures/ood_confirmation/20261013-opening01/ood_confirmation.png)
(with an SVG alongside):

- **Panel A:** C2a E_val is below 0 for every participant, for CLIP and
  SigLIP2; PEcore is secondary.
- **Panel B:** C3 deltas are positive for every participant and VLM; DINOv2 is
  shown in gray as "not a prediction".
- **Panel C:** the CLIP − DINOv2 native gap by category, inside the ±0.02
  margin band, with a deficit only on selfmade.

## 2026-10-13 02:10 UTC — reporting — result summary and figure delegated to Codex (text-only)

The scientist set up a split: Opus orchestrates, and Codex `gpt-6-astra` at
`max` does the work. The Codex CLI default was set to `gpt-6-astra`/`ultra`
at the scientist's request; the backup is
`~/.codex/config.toml.before-astra-default-20261013`.

On Sherlock, Codex runs with `-s read-only` and executes no commands. Its
sandbox cannot start here, and disabling the sandbox or auto-approving
escalations is not used.

The prompt sends only verified summary numbers, extracted programmatically
from the D0–D3 and confirmation reports, together with the report schema. No
neural data or raw images are sent. Prompt and response are under
`$SCRATCH/br_autoresearch/episode22_geometry_gap_decomposition/codex/taskA_summary_figure/`.

Codex is asked for two deliverables:

- `outputs/summary.md`;
- `outputs/code/plot_ood_confirmation.py`.

The orchestrator reviews both, renders the figure in Slurm, and checks it.

## 2026-10-13 01:29 UTC — ood_confirmation — single opening complete; terminal class `confirmation_complete`

**Run.** Job 47580993 ran on sh02-06n11 from 00:50:00 to 01:08:02 UTC and
ended `COMPLETED 0:0`: 18 min 15 s, 1.83 allocated CPU-hours. EP22 now totals
26.43 CPU-hours, with no GPU.

- MaxRSS reached 33.55 GB, at the 32 GiB request. There was no OOM and no
  error, and the report is complete. A future rerun would need more memory.
- The report
  [confirmation_report.json](runs/ood_confirmation/20261013-opening01/confirmation_report.json)
  was written once and then made read-only. Its sha256 is
  `93c60a0fca41cd33c483aa2460e91b33066053b063744811cbaa441143e2307e`.
- The report records the lock sha256 `efb07381…` and the approval anchor.
- The reader decoded only OOD rows; for example, sub-01 ses-01 decoded 36
  volumes and discarded 1,008 as opaque bytes.
- All 371 images were eligible in every participant: 1,484 presentations
  each, and 1,483 for sub-03.

**Locked outcomes.** The opening was procedurally sealed, with five
participants.

| Prediction | Result | Evidence |
| --- | --- | --- |
| Gate | **passed** (scored) | DINOv2 within-category RSA 0.187 (participants 0.171–0.196), threshold 0.033, null mean −0.000 |
| C1, CLIP | **indeterminate** | mean gap −0.023, but only 2/5 participants ≤ −0.02 (−0.059, +0.005, −0.017, +0.003, −0.046). Leaving out selfmade gives −0.000 |
| C1, PEcore (secondary) | **indeterminate** | mean −0.027; 3/5 participants ≤ −0.02 |
| C2a, CLIP | **passed** | within-category E_val −0.020, negative in 5/5 |
| C2a, SigLIP2 | **passed** | within-category E_val −0.023, negative in 5/5 |
| C2b (descriptive by lock) | not evaluable | the all-pairs gap is positive: CLIP +0.074, SigLIP2 +0.021 |
| PEcore secondary | **failed** | E_val −0.017 was negative in 5/5, but E_vec was −0.010, positive in only 1/5 |
| C3 | **passed** for every VLM | development-fitted α raised OOD RSA in 5/5 participants: CLIP +0.022, PEcore +0.017, SigLIP2 +0.035 (DINOv2 +0.045, not a prediction) |

**Descriptive results, outside the locked predictions.**

- **CLIP − DINOv2 gap by category:**

  | Category | Gap |
  | --- | ---: |
  | cropped | −0.005 |
  | relations | +0.008 |
  | selfmade | −0.090 |
  | unusual | −0.004 |

  The mean deficit comes from selfmade alone.
- **Within-category decomposition:**

  | Model | Gap | E_val | E_vec | Share of gap from the spectrum |
  | --- | ---: | ---: | ---: | ---: |
  | CLIP | −0.023 | −0.020 | −0.003 | about 89% |
  | SigLIP2 | −0.051 | −0.023 | −0.029 | — |

- **All-pairs over the confirmatory images:** every VLM beats DINOv2. The
  advantage comes from the eigenvectors (E_vec +0.04 to +0.09), while E_val
  stays negative (−0.006 to −0.020).
- **Synthetic types:** DINOv2 leads on gabor, illusion-classic and
  illusion-natural; the VLMs lead on shape and gaudy. In EVC, DINOv2 reaches
  0.160 against 0.05–0.09 for the VLMs.
- **The superseded original C2** would also have failed on OOD. CLIP's mean
  closure was 0.99, but 0/5 participants met the per-participant rule;
  SigLIP2 reached 0.53, with 2/5 participants.

**Interpretation.**

- **Confirmed** on the sealed OOD images:
  - Every VLM's spectrum lowers its within-category RSA relative to DINOv2
    (C2a).
  - Brain geometry prefers spectra steeper than any model's native spectrum,
    with α ≈ 1.6–2.3, and this transfers out of sample (C3).
- **Not confirmed:**
  - The spectral-versus-selection test for CLIP is indeterminate (C1).
  - PEcore's eigenvector advantage seen in development does not hold within
    OOD categories.
- **Post hoc readings**, formed after this opening; these are hypotheses, not
  confirmations:
  - CLIP shows no deficit in three of the four photographic categories, which
    is compatible with selection inflating its LAION deficit.
  - Eigenvector agreement is scale-dependent. VLMs are better at
    between-category structure and not better within categories. This is
    consistent across D1, the dev-test and the OOD descriptives.

**Claim boundary.**

- The evidence comes from the same five participants as development.
- The seal is procedural.
- C2 was rewritten post hoc before the lock.
- No population or causal claim is made.

## 2026-10-13 00:32 UTC — ood_confirmation — scientist approved the single OOD opening

**Approval.** The scientist explicitly approved the opening: "批准开启 OOD".
This entry is the `--approval-record` for the single procedurally sealed run.

**Pre-submission checks.**

- The evaluator's sha256 prefix is `817145a77ddfbf01`, matching the lock.
- No `runs/ood_confirmation/` directory exists, so there has been no prior
  opening.

**Run.** One job runs
`ood_confirmation.py --mode confirm --workers 5`, writing to
`runs/ood_confirmation/20261013-opening01/`.

- Resources: 6 CPUs, 32 GiB, a 2-hour limit and no GPU.
- OOD values are held only in memory.
- The single report is created write-once.
- Nothing is released until it completes.

Submitted with `sbatch --wait` as job **47580993**
([launcher](code/ood_confirmation.sbatch)).

## 2026-10-13 00:00 UTC — configuration_lock — revalidation passed; write-once lock written

**Revalidation run.** Job 47575628 ran on sh03-01n50 and ended
`COMPLETED 0:0` in 4 min 12 s (0.42 allocated CPU-hours; EP22 total 24.60).
It read no OOD response value. Results:

- synthetic tests 4/4 passed;
- `coverage_ok` true;
- reader bit-for-bit identical on sub-03 ses-05 and ses-17;
- gate passes (0.125 against 0.015).

On development pseudo categories:

- **C2a** holds: within-category E_val is −0.014 for CLIP and −0.014 for
  SigLIP2, negative in 5/5 participants for each.
- **C2b** cannot be evaluated: the native all-pairs gap is **positive**
  (CLIP +0.035, SigLIP2 +0.005).
- The retention rule therefore makes **C2b descriptive**
  (`c2b_confirmatory: false`).
- **C3** passes for every VLM.
- **PEcore secondary** passes.
- **C1** for CLIP stays indeterminate.

**Post hoc observation.** E_val is negative at both scales: within category
(CLIP −0.014) and over all pairs (CLIP −0.027). What changes sign between
scales is the eigenvector component: negative within category, positive over
all pairs.

**Lock.** [ood_confirmation_lock.json](ood_confirmation_lock.json) was
written at 2026-10-13T00:00:35Z. Its sha256 is
`efb07381e6c2ed90e14f68337d0fe4c7f311234b09d82822af9f1efe3488b6e1`, and it is
read-only (`r--r-----`). Its status is
`locked_pending_scientist_opening_approval`. It binds:

- the evaluator checksum `817145a7…` and four EP04 module checksums;
- all settings;
- the category table checksum;
- the corrected-embedding ETags;
- the C3 α source;
- the exposure review and the six passed checks;
- `c2b_confirmatory: false`, together with its basis.

The single OOD opening is **not** authorized.

## 2026-10-12 23:48 UTC — confirmation design — C2 rewritten (post hoc, before the lock)

The scientist chose option 1 ("选 1，改写 C2，重跑验证后给我看 lock").

**Post hoc label.** This revision was formed after seeing the dev-test result
above, which compared three options: keep, rewrite, or make descriptive. It
precedes any OOD access and the lock.

C2 is replaced by two parts:

- **C2a, within-category sign.** For CLIP and SigLIP2, the within-category
  eigenvalue component E_val is below 0, both for the mean and in at least
  4/5 participants. The development pseudo-categories showed this in 5/5
  participants for both models.
- **C2b, all-pairs scale.** This mirrors D1. It is computed on all pairs among
  each participant's eligible confirmatory images. For CLIP and SigLIP2,
  imposing DINOv2's eigenvalues on the VLM's eigenvectors must close at least
  half of the native all-pairs gap, for the mean and in at least 4/5
  participants. It can be evaluated only if the mean native all-pairs gap is
  ≤ −0.02.
  - C2b is confirmatory only if the rerun dev-test supports it on development
    pseudo data: closure ≥ 0.5 for both models, meeting the same mean and 4/5
    rule.
  - Otherwise it is reported descriptively. The lock records which case
    applies.

The PEcore secondary prediction (within-category E_val < 0 and E_vec > 0) is
unchanged. So are C1, C3 and the gate.

**Implementation.** The evaluator now also computes all-pairs geometries
(native, swaps and C3) on each participant's eligible confirmatory union. It
reports C2a, C2b and the PEcore secondary through a shared exact-decomposition
helper; C2b is labelled confirmatory from the lock's `c2b_confirmatory`
flag. The synthetic tests were updated for the new keys and now also check
gap = E_val + E_vec exactly.

**Revalidation.** The same three-part validation reruns into
`runs/ood_evaluator_validation/20261012-attempt02/`. Attempt01 is preserved
as superseded: its evaluator checksum no longer matches. The lock script
reads attempt02 and writes the real write-once lock only if every check
passes.

Submitted with `sbatch --wait` as job **47575628**.

## 2026-10-12 23:31 UTC — configuration_lock preparation — validation passed; C2 mismatch found; lock not written

**Run.** Job 47570239 ran on sh03-08n42 and ended `COMPLETED 0:0` in
4 min 4 s (0.41 allocated CPU-hours; EP22 total 24.18). It read no OOD
response value. Outputs are in
[runs/ood_evaluator_validation/20261012-attempt01/](runs/ood_evaluator_validation/20261012-attempt01/).

**Synthetic tests:** 4/4 passed.

**Coverage:**
[ood_coverage_check.json](runs/ood_evaluator_validation/20261012-attempt01/ood_coverage_check.json)
reports `coverage_ok`.

- OOD presentations per participant are 1,484, 1,483, 1,484, 1,484 and 1,484,
  matching EP04's frozen role accounting.
- Every image has at least 2 presentations. The single 3-presentation image is
  sub-03's `shared_4rep_OOD_unusual_bear_fishing.jpg`.
- All 30 sessions per participant contain OOD trials, and every one has
  development trials for standardization.
- TSV row counts match the NIfTI axes.

**Dev-test** (pseudo-OOD development images,
[dev_test_report.json](runs/ood_evaluator_validation/20261012-attempt01/dev_test_report.json)):

- **Reader:** the trusted reader is bit-for-bit identical to EP04's handoff on
  sub-03 ses-05 and ses-17 (631 rows each, same row order).
- **Pseudo targets:** all 371 eligible in every participant.
- **Gate:** passes, 0.125 against a 1,000-permutation threshold of 0.015.
- **C3:** passes for every VLM (+0.040 to +0.050).
- **PEcore secondary:** passes.
- **C1:** CLIP is indeterminate. The pseudo gap is −0.024, but only 3/5
  participants are ≤ −0.02.
- **C2:** fails for both CLIP and SigLIP2. Within categories the eigenvalue
  swap closes only 30% (CLIP) and 22% (SigLIP2) of the gap, against the 50%
  required. E_val is negative in 5/5 participants for both, but E_vec
  dominates within categories. Example: CLIP on the LAION pseudo-category
  "relations" has gap −0.073, E_val −0.021 and E_vec −0.052.

**Post hoc interpretation** (formed after this dev-test and after D1). The
gaps are spectral at the coarse scale: D1 used whole folds of about 942 images,
dominated by between-stratum structure. Within homogeneous categories, VLM
eigenvectors capture fine-grained brain structure worse than DINOv2's. This is
a hypothesis, not a confirmed result.

**Consequence.** The adopted C2 had been justified by D1's whole-fold
evidence. Its estimand, however, is within-category, and at that scale the
development data do not support the 50% closure. All six pre-lock checks are
true. A
[lock preview](runs/ood_evaluator_validation/20261012-attempt01/lock_preview.json)
(`PREVIEW_NOT_A_LOCK`) was generated. The write-once lock was **not**
written, pending a scientist decision on C2.

## 2026-10-12 23:13 UTC — configuration_lock preparation — evaluator implemented; validation selected

**Authority.** The scientist asked for the coverage check and the locked
evaluator ("好，做 coverage check 和锁定版 evaluator"). The lock is written
after validation and shown to the scientist. The single OOD opening still
needs explicit approval.

**Evaluator.** [ood_confirmation.py](code/ood_confirmation.py) runs in one of
two modes. In `confirm` mode it:

- reads OOD trial rows from the trial TSVs;
- streams them with EP04's production trusted reader (`stream_selected`);
- standardizes each session fold-free, using EP04's `training_statistics`
  with no held-out fold, i.e. all development trials of that session;
- applies EP04's validity, repeat and mean-pattern functions;
- computes within-category RSA for native, eigen-swapped and C3-transformed
  geometries;
- evaluates the adopted C1–C3 and the 10,000-permutation gate.

Safeguards in `confirm` mode:

- OOD values stay in memory, and the report is created write-once.
- It refuses to run unless the lock's code checksums (the evaluator plus four
  EP04 modules) and its settings match, the exposure review passed, and an
  approval record is given.
- It refuses to run if a confirmation report already exists.

**Validation job.** [ood_evaluator_validation.sbatch](code/ood_evaluator_validation.sbatch)
uses 6 CPUs, 32 GiB and a 90-minute limit. It reads no OOD response value and
runs three checks:

1. **Synthetic known-answer tests**
   ([test_ood_confirmation.py](tests/test_ood_confirmation.py)): a spectrum-only
   difference gives "spectral" and passes C2; identical models give
   "selection_only" with C2 not evaluable; a random brain fails the gate; the
   null is deterministic.
2. **Outcome-blind coverage** ([ood_coverage_check.py](code/ood_coverage_check.py)):
   counts OOD presentations per image from the 150 trial TSVs, checks the TSV
   against the NIfTI axis, checks that development trials support each session
   with OOD trials, and compares the totals with EP04's frozen role accounting.
   That accounting lists 1,484 OOD trials for each participant except sub-03,
   which has 1,483.
3. **Dev-test:** the full pipeline on pseudo-OOD development images. Category
   sizes mirror OOD: confirmatory 60/72/30/64, descriptive 8/10/24/21/82. It
   includes a bit-for-bit reader check against EP04's handoff on sub-03 ses-05
   and ses-17, and uses 1,000 permutations.

Submitted with `sbatch --wait` as job **47570239**.

## 2026-10-12 23:01 UTC — ood_exposure_review — passed (outcome-blind)

The full record is
[exposure_review.json](runs/ood_exposure_review/20261012-review01/exposure_review.json).
The review read no neural value. It checked six scopes:

- **EP22 itself:**
  - D0 read only embeddings and identity metadata.
  - D1–D3 read only EP04's saved development patterns. For every participant,
    all 4,712 pattern rows are `unique_*` images; none is shared or OOD.
- **EP04, current contract:** 271 of 273 run-report JSONs were scanned. Every
  flag named OOD, audit or shared is false, for example `OOD_outcomes_opened`,
  `audit_opened` and `excluded_regular_shared_or_ood_images_read`. The logs
  agree.
- **EP04's earlier round:** this was a caption–LLM reproduction on the
  regular-shared images.
  - The legacy scratch `outputs/` holds only `validate_neural_inputs.py` and
    its test, with no result files.
  - That validator reads metadata, split JSON, trial TSVs and NIfTI headers
    only, and it explicitly excludes the 371 OOD images.
- **Canonical BR (read-only):**
  - The EP04 loop `arl_0c0059fbea38a5b1889532163cf34d7e` stopped at
    REVIEWING, with 0 real executions.
  - Of the 2,508 runs in the last 60 days, 2,489 are research-logging
    snapshots and 12 are `pipeline_execute`. None of the latter is a LAION-fMRI
    operation.
  - Two memory searches returned 16 nearest cards. An outcome-blind keyword
    scan found none that mentions LAION or OOD.
- **EP21:**
  - Its contract closes every OOD derivative.
  - Its stimulus and neural exports are regular-only. The neural export
    record states that excluded bytes were discarded and that no OOD numerical
    arrays were formed.
  - No output file name contains "ood".
- **Steward acquisition and public sources:**
  - The acquisition was checked with whole-file integrity hashes only.
  - No published OOD brain–model result was consulted.

**Disclosed boundaries.**

- The readers are trusted readers that traverse excluded compressed bytes;
  this is not a byte seal.
- Separation is procedural, within the same account.
- A documentary review cannot rule out access that was never recorded.

**Conclusion.** No evidence of OOD outcome exposure was found. The 371 OOD
images are eligible as EP22's procedurally sealed one-shot confirmation set.
`DATA_LOCATION_MANIFEST.json` is updated: the EP22 asset association and the
binding state.

**Remaining before the lock:**

1. An outcome-blind coverage check of the OOD trial rows and valid
   presentations, from trial metadata only.
2. The locked evaluator and its code checksum.
3. The write-once lock.

The lock and the OOD opening are not yet authorized.

## 2026-10-12 22:55 UTC — confirmation design — scientist adopted the restated predictions; exposure review authorized

The scientist agreed to all four proposed items ("同意这四项，开始做 OOD
exposure review"). They are now the agreed confirmation design. They are
frozen only at the write-once lock, and they are recorded in
[SEARCH_POLICY.yaml](../SEARCH_POLICY.yaml) and [GOAL.md](../GOAL.md).

1. **C1 is restated as a discriminating test** of spectrum versus selection on
   the confirmatory photographic OOD categories:
   - a native CLIP − DINOv2 gap ≤ −0.02 supports the spectral account;
   - a gap > −0.02 supports the selection-only account.
2. **C2 is extended to CLIP.** For CLIP and SigLIP2, imposing DINOv2's OOD
   eigenvalues closes at least half of the native OOD gap. PEcore adds a
   secondary sign prediction: E_val < 0 and E_vec > 0.
3. **C3 is kept.**
4. **Confirmatory categories are restricted to those with at least 20 images:**
   cropped (60), relations (72), selfmade (30) and unusual (64), weighted
   equally. illusion-natural (8) and the synthetic types are reported
   descriptively.

The outcome-blind OOD exposure review is authorized and starts now. The lock
and the OOD opening are not yet authorized.

## 2026-10-12 22:18 UTC — d1/d2/d3 — figure re-rendered (collector job 47560472)

The panel-A legend overlapped the CLIP bar, so the legend was moved to the
top-left. Collector job 47560472 reran the deterministic aggregation
(`COMPLETED 0:0`, 30 s, 0.008 CPU-hours). The summary values are unchanged;
only `created_utc` differs. EP22 now totals 23.77 allocated CPU-hours.

## 2026-10-12 22:05 UTC — d1/d2/d3 — completed (array 47548992, collector 47549215)

**Run.** Array job 47548992 completed 25/25 tasks with `COMPLETED 0:0`. Each
took 11–20 min, at most 8 at a time, ending about 22:02 UTC. Collector job
47549215 completed in 25 s.

| Job | Allocated CPU-hours |
| --- | ---: |
| Array 47548992 | 23.08 |
| Collector 47549215 | 0.01 |
| Test cell 47547277 | 0.45 |
| D0 47545539 | 0.21 |
| **EP22 total so far** | **23.76 of 400** |

**Checks.**

- R0 reproduced EP04 in all 400 rows; the maximum |Δz| is 5.6e-17.
- ASTRA validation passed.

**Outputs.**

- [d123_summary.json](runs/d123_development/20261012-attempt02/summary/d123_summary.json)
- [Figure](figures/d123_development/20261012-attempt02/d123_results.png)
- [Per-cell values](runs/d123_development/20261012-attempt02/cells/)

All values below are exploratory development evidence: equal weights over
folds, then primary sectors, then participants.

**D1 — exact split of the native gap to DINOv2.** Brackets give the 95%
image-bootstrap interval of the five-participant mean.

| Model | Native gap | E_val (spectrum) | E_vec (eigenvectors) | Participants |
| --- | ---: | ---: | ---: | --- |
| CLIP | −0.035 | −0.045 [−0.046, −0.044] | +0.011 [+0.007, +0.013] | E_val dominates in 5/5 |
| SigLIP2 | −0.039 | −0.038 [−0.039, −0.038] | −0.001 [−0.003, +0.001] | E_val dominates in 5/5 |
| PEcore | +0.001 | −0.038 | +0.039 | Both signs hold in 5/5 |

- PEcore's parity with DINOv2 is a cancellation. It carries the same spectral
  penalty as the other VLMs, offset by eigenvectors that align with the brain
  better than DINOv2's.
- EVC (descriptive) shows smaller gaps (−0.004 to −0.010), mostly from
  eigenvalues.
- Given DINOv2's eigenvalue profile, every VLM's eigenvectors match the brain
  at least as well as DINOv2's do. EP04's native-RSA ranking reflects the
  models' spectra, not their eigenvectors.

**D2 — held-out RSA at matched flexibility.** Each rung is applied to every
model. Every sign below holds in 5/5 participants.

| | R0 native | R1 exponent | R2 64 axes | R3 encoding geometry |
| --- | ---: | ---: | ---: | ---: |
| DINOv2 RSA | 0.129 | 0.234 | 0.258 | 0.532 |
| CLIP − DINOv2 | −0.035 | −0.098 | −0.081 | +0.013 |
| PEcore − DINOv2 | +0.001 | −0.058 | −0.039 | +0.019 |
| SigLIP2 − DINOv2 | −0.039 | −0.076 | −0.047 | +0.024 |

- Every model prefers a steeper spectrum than its native one. The selected
  exponents, all interior to the widened 0–4 grid, are:
  - DINOv2 2.1–2.3
  - CLIP 1.7–1.8
  - PEcore 1.8–2.1
  - SigLIP2 1.6–2.1
- Truncating to the leading 256 training axes alone raises every model's RSA:

  | Model | R0 | Truncated |
  | --- | ---: | ---: |
  | DINOv2 | 0.129 | 0.155 |
  | CLIP | 0.094 | 0.111 |
  | PEcore | 0.130 | 0.145 |
  | SigLIP2 | 0.090 | 0.119 |

- Reweighting within each model's own principal axes (R1, R2) helps DINOv2
  most, so the gap widens. Only a full linear readout (R3) closes it, and
  there every VLM is slightly ahead of DINOv2. This matches EP04's encoding
  parity.
- The brain-relevant dimensions are present in all four spaces. DINOv2
  concentrates them in its leading principal axes, whereas the VLMs spread
  them across axes that no diagonal reweighting can isolate.

**D3 — own-selection effect O(s) on native RSA.** Computed on LAION
evaluation images.

| Selector | O(s) | Sign consistency | Selector's ED change |
| --- | ---: | --- | ---: |
| DINOv2 | −0.039 | negative in 5/5 | ×1.39 |
| PEcore | −0.008 | negative in 5/5 | ×1.15 |
| CLIP | +0.011 | negative in 0/5 | ×1.18 |
| SigLIP2 | +0.004 | negative in 0/5 | ×1.19 |

With DINOv2 as selector, its own RSA dropped 0.061, against 0.016–0.025 for
the other models.

- The mechanism exists: ED-maximizing selection in a model's own space lowers
  that model's fixed-geometry RSA. For DINOv2 the drop exceeds the whole
  CLIP–DINOv2 gap.
- CLIP and SigLIP2 show no own-selection effect on these LAION images. That is
  consistent with saturation from the provider's CLIP-space selection, but it
  does not demonstrate it.

**Interpretation (exploratory).**

- **S, spectrum:** this explains EP04's native-RSA gaps.
- **V, eigenvectors:** this appears in the leading axes once steeper weighting
  is allowed. Information parity holds only for full linear readouts.
- **C, selection:** the mechanism is real and large for a non-CLIP selector.
  Its role in CLIP's LAION-specific gap remains indirect.
- The brain's preferred exponent is about 2 for all four models. Fixed-weight
  RSA therefore weights brain geometry toward fewer dimensions than any
  model's native spectrum. fMRI voxel pooling may contribute to this.

**Implications for the sealed confirmation.** These need a scientist decision
before the lock.

- **C2:** supported in development for SigLIP2 (98% of its gap from
  eigenvalues) and CLIP (130%). Proposal: extend C2 to CLIP.
- **C1 as stated:** conflicts with D0 + D1. OOD spectra differ even more than
  LAION spectra, so the spectral account predicts a CLIP deficit on
  photographic OOD images, while the selection-only account predicts none.
  Proposal: restate C1 as this discriminating test, with C2 decomposing the
  result.
- **C3:** supported, since α̂ > 1 for every model in 25/25 cells.
- **PEcore:** gives a distinctive OOD prediction, E_val < 0 together with
  E_vec > 0.
- **Category weighting:** illusion-natural has n = 8; its weighting is still
  open.

## 2026-10-12 20:49 UTC — d1/d2/d3 — representative cell passed; grids widened; full run selected (attempt02)

**Representative cell.** Job 47547277_0 (sub-01, fold 0) ran on sh04-01n14
and ended `COMPLETED 0:0`: 6 min 44 s, MaxRSS 3.4 GB, 4 CPUs.

- **R0 reproduction:** all 16 evaluable rows match EP04's observed native
  fisher z; the maximum absolute difference is 5.6e-17.
- **D1:** native geometry equals R0 exactly.
- **Timing:** D1 185 s, D2 181 s, D3 13 s.

The cell is preserved at
[attempt01/cells/sub-01-fold-0.json](runs/d123_development/20261012-attempt01/cells/sub-01-fold-0.json).
It is superseded for aggregation because its α and λ grids were too narrow
(below). Its single-cell values are not interpreted.

**Grid defect and fix.** Both are open exploratory settings in a policy that
is not frozen.

- DINOv2's R1 exponent was selected at the upper bound, α = 2.0, so the
  optimum may lie beyond the grid. The α grid is widened to 0–4 in steps of
  0.1.
- R2's split-half-validated λ was often the largest value, 1.0. The λ grid
  is widened to {0, 1e-3, 1e-2, 1e-1, 1, 10, 100}.

These changes only fix optimizer bounds. The estimands are unchanged.

**Full run.** All 25 cells are rerun with the widened grids into
`runs/d123_development/20261012-attempt02/`.

- The array runs at most 8 tasks at once (`%8`, 32 cores), matching the
  policy's `max_parallel_cores`.
- A dependent collector (`afterany`) aggregates the cells and renders
  `figures/d123_development/20261012-attempt02/`. It also re-renders the D0
  figure, changing only the tick labels, and validates ASTRA.
- Estimated cost is about 25 × 9 minutes × 4 CPUs, roughly 15 allocated
  CPU-hours.

**Submitted at 20:50 UTC.**

- The array is job **47548992** (`--array=0-24%8`).
- The first collector submission, 47548994, was cancelled before it started.
  It was resubmitted as **47549215** with `sbatch --wait`
  (`--dependency=afterany:47548992`), so completion arrives as a notification
  instead of by polling.

## 2026-10-12 20:39 UTC — d1_eigen_factorial, d2_flexibility_ladder, d3_selection_intervention — representative cell selected

One runner, [d123_cell.py](code/d123_cell.py), computes D1–D3 for a single
participant × held-out fold. The launcher is
[d123_cells.sbatch](code/d123_cells.sbatch): array tasks 0–24, each with
4 CPUs, 16 GiB and a 1-hour limit.

**Inputs, all read-only from EP04's closed artifacts:**

- development embedding handoff;
- saved per-fold mean patterns and image validity in
  `$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/neural_preprocessing01/`;
- EP04's observed native panel, used only for the R0 reproduction check.

Image support is the same as EP04's:

- evaluation: held-fold images that are primary-eligible;
- training: primary-eligible images from the other folds (EVC rows also
  require EVC eligibility).

**Engineering choices, within the policy's open exploratory choices:**

- **D1:** 100 whole-image bootstrap replicates of the evaluation images.
- **D2 R1:** α grid from 0 to 2 in steps of 0.1, applied in the leading-256
  training-PCA basis. α is selected on mean primary-sector training RSA.
- **D2 R2:** non-negative weights on the leading 64 training axes, fit to
  correlation distances for 500k training pairs. λ ∈ {0, 1e-3, 1e-2, 1e-1, 1}
  is chosen by split-half validation within the training images.
- **D2 R3:** EP04's ridge settings (normalized λ = 0.001, training-mean
  intercept), followed by the correlation-distance geometry of the predicted
  responses.
- **D3:** a greedy participation-ratio maximization over half the fold's LAION
  evaluation images, in each model's own 64-d PCA space, starting from 10
  seeded images shared by all selectors. Baselines are 20 random half-subsets.

**Test before scale-up.** Run task 0 (sub-01, fold 0) first. The runner exits
with status 3 unless R0 reproduces EP04's observed native fisher z for every
evaluable row of the cell within 1e-8.

Task 0 was submitted with `sbatch --wait --array=0` as job **47547277**.

## 2026-10-12 20:38 UTC — d0_outcome_blind_spectra — completed (job 47545539)

**Run.** Job 47545539 ran on sh03-01n36 from 20:28:40 to 20:31:46 UTC and
ended `COMPLETED 0:0`: 3 min 13 s, MaxRSS 2.1 GB, 4 CPUs, about 0.21
allocated CPU-hours. Outputs:
[results](runs/d0_outcome_blind_spectra/20261012-attempt01/results/) and
[figure](figures/d0_outcome_blind_spectra/20261012-attempt01/d0_outcome_blind_spectra.png).
No neural value was read.

**Binding — passed.** See
[ood_embedding_binding.json](runs/d0_outcome_blind_spectra/20261012-attempt01/results/ood_embedding_binding.json).

- The four corrected files were downloaded with their ETag, Last-Modified
  (2026-09-28 09:52:06 GMT) and size recorded. They are now read-only in
  `inputs/laion_ood_embeddings_corrected/`. The group was reset from `nobody`
  to `oak_russpold` after the cross-filesystem move.
- Each model changed exactly the same 116 rows, all OOD: cropped 60,
  illusion-classic 21, shape 35.
- The other 255 OOD rows, and all 24,681 non-OOD rows, are bitwise identical
  to the 2026-05-12 files.
- All 371 OOD rows are valid.
- EP04's development handoff equals the current release.
- The [category table](runs/d0_outcome_blind_spectra/20261012-attempt01/results/ood_category_table.csv)
  flags the corrected rows.

**Spectra.** The comparison measure is log(ED_VLM / ED_DINOv2) on random
128-image subsets, 200 per stratum. Values at 512 images agree in sign.

- All three VLMs have flatter spectra than DINOv2 in every stratum.
- The VLM excess is larger on LAION than on THINGS or THINGSplus. LAION minus
  THINGS is positive in 5/5 participants for every VLM:

  | Model | LAION − THINGS, per participant | Pooled LAION − THINGS | Pooled LAION − THINGSplus |
  | --- | --- | ---: | ---: |
  | CLIP | +0.17 to +0.25 | +0.21 | +0.18 |
  | PEcore | +0.25 to +0.35 | +0.29 | +0.22 |
  | SigLIP2 | +0.16 to +0.23 | +0.20 | +0.20 |

- The OOD images were not selected in CLIP space, yet they show an even
  larger VLM excess than LAION:

  | Model | OOD photographic | OOD synthetic | LAION |
  | --- | ---: | ---: | ---: |
  | CLIP | 0.55 | 0.73 | 0.41 |
  | PEcore | 0.50 | 0.59 | 0.39 |
  | SigLIP2 | 0.56 | 0.48 | 0.35 |

  This happens mainly because DINOv2's ED collapses there (31 and 13 against
  roughly 51 on LAION).
- Cross-model agreement with DINOv2 is lowest on LAION. Linear CKA is
  0.70–0.72 on LAION against 0.75–0.79 elsewhere. Leading-10 subspace overlap
  is about 0.39 on LAION against about 0.46 on THINGS, with a chance level of
  0.08.

**Interpretation, exploratory.**

- The LAION-versus-THINGS pattern fits CLIP-space selection flattening every
  CLIP-like VLM, not only CLIP.
- Relative ED is also strongly content-dependent, so on its own it cannot
  isolate selection. D3's intervention is needed.
- PEcore's relative flattening on LAION is as large as CLIP's, yet its LAION
  RSA gap is only −0.016, against CLIP's −0.054. So relative ED does not
  predict the size of the RSA gap across models.
- C1 is affected. If D1 shows that the gaps are eigenvalue-driven, C1's
  premise of "no deficit on non-selected images" conflicts with the larger
  OOD spectral difference. C1 should then be restated in spectral terms before
  the lock.

**Checks.** ASTRA schema and semantic validation passed. The YAML parse check
could not run, because the ASTRA venv has no PyYAML module; this is not a
gate.

## 2026-10-12 20:29 UTC — d1_eigen_factorial, d2_flexibility_ladder, d3_selection_intervention — authorized

The scientist asked to run D1–D3 after D0 ("D0 跑完后继续跑 D1–D3"). They use
only development responses for the 23,560 subject-unique images, which EP04
already exposed, so their results are exploratory development evidence. OOD
responses stay closed, and no lock or OOD opening is authorized. Implementation
reuses EP04's frozen development definitions read-only, as
[DATASETS](../DATASETS.md#reused-frozen-ep04-definitions-read-only) describes.
D0 job 47545539 started at 20:28:40 UTC on sh03-01n36.

## 2026-10-12 20:26 UTC — d0_outcome_blind_spectra — selected; submitted as job 47545539

Submitted at 20:27 UTC with `sbatch --wait` as Slurm job **47545539**. No
outcome has been observed yet.

**Authority.** On 2026-10-12 the scientist adopted the default procedural OOD
separation and authorized D0 ("同意默认封存，开始跑 D0"). C1–C3 remain
proposed, not frozen. No other analysis is authorized by this entry.

**Selected work.** D0 opens no neural values. It reads only:

- EP04's frozen development embedding handoff
  (`20261003-development_embeddings01`, read-only), covering 23,560 rows × 4
  models;
- provider image names and roles from `task-images_metadata.csv`;
- the original (2026-05-12) and corrected (2026-09-28) provider embedding
  files.

The job runs four steps:

1. **Acquire** the four corrected files (public CC0) from
   `https://laion-fmri.s3.us-west-2.amazonaws.com/stimuli/` into
   `inputs/laion_ood_embeddings_corrected/`, then make them read-only. A HEAD
   check on 2026-10-12 showed Last-Modified 2026-09-28 09:52:06 GMT, with ETags
   and sizes that differ from the 2026-05-12 acquisition manifest.
2. **Bind** the corrected rows. Check that changed rows are OOD only, count
   them against the provider's 116, confirm the other 255 OOD rows and all
   non-OOD rows are identical, and confirm the EP04 development handoff equals
   the current release.
3. **Compute spectra.** Draw size-matched subsets (128 images, 200 per stratum;
   512 for pooled development strata). The strata are each participant ×
   {LAION, THINGS, THINGSplus}, pooled development strata, and OOD
   photographic / synthetic / all. Measures:
   - ED and relative ED versus DINOv2;
   - top-eigenvalue share;
   - spectral decay;
   - nearest-neighbour spacing CV;
   - linear CKA;
   - subspace overlap at k = 10 and 32.
4. **Render** the figure from the saved summary, then run ASTRA and YAML
   parse checks. These checks are not gates.

**OOD category table (outcome-blind, from provider image names).** 371 images:

| Photographic | Images | Synthetic | Images |
| --- | ---: | --- | ---: |
| cropped | 60 | gabor | 10 |
| illusion-natural | 8 | gaudy | 24 |
| relations | 72 | illusion-classic | 21 |
| selfmade | 30 | shape | 82 |
| unusual | 64 | | |
| **Total** | **234** | **Total** | **137** |

All 131 PNGs are in cropped (60), illusion-classic (21) and shape (50).
Design note for the lock: illusion-natural has only 8 images, so equal
category weights would make it disproportionately noisy. Decide a minimum
category size or pair weighting before the lock, without looking at outcomes.

**Resources.** One job: 4 CPUs, 16 GiB, 30 minutes, no GPU. Code:

- [d0_outcome_blind_spectra.py](code/d0_outcome_blind_spectra.py)
- [d0_outcome_blind_spectra.sbatch](code/d0_outcome_blind_spectra.sbatch)
- [plot_d0_spectra.py](code/plot_d0_spectra.py)

Outputs go to `runs/d0_outcome_blind_spectra/20261012-attempt01/`, and the
figure to `figures/d0_outcome_blind_spectra/20261012-attempt01/`.
