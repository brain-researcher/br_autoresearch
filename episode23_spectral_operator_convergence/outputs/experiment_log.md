# EP23 experiment log

Newest entries first. Only operator feasibility checks, which use no model or
neural data, have been executed. An ASTRA program will be authored when
execution is scoped.

## 2026-10-14 17:10 UTC — a1 — rank-32 sensitivity launched

- Submitted RUN=20261014-rank32-attempt01 with RANK=32: array 47793391 (5 tasks, one participant each running its 5 folds in sequence, so each job lasts over 10 min) and collect 47793396 (`afterok`).
- New launcher [a1_metric_family_grouped.sbatch](code/a1_metric_family_grouped.sbatch), sized from the rank-64 cells (≤10.3 min, ≤1.5 GB each); analysis code and settings otherwise unchanged.
- Next: compare both contrasts with rank 64, then the stage figure and three-line summary.

## 2026-10-14 03:15 UTC — a1 — rank-64 complete

- Ran: A1 array 47703628 (25 participant × fold cells, rank 64) and collect 47703629, all `COMPLETED`; [a1_report.json](runs/a1_metric_family/20261014-attempt01/a1_report.json).
- Primary, full − diagonal held-out Δz: +0.19 mean over models (DINOv2 0.13 to CLIP 0.23), 0.17–0.23 per participant; positive 5/5 for every model; exact sign p = 0.031 (the n = 5 floor).
- Mean held-out RSA z, primary ROIs: reference 0.11, native 0.15, power 0.18, diagonal 0.21, full 0.41. EVC: 0.04, 0.04, 0.04, 0.05, 0.13.
- Secondary, power − native: +0.023 (0.010–0.046), 5/5, p = 0.031. Selected α: CLIP 1.5–1.6, DINOv2 1.9–2.1, PEcore 1.8–2.1, SigLIP2 1.2–1.9; never at a grid edge.
- λ0 = 0 (no shrinkage toward native) chosen in 100/100 primary model-cells for both fitted families; this is the grid's lower limit, not a truncation.
- Reading: held-out brain geometry needs rotation within the models' top-64 PCs, not only per-axis reweighting. Per P4 this is predictive sufficiency, not native geometry. Exploratory, development data.
- Next: rank-32 sensitivity (same code, RANK=32, new RUN), then the stage figure and summary.

## 2026-10-13 23:40 UTC — p1_p5, a2, a3 — complete; A1 running

**Execution.** All jobs `COMPLETED` with exit 0:

| Stage | Job | Wall time |
| --- | --- | --- |
| Theory checks | 47703623 | 14 s |
| A2 array | 47703624 | 18–21 s per task |
| A2 collect | 47703625 | 15 s |
| A3 array | 47703626 | 26–30 s per task |
| A3 collect | 47703627 | 13 s |

**P1–P5 refutation attempts**
([theory_checks.json](runs/p1_p5_theory/20261014-attempt01/theory_checks.json)).
All 17 checks pass. Passing is not proof.

- **Inequalities.** Maximum left-hand side / bound:

  | Check | Max ratio |
  | --- | ---: |
  | Yu–Wang–Samworth sinΘ | 0.31 |
  | Mixed gap | 0.69 |
  | Two learners | 0.29 |
  | Weyl | 0.66 |
  | Distance bound | 0.91 |
  | Haar mean deviation, relative to its Hoeffding threshold | 0.09 |

- **Exact checks.** The identities and explicit counterexamples hold to
  ≤7×10⁻⁶ relative tolerance. The kNN margin produced no mismatches. No
  random competitor beat the P2 optimum (maximum loss ratio 0.998).

**A2, coarse versus fine**
([a2_report.json](runs/a2_coarse_fine/20261014-attempt01/a2_report.json)).

- **A2a, model–model** (pooled; 19 categories, r = 8, 1,138 images;
  noise-free; descriptive):

  | Pairs | κ_coarse | κ_fine |
  | --- | --- | --- |
  | VLM–VLM | 0.80–0.85 | 0.51–0.61 |
  | VLM–DINOv2 | 0.58–0.66 | 0.21–0.23 |

  κ_coarse − κ_fine is 0.23–0.43 for all 6 pairs. Fine structure is less
  shared between models, especially between VLMs and DINOv2.
- **A2b, model–brain** (C = 7–10, r = 3–4):
  - **Raw κ:** coarse exceeds fine in 5/5 participants (Δκ = +0.14 to
    +0.41; sign p = 0.031).
  - **Frozen primary, reliability-relative ρ_coarse − ρ_fine:** −0.36,
    −0.20, +0.48 and −0.17 for sub-03, 05, 06 and 07. Positive in only 1/4,
    sign p = 0.94, so it is **not supported**.
  - **sub-01 is null.** Its coarse self-consistency is not significant in
    any primary ROI (0/12 reliable cells).
  - **Reliability by scale.** Brain coarse self-κ is large but imprecise:
    0.10–0.76, often not significant with only 7–10 categories. Fine self-κ
    is small (0.03–0.16) but always significant.
  - **Reading.** The raw brain coarse advantage tracks the brain's own
    scale-specific reliability. This matches the "measurement ceiling only"
    account for brains.
  - **Caveat.** The ratios are noisy because the denominators are small.

**A3, pooling simulations**
([a3_report.json](runs/a3_pooling_simulation/20261014-attempt01/a3_report.json)).
These are synthetic and support no biological claim. In every case the
identity measurement gives α\* = 1.005 (the sanity check).

| Topography | Mixing fraction | Attenuation | α\* | Top-8 subspace overlap |
| --- | --- | --- | --- | --- |
| Frequency-ordered DCT | 0 | Spearman −0.999 | Rises with pooling width: 1.01 (σ = 0, downsampled), 1.03, 1.06, 1.18 (σ = 4) | — |
| Shuffled DCT | — | Uncorrelated with mode index | Stays at 0.99–1.09; at σ = 4, sd = 0.6 | — |
| White random | Rises to 0.58 | — | Stays at 0.99–1.01 | Falls to 0.66 |
| Smooth random | Rises to 0.42 | — | Rises slightly to 1.06 | — |

- **Topography dependence.** The range of α\* across topographies grows from
  0.005 to 0.185 with σ.
- **Verdict.** Pooling steepens the preferred power only when topography
  orders modes by spatial frequency. Even then it reached only about 1.18
  here, not α ≈ 2. This is a counterexample to a universal pooling exponent
  under these settings.

## 2026-10-13 22:20 UTC — p1_p5, a1, a2, a3 — deliverables reviewed; launching (RUN=20261014-attempt01)

**Review.** The orchestrator read all four deliverables in full.

- [propositions.md](theory/propositions.md), P1–P5, 693 lines.
  - Every explicit example was re-derived by hand:
    - the 3×3 reordering matrix, with eigenvalues 1, 1/5 and −3/5;
    - the nonsymmetric TTᵀ case;
    - the observation-compression example;
    - the entrywise-log commutator;
    - the P2 losses;
    - the block model T₀ with β = 1/2 and fine eigenvalues ±1/20;
    - the kNN counterexample;
    - the ridge, rank-truncation and rank-deficient OLS cases;
    - the 2×2 blur mixing, (1/9)[[20,8],[8,5]];
    - the Rademacher non-PSD cross-Gram.
  - The proof steps of the Yu–Wang–Samworth, Sylvester/mixed-gap and Haar
    expectation arguments are standard and correct.
  - Applied without edits.
- [theory_checks.py](code/theory_checks.py): seeded refutation attempts.
  Applied without edits.
- [a2_coarse_fine.py](code/a2_coarse_fine.py): matches the frozen A2
  specification. Fix: generic input SHA-256 hashing removed.
- [a1_metric_family.py](code/a1_metric_family.py): reuses EP22's loaders and
  RSA helpers. The closed forms check out:
  - Q = ZᵀG̃Z, computed through the factored target;
  - the diagonal and ridge-toward-native full metrics, the latter with PSD
    projection;
  - power reweighting as Z·d^(α−1).

  Inner selection uses the interleaved training-fold groups. Applied without
  edits; leftover EP22 constants are unused.
- [a3_pooling_simulation.py](code/a3_pooling_simulation.py): matches the
  frozen A3 grid of 4 topographies × 5 measurements (identity, then σ = 0, 1,
  2 and 4 with 4×4 downsampling) × 2 noise levels × 20 seeds. Applied without
  edits.

**Launch.** Record first, then submit:

- theory checks: 1 CPU, 20 min;
- A2: array of 6 × 2 CPUs, 1 h, reusing A0's brain patterns;
- A3: array of 4 × 1 CPU, 1 h;
- A1: array of 25 × 2 CPUs, 1.5 h, at rank 64;
- collect jobs that depend on each array.

The rank-32 A1 sensitivity follows after the rank-64 cells finish.

**Submitted (RUN=20261014-attempt01):**

| Stage | Job | Collect job |
| --- | --- | --- |
| Theory checks | 47703623 | — |
| A2 | 47703624 | 47703625 (`afterok`) |
| A3 | 47703626 | 47703627 (`afterok`) |
| A1, rank 64 | 47703628 | 47703629 (`afterok`) |

## 2026-10-13 21:20 UTC — p1_p5, a1, a2, a3 — initial stage continued; A1–A3 frozen; implementation delegated

**Authorization.** The scientist asked "shall we continue from here?" The
initial-stage budget (64 CPU-hours, no GPU, proofs drafted by Codex) was
approved on 2026-10-13. EP23 has no concurrent owner: the newest log entry and
file times are this session's, and `squeue` shows only EP21 jobs.

**Freeze.** `frozen_specification` blocks were added for A1, A2 and A3 in
[SEARCH_POLICY.yaml](../SEARCH_POLICY.yaml) before any of their outcomes
existed.

- **A1:** closed-form ridge-toward-native full metric and diagonal metric in
  the training PCA span, with inner selection on interleaved training folds.
- **A2:** chance-corrected κ, and reliability-relative ρ that applies only
  where brain self-consistency is significant.
- **A3:** four topographies × four pooling widths × two noise levels × 20
  seeds.
- **Timing caveat.** A2's frozen form (calibration, reliability rule) was
  written after the A0 results. The A2 outcomes themselves are unseen.

**Delegation.** Four parallel Codex `gpt-6-astra` (`max`, text-only) tasks:

| Task | Deliverable |
| --- | --- |
| J | P1–P5 propositions with proofs and counterexamples, plus a numerical refutation-check script |
| K | A2 script |
| L | A3 script |
| M | A1 script |

The orchestrator reviews every deliverable before any Slurm run.

**A4.** The independent source still awaits a scientist decision.

**Observed failure, then fix (engineering).**

- **Failure.** The first two launches of tasks J–M (4 concurrent, then 2
  concurrent) exited immediately with code 135 and empty logs.
- **Cause.** `$HOME` was at its 15.0/15.0 GB quota (`sh_quota`: "quota
  exceeded"). `~/.codex/sessions` held 5.43 GB of old Codex session logs.
- **Fix.** The scientist chose to delete Codex session logs older than 14
  days: 2,044 files dated 2026-03-05 to 2026-09-28. The 619 more recent
  files were kept.
- **Result.** `$HOME` is now 10.0/15.0 GB. Tasks J–M were relaunched two at
  a time, and have started normally.

## 2026-10-13 09:35 UTC — a0_operator_alignment — attempt01 complete: operator predicts brain subspaces 5/5, but not beyond taxonomy for VG; models align beyond taxonomy

**Execution.** All jobs `COMPLETED` with exit 0:

| Stage | Job | Wall time |
| --- | --- | --- |
| Audited operators | 47630147 | 54 s |
| Alignment, participant tasks 0–4 | 47630149 | 3 min 27 s – 4 min 7 s each |
| Alignment, pooled task 5 | 47630149 | 11 min 13 s |
| Collect | 47630151 | 21 s |

- ASTRA schema and semantic validation passed
  ([astra_validation.json](runs/a0_operator_alignment/20261013-attempt01/astra_validation.json)).
- Report: [a0_report.json](runs/a0_operator_alignment/20261013-attempt01/a0_report.json);
  summary: [a0_summary.txt](runs/a0_operator_alignment/20261013-attempt01/a0_summary.txt).
- The audited rebuild gives VG S = 706 and captions S = 893 after
  excluding `orange` and `baseball`.
- Participant concept sets: VG 192–205; captions 238–264.
- **Accounting.**
  - A0 used about 2.41 allocated CPU-hours across jobs 47629806, 47629807,
    47630147, 47630149 and 47630151.
  - Feasibility and downloads used about 0.13.
  - The EP23 total is therefore about 2.55 of 64 CPU-hours, with no GPU.

**Evidence status.** This is exploratory, on exposed LAION development data.
Two operators × five statistics were evaluated. The prespecified primary
cell is VG / `primary`; the taxonomy residual was named in advance as the
decisive control. No explanation below was formed before these results,
except where stated.

**Brain: participant summaries** (mean of 3 primary sectors, q = 8; Δ = ω −
null mean; per-participant p from 1,000 permutations):

| Operator / statistic | Δ for sub-01, 03, 05, 06, 07 | k positive | Exact sign p | Participants with p < .05 |
| --- | --- | --- | --- | --- |
| VG primary | 0.054, 0.053, 0.028, 0.052, 0.040 | 5/5 | 0.031 | 5 |
| VG frequency-preserving | 0.047, 0.051, 0.029, 0.046, 0.040 | 5/5 | 0.031 | 5 |
| VG minus low-level visual | 0.051, 0.033, 0.024, 0.036, 0.030 | 5/5 | 0.031 | 5 |
| **VG minus taxonomy** | 0.018, 0.008, −0.000, 0.016, −0.001 | 3/5 | 0.50 | 2 |
| VG minus taxonomy and visual | 0.021, 0.000, 0.006, 0.018, 0.006 | 5/5 | 0.031 | 2 |
| Captions primary | 0.047, 0.041, 0.027, 0.051, 0.042 | 5/5 | 0.031 | 5 |
| Captions minus taxonomy | 0.016, 0.018, 0.013, 0.004, 0.006 | 5/5 | 0.031 | 3 |
| Captions minus taxonomy and visual | 0.021, 0.016, 0.013, 0.001, 0.002 | 5/5 | 0.031 | 3 |

- VG primary ω = 0.067–0.095 against null means of about 0.04.
- **By sector** (VG, participant means):

  | Sector | Primary ω | Taxonomy-residual ω | Self-consistency |
  | --- | ---: | ---: | ---: |
  | Ventral | 0.101 | 0.063 | 0.196 |
  | Lateral | 0.085 | 0.058 | 0.174 |
  | Dorsal | 0.072 | 0.050 | 0.096 |
  | EVC (descriptive) | 0.060 | 0.052 | 0.205 |

  The primary null is 0.041 and the taxonomy-residual null 0.049.
  Self-consistency is the split-half q = 8 overlap.
- **Bands** (primary ROIs; 15 participant × sector cells):

  | Modes | ω | Null | Cells with p < .05 |
  | --- | ---: | ---: | ---: |
  | 1–8 | 0.086 | 0.041 | 14 |
  | 9–16 | 0.043 | 0.041 | 0 |
  | 17–32 | 0.083 | 0.081 | 1 |

- **Sensitivities hold** (cells with p < .05 out of 15): q = 4, 14; q = 16, 15;
  full-mean Gram, 14; direct VG operator, 15.

**Models (descriptive).** Pooled over participants: 706 VG concepts and
1,045 images.

- **Operator versus model:** ω is 0.239 (CLIP), 0.287 (PEcore), 0.259
  (SigLIP2) and 0.252 (DINOv2), against a null of 0.011.
- **Taxonomy residual:** 0.109–0.131 against a null of 0.013, for every
  model. This includes image-only DINOv2 (0.130).
- **Participant concept sets** give the same picture: every model has
  taxonomy residual p < .05 in 5/5.
- **Model–model agreement falls from coarse to fine bands** (modes 1–8,
  then 9–16, then 17–32):

  | Pairs | 1–8 | 9–16 | 17–32 |
  | --- | --- | --- | --- |
  | VLM–VLM | 0.69–0.75 | 0.42–0.58 | 0.39–0.46 |
  | VLM–DINOv2 | 0.49–0.53 | 0.18–0.20 | 0.14–0.16 |

  The null is at most 0.023.

**What this establishes, within its limits:**

- An externally measured co-occurrence operator predicts each participant's
  leading brain subspace in high-level visual cortex. That prediction is
  robust to frequency and to low-level pixel moments.
- For VG, the brain-level prediction is **not established beyond the
  27-category taxonomy** (3/5 positive, sign p = 0.50). The caption
  operator's residual is small, positive in 5/5 but significant in 3/5, and
  it is the secondary operator.
- For all four models, including a language-free one, the same operator
  predicts subspace structure well beyond taxonomy.

**Strongest surviving alternatives:**

- **Measurement reliability.** Brain split-half self-consistency at q = 8
  is only 0.10–0.20, so weak beyond-taxonomy and fine-band brain alignment
  cannot be separated from noise.
- **Experience mismatch.** Photo-annotation statistics may differ from what
  brains experienced.

Under the GOAL decision rule ("models align but brains do not"), measurement
and experience mismatch are both implicated, and neither is identified.

**Post hoc, descriptive only.** Primary ω divided by self-consistency is
about 0.5 in ventral and lateral cortex. This ratio was not prespecified. The
self-consistency comes from half-data Grams, so it is not a strict ceiling.

**Figure.** Codex `gpt-6-astra` (`max`, text-only) wrote
[plot_a0.py](code/plot_a0.py) from the report schema.

- The orchestrator reviewed it and it was applied without edits.
- Job 47646330 (`COMPLETED`, 15 s) rendered
  [a0_operator_alignment.png](figures/a0_operator_alignment/20261013-attempt01/a0_operator_alignment.png),
  with an SVG alongside, from the report only.
- On inspection the panels match the tables above:
  - A: participant summaries for each statistic and operator;
  - B: sectors, with self-consistency;
  - C: bands, plus model–model curves;
  - D: pooled models.
- The band 17–32 null (0.081) is drawn, so the return of the brain curve to
  0.083 there reads correctly as chance.

**Next.** The README has been refreshed. Remaining initial-stage work is
P1–P5, A1, A2 and A3. A4 needs a scientist decision on an independent
source.

## 2026-10-13 09:05 UTC — a0_operator_alignment — stages 1–2 complete; audit decision; operators rebuilt; stages 3–4 launching

**Stage 1 (operators).** Job 47629806 `COMPLETED` in 1 min 14 s, with
1.7 GB MaxRSS.
[operators_report.json](runs/a0_operator_alignment/20261013-attempt01/operators/operators_report.json).

- **WordNet crosswalk:**
  - All 1,797 THINGS ID4 synsets decode; 57 concepts have none. No offset
    is shared between concepts. ID4 agrees with the THINGS `WordNet ID`
    column for 1,760 of 1,796 concepts.
  - All 7,605 VG synsets decode. Offset matching reaches 1,211 THINGS
    concepts, against 1,029 by exact name: 182 more, e.g. glasses, boy,
    bush, couch, cellphone, donut.
- **Concept sets:** VG S = 706 (S_p = 192–205). Captions S = 895
  (S_p = 239–265), with 58 ambiguous variants dropped.
- **Stage 2 (brain patterns).** Array 47629807, tasks 0–4 `COMPLETED` in
  39–48 s each, about 1.1 GB MaxRSS each.
  - Each participant has 30 sessions and 431–435 selected THINGS-stratum
    images, all with ≥2 valid presentations.
  - ROI voxel counts match EP04 (e.g. sub-01 ventral 3,767).
  - Reports are in [brain_patterns](runs/a0_operator_alignment/20261013-attempt01/brain_patterns/).

**Audit decision.** This was outcome-blind: it was made from the frequency
lists alone, before any alignment.

- **VG:** the 40 most and 40 least frequent concepts all carry plausible
  scene senses, so there are no exclusions.
- **Captions:** two of the 40 most frequent concepts have a dominant caption
  sense that differs from the THINGS sense. They are excluded from the
  caption lexicon, so they are neither targets nor contexts:
  - `orange`: THINGS means the fruit; captions mostly use the colour.
  - `baseball`: THINGS means the ball (baseball.n.02); captions mostly mean
    the sport.
- **Kept:** `game` (THINGS game.n.01 is "a contest with rules", which
  matches captions), `board` (mostly physical boards) and `glass` (mixed
  drinking-glass and material senses). They are retained and noted.
- **Mechanism:** the option `--exclude-caption-concepts orange,baseball` was
  added to [a0_operators.py](code/a0_operators.py). The rebuild writes to
  `operators_audited/`, which the alignment uses. The first build in
  `operators/` is preserved.

**Submitted chain:**

| Stage | Job | Dependency |
| --- | --- | --- |
| Audited operators | 47630147 | — |
| Alignment array (tasks 0–5) | 47630149 | `afterok:47630147` |
| Collect plus ASTRA validation | 47630151 | `afterok:47630149` |

## 2026-10-13 08:45 UTC — a0_operator_alignment — scripts reviewed; stages 1–2 launching (attempt 20261013-attempt01)

**Review.** The orchestrator read the three Codex scripts in full.

- [a0_operators.py](code/a0_operators.py): the WordNet `index.noun` parser,
  the name decoding, ID4-first offsets with a column fallback, Visual Genome
  offset contexts, the Word/uniqueID caption lexicon, and the PPMI kernel
  with own-label exclusion all match the brief. One fix: hyphens now become
  spaces before character filtering ("t-shirt" → "t shirt").
- [a0_brain_patterns.py](code/a0_brain_patterns.py): matches the brief
  without changes. It uses session standardization over all development
  trials, EP04 validity, interleaved halves in (session, beta-axis) order, and
  ≥2 valid presentations. All 150 session reports have the required keys,
  including sub-01's reused pilot `ses-01`.
- [a0_alignment.py](code/a0_alignment.py): the statistics, the shared
  permutation banks, the residual nulls with recomputation, and the
  participant mean over sectors with synchronized draws all match the frozen
  specification. Two fixes:
  1. The 166 truncated THINGSplus rows stayed unmapped (−1). As drafted,
     they aborted the run.
  2. Generic input SHA-256 fingerprints were removed, per repository rules.

**Plan.** Stage 1 (operators, one job) and stage 2 (brain patterns, a
five-task array) run in parallel. The operator audit lists (the 40 most and
40 least frequent concepts per source) and the crosswalk report will be
reviewed and recorded before stage 3 (alignment, a six-task array) and
stage 4 (collect plus ASTRA validation) are submitted.

## 2026-10-13 06:40 UTC — a0_operator_alignment — design frozen before any A0 outcome; implementation delegated

**Authorization.** The scientist approved the three recommendations and requested
notification when the run finishes:

1. bind the raw stimulus images for a low-level visual operator;
2. freeze A0 and run it within 64 CPU-hours and zero GPU;
3. defer ADE20K.

**Freeze.** The `A0.frozen_specification` in
[SEARCH_POLICY.yaml](../SEARCH_POLICY.yaml) was written before any model or
neural value was computed for A0. Decisions recorded at freeze:

- **LVIS is excluded from A0.** It failed the prespecified per-participant
  coverage criterion, which is outcome-blind.
- **Visual operator.** It reuses EP04's frozen RGB block-moment features
  (`ep04_rgb_block_moments_5_10_20_v1`, 23,560 × 3,150), read-only. This
  binds the raw-image asset through derived features, so A0 reads no pixels.
  The features cover colour, luminance layout and local contrast, not
  orientation or shape, so a surviving result is "beyond this pixel-moment
  control" only.
- **Brain preprocessing.** It reads EP04's development single-trial cache
  (`$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/neural_full01/`,
  through EP04's participant handoffs). This is permitted by the EP22/EP23
  reuse rule while the cache matches the frozen exports. Standardization uses
  all development trials per session, because A0 fits nothing to neural data.
  Splits are interleaved halves of the valid presentations.
- **WordNet crosswalk.** It uses NLTK's WordNet 3.0 package (`wordnet.zip`,
  10,775,600 bytes), downloaded to the feasibility downloads directory and
  recorded in `download_record.tsv`.

**Delegation.** Codex `gpt-6-astra` (`max`, read-only, text-only) is drafting
three scripts from exact interface briefs:

- `a0_operators.py`;
- `a0_brain_patterns.py`;
- `a0_alignment.py`.

The briefs are under
`$SCRATCH/br_autoresearch/codex_orchestration/ep23_a0/`. The orchestrator
reviews the scripts before any Slurm run.

## 2026-10-13 06:20 UTC — operator_feasibility — check01 results: Visual Genome usable; captions need a cleaned matcher; LVIS fails

**Run.** Job 47618889 `COMPLETED` on sh03-01n38. It took 1 min 19 s wall
time, 2.6 GB MaxRSS and about 0.1 allocated CPU-hours.
[feasibility_report.json](runs/operator_feasibility/20261013-check01/feasibility_report.json)
and [stdout](runs/operator_feasibility/20261013-check01/stdout.txt).

**Coverage and structure.** The pooled set S is the 1,526 dev concepts with
≥10 images; S_p is the per-participant subset.

| Source | Images with a THINGS concept | Pooled S | S_p | Largest component | Taxonomy ω (direct / shared-context) | Null 95th percentile | Verdict |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| Visual Genome v1.4 | 92,332 / 108,077 | 615 | 166–181 | 100% | 0.292 / 0.434 | 0.077 / 0.061 | Usable: all criteria pass |
| COCO captions | 122,789 / 123,287 | 880 | 234–257 | 100% | 0.200 / 0.390 | 0.051 / 0.045 | Numeric criteria pass; matcher defect (below) |
| LVIS v1 | 111,997 / 119,979 | 473 | 121–151 | 100% | 0.264 / 0.428 | 0.073 / 0.062 | Fails S_p ≥ 150 in 4/5 participants |

Additional numbers:

- **Taxonomy p-values:** all p = 0.001, at the 1,000-permutation floor.
- **Cross-source agreement** of the leading 8 eigenspaces, on 275 common
  concepts (direct / shared-context; null mean ≈ 0.03, null 95th percentile
  ≤ 0.043):

  | Pair | Direct | Shared-context |
  | --- | ---: | ---: |
  | VG–LVIS | 0.746 | 0.790 |
  | VG–captions | 0.674 | 0.691 |
  | LVIS–captions | 0.555 | 0.614 |

- **Open Images:** 1,117 pooled-dev concepts match by name. That is coverage
  only; no counts were computed.

**Operator conditioning (descriptive).** The direct normalized operator is
frequency-dominated, whereas the shared-context PPMI kernel is better
conditioned:

| Diagnostic | Direct (VG / LVIS / captions) | Shared-context (VG / LVIS / captions) |
| --- | --- | --- |
| Spearman of \|u₂\| with log degree | 0.69 / 0.82 / 0.66 | 0.26 / 0.30 / 0.35 |
| IPR of the leading modes | up to 0.08–0.17 (reference 0.001–0.002) | 0.003–0.016 |

The shared-context kernel is therefore the natural primary operator. About
57–61% of its leading-8 subspace lies outside the category27 span, which
leaves room for structure beyond taxonomy.

**Quality defects found on inspection** (the frequency lists in the report):

- **Captions.** Matching WordNet synonyms creates false positives from
  common words and verbs:

  | Concept | Matched via | Images |
  | --- | --- | ---: |
  | `rack2` | "stand(s)" | 9,208 |
  | `backdrop` | "background" | 5,661 |
  | `photograph` | "picture" | 6,522 |
  | `lighter` | "light" | 3,903 |
  | `handle` | "hold" | 3,951 |

  Fix: match only THINGS `Word` and `uniqueID` names, and audit the
  high-frequency matches.
- **Visual Genome.** Exact synset-name matching is a lower bound. For
  example, THINGS `glasses.n.01` versus canonical `spectacles.n.01`, and
  `glass.n.02` versus VG's `glass.n.01`, both give about 0 images. Fix: a
  WordNet 3.0 offset crosswalk. NLTK is not installed; WordNet's index files
  are a small download.
- **LVIS.** Frequent categories cluster at about 2,300 images (e.g. sock,
  bench, dog), consistent with federated sampling caps, not natural
  frequency. 9.8% of positives are in non-exhaustive categories. LVIS is
  therefore a robustness source only, with verified-pair denominators.
- **Shared images.** COCO, LVIS and part of VG share images. Cross-source
  agreement shows robustness to the annotation channel, not an independent
  replication.

**Verdict.** The operator test is feasible on Visual Genome now, and on COCO
captions once the matcher is cleaned. No model or neural data were read.


**Scope.** The scientist agreed to all four open-decision recommendations:

- add the measured co-occurrence-operator test as the primary empirical
  question;
- accept the A2 labels and thresholds;
- defer the A4 source;
- approve the initial-stage budget of 64 CPU-hours and zero GPU.

The scientist then asked for the operator's feasibility and novelty checks.
These checks use annotation files and THINGS metadata only.

**Audits.** Codex `gpt-6-astra` (`max`, read-only, web search) produced both;
the orchestrator reviewed them.

- [operator_novelty_audit.md](design/operator_novelty_audit.md). Spot checks:
  Sadeghi et al. 2015 (Europe PMC) and Johnson et al. 2023 (arXiv).
- [operator_source_audit.md](design/operator_source_audit.md).

**Downloads.** Job 47615301 (`COMPLETED`, 25 s) fetched public annotation
files, with no images, into transient scratch
(`$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/feasibility/downloads/`,
read-only). Byte counts:

| Source | File | Bytes |
| --- | --- | ---: |
| Visual Genome v1.4 | `objects.json.zip` | 55,323,929 |
| Visual Genome v1.4 | `object_synsets.json.zip` | 357,161 |
| LVIS v1 | train | 350,264,821 |
| LVIS v1 | val | 64,026,968 |
| COCO 2017 | annotations | 252,907,541 |
| Open Images V7 | class descriptions | 501,178 |

All are CC BY 4.0 annotations. URLs and Last-Modified dates are in
`download_record.tsv`.

**Feasibility script.** Codex drafted
[operator_feasibility.py](code/operator_feasibility.py) from an exact schema
brief. The orchestrator reviewed it, tested its metadata and caption matching
on the login node, and changed the taxonomy basis to the full projected
category span. THINGS has 1,797 distinct synsets and none is shared by two
concepts. 336 ambiguous name variants are dropped.

**Run.** [operator_feasibility.sbatch](code/operator_feasibility.sbatch), job
47618889 (4 CPUs, 32 GB, 1 h), writes `runs/operator_feasibility/20261013-check01/`.

**Criteria, fixed before any output was read.** A source is *usable* for the
operator test if all of the following hold:

- each participant has ≥150 dev concepts with ≥10 images;
- the largest connected component of the direct operator holds ≥90% of the
  pooled concept set S;
- the operator's leading subspace aligns with category27 above the
  permutation null (p < 0.05), but leaves room beyond taxonomy (ω < 0.8);
- at least one other source agrees with it above the null 95th percentile.

Hub dominance (the Spearman correlation of |u₂| with log degree) and
localization (IPR) are reported descriptively. They inform the choice of
primary operator.

## 2026-10-13 04:55 UTC — design — literature verification reviewed; contribution narrowed

**How the audit was run.** Codex `gpt-6-astra` (`max`, read-only, web search)
audited the draft's 20 literature claims plus the novelty and mathematics
checks. It received the claim list only, with no data. The reviewed audit is
[literature_verification.md](design/literature_verification.md).

**Spot checks.** The orchestrator checked these against primary pages:

- Nava & Wyart, arXiv 2605.23821;
- Kang, Canatar & Chung, arXiv 2502.19648, whose venue is not stated on the
  abstract page;
- Khaligh-Razavi & Kriegeskorte, PLOS Computational Biology 2014;
- Yamins & Nayebi, arXiv 2607.08561;
- Karkada et al., arXiv 2602.15029 (ICML 2026).

**Consequences now in GOAL.md:**

- **Corrected summaries.** Five corrections from the audit were applied:
  - Gröger et al.: calibrated capacity trends in global measures;
  - Storrs et al.: fitting reduced architecture differences;
  - FR-RSA: signed weights;
  - Rosetta Stone: broader than NSD; MEG–text at chance;
  - Marcos-Manchón et al.: within-subject correspondence and seed selection
    are used.
- **P3** now uses Yu–Wang–Samworth Theorem 2, which needs no small-ε
  condition. Weyl's inequality preserves the cluster for ε < δ/2, and the
  draft's ε < δ/4 is valid but conservative. P3 also gains the two-learner
  bound 2ε/(δ − ε) and a kNN distance-margin condition.
- **P1, P2, P4 and P5** gain these points:
  - observation-map compression (P1);
  - the indefinite-K truncation and the scope of the objective (P2);
  - non-identifiability of latent geometry from an unrestricted readout (P4);
  - cross-repeat Gram unbiasedness (P5). A2b now uses this estimate.
- **Novelty:**
  - The co-occurrence → coarse-to-fine eigenspace mechanism is already
    proven and tested for language (Nava & Wyart 2026; Karkada et al.
    2026).
  - Reweighting versus remixing, and categorical versus within-category
    contrasts, have a direct precedent in Khaligh-Razavi & Kriegeskorte
    2014.
  - Measurement effects on eigenvectors have precedents in Kriegeskorte &
    Diedrichsen 2016 and Kang et al. 2025.
  - P1–P5 are therefore conditional restatements, not new theorems. The
    proposed contribution is narrowed to the joint test: transfer to vision
    models and brains, subspace separated from metric at participant level,
    and noise-free model–model fine structure. Priority is not established.
- **Scoping option.** Recorded for the scientist: test the co-occurrence
  intuition directly with an independently measured concept co-occurrence
  operator.

## 2026-10-13 04:30 UTC — design — proposed design written; metadata feasibility checks

**Origin.** After EP22 reported (`confirmation_complete`), the scientist asked
for a parallel design of a theory-level episode. The scientist set up a split:
Opus orchestrates, and Codex `gpt-6-astra` at `max` drafts.

- Codex ran read-only and executed no commands. It received only text: EP22's
  published summary numbers, the data inventory and the design rules.
- Codex received no data, images or closed outcomes.
- Briefs and responses are under
  `$SCRATCH/br_autoresearch/codex_orchestration/ep23_theory_design/`.
- The orchestrator reviewed the draft. Changes made:
  - corrected the data facts below;
  - split A2 into model–model (A2a) and model–brain (A2b);
  - made A2b reliability-relative;
  - aligned A1's α grid with EP22;
  - marked A4 as blocked;
  - added the competing account "measurement ceiling only".

**Metadata checks (login node, light reads of label tables and EP04's frozen
development rows; no neural or model values).**

- **Development strata:** every participant has 4,246 LAION, 322 THINGSplus
  and 144 THINGS images. No development image is shared by all five
  participants.
- **THINGS category53 (single-label concepts).** Participants have only 1–2
  categories with ≥20 images, so the draft's A2 prerequisite (≥5 categories
  of ≥20) fails.
- **THINGS category27 manual (single-label concepts).**
  - Categories with ≥20 images per participant: 3, 4, 2, 2, 4.
  - Categories with ≥10 images per participant: 9, 7, 10, 8, 8
    (sub-01/03/05/06/07).
  - Pooled: 2,330 THINGS-stratum images, 1,166 of them single-label. Of the
    categories, 19 have ≥20 images and 10 have ≥40.
  - Rows align with `concepts-metadata_things.tsv`, checked on 10 concepts.
  - Concept–participant pairs: 2,118 have one image and 106 have two.
- **A2 decision.** A2a (pooled, model–model) uses ≥20 images per category;
  A2b (per participant, model–brain) uses ≥10. Both thresholds were chosen
  from these counts alone.
- **Repetitions.** There are four planned presentations per development
  image (EP04 DATASETS: 23,536 four-repeat and 24 three-repeat images), so
  split-half reliability is available.
- **Spatial metadata.**
  - GLMsingle-tedana betas are in T1w volume space.
  - Noise-ceiling maps exist for 4-repetition, 12-repetition and
    all-repetition stimulus sets.
  - ROI masks are fsnative surface labels.
  - Surface meshes and smoothing are not verified.
- **α convention.** In EP22's `d123_cell.py`, Φ_α = U_k D_k^α in the
  leading-256 training-PCA basis, with α on the singular values. So α = 2
  squares the feature Gram matrix.
- **Confirmation candidates.**
  - EP18 (THINGS-EEG1) holds 30 development and 16 audit participants. Its
    audit is closed and its development work is active.
  - EP17 (CNeuroMod-THINGS) holds 480/120/120 concept roles in four
    participants, with a closed audit. Four participants cannot reach
    α = 0.05 with an exact sign test.
  - A4 therefore has no eligible source.

**Registrations.**

- `EPISODE_REGISTRY.yaml`: ep23 entry, using the new `local_role`
  `current_theory_episode`, which was added to the schema. ep23 also joins the
  LAION shared evidence group.
- `DATA_LOCATION_MANIFEST.json`: EP23 binding, plus EP23 in the
  `asset_ep04_laion_mixed_role` and `asset_ep18_things_metadata` episode IDs.
  The stale EP22 binding state was corrected to "OOD consumed".
- `DATA_LOCATIONS.md` and the root `README.md` were updated to match.

**Next.** Await the scientist's decisions on the open design choices. The
literature verification is recorded in the entry above.
