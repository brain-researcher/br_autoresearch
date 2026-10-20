# EP03 data: separating task-family recognition from contrast information

The data must allow a question that a familiar-looking map cannot answer:
does a signed comparison described in methods predict another dataset's map
better than knowing only its task family? The number of independent dataset
groups, the support within each family, and the reliability of their maps
matter more than the number of contrast files.

This is a design contract, not a report that the collection has been imported
or reanalyzed. Only the episode's plain-text contracts and local status docs
were read for the 2026-09-30 writing milestone; no map payload or protected
outcome was accessed.

## Status and source exposure

The completed metadata work is narrower than the reported 51-group corpus.
The [source census](outputs/runs/source_handoff_qualification/20261003-metadata01/source_metadata_report.json)
found 31 accessions, 46 dataset/task routes and 3,667 acquisitions; the
[corrected design join](outputs/runs/source_handoff_qualification/20261003-design-support03/design_support_report.json)
matched those acquisitions uniquely, including 60 authenticated session-case
aliases. The [candidate group-map census](outputs/runs/source_handoff_qualification/20261003-group-map-metadata02/group_map_metadata_report.json)
found 297 Z-named files, and their [readable headers](outputs/runs/source_handoff_qualification/20261003-group-map-headers01/group_map_headers.json)
share a 97 × 115 × 97 grid. Filename/header availability does not authenticate
Z statistics, independence, repeatability, a common mask or the missing
577-program handoff. No map voxel arrays were decoded. The preserved spatial
source pack is BOLD/transform lineage, not the requested group-map corpus.

The completed October 5 first-route qualification covers 13
`ds000001/balloonanalogrisktask` maps. All 16 configured subjects have matching
available subject-effect paths. **Fitted N=16 (sub-01…16) was confirmed on
October 17** by the lineage check (job 48221009): the df was 15 and the Dataset maps
recompute exactly from the 16 subject effect maps. Orientation is frozen for all 13
(9 resolved, 4 under adopted rules); see [source facts resolved
2026-10-16](#source-facts-resolved-2026-10-16) and the
[orientation freeze](outputs/orientation_freeze_proposal.md). These facts do not establish
group independence or scientific eligibility. The first bounded route attempt
was incomplete; its diagnosed repair completed in job 46660232.

The [public methods references](outputs/paper_plan.md#october-6-public-methods-sources)
retrieved on October 6 provide the producer's model specification and event-field
documentation. They establish configured definitions, not the historical
Dataset-node inclusion list, and do not replace the episode's source binding.

This contract does not assert that any absent asset has been acquired. The
intended development source is the NeuroEffect/OpenNeuro collection:

| Reported asset | Current knowledge | Exposure classification |
| --- | --- | --- |
| 51 OpenNeuro dataset groups | Reconstructed October 16 as 51 NeuroEffect candidate accessions: 31 included, 20 with no eligible subjects. Independence is still unchecked | Fully exposed by prior corpus-wide development |
| 577 analysis/contrast programs | Repeated observations nested within dataset groups. The list cannot be recovered on Sherlock; 628 Dataset-level Z files exist today | Fully exposed; never independent sample size |
| approximately 45 GB of statistical maps | Source locations, stable release identifiers, and map eligibility not yet verified for this episode | Source outcomes previously used in pilots |
| reported 36-dataset / 341-contrast repeatability subset | Lineage and meaning of repeats require authentication | Development/reliability estimation only |
| NeuroEffect/FitLins provenance | October 16: build, executed spec, input rewrite and event edits recovered for all 96 routes with maps ([route census](#route-census-and-orientation-proposal-2026-10-16)) | Execution metadata only; no outcome read |

Historical reports of approximately 2.2% improvement over a global mean map
and 40--46% reduction in repeated-map disagreement are prior feasibility
signals, not current results and not thresholds to optimize against.

In particular, a small gain over a **global mean** does not establish gain over
a **task-family prototype**, the current primary comparator. The reported
repeatability change does not establish the reliability of a family-residual
map or a noise ceiling for semantic prediction. The meaning of those reports
has to be recovered from the original methods before it can inform precision.

## What the source can identify

### Source facts resolved 2026-10-16

Under the October 16 rule in [AGENTS.md](../AGENTS.md), the open source facts
were recovered from the producer's execution records, code and events, or
adopted as stated assumptions. The [log](outputs/experiment_log.md#2026-10-16-1528-pdt--source-fact-rules-fixed-before-the-events-check)
fixed those assumptions and the orientation rule before any map access. No
producer was contacted and no map value, header, result PNG or score was
opened. The following were read:

- the run's FitLins report (text, model JSON and image paths only);
- the producer's methods and QC text;
- read-only git metadata;
- the 48 events TSVs;
- the staged NeuroEffect metadata.

Check outputs are under [job 48169324](outputs/runs/source_handoff_qualification/20261016-source-facts-48169324/).
Producer paths are relative to `$OAK/data/OpenNeuro_analyses/openneuro_fitlins`.

| Open item | Status | Evidence or rationale | Check / sensitivity plan |
| --- | --- | --- | --- |
| BART spec that ran | **Resolved** | `analyses/ds000001/task-balloonanalogrisktask/reports/model-ds000001.html` was rewritten by the run that produced the current maps (data commit `d723670`, 2025-09-26). Its embedded model equals the local `ds000001-balloonanalogrisktask_specs.json`, which is byte-identical to public `main` and unchanged since producer commit `4dc69d7` (2025-06-05). The June run embeds the same model. | None needed. |
| FitLins build that ran | **Resolved** version; **assumed** commit | Runtime `dataset_description.json`: FitLins `0.11.0.post0.dev94` (dev16 before `d723670`), nilearn, `5:run:iso`, `MNI152NLin2009cAsym`, `drop_missing=true`, `participant_label=null`. FitLins versioneer `pep440-pre` places dev94 94 commits after tag 0.11.0. Assumed commit: poldracklab/fitlins `dev` `55ae712e72d3`, the only first-parent `dev` commit at that distance and `dev` HEAD from 2025-09-22 to 2025-10-13. The producer switched to `poldracklab/fitlins@dev` on 2025-09-24 (`1363102`), so the copied `uv.lock` (jmumford `paddedint` `1bed1a2b`, dev42) is stale. | No EP03 estimand uses the commit; only an exact regeneration would test it. |
| `_demean` construction | **Resolved** | These are original OpenNeuro ds000001 columns. The producer's only edit (`modify_events.ds000001`, the same on `main`) sets n/a to 0 in them and in `response_time`; no other change appears in 48/48 runs. Each column is non-missing only on its own event type. `pumps_demean` = pump ordinal minus its balloon's mean ordinal (3,070/3,070 steps +1; 1,136/1,136 balloons centered). `control_pumps_demean` rises +1 per control pump (1,984 steps; 86 resets and one zero step between back-to-back balloons; range ±5.5; run mean 0). `cash_demean` and `explode_demean` = the balloon's pump count minus the run mean over that outcome (slope 1, residual <2e-15 in 48 and 47 runs). | This agrees with Schonberg et al. (2012). The control centering scope cannot be tested (control balloons have no terminal event) and does not affect the sign. |
| Orientation of the newly resolved contrasts | **Resolved** 4 more, now 9/13 | The pre-run rule "higher minus lower pump count" gives `s=+1` for `pumpspara`, `controlpara`, `cashpara` and `explodepara`. Four remain sign-ineligible: `pumps` and `allpumps` (event vs implicit baseline), `explodevcash` (category order) and `rt` (duration regressor). They need an ontology rule, not a source fact, so they stay unresolved until the corpus taxonomy freezes. | A failed construction would have left its map unresolved; none failed. |
| Fitted N and contributors, 13 Dataset maps | **Resolved** N=16, sub-01…16 (job 48221009) | The run used `participant_label=null` and model input 01–16. Its FitLins report lists "Participants (16)", 48/48 run design matrices and 16×13 subject sections. Commit `d723670` rewrote all 208 subject effect maps together with all 65 Dataset files. FitLins feeds Subject-node outputs to the intercept-only Dataset node, and `drop_missing` drops only missing inputs; none were missing. The producer report states "For the 16 subjects", a one-sample t-test and no flagged subjects. No Dataset-node input list, dof map or work directory survives. | Confirmed under the pre-registered rule (log 2026-10-17 00:21). The best integer df from paired t/z voxels is 15 (median |z_pred−z| = 0; df 16 errs by 4e-3–1.2e-2). The Dataset effect equals the mean of the 16 subject effect maps (max relative Δ ≤ 1e-15), the recomputed t is identical (r = 1.0000000), and the inputs are sub-01…16. |
| "51 groups" provenance | **Resolved** (reconstructed) | The 51 are `candidate_dataset_ids` of NeuroEffect's policy-3 within-dataset subject split (seed 20260904, built on Marlowe). It is staged only at `$SCRATCH/neuroeffect-full51-spatial-build/20260917T201500Z.2230efb/release-stage-v1/subject_partitions.json`. 31 accessions are included (946 subjects, 46 task pairs); 20 have `fewer_than_2_eligible_subjects`. "51 vs 46" thus counts accessions vs task pairs, and the 51 were never authenticated groups. 49/51 have shared FitLins analyses; ds002785 and ds002790 do not. | The [census](outputs/runs/source_handoff_qualification/20261016-source-facts-48169324/neuroeffect_candidate_provenance.json) preserves the IDs, statuses and generating inputs. |
| 577 programs and ~45 GB | **Unrecoverable** on Sherlock | The defining estimand catalog and pair manifest are on Marlowe (`/projects/m000264`), which Sherlock cannot reach. Today's pool holds 628 Dataset-level Z filenames in 103 routes (322 in included, 306 in excluded accessions), not 577. | Not needed: contrasts are never sample size, and EP03's own manifest defines its corpus. |
| Independence unit | **Assumed** | One group per OpenNeuro accession, with its tasks pooled because they share people (e.g., ds000008, ds000009, ds000052). No cross-accession overlap is assumed unless a lineage check finds one. | This is the required duplicate-lineage falsifier: merge accessions that share participants, first-level inputs or derivative lineage, then rerun grouped folds. |

### Route census and orientation proposal 2026-10-16

The BART treatment was extended to all 103 candidate routes (49 accessions, 628
Dataset Z maps) using git log/show, committed JSON/HTML text, events and design
headers. Census jobs 48184490 and 48185964 ran first; build jobs 48188582 and 48189996 followed.
Per-route rows are in the
[route manifest](outputs/runs/source_handoff_qualification/20261016-orientation-proposal/route_manifest.tsv)
and per-map rows in the
[orientation table](outputs/runs/source_handoff_qualification/20261016-orientation-proposal/orientation_by_map.tsv).
No map value, header, result PNG, group report or score was opened.

| Fact | Status | Evidence | Check / sensitivity plan |
| --- | --- | --- | --- |
| Runtime FitLins build (96 routes) | **Resolved** | `dataset_description.json` as committed with each map: `0.11.0.post0` dev16 (58 routes), dev42 (37), dev94 (BART). All are nilearn, `5:run:iso`, MNI152NLin2009cAsym and `participant_label=null`. `drop_missing` is false in ds000002 (3 routes); ds000148 uses a cosine drift model. 49 routes were read from the later in-git copy of the then-annexed file. | None needed. |
| Spec that ran | **Resolved** in 95; **assumed** in ds000009 BART | The model embedded in the report written by the map commit. ds000009 BART's report predates its maps, while its spec and description are unchanged. Against the local spec: 86 exact, 2 pybids defaults only, 2 subject/run filters only, 6 with changed contrasts (ds003441 ×2, ds003481 `verbs`, ds003858 ×3). | Orientation uses the executed definitions, never the local file or filenames. |
| Fitted N and contributors (628 maps) | **Resolved** (October 17) | Lineage check jobs 48221634 and 48245679. 622 maps match exactly: the t→z df equals n−1, the Dataset effect equals the input mean within 1.2e-15, and t is identical. The 6 ds003545 compL1 maps were fitted on N=20, without sub-13 and sub-14, even though 22 subject maps exist (exact leave-two-out match). N ranges from 2 to 108; included accessions 8–73. 57 maps (ds003441, ds000255) have N=2. | Use the corrected N=20 wherever sample-size nuisance or reliability caps apply. The census JSON keeps the original counts. |
| Producer event edits (49 accessions) | 26 **resolved**, 2 **assumed**, 21 none | [Edit review](outputs/runs/source_handoff_qualification/20261016-orientation-proposal/event_edit_review.json) of the function current at each run, design columns and retained input copies. ds000009 was edited by code first committed after its run, as the design columns show. ds002872's T/P→high/low mapping is undocumented. ds000002's function never applies (it raises an error). | ds000009: recompute the committed constructions against design columns. ds002872: report both orientations. ds003545 trimmed 4 dummy volumes without shifting onsets; the onset check was inconclusive, so it is proposed to stay out of the primary set until settled. |
| Definition defects | **Resolved** | 1 null contrast, 2 duplicate definitions, 4 maps on never-convolved regressors (ds000009 discounting, ds004556 `rate`) and 5 names that contradict their definitions: 10 distinct maps. | Screens in the proposal. |
| Canonical orientation (628 maps) | Frozen 2026-10-16: 208 **resolved** (9 BART decisions frozen earlier), 51 **assumed**, 345 under adopted rules (323 + 22 assumed), 21 unresolved, 3 ineligible | Frozen R1–R7 with precedence and methods-only condition roles (build job 48189996); see the [frozen proposal](outputs/orientation_freeze_proposal.md). Included accessions (31; 322 maps): 106 / 33 / 163 / 19 / 1 in the same status order. | First target load: fitted-N lineage check, BART first, then all routes subject to the registered BART gate. Retain P/I and R5 sensitivities. |

#### Orientation freeze (2026-10-16)

Exclude from the primary set the 1 null map, 2 duplicate maps, 4 maps on never-convolved regressors and 57 N=2 maps; hold ds003545’s 40 maps out until onset timing is settled. The build will count overlaps.
The family taxonomy, negative-control pass threshold and normalisation details remain unfrozen and must be frozen before target access for the family-prototype baseline.

### Public online resources — checked October 6, 2026

- [Producer BART model specification](https://raw.githubusercontent.com/demidenm/openneuro_glmfitlins/main/statsmodel_specs/ds000001/ds000001-balloonanalogrisktask_specs.json): public Run/Subject/Dataset design, 13 contrast definitions and configured subjects 01–16. It explicitly assigns `response_time` to the RT regressor's duration and defines `allpumps` with literal 0.33 weights; these meanings should not be inferred from filenames.
- [Producer dataset details](https://raw.githubusercontent.com/demidenm/openneuro_glmfitlins/main/statsmodel_specs/ds000001/ds000001_basic-details.json): public event-column names/types, three configured task runs, preprocessed-events flag and the original methods citation. This supports methods-only source interpretation, not a new map observation.
- [FitLins usage documentation](https://fitlins.readthedocs.io/en/latest/usage.html): documents participant/derivative filters and `--drop-missing`, which can remove missing inputs or contrasts. Consequently, the configured list and available subject files cannot substitute for actual historical Dataset-node contributors.

These public `main` files were read as current documentation, not certified as
the exact historical execution snapshot. They narrow the existing local
producer request to fitted input identities and unresolved canonical
event/ontology definitions. Actual contributor IDs/N remain unknown and only
five of thirteen canonical directions are resolved. No map values, results
images or score-bearing artifact were opened by this online check. *The
October 16 [source-fact resolution](#source-facts-resolved-2026-10-16)
supersedes this status.*

| Data object | Scientific role | What it does not establish |
| --- | --- | --- |
| Methods/events and signed contrast definitions | Model input and a map-independent family/semantic taxonomy | Neural results, intended psychological mechanism, or causal intervention validity |
| Unthresholded group Z contrast map | Sign-preserving normalized statistical-map geometry | Effect amplitude, an individual's brain state, or causal neural implementation |
| Independent dataset group | Unit of exclusion, equal weighting, uncertainty, and influence | Independence of each contrast, voxel, paper, or derivative |
| Authenticated repeated maps | Development reliability and sensitivity analysis | A fresh audit or interchangeable repeats without participant/processing lineage |
| Prospective independent group | One locked evaluation of transfer after search | Freshness merely from a new filename or another release of the same participants |

Z-map normalization is not an effect-size transformation and does not remove
all processing or sampling differences. A prototype-dominated outcome is
interpretable only with adequate group support and informative uncertainty;
otherwise the source yields an unresolved comparison. The independent-group
inventory and source handoff remain missing, so no such conclusion is now
available.

The 2026-10-02 sign-consistency amendment applies one methods/ontology-defined
canonical comparison ordering to every eligible map and description. With
source-to-canonical polarity `s`, fitting uses `Z*=s Z`; both the family
prototype and prototype-plus-residual prediction are multiplied by `s` on
return to the requested comparison. Reversal changes only that polarity,
not family membership or evidence roles. Canonical ordering cannot use map
values or scores; unresolved ordering remains sign-ineligible. The family
baseline receives the same orientation convention as the candidate. See
[the exact rule](GOAL.md#one-orientation-rule-for-the-baseline-model-and-evaluator).

## Read-only development handoff

Before search, bind a read-only handoff under the episode input path. It must
contain only the information needed to establish scientific eligibility and
run the frozen analysis:

1. OpenNeuro accession/snapshot, participant and derivative lineage, license,
   and exposure flag for every dataset group;
2. one row per map with a deterministic map ID, group, participant and
   derivative lineage, task, canonical signed contrast, target observability
   (`detected`, `verified_absent`, or `unknown`), statistic, threshold status,
   sample size, source release, space, affine, mask coverage, smoothness, and
   path;
3. raw methods/events/contrast excerpts, standardized signed description,
   redacted model input, and provenance;
4. task-family, condition-cluster, duplicate, near-duplicate, and repeatability
   tables made without target performance;
5. stable code, container, and provider/model version identifiers needed to
   execute the frozen pipeline, plus map-generation provenance.

Only unthresholded, sign-authenticated group Z maps that can be placed in the
frozen common space and mask are primary eligible. Do not silently mix Z, T,
beta, effect-size, posterior, or thresholded maps.
`unknown` must not be recoded as `verified_absent`; only `detected` eligible
map targets enter geometry scoring, while verified absences and unknowns are
reported separately.

The task-family taxonomy must come from methods and task definitions rather
than target maps. Every prototype, feature vocabulary, residualization,
semantic/map basis, and fitted readout is an outer-training-fold object.
Splitting contrast rows at random, retaining another derivative of a held-out
group, or creating family labels after seeing prediction performance would
answer a different, leaky question. Semantic-cluster exclusion remains a
required falsifier in addition to independent-dataset exclusion.

## Dataset roles

| Role | Source | Allowed use | Freshness |
| --- | --- | --- | --- |
| structural/QC fixtures | synthetic maps, permuted labels, metadata-only tables | implementation, leakage, sign, and split tests | no neural claim |
| adaptive development and selection | all authenticated members of the historically exposed 51 dataset groups | grouped nested CV, operator search, falsifiers, ablations, influence and reliability analysis | exposed exploratory evidence |
| prospective audit | future independent dataset groups, sealed before any target-map or result-text access | exactly one locked evaluation after search and config lock | potentially fresh only after exposure audit |

No subset of the 51-group source may be promoted to audit. Releases sharing
participants, first-level inputs, or derivative lineage remain one group. A
future audit group may supply methods-only text for locked inference, but its
map, reliability, result text, and model-comparison score remain inaccessible
until audit opening.

## Audit firewall

- Store audit bytes and decryption/access authority outside the adaptive
  worker's readable paths until the immutable, write-once configuration lock
  exists.
- Maintain separate development and audit group tables and verify that no
  participant, derivative, map, semantic-cluster duplicate, or source file
  crosses them.
- Do not use audit metadata distributions to select encoders, ranks,
  nuisance handling, stopping, or thresholds beyond the fields required to
  establish eligibility.
- Permit one deterministic execution of the locked container. Preserve all
  outputs even if it fails. A software failure may be retried only from the
  identical pre-outcome lock and may not reveal partial scores.
- After any score is revealed, the audit set is permanently exposed and no
  alternative pipeline may be substituted.

## Missing assets and blockers

- The 51 candidate accessions have been identified (October 16). Execution
  provenance is recorded for all 96 routes with maps (October 16 route census).
  Canonical orientation, the eligibility screens and the negative-control identity
  were [frozen](outputs/orientation_freeze_proposal.md) on 2026-10-16. On October 17
  the fitted-N lineage check resolved N for all 628 maps (ds003545 compL1 corrected
  to N=20).
- Exact independent-group, duplicate, sign, statistic, space, task-family,
  semantic-cluster, and license eligibility is unresolved.
- The methods-only redaction audit and frozen text/model cache do not yet
  exist for the episode.
- No prospective independent audit dataset group has been acquired, sealed,
  or authenticated.
- Audit power and represented-family requirements must be calculated after the
  development inventory; this document does not fabricate a sample count.
- The executable evaluator has not been qualified. This blocks candidate
  scoring and audit opening even if source bytes exist elsewhere; it does not
  block explicit task startup.

## Storage boundary

Large immutable inputs must remain outside Git and be exposed read-only.
Durable group/map tables, ledgers, code, compact results, and the final lock
record belong in the episode workspace; transient matrices belong under a
dedicated `$SCRATCH/br_autoresearch/episode03_condition_semantics_effect_maps/`
path. The old EP03 directory and any remote live cache are provenance sources
only, never mutable episode storage.
