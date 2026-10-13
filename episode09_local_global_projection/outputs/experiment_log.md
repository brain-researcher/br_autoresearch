# Experiment log

## 2026-10-06 Pacific — metadata_feasibility: provider alias confirmed

The selected bounded metadata check exited 0 in coordinator-owned job
46842156 on `sh04-04n22.int`. Its
[direct observation](runs/metadata_feasibility/attempt002/provider_alias236174_observation_job_46842156.json)
returns exactly one matching row from the retained public supplement, row 81:
Animal ID `236174`, Image ID `16124`, fMOST,
`Gnb4-IRES2-CreERT2;Ai139`, provenance `BIL meta (20231223)`, and
`Claustral and cortical Car3+ IT`, referencing Peng et al. 2021. The provider
locator is `https://download.brainimagelibrary.org/7f/53/7f537a62e521a26a/mouseID_236174/`.
The animal ID does not already occur in the retained 38-animal join. Its
original unmatched record remains brain `236174` / 43 cells.

This resolves the factual alias/genotype gap: the original implementation
searched the Image ID column, while this source group uses the animal ID.
The observation agrees with the independently checked BrainTell string and
TeraVR Methods. No EP11 crosswalk, source-cell workbook, morphology or axonal
outcome was read. The shared parent allocation is charged once to EP18;
no separate EP09 allocation or additional charge is introduced.

Updated current GOAL, DATASETS, paper plan and README so they no longer ask
for this resolved identity/genotype. Preserved `brain_genotype_join.json`,
`genotype_support_summary.json` and all historical entries. Reuse the direct
observation as correction evidence; any future integrated join must use a
separate output and retain original brain, animal and image IDs explicitly.
No blanket cross-namespace fallback, primary IT eligibility, layer recovery,
donor support or role assignment follows. The 12-driver-animal and
two-potential-donor-animal feasibility limits, original 12/8 design and
unadopted nine-animal proposal remain unchanged. Remaining source questions
concern authentic layer/class and matched-donor support, not this alias.

## 2026-10-06 Pacific — metadata_feasibility: exact provider alias check prepared

Rechecked the original BrainTell genetic table and retrieved the primary
[TeraVR Methods through PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6677772/).
They retain the recorded `236174`/`16124` Gnb4 lead. The named join implementation
matches source brain IDs only to the supplement's Image ID column B; the saved
schema separately identifies Animal ID in column A. Selected a bounded check
of those two exact keys in the already retained public brain-metadata workbook.
No neuron/source-cell workbook, morphology, axon, projection class or sibling
output is needed. BrainTell's 45-neuron gallery count is distinct from the 43
retained SEU-A1876 source cells and will not substitute for cell lineage.

Prepared `tools/reconcile_brain236174_provider_alias.py`: actual non-login Slurm
shell, one CPU and explicit single numerical-thread caps required. It reads
only `runs/metadata_feasibility/attempt002/provider_brain_metadata.xlsx` and
the existing non-outcome `brain_genotype_join.json`; stdout contains selected
identity/genotype/provenance fields and whether the animal ID is already in
that join. The coordinator will run it for at most five seconds inside the
substantive EP18 provisioning allocation and redirect stdout to
`runs/metadata_feasibility/attempt002/provider_alias236174_observation_job_JOBID.json`.
No separate metadata job is requested. This is the existing
`metadata_feasibility` analysis, not a new scientific experiment.

Code was prepared and read back on sh04-ln04 without an allocation; no test,
reader execution, scheduler call, outcome or frozen-artifact mutation occurred.
Actual observation and reconciliation remain pending. The current 12/8 roles,
IT-driver support, donor bound and unadopted nine-animal proposal are unchanged.

## 2026-10-06 Pacific — public brain identity/genotype lead recorded

At the user's request to add the verified online findings, recorded the
original TeraVR Methods and BrainTell genetic table in the
[existing paper plan](paper_plan.md#october-6-public-source-lead--brain-236174-and-provider-key-16124)
and DATASETS. The paper names brain 236174 and Gnb4-IRES2-CreERT2;Ai139;
the provider table links its key 16124 to the genetic-background string
containing 236174-16124. This is a concrete reconciliation lead, not a
completed EP09 join, extra authenticated IT animal or qualified layer/donor
pool. Existing saved joins, feasibility counts, roles and policy are unchanged;
no EP11 crosswalk or live sibling output was consumed.

Documentation-only work on verified login host sh04-ln04 without allocation.
The preceding public-source search returned incidental published-result text;
none selected a model or criterion. No morphology/axon payload, closed outcome,
test, import, scheduler call, contact, job or scientific amendment occurred.

## 2026-10-06 Pacific — retained source/layer counts limit donor support to two animals

Under the instruction to continue, inspected only the already permitted
October 3 `brain_genotype_join.json` and its recorded per-animal
`source_layer_counts`. The original metadata authority covers this bounded
refinement; the earlier proposal's extra count-approval step was unnecessary.
No new workbook/payload access, code execution, allocation or count job was
needed. Actual shell: sh04-ln04.stanford.edu, no Slurm allocation; work was
limited to reading the saved small tables and editing documentation.

The [living plan](paper_plan.md#october-6-metadata-feasibility-and-unadopted-exploratory-route)
now transcribes all 12 authenticated IT-driver animals and the ten exact
source/layer strata with at least five recorded cells. The active donor floor
of four nonself donors requires at least five cells, without selecting the
still-unset `minimum_donors_per_pool`. Only Cux2 animals 374712 and 382148 have
such driver-compatible recorded strata: seven strata / 53 cells and three
strata / 19 cells respectively. The other ten animals have maximum compatible
stratum size at most three, or no compatible recorded layer. Thus the current
documented labeling route supplies at most two potentially donor-supported
animals and 72 cells before finer class, calipers, duplicate/paired identity,
quality and target support. This is not qualified eligibility or a finding
about dendrite–axon coupling.

Missing layers and brain 236174 remain unresolved. Authentic corrections or
additional independent annotations could change the labeling route and this
bound; they were not inferred from projection class, axonal features or EP11.
The nine-animal exploratory draft remains unadopted and unsupported by current
metadata. Original roles, thresholds, budgets and access are unchanged. No
code, test, simulation, scheduler action, empirical fit or new authorization
was produced. Historical attempts and the earlier proposal entry below are
preserved; the unnecessary metadata-count job is superseded by this result.

## 2026-10-06 Pacific — authenticated label count corrected; fallback unadopted

Read the existing EP09-only genotype support metadata: 38 matched brains
map to 38 distinct animals; 6 Cux2 + 5 Plxnd1 + 1 Tlx3 give at most 12
authenticated cortical IT-driver animals before restrictions. The historical
parenthetical below allowing a hypothetical thirteenth is not an eligible
count: brain 236174 remains unmatched. The external critique reports a
different EP11 crosswalk; no EP11 data or assignments were consumed, and an
original-provider identity reconciliation remains necessary.

Updated GOAL, DATASETS and the existing paper plan with those facts and a
clearly unadopted metadata-count/at-least-nine-animal exploratory LOAO proposal.
The metadata count cannot settle morphology/donor quality or outcome support,
and neither 20 nor nine is represented as a demonstrated power result. Active
12/8 roles, thresholds, resource ceilings and access remain unchanged. No
metadata job, source workbook/payload read, test, simulation, scheduler action
or empirical model was run. Original preparation records remain preserved.

## 2026-10-03 — metadata_feasibility selected

The scientist authorized feasible work across EP01–EP21. The official SEU-A1876
source is already preserved in the steward quarantine; it remains read-only
and will not be mounted in a candidate workspace. First qualify the spreadsheet
schema with `outputs/tools/inspect_metadata_schema.py` in Slurm job 46425841,
one CPU, 1 GiB and at most five minutes. This steward operation reads worksheet
headers and the exact shared-string entries they reference. It does not decode
cell-row values, inspect axonal morphology, assign roles or infer IT membership
from projection class. Preserve the result in
`runs/metadata_feasibility/attempt001/metadata_schema.json`.

The purpose is to identify a potential non-axon IT and biological-group source
before any scientific pipeline can use these mixed-role inputs. Acquisition
alone is not a qualified handoff. Shared EP09/10/11 exposure and audit rules
remain binding; existing EP10/11 jobs and results are untouched.

Schema inspection completed on sh03-06n50, job 46425841. The 12 columns include
cell and brain identities, soma coordinates, manual-check status, source region,
projection class and cortical layer. No independent genetic label is present.
No cell-row values were decoded. This is not proof that a usable genetic
provenance join is unavailable elsewhere.

The [provider paper, Methods](https://www.nature.com/articles/s41467-024-54745-6)
identifies Supplementary Data 1 as brain-level acquisition, transgenic-label and
targeted-neuron metadata. Selected next step: acquire that exact public XLSX
into `runs/metadata_feasibility/attempt002/` and inspect its schema, using one
CPU/1 GiB/<5 minutes in job 46425841. A potential join will use brain identity,
genotype and acquisition metadata only; provider projection classes and axonal
values remain excluded. The earlier [molecular-lineage source](https://www.nature.com/articles/s41586-021-03941-1)
provides a lead for Cux2 and Plxnd1 labels, but neither source substitutes for
an authenticated cell-to-brain-to-genotype join or the minimum group support.

The public supplement was acquired successfully (93,468 bytes; SHA256
`293852b3c524de509886d68096606716c53c39338ca2c1a92b0569a7afc439f7`).
Its schema supplies animal ID, image ID, transgenic line, provenance and
targeted-neuron references. Selected `tools/join_brain_genotype_metadata.py`
next, one CPU/2 GiB/<5 minutes in job 46425841. It decodes only Full_morphometry
columns A/B/J/L (cell ID, brain ID, soma region and layer), excluding projection
class and every morphology/outcome value. It performs an exact brain-ID/image-ID
join and reports non-outcome support; no role or eligibility rule is selected.

The first join stopped on conflicting public supplement image identities
(`ValueError: Conflicting provider metadata for image identity`), before any
joined report or eligibility decision. The diagnosed repair retains all distinct
provider records and labels a source brain ambiguous if its exact key has
multiple records. Unrelated ambiguous public keys no longer abort all joins;
no ambiguous source brain is silently assigned a genotype. Retry selected
under the same one-CPU, five-minute bounds; no outcome or science rule changed.

The repaired join completed: 1,876 cell identities in 39 brain IDs; 38 brains
match the public genotype supplement exactly, with no ambiguous matched brain.
Brain 236174 (43 cells) lacks a matching image ID. Projection class, morphology
and target values were not decoded. Selected a one-CPU summary of animal/group
and layer support by genotype before judging whether 12 development plus 8
audit biological groups could exist; this is descriptive metadata support,
not cohort selection or an interpretation of axonal outcomes.

Support summary completed in job 46425841. The 38 matched brain IDs map to
38 distinct animal IDs. Explicit cortical L2/3/4/5 IT-targeting genotypes
span 6 Cux2, 5 Plxnd1 and 1 Tlx3 animals before any layer/donor/QC restriction.
This route therefore supplies at most 12 groups (13 even if the unmatched
brain were eligible), below the frozen 12 development plus 8 audit groups.
Gnb4's documented L6/claustral IT population does not meet the primary layer
scope. Other lines cannot acquire an IT label from forbidden projection-class
values. The result is a source-support limitation for this authenticated
labeling route, not a dendrite–axon association result or a proof that no
additional non-axon annotation could exist. Further primary execution needs
such an annotation or an explicitly approved downgrade. No roles, dendritic
features, targets, model fits or audit opening were produced.

## 2026-10-02 — Concept figure v3

Built-in imagegen produced the lean [own-versus-donor schematic](ep09_conceptual_question-v3.png);
targeted corrections removed ambiguous axon joins and a cross-species icon.
The [exact prompt set](ep09_conceptual_question-v3-prompt.md) is saved.
Reader links/captions now match the three-panel design and no longer describe
equivalence as a shared-class cause. Earlier assets remain. No scientific
contract, endpoint, data access or execution state changed.

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md) completed from existing design and primary literature. Correct-versus-donor equivalence now means no meaningful own-dendrite advantage at the tested scale, not identification of a shared-class cause. Prior read-only BR retrieval on the related projection question returned no relevant evidence; no outcome or execution state was newly inspected.

## 2026-09-27 — scientific design revision

- Reframed the question as whether a neuron’s own native dendrites add
  information about distal target families beyond source anatomy and
  axon-independent acquisition or dendrite quality.
- Separated an overall prediction gain from the stronger scale-by-target-family
  explanation needed for an in-depth paper.
- Defined the exact equal-target, equal-group Brier estimand and kept
  biological groups as the inference unit.
- Added 20 outcome-free, independent-block residual derangements as
  locked-pipeline capacity controls. Kept the cell-identity test distinct by
  swapping an intact dendrite within a frozen dendrite-only morphology class.
- Changed the target-hierarchy analysis to compare the dendritic increment under
  true versus matched shuffled families, so context-only base rates cannot
  create the explanation.
- Defined train-and-test conditional block replacement with 10 full refits
  rather than a post-fit feature permutation.
- Added fixed E0/E1 explanatory models containing every scale block, so an
  adaptively omitted feature cannot be mistaken for biological absence.
- Defined matrix replication through development-scaled cosine similarity,
  minimum norms, required signs, and nested pseudo-audits that relearn the
  entire template.
- Split small-group and max-`t` simulations into frozen 50,000-replicate
  calibration and validation sets.
- Split lower and upper small-group tails, including the cell-swap equivalence
  interval, so skewed group contrasts cannot silently miscalibrate one side.
- Made the development-frozen block-by-family matrix a prospective audit
  prediction rather than a post-hoc heat map.
- Added an outcome-dependent decision tree that distinguishes cell-specific
  organization, shared morphology-class organization, unresolved cell identity,
  diffuse information, unavailable explanation, invalid evidence, informative
  null, and unresolved data.
- No source morphology, development score, or audit outcome was opened during
  this revision.
- Generated `ep09_conceptual_question.png` with the built-in image-generation
  tool from a synthetic prompt; it contains no source neuron or outcome value.

Planning validation passed with no warnings. The final independent rereview
confirmed the tail calibration, nuisance logic, cell-identity branches, matrix
replication language, and revised decision figure; no protected outcome was
opened.

## 2026-09-29 — novelty-driven primary amendment

- Reclassified M1-correct versus M0 as a required benchmark rather than the
  paper's novelty claim after reviewing Peng 2021, Gao 2023, Liu 2024/2026,
  and Sorensen 2026.
- Promoted correct-versus-coherent-matched-donor prediction to the primary
  biological estimand, averaged across frozen donor plans and inferred at the
  biological-group level.
- Restricted the proposed headline population to cortical IT neurons whose IT
  membership is independent of axonal target outcomes.
- Made exact CCFv3 targets the decisive outcome resolution and changed the
  explanatory template to correct-minus-donor scale × exact target. Broad
  families remain secondary summaries and falsifiers.
- Corrected the source citations and recorded that SEU-A1876 is predominantly
  a refinement of the Peng-2021 data lineage, so held-out groups support
  internal grouped reproducibility rather than independent replication.
- Added a metadata-only feasibility gate for IT-label lineage, original versus
  newly annotated source lineage, exact-target support, and donor support under
  the frozen matching strata.
- Regenerated the conceptual figure to show the new primary hierarchy.
- No source morphology, development association, target prevalence, or audit
  outcome was opened during this amendment.

## 2026-09-30 — amended design completion

- Preserved the earlier revision records above; their prior implementation
  descriptions do not override the amended GOAL and search policy.
- Synchronized DATASETS with the fixed-model held-out donor substitution used
  by both the primary and fixed all-block explanation. The old ten full
  training-and-test replacement refits are superseded, not performed results.
- Added non-axon class to the M0 prose and aligned the positive primary,
  benchmark, E1 increment, norm and narrower-result interpretation across docs.
- Visually inspected the previously generated built-in imagegen output and
  copied it non-destructively to `ep09_conceptual_question-v2.png`. The old
  `ep09_conceptual_question.png` remains historical. The new schematic has no
  source neuron or observed heat-map/bar value. No image was generated or
  edited in this completion task.
- Replaced the obsolete single-analysis ASTRA snapshot with an experiment
  program for the amended contract. Pinned schema/official semantic tooling
  has not been located; official conformance is not claimed.
- No payload acquisition, source morphology or outcome read, job submission,
  model fitting, Git mutation, or canonical scientific-state mutation occurred.

### Current execution handoff, keyed to ASTRA analyses

| Analysis ID | State | Actual blocker / next action |
| --- | --- | --- |
| `metadata_feasibility` | Not started; no attempt or process/job ID | Source metadata not provisioned. On a separately authorized data task, obtain metadata and test independent IT labels, biological groups, lineage and donor support without outcomes. |
| `measurement_and_roles` | Not started | Requires a usable primary cohort; then establish cell/coordinate/measurement rules and whole-group shared roles. |
| `grouped_development` | Not started | Requires instantiated development measurements and roles; no fitted model exists. |
| `matched_donor_qualification` | Not started | Requires morphology, outcome-independent matching support and the development pipeline; no donor bank exists. |
| `nuisance_qualification` | Not started | Requires morphology and the development pipeline; no nuisance generator or fitted controls exist. |
| `explanation_and_inference_freeze` | Not started | Requires development and qualified controls; margins, exact-target template, simulations and resource feasibility remain unresolved. |
| `one_shot_group_audit` | Closed, not started | Requires every EP09/10/11 shared contract to be frozen and all primary dependencies satisfied; no audit has opened. |
| `paper_synthesis` | Design/paper plan only; evidence synthesis not started | Requires the locked audit report; no scientific finding exists. |

Completed durable deliverables are the revised design documents and v2
conceptual schematic only. No empirical artifact, active job, or completed
experiment is reported.

Focused completion checks passed: whitespace check, all seven current local
Markdown links, retention of the old PNG plus integration of v2, and search-
policy/ASTRA YAML parsing with duplicate-key detection in the existing
`encoding_model` environment (PyYAML 6.0.2). All eight ASTRA analysis IDs, 24
input references, output-input declarations and the acyclic dependency graph
pass the focused local check. Official ASTRA schema and semantic conformance
remain unverified. No package was installed.
