# Brain Researcher Autoresearch

**EP24 — harness improvement, October 13:** [Result-driven research harness](episode24_result_driven_harness/GOAL.md)
uses the original session's recorded action history to select a small execution
change and evaluate it on actual authorized result-producing work. See its
[current evidence](episode24_result_driven_harness/outputs/README.md).
Creating this episode does not validate a harness improvement or change the
scientific contracts and job ownership of the existing episodes.

Browse the [EP01-EP21 figure gallery](FIGURES.md) for current study schematics
and links to each episode's scientific goal. Scope, prior work, and claim limits
are maintained in the episode documents, not separate review reports.

We have Brain Researcher as an analytical tool. The next step is to see whether
this kind of agentic workflow can be extended into autonomous, self-evolving
research episodes.

We plan to run more than 100 neuroimaging research episodes. We want to use
them to do two things:

1. accelerate neuroimaging discovery; and
2. ask whether we are truly accelerating discovery, or simply automating
   p-hacking and analysis selection.

We assume that an autoresearch episode can be launched, can run, and can reach
a clear ending. We do not assume that it will produce a positive scientific
result. A good episode may find a candidate, rule out an idea, finish with no
candidate, or identify a technical limitation. All of these outcomes should be
kept.

The research question can be narrow or very open. We deliberately want a large
search space: different explanations, estimands, representations, models,
controls, validation strategies, and datasets may all be considered. One aim
of the campaign is to see how broadly an agent can search, how deeply it can
follow the most promising branches, and whether that produces more informative
and reproducible science.

The episode's scientific policy sets the breadth, depth, and resource limits
of its search. An ordinary bounded-search lane maps competing directions and
selects a small set for deeper falsification. The
[adaptive-search protocol](ADAPTIVE_SEARCH_PROTOCOL.md) supports many more
outcome-adaptive successor cycles inside a frozen grammar, append-only ledger,
finite round budget, and one-shot audit. Across more than 100 episodes, we can
study both breadth and depth without confusing a larger search with a license
to select whichever analysis happens to look best. Here, a better result means
a sharper test, a more reproducible or transportable finding, or a
well-supported reason to stop. It does not mean a smaller `p`-value.

## Online resources in the episode notes — October 6, 2026

Primary-source web lookups now cover every episode. Each linked `DATASETS.md`
section records concrete available information, its use for that episode, and
what the source cannot establish. These are documentation and source-discovery
additions; acquired versions and scientific data roles remain as recorded.

| Episode | Online information added directly to its data notes |
| --- | --- |
| [EP01](episode01_narps_deep_search/DATASETS.md#public-online-resources--checked-october-6-2026) | NARPS participant release, results-version history and original analysis code. |
| [EP02](episode02_dopamine_learning_rate_causality/DATASETS.md#public-online-resources--checked-october-6-2026) | Author learning-model and event-alignment utilities; empirical release lookup limits. |
| [EP03](episode03_condition_semantics_effect_maps/DATASETS.md#public-online-resources--checked-october-6-2026) | BART contrasts, event fields and FitLins input-filter semantics. |
| [EP04](episode04_shared_scene_geometry/DATASETS.md#public-online-resources--checked-october-6-2026) | LAION captions, embeddings, segmentation coverage, beta mapping and access terms. |
| [EP05](episode05_sensorimotor_lfp/DATASETS.md#public-online-resources--checked-october-6-2026) | Public recording inventory, author CCA preprocessing and TrialData dependency. |
| [EP06](episode06_lfp_regional_fingerprints/DATASETS.md#public-online-resources--checked-october-6-2026) | Paired-session release and guide-based spike/LFP analysis methods. |
| [EP07](episode07_lfp_session_transfer/DATASETS.md#public-online-resources--checked-october-6-2026) | LMP versus power construction and the limits of downstream spike transformations. |
| [EP08](episode08_lfp_electrode_trial_policy/DATASETS.md#public-online-resources--checked-october-6-2026) | Exact public session listings, task documentation and source-code organization. |
| [EP09](episode09_local_global_projection/DATASETS.md#public-online-resources--checked-october-6-2026) | Native/CCF/arbor archives and provider specimen/genetic identity metadata. |
| [EP10](episode10_single_cell_coprojection/DATASETS.md#public-online-resources--checked-october-6-2026) | Original Gao reconstruction archive and alternative distribution locations. |
| [EP11](episode11_projection_types_vs_gradients/DATASETS.md#public-online-resources--checked-october-6-2026) | Reconstruction release identity and published access routes. |
| [EP12](episode12_malecns_type_sufficiency/DATASETS.md#public-online-resources--checked-october-6-2026) | MaleCNS graph, annotations, API interfaces and skeleton coordinate units. |
| [EP13](episode13_bold_target_identifiability/DATASETS.md#public-online-resources--checked-october-6-2026) | Primary classical reference for estimability and generalized inverses. |
| [EP14](episode14_openbhb_roi_site_generalization/DATASETS.md#public-online-resources--checked-october-6-2026) | OpenBHB preprocessing/ROI definitions and pinned benchmark documentation. |
| [EP15](episode15_mdtb_topography_vs_geometry/DATASETS.md#public-online-resources--checked-october-6-2026) | MDTB derivative release, session/condition conventions and spatial sampling code. |
| [EP16](episode16_braingate_cross_session_failure_modes/DATASETS.md#public-online-resources--checked-october-6-2026) | BrainGate units/indexing codebook, participant/array metadata and decoding methods. |
| [EP17](episode17_cneuromod_model_ranking_stability/DATASETS.md#public-online-resources--checked-october-6-2026) | CNeuroMod processing paths, stimulus/annotation documentation and access guide. |
| [EP18](episode18_things_eeg_concept_stability/DATASETS.md#public-online-resources--checked-october-6-2026) | THINGS-EEG study documentation, exact runtime distributions and CLIP configuration. |
| [EP19](episode19_causal_motor_forecasting_benchmark/DATASETS.md#public-online-resources--checked-october-6-2026) | WAY event/context codebook, sensor conventions and EEG importer scaling. |
| [EP20](episode20_neurocam_prior_guided_codesign/DATASETS.md#public-online-resources--checked-october-6-2026) | NeuroCam DAQ/processing methods, manufacturer manuals and fixed DANDI release metadata. |
| [EP21](episode21_wang_neural_code_conversion_reproduction/DATASETS.md#public-online-resources--checked-october-6-2026) | Pinned reproduction code/pretrained manifest and THINGS, NSD and LAION access documentation. |

## Which skill to use

### Active execution coordination — 2026-10-03

The scientist instructed the coordinator to resume possible work and finish
the existing questions across EP01–EP21. Resume the operative contracts;
unactivated narrative follow-ups remain separate. One coordinator owns Slurm
submissions, scheduler checks and shared compute sessions; each worker owns
one explicit episode scope at a time. Keep actual attempts, job IDs, consumed
resources and evidence in each episode's `outputs/experiment_log.md`.

The user additionally requested multi-agent monitoring and job/process
babysitting. Episode workers prepare code and independent scientific reviews
in separate writable scopes; the coordinator checks scheduler and resource state, owns compute
sessions, and acts on diagnosed failures. No duplicate pending jobs, background
supervisors or automatic allocation renewals are created. Detailed evidence
and actual job/attempt records remain in the linked episode logs.

### Current episode navigation — October 6, 2026 Pacific

The table summarizes the recorded episode evidence, not a live scheduler check.
The October 6 document revision incorporates the scientist's accepted
judgments: EP02 article/task semantics as primary, EP15 exploratory development
analysis, original-comparison closeouts for EP10/11/12/16, EP20's explicit
simulation exception, and EP21's bounded parallel decoder continuation.
The episode contracts define each amendment's exact scope; optional proposals
outside those decisions remain unadopted.

The October 6 public-source follow-up is incorporated into the existing
handoffs and logs for EP02, EP03, EP06, EP07, EP09, EP11 and EP16–EP20.
Each linked episode distinguishes
verified online documentation from unresolved acquisition/execution evidence;
these additions do not adopt draft scientific amendments or new jobs.

The broader October 6 reconciliation also covers EP01–EP21's existing
results, data preparation and accepted decisions. Episode DATASETS, plans and
current navigation now connect to that evidence; stale claims that completed
work or adopted choices are still pending have been corrected. Already
complete accounts are retained, and descriptive policy fields are synchronized
only where their established provenance is explicit. Historical attempts and
unadopted proposals retain their original status.

| Episode | Recorded result or actual obstacle | Next action / judgment |
| --- | --- | --- |
| [EP01](episode01_narps_deep_search/outputs/README.md) | Observed behavior and twelve held-out fit directions complete for 108 people / 27,648 trials; saved-score contrasts complete. Mechanisms remain unidentified. | Preserve exploratory results. Old simulation requests are superseded by the October 5 stop. |
| [EP02](episode02_dopamine_learning_rate_causality/outputs/README.md) | Authenticated article/task semantics are adopted as primary; the pinned code is report-only sensitivity. A generic training-only grouped-binomial optimizer is implemented and reviewed; all six checks and updated ASTRA validation passed in 46845646. Exact time axis, trigger/codebook and assignment facts remain unresolved. | Complete the observation recipe and population-predictive integration. Retain the targeted provider request. No scientific observation family, penalty or missing empirical fact was invented by the optimizer. |
| [EP03](episode03_condition_semantics_effect_maps/outputs/README.md) | Public BART model/details provide 13 contrast definitions and 16 configured subjects; actual historical fitted contributors remain unknown and only 5/13 canonical orientations are resolved. | Use the public methods sources to narrow the producer request to actual fitted-input records and remaining event/ontology ambiguity. Configured subjects do not establish fitted N. |
| [EP04](episode04_shared_scene_geometry/outputs/README.md) | All 23,560 development images have finite low-level controls; the 400-comparison panel, sensitivities and null panel are complete. Closed with no candidate. | Data/source notes now reflect complete development coverage. No pending raw-image-control approval or automatic successor. |
| [EP05](episode05_sensorimotor_lfp/outputs/README.md) | Original 45-comparison search and selected calibration/decomposition follow-ups complete; practical margins still fail. | Preserve results and stop; old pending requests are not current work. |
| [EP06](episode06_lfp_regional_fingerprints/outputs/README.md) | All 12 sessions' metadata ready; the public Dryad/code review does not resolve EP07's shared count defect or the role/exposure conflict. | Corrected counts first. Joint session-drop fallback requires sample-size review and dated-role preservation. |
| [EP07](episode07_lfp_session_transfer/outputs/README.md) | Mihili March 6 has invalid native counts in both regions/all 217 trials. The inspected public Dryad release has no visible correction notice; analysis transformations do not authenticate an inverse repair. | Retain the targeted corrected-export request. Dropping that day also changes EP07's new-day rules; no such amendment was adopted by adding sources. |
| [EP08](episode08_lfp_electrode_trial_policy/outputs/README.md) | The 41-comparison real-data search across eight development sessions is terminal without a robust initializer policy. The +0.003353 Q average has session/initializer reversals. | Preserve the bounded result and all four closed held-out sessions; the original reference design is not unfinished mandatory work. |
| [EP09](episode09_local_global_projection/outputs/README.md) | Current metadata bounds remain two animals / 72 cells in ten sufficiently large recorded strata, not qualified eligibility. Job 46842156 authenticated the provider alias: Animal ID 236174 = Image ID 16124, genotype Gnb4-IRES2-CreERT2;Ai139. The historical join remains preserved. | Identity/genotype no longer needs a provider answer. Obtain authentic layer/non-axon-class and matched-donor support; the alias adds no qualified IT animal. Neither the original 12+8 nor proposed nine-animal route is supported. |
| [EP10](episode10_single_cell_coprojection/outputs/README.md) | Real-data M1's 949 fits passed postflight across 194 cells / 12 conservative groups, with +0.0026378637 bits/cell exploratory gain. The over-budget M2 branch is closed as `incomplete_search`. | Preserve exploratory M1 and the measured compute limitation. The accepted closeout establishes no negative M2 finding. |
| [EP11](episode11_projection_types_vs_gradients/outputs/README.md) | The original cross-animal comparison is closed as incomplete: qualification, identity/completeness and support remain unresolved after the execution window. | Preserve completed evidence. The optional ≤48 CPU-hour exploratory design remains a draft; no recovery successor or new budget is activated. |
| [EP12](episode12_malecns_type_sufficiency/outputs/README.md) | Observed tables and figure complete for 816 types / 84,735 neurons, including 126 zero-mass neurons. Original T/U/M comparisons are closed as incomplete. | Preserve the one-specimen descriptive result and closed final/incoming scope. The separate decoder proposal remains unadopted. |
| [EP13](episode13_bold_target_identifiability/outputs/README.md) | Theoretical result, proof package, 12 exact fixtures, seven rejection cases and independently authored replay are complete. | Source/plan notes now link the completed mathematical evidence. No empirical replacement needed. |
| [EP14](episode14_openbhb_roi_site_generalization/outputs/README.md) | Original no-candidate result and selected real-MRI follow-ups complete; no overall improvement established. | Preserve results and closed holdout; no automatic successor. |
| [EP15](episode15_mdtb_topography_vs_geometry/outputs/README.md) | All 24 participants / 36 containers / 36,576 members are materialized; roles, support, metrics and M3 fitting checks are established. Exploratory analysis of the existing 12 development participants is adopted, with τ_remaining=.20 and τ_increment=.05. The paired M4 fitter is implemented; review corrected a scaling defect, then all seven checks and updated ASTRA validation passed in 46845646. | Deploy the enforced development reader and finish the fixed model recipe. Local optimization does not certify uniqueness or locality. Formal class selection, uncertainty qualification and audit remain deferred; the other five margins are not silently adopted. |
| [EP16](episode16_braingate_cross_session_failure_modes/outputs/README.md) | Structural inventory covers 729 sessions / 328,548 trials / 5,150 blocks. Only T11/T9 conditionally qualify; the original three-held-out-participant claim is closed as `support_or_uncertainty_prevents_an_answer`. | Preserve the inventory and completed arithmetic. An optional T11/T9 case study remains unadopted; general field documentation does not establish missing session facts or expand support. |
| [EP17](episode17_cneuromod_model_ranking_stability/outputs/README.md) | The 720-concept universe, 480/120/120 roles, N=256/q=32 structural rule, numerical conventions and margins are adopted; exact EP18 overlap is recorded. The provider README identifies password_images.txt; private delivery and reader separation remain unresolved. | Policy fields and current methods now reflect the October 4 decisions. Prepare the documented private delivery route under existing accepted terms; no repeated margin ratification is needed. |
| [EP18](episode18_things_eeg_concept_stability/outputs/README.md) | Metadata retain all 1,133,808 events and the fixed 30/16 participant assignment; EP17 overlap is resolved as correlated exposure. Exact stable runtime, all five public model assets and all 1,854 name features are complete. The sequence-local joint ridge/ablation core is implemented; all ten checks and updated ASTRA validation passed in 46845646. | Preserve completed preparation. Enforced image/EEG separation, remaining image features and the empirical runner are unfinished. The attended shell is closed and the stopped namespace route remains stopped. No evaluation result or overall feature freeze is claimed. |
| [EP19](episode19_causal_motor_forecasting_benchmark/outputs/README.md) | The 12-person S1 support panel is complete: seven pass conditionally, five remain unresolved. All 16 classical/non-neural backend checks passed. Public event/sensor conventions are documented; upstream stream origin, physical scaling and full clock bound remain unresolved. | Obtain authentic exporter/origin and calibration evidence. Neither direct StartTime copying nor 2 ms window indexing justifies a 2 s correction. Preserve the 300–600 ms horizon and separate scientific-gain judgment. |
| [EP20](episode20_neurocam_prior_guided_codesign/outputs/README.md) | The scientist explicitly allows simulated co-design. Seven R1, six loading/noise and eight dense coupled-circuit checks have passed. The sparse implementation retains all 4,096 pads plus 64 source states; all five checks and updated ASTRA validation passed in 46845646. DAQ/calibration context and Rat5 mapping remain unresolved. | Advance the remaining contact/CPE/noise model with supported parameters. Simulation approval does not supply missing acquisition facts, qualify the reference, or establish measured candidate performance. |
| [EP21](episode21_wang_neural_code_conversion_reproduction/outputs/README.md) | All 24,681 regular-image feature rows / 19 layers and 151,861 regular neural repetitions are provisioned. The serial job was gracefully checkpoint-stopped, retaining 716,160 total units. Full continuation is adopted: 500 allocated CPU-hours including job 46801996, up to eight 1 CPU / 7 GiB workers, zero GPUs and 500 GB new durable output. All four new participant states completed in job 46833971 (3,479 CPU-seconds); all required storage checks and changed document validations passed. At October 6, 21:08 Pacific, all eight workers remain running: 46839670_0–6 and corrected task 46840840_7. The replacement passed all 27 repair/accounting checks; the batch targets 8,015,424 additional units and logs report 3,600,064 newly fitted units across all eight workers, pending terminal validation. A 15-second pre-admission scheduler-ID failure is preserved. | Monitor the eight-worker batch and reconcile exact terminal results before the next finite batch. Actual spend plus all current full reservations totals 69.7456 of the 500 CPU-hour cap. The [adopted continuation](episode21_wang_neural_code_conversion_reproduction/outputs/decoder_training_proposal.md) covers decoder training only; scoring, converters, reconstruction, tau and OOD remain closed. |
| [EP22](episode22_geometry_gap_decomposition/outputs/README.md) | Terminal: `confirmation_complete` (2026-10-13). This is a diagnostic of EP04, which found that native-RSA gaps are spectral; VLM eigenvectors match DINOv2 at coarse scale. The single procedurally sealed OOD opening passed the gate. C2a passed: VLM spectra penalize within-category RSA, 5/5 for CLIP and SigLIP2. C3 passed: development-fitted steepening (α ≈ 2) transfers, 5/5 for every VLM. C1 (spectrum vs. selection) is indeterminate, with a CLIP deficit only on selfmade. The PEcore secondary prediction failed. Total compute was 26.43 CPU-hours. | Report the result. A theory episode could build on the confirmed spectral findings and on the post hoc observation that eigenvector agreement depends on scale. |
| [EP23](episode23_spectral_operator_convergence/outputs/README.md) | A0 complete (exploratory, 2026-10-13). A Visual Genome co-occurrence operator predicts each participant's leading high-level visual brain subspace in 5/5 participants (ω 0.067–0.095 vs null ≈0.04; sign p = 0.031). The result is robust to frequency and low-level pixel controls, but not established beyond the 27-category taxonomy (3/5 positive, sign p = 0.50; captions operator 5/5 positive, 3/5 significant). Brain alignment is confined to the leading 8 modes. Models align beyond taxonomy, including DINOv2 (descriptive). About 2.55 CPU-hours, no GPU. | Remaining initial-stage work: P1–P5 and A1–A3. A4 (independent confirmation, ≥5 participants) needs a scientist decision on its source. |

### Remaining source and design dependencies

- **Evaluation separation (EP02/15/17/18):** a separate agent or same-account
  file permissions do not enforce separation. A correctly restricted runtime
  or an independent human steward may enforce it. Procedural separation is a
  weaker, explicit alternative requiring revised access/claim rules; it is not
  silently substituted for an existing enforced boundary.
- **Bounded validation:** fixed-seed known-answer and leakage checks can be
  part of real-data work. Proposed real-data permutations require justified
  exchangeability and a finite compute cost. Recovery, power, qualification
  sweeps and refit-null expansions remain outside the October 5 priority unless
  separately selected. A handful of checks cannot establish uncertainty coverage.
- **Provider requests:** prioritize EP07 corrected counts, EP03 fitted-input
  provenance and EP02 field/trigger/assignment definitions. The episode
  handoffs contain drafts for the user to send; agents do not contact providers.

Remaining consequential choices include evaluation separation, the EP06/07
fallback and EP19 meaningful gain; EP15's formal inference is deferred under
its accepted exploratory route. Closeouts for EP10/11/12/16, EP20 simulation
permission and EP21's aggregate compute/storage envelope are answered.
BR advisory review is not a canonical BR-harness launch or acceptance;
execution evidence remains in the episode logs.

### Historical execution snapshots

The dated observations below preserve prior states. They do not override the
current navigation or authorize restarting cancelled/terminal branches.

**October5,18:37UTC — real-data priority across EP01–EP21.** The scientist
instructed the coordinator to stop expanding simulations, cancel running
simulation jobs and prioritize useful analyses of available real data. All88
remaining EP01 synthetic parents are cancelled; the18:34 user queue is empty.
Existing outputs, failed attempts and allocation charges are preserved. Do not
resume EP01 recovery/repair/bootstrap waves, EP11 recovery successors, EP12
oracle successors or other unselected simulation expansions under this priority.
Ordinary necessary code checks remain part of the substantive empirical work.

**October 5, 19:18 UTC:** [EP01's observed-data branch](episode01_narps_deep_search/outputs/README.md)
is complete: 108 people, 27,648 trials, all twelve held-out fit directions for
six fixed models converged with complete scoring support. The saved report and
figure describe actual behavior and exploratory within-session prediction.
All parents are terminal; the branch consumed 0.6758333333 allocated core-hours,
including checks and two shared EP04 diagnostics. Recovery and psychological
mechanism/neural claims remain incomplete. By 19:35 UTC, EP04's training-only
reliability/noise-ceiling components completed for all five participants,
25 folds and 100 ROI units. Codex and Slurm execute this work. BR has supplied advisory
material; no canonical BR-harness launch or acceptance is represented here.

October5,15:13UTC: [EP01's saved offer-sequence influence](episode01_narps_deep_search/outputs/README.md)
completed all16 single and120 pair omissions with positive EI/ER mean gains
throughout. [EP15's role inventory and sub-08 copy pilot](episode15_mdtb_topography_vs_geometry/outputs/README.md)
completed; full unreleased materialization46625544 also completed all36
containers/36,576 members with four workers joined and exact-byte checks.
Its0.2166666667core-hour parent cost is charged once. Reader deployment and
scientific qualification remain separate. [EP18 runtime preparation](episode18_things_eeg_concept_stability/outputs/README.md)
installed the selected project dependencies, but model work is stopped on
actual Torch/vision versions differing from the pinned releases and separate
DTN authentication. No EP18 inference or EEG audit opening occurred.
Shared attended46624388 is closed and its actual parent0.6377777778core-hours
is charged once across EP01/EP15/EP18. The15:12UTC queue is empty. EP20's
passed11 component checks do not
resolve its missing electronics/DAQ context or Rat5 acquisition mapping.

October5,14:23UTC: EP01 validates199 MZ menus, all with MZ alone maximal
eligible (held index1 unknown), and all200 null menus, all with M0 alone
maximal eligible. All selected old numerical repairs are terminal and accounted
once. The twice-effect pool has200 valid necessary successes/no valid failures;
all200 half-effect components supply continuous diagnostics without a90% criterion.
Both effect arrays are terminal and accounted. The original nonlinear M0/MZ
pair ended with iteration-limit failures; M0 repair46611338 also exhausted1500
first-half sweeps, producing no nonlinear scores. Its1.7575 core-hours are
charged once. Direction-specific repair46618614 completed the fresh reverse
M0 fit at348 sweeps and first direction at1546 sweeps under its2000 ceiling.
All108 participants are scored. Both four-worker fits retain the original
priors, likelihood and tolerances. The parent is COMPLETED and its4.2577777778
core-hours are charged once, including the two urgency diagnostic steps.
All twelve focused checks, ASTRA and actual source/context/replay checks pass.
MZ reverse repair
46612046 completed: fresh reverse converged at736 sweeps, the preserved153-sweep
first fit was rescored, and its1.0144444444 core-hours are charged once.
The combined nonlinear comparison is complete: mean MZ-minus-M0 gain0.01104355
nat/trial, SE0.00110184, positive EI/ER signs and both nested numerical criteria
met. This one-cohort result is not full robustness or recovery qualification.

Both alternate partitions and four leave-one-run-out fits completed; pooled
LOO gain is0.0134485815 nat/trial. The urgency initialization pilot completed;
its48-call batched comparison exactly matches saved scalar scores with10.41x
coarse/6.26x refined speedups, without fitting or inference. Eight parallel
conditional optimization-cost probes in46612782 are terminal: three returned
all starts and five reached the3000s limits. Fourteen of17 returned starts
report line-search failure. The6.6977777778 core-hour cost is charged once;
a six-call endpoint diagnostic completed with exact scalar/batched agreement
and a-0.2040 discrepancy from the unsuccessful optimizer's reported objective.
The unchanged original-start trace completed490 evaluations and reproduced
the unsuccessful optimizer result exactly: returned x and reported fun match
different recorded evaluation states. The line-search cause remains unresolved;
full urgency fitting remains unselected. Saved-trace analysis completed inside
46618614 without new likelihood calls: repeated x reproduces scores exactly,
while one trial's bin13→12 change accounts for most of the nearby objective
increase. Five derivative stencils also cross grid/bin boundaries. This
supports further numerical diagnosis, not a causal failure or repair verdict.
The subsequent five-call crossed diagnostic completed in55.73s with four
focused checks and ASTRA passing. A0.951-microsecond nondecision-time change
reproduces about0.2041 objective increases under both fixed backgrounds;
scalar/batched scores agree exactly and the same single trial changes bins.
Horizon/actualdt also change, so no bin-only or optimizer-repair claim follows.
That selected diagnostic is complete; no further unchanged repeat is selected.
A separate two-point A/C check at the registered finer mesh completed in12.11s:
the objective contrast changes from+0.204105 to-0.000024538 and no fine-grid
trial changes bins. This establishes resolution dependence, not fitted-support
precision or an adopted repair. Three focused tests and ASTRA passed; attended
46621405 closed, costing0.0433333333 core-hours once. The14:21 UTC user queue
is empty. A [saved-result figure](episode01_narps_deep_search/outputs/figures/mz0_robustness_diagnostics.png)
now compares the completed primary, common-gain and nonlinear diagnostics.
The [reviewed numerical-method proposal](episode01_narps_deep_search/outputs/urgency_numerical_method_followup.md)
awaits the scientist's decision on an isolated twelve-call pilot; no method
change or production adoption is activated.
The required common-gain10–20 sensitivity46613139 ended at
both first M0 fits'500-sweep limit; no comparison was produced. Four independent
components in46615238 completed both M0 fits and direction0 MZ; only reverse
MZ exhausted500. Their1.59 core-hour parent cost is charged once. Reverse MZ
repair46617879 converged at565 sweeps using four participant workers; the
three prior complete components are preserved. The common-gain pool now
validates all four: mean MZ-minus-M0 gain0.01276781 nat/trial, SE0.00148314,
positive EI/ER gains and both nested criteria met. This is a one-cohort
synthetic diagnostic, not full robustness or recovery. All eighteen new
repair/pool/trace checks and ASTRA passed. The5CPU parent completed both
branches and its1.1902777778 core-hour allocation cost is charged once.
The finished
interactive allocation has been exited. Full recovery, interval coverage and false-selection
rates remain unresolved; observed behavior/BOLD stay closed. Exact attempts,
consumed resources and evidence remain in the EP01 log.

EP20 has no running or pending job. Its eleven contact-equation checks passed;
reference fitting and candidate search still need measurement/DAQ inputs and
Rat5 geometry/acquisition history. These are standalone Slurm jobs with BR
advisory assistance, not a canonical BR launch. See
[EP01's current evidence](episode01_narps_deep_search/outputs/README.md) and
[EP20's required inputs](episode20_neurocam_prior_guided_codesign/outputs/README.md).

Earlier execution observations follow.

October5,01:44UTC: EP01 has217/219 selected array tasks completed/exit0 and
two still running, plus two active MVZ/MVC jobs. Dependent job46547708 will
summarize all200 MZ draws and run one paired bootstrap cost pilot only if the
complete group-bias gate passes. The attended allocation is released. Latest
scientific pooling remains186 valid necessary successes; full recovery is
unresolved. See [EP01's current evidence](episode01_narps_deep_search/outputs/README.md).

Earlier execution observations follow.

October5,01:37UTC: EP01 has186 valid MZ necessary successes, already exceeding
the required180;14 fixed draws continue for the complete group-bias assessment.
MVZ/MVC46545300/46545303 are running. The next paired-refit cost pilot passes
nine tests, independent review and ASTRA validation; it awaits the complete
point-bias result before execution selection. Full recovery remains unresolved.
See [EP01's current evidence](episode01_narps_deep_search/outputs/README.md).

Earlier execution observations follow.

October5,01:29UTC: EP01's MV necessary cell fails with21 valid failures
(maximum179/200). MZ's latest pool has140 valid successes; its group estimates
are provisional. The MZ0 MC/MV diagnostic fits completed, and22 new tests pass.
MVZ/MVC jobs46545300/46545303 are submitted after a diagnosed wrapper-startup
repair; failed attempts and costs are preserved. The recovery arrays continue.
See [EP01's current evidence](episode01_narps_deep_search/outputs/README.md).

Earlier execution observations follow.

October5,00:51UTC: EP01 has219 selected array tasks across three arrays.
Terminal pooling records11 valid MV failures and10 valid MZ successes;
neither is a full recovery verdict. Nine new focused tests and ASTRA validation
pass, and four MC/MV rival fits on the saved MZ0 cohort are running in the
existing attended allocation, with a50-minute process cap. See
[EP01's current evidence](episode01_narps_deep_search/outputs/README.md).

Earlier execution observations follow.

October5,00:38UTC: **179 further MZ draws submitted as46539590** after the
representative converged and passed its necessary screen in20m24.3s. This
schedules all200 fixed MZ draws; the earlier MV/MZ arrays remain, totaling
219 selected array tasks. One MZ success and one MV failure are not cell
verdicts. Existing scientific/data gates stay intact; root owns monitoring
and pending-task curtailment. See [EP01's log](episode01_narps_deep_search/outputs/experiment_log.md).

Earlier execution observations follow.

October5,00:33UTC: MV0 completed all four fits but its0.006791nat/trial gain
misses the fixed0.01margin: one valid recovery failure, no cell verdict.
MZ0 and all40 array draws continue. EP01's seven pooling tests and EP02's
nine pinned-critic checks passed; both changed ASTRA programs validate.
The attended allocation now supports terminal-result analysis. See
[EP01's result](episode01_narps_deep_search/outputs/experiment_log.md)
and [EP02 checks](episode02_dopamine_learning_rate_causality/outputs/experiment_log.md).

Earlier execution observations follow.

October5,00:32UTC: **all42 selected EP01 recovery tasks are running**—two
representatives plus40 array draws. All array model/index/CPU startup headers
are verified; no terminal recovery result is claimed. Root monitors and owns
pooling in the bounded attended allocation when its CPU becomes free.
See [EP01's current log](episode01_narps_deep_search/outputs/experiment_log.md).

Earlier execution observations follow.

October5,00:25UTC: the scientist requests more parallel jobs. **Forty new
EP01 recovery tasks are submitted**: MV array46538647 and MZ array46538652,
each fixed indices1–20 and up to20 concurrent1CPU/2GiB tasks. The two original
representatives remain running; total selected concurrency can reach42CPUs.
Scientific cells, seeds and21-valid-failure stopping rule stay fixed. Root
monitors scheduler/processes; workers implement and independently review the
required pooling arithmetic. See [EP01's log](episode01_narps_deep_search/outputs/experiment_log.md).

Earlier execution observations follow.

October5,00:10UTC: **EP01's declared MV/MZ recovery alternatives are launched**:
MV representative0 in attended46535708; MZ representative0 submitted46536895.
Eight runner checks and independent review pass. Each bounded process uses one
CPU/2GiB for at most90minutes; original MC/full-family failure remains intact.
EP02's required scoring arithmetic passed12 grouped tests without source access.
These remain standalone Slurm launches; BR criteria are advisory, not harness
dispatch. See [EP01's log](episode01_narps_deep_search/outputs/experiment_log.md)
and [EP02 outputs](episode02_dopamine_learning_rate_causality/outputs/README.md).

Earlier execution observations follow.

October4,23:55UTC: **allocation46531931 completed normally**: EP17's33
numerical checks pass and the exact EP17/EP18 overlap is now documented in both
episodes. All3,634 eligible EP17 images/720concepts occur in EP18's main
inventory; retain both designs and disclose correlated, design-exposed evidence.
Both changed ASTRA programs validate and the compute shell is closed. Bulk
EP17 work still needs the existing private image credential interface and
resolution of the stopped quota/copy route; EP18 runtime isolation remains
unavailable. See [EP17's log](episode17_cneuromod_model_ranking_stability/outputs/experiment_log.md)
and [EP18's exposure record](episode18_things_eeg_concept_stability/outputs/config/ep17_ep18_exposure_record.json).

Earlier execution observations follow.

October 4, 23:50 UTC: **EP17 passed33 focused numerical checks** in bounded
allocation46531931. Exact endpoint/audit conventions are adopted; two workers
and root are completing a metadata-only EP17/EP18 overlap comparison in the
remaining allocation. Image decoding still needs the private credential
interface, and the quota/copy route remains stopped. These remain standalone
Slurm launches with BR scientific criteria used as advisory guidance.
See [EP17 outputs](episode17_cneuromod_model_ranking_stability/outputs/README.md).

Earlier execution observations follow.

October 4, 23:30 UTC: EP17's **anatomical mapping, archive metadata inspection
and structural quota selection completed** in allocation 46527997: 881,359
columns preserved, all 32 image members matched, and N=256 with 32 voxels per
spatial bin frozen. The reviewed prediction core passed its compute tests.
The allocation closed normally; no active job remains. Neural copying remains
stopped after the quota-query failure, and image decoding needs the existing
private credential interface. These are standalone Slurm launches; BR
scientific criteria remain advisory. See EP17's
[current outputs](episode17_cneuromod_model_ranking_stability/outputs/README.md).

Earlier coordination observations follow.

October4,22:55UTC: EP17 **46525781 is queued**,1CPU4GiB/20minutes, for
anatomical tests and the adopted structural mapping. Model-core checks will
share that allocation if ready. The feature definition has passed independent
review and is adopted, with unknown image exposure stated explicitly. No
neural-copy or model-score run is launched; the quota route remains stopped.
See EP17's [current outputs](episode17_cneuromod_model_ranking_stability/outputs/README.md).

Earlier coordination observations follow.

October4,22:46UTC: EP17 copy safeguards passed10 tests, but Oak's quota query
timed out and returned cached data. The required current capacity check is
unresolved, so the neural-copy pilot stopped before launch or payload access.
Independent anatomical mapping implementation is adopted and underway;
model-registry preparation is delegated separately. EP05's final development
closeout remains complete. Details are in EP17's
[execution log](episode17_cneuromod_model_ranking_stability/outputs/experiment_log.md).

Earlier coordination observations follow.

October4,22:33UTC: **EP17's full metadata assignment completed** in21.60seconds
under46523446. All720concepts now have480/120/120roles plus fixed folds/blocks,
preserving4320images/49980presentations. Neural/model outcomes remain closed.
The next adopted task is the bounded trusted-builder literal-copy pilot,
conditional on capacity/layout/implementation checks; it will not activate
reader handoffs or scoring. EP05's45-comparison development closeout is final.

Earlier coordination observations follow.

October4,22:27UTC: **EP17 full assignment submitted as46523446** after23
fixtures and independent reviews passed,1CPU6GiB/35minutes,no-requeue. Root
owns monitoring. Shared preparation46521343 closed normally at22:27:18UTC:
1342seconds/oneCPU,0.3727777778core-hours total, split once across EP05/EP17
as0.1863888889 each. No interactive compute shell remains. EP05's inclusive
final documented bound is19.249core-hours; its scientific closeout is complete.

Earlier coordination observations follow.

October4,22:21UTC: EP17's metadata pilot completed16.31seconds, with720
concepts/237 fields,4096 exact delta checks and no roles assigned. Conservative
all-four-assignment planning estimate1485.70seconds supports1800s internal/
1860s external in one fresh35-minute1CPU6GiB job after final code/tests.
The existing allocation46521343 is reused for preparation and will close
before that submission. EP05's scientific development closeout remains final.

Earlier coordination observations follow.

October4,22:17UTC: **EP05 primary development search stopped validly** at45
comparisons/patience12, with no candidate meeting its practical requirements.
Its [scientific report](episode05_sensorimotor_lfp/outputs/primary_development_report.md)
preserves small positive predictive effects, negative practical eligibility,
all evidence limits and the conservative resource bound. EP17's metadata-only
pilot is next in the currently owned allocation after fixture review.

Earlier coordination observations follow.

October4,22:16UTC: EP05's complete16-session basis diagnostic passed with one
distinct comparison:45 valid, patience12/12, unchanged incumbent and no
primary-eligible candidate. Final stopping/resource review is underway; figures
and complete compact evidence are linked in its outputs. Current46521343 is
verified RUNNING on sh03-01n68 through22:39:59UTC and can sequentially host
EP17's bounded metadata pilot after focused checks. EP17 role code and
independent review are underway; no neural outcomes or role maps are released.

Earlier coordination observations follow.

October4,22:07UTC: EP05 basis-only allocation46521343 is RUNNING on sh03-01n68
(1CPU/6GiB,Requeue0,ends22:39:59UTC). Ten focused fixtures and changed ASTRA
validation passed; pilot completed22.868s and remaining15-session/full-summary
basis diagnostic runs in root's PTY36735. EP17 concept-role specification is
adopted for implementation/metadata timing pilot; no roles or outcomes opened. Earlier EP15/EP17 preparation completed below.

October4,22:00UTC: shared allocation46518769 **closed normally** at22:00:53UTC,
1962seconds/oneCPU =0.545 allocated core-hours total. Bookkeeping shares for
EP05/EP15/EP17 are0.1816666667 each, charged once; TotalCPU61.855s and
stepMaxRSS1093668K are separate measurements. PTY34288 is closed; no compute
session remains. EP05 basis-only diagnostic implementation/review and EP17
concept-role specification continue as bounded code/document preparation.
A fresh attended allocation will follow prepared work. EP15 roles and condition
plan are complete; EP17 eligibility and chronology are complete. EP05's full
PLS panel46519201 completed separately with exit0/153s/0.0425coreh.
BR provides advisory criteria; these standalone jobs use Slurm execution.

Shared preparation allocation46515379 closed normally on October4 at
21:09:28UTC after28m10s on one CPU: **0.469444444 allocated core-hours total**.
For budget bookkeeping, split that shared preparation/coordination cost equally
across its three execution scopes EP05/EP15/EP17:0.156481481 core-hours each.
These are allocation-cost shares, not measured per-episode process CPU. The
three shares sum to one allocation; never add the shared total again. Slurm
TotalCPU94.957seconds and stepMaxRSS6265580K are separate measurements. The
EP05 substantive remainder46516505 adds its separately recorded221core-seconds.
The test PTY is closed; subsequent computation requires another verified
task-owned allocation. EP17 substantive batch46516829 subsequently completed
at21:13:24UTC in589seconds on one CPU (0.163611111 allocated core-hours,
separate from the shared preparation cost); all49,980 retained event rows matched.

| Episode | Current execution lane / next dependency |
| --- | --- |
| [EP01](episode01_narps_deep_search/outputs/README.md) | All13 null repairs running; MZ repairs106 completed/87 running, initial3 completed/2 running. New half/twice and alternate-partition pairs running after13 focused checks. Checked pools:104 MZ and187 null menus. Rates/intervals/full recovery unresolved; observed behavior/BOLD closed. |
| [EP02](episode02_dopamine_learning_rate_causality/outputs/README.md) | Independent equation replay and12 development-score tests passed; full models, source semantics, role handoff and scientific margins remain. |
| [EP03](episode03_condition_semantics_effect_maps/outputs/README.md) | Source identities reconciled; contributor metadata join stopped after source I/O failed to finish within its bound. Actual contributors and original producer handoff remain unresolved. |
| [EP04](episode04_shared_scene_geometry/outputs/README.md) | All 150 source handoffs and 25 preprocessing folds completed. Sub-05 reliability/NC completed; remaining sequence stopped at sub-01's timeout with no report, three participants unlaunched. Raw-image approval and matched RSA remain. |
| [EP05](episode05_sensorimotor_lfp/outputs/README.md) | Primary development search complete:45 comparisons/patience12 ordinary stop, no eligible candidate. Final scientific report/figures/accounting complete; no future-session confirmation or spike-population follow-up. Separate blend-penalty amendment remains unadopted. |
| [EP06](episode06_lfp_regional_fingerprints/outputs/README.md) | Shared source handoff recovered; scoring awaits EP07's fixed primary, and broader transfer needs another paired animal. |
| [EP07](episode07_lfp_session_transfer/outputs/README.md) | One fixed session needs authenticated native spike counts; source correction requested. |
| [EP08](episode08_lfp_electrode_trial_policy/outputs/README.md) | Development search stopped after 41 valid comparisons and reviewed stopping reconciliation. Mean initializer-order correspondence benefit is +0.003353 Q, but Mihili reverses in one order and only one of its four mean-order session effects is positive. A stable initializer-independent acquisition policy is not established. The limited [development report](episode08_lfp_electrode_trial_policy/outputs/final_report.md) preserves the positive average finding; all four held sessions remain closed. |
| [EP09](episode09_local_global_projection/outputs/README.md) | Recovered non-axon labels supply insufficient groups by the documented route; additional labels or approved downgrade needed. |
| [EP10](episode10_single_cell_coprojection/outputs/README.md) | M1 completed; a numerically faithful solver repair gave no material overall speedup. M2 qualification and the frozen resource gate remain unresolved. |
| [EP11](episode11_projection_types_vs_gradients/outputs/README.md) | Bounded-attempt limitation closeout complete: synthetic recovery46143347 timed out at72 hours,216.025 core-hours retained. Final artifact evidence remains unverified after the stopped scratch read; qualification is incomplete. No successor; real outcomes remain closed. |
| [EP12](episode12_malecns_type_sufficiency/outputs/README.md) | Final generated oracle refinement completed; precision remains unresolved at its authorized bound:32/35 cells pass,8/105 coordinates fail. Conditional qualification refused this report and exited1 before child qualification. Empirical deadline expired October1; launcher/readiness and canonical-target gaps remain. Decoder proposal unadopted; historical36 trials remain attempts, no corrected prefix or renewed budget. |
| [EP13](episode13_bold_target_identifiability/outputs/README.md) | Local exact theorem/certificate experiment completed. |
| [EP14](episode14_openbhb_roi_site_generalization/outputs/README.md) | Complete fixed development panel finished with no eligible candidate; public holdout remains closed. |
| [EP15](episode15_mdtb_topography_vs_geometry/outputs/README.md) | All24 metadata/anatomy and private Task-A reliability complete; fixed208-location support. Conservative potentially-correlated EP03 exposure review completed without claiming independence. All2,704,156 candidate splits enumerated and fixed12/12 roles published in8.44s. Metadata-only condition/partition/member plan completed all36480 rows after10 tests pass. Physical reader-separated handoffs, seven margins and Task-B scoring remain unactivated. |
| [EP16](episode16_braingate_cross_session_failure_modes/outputs/README.md) | Adopted numerical support rules evaluated: only T11/T9 components pass conditionally, insufficient for the planned cohorts on this panel; specific schedule eligibility remains unresolved. |
| [EP17](episode17_cneuromod_model_ranking_stability/outputs/README.md) | Metadata roles, anatomy, archive membership and structural quotas complete; N256/q32 fixed. Prediction core and33 additional numerical checks pass. Exact EP17/EP18 overlap complete;46531931 closed. Neural copying stopped after quota timeout; pixels await private credential interface. Reader-separated handoffs and scoring remain unfinished. |
| [EP18](episode18_things_eeg_concept_stability/outputs/README.md) | Roles, event joins, human features and opaque-ID/name metadata prepared; feature-formula checks passed. Exact qualified EP17 overlap recorded; correlated/design-exposed decision adopted with unchanged roles. Actual model extraction and a working isolated evaluator remain. |
| [EP19](episode19_causal_motor_forecasting_benchmark/outputs/README.md) | First fitting-series pilot has 13/34 stable candidate rest windows. Expanded fitting-only support pass timed out with no report; affected source route stopped. Timing and return semantics remain unresolved. |
| [EP20](episode20_neurocam_prior_guided_codesign/outputs/README.md) | Contact equations implemented; eleven analytic checks and ASTRA validation pass in46574266. Reference fitting and candidate search are not running: measurement/DAQ details and Rat5 geometry/history remain unresolved. |
| [EP21](episode21_wang_neural_code_conversion_reproduction/outputs/README.md) | Terminal closeout documented: regular-image export46424955 failed after7:56:09 with deadline_reached and incomplete publication. No complete feature handoff or neural result; no extension/relaunch. External writer's records and partial artifacts preserved. |

BR literature and review criteria inform specific scientific decisions; they
do not themselves establish a completed experiment or scientific acceptance.

| What you want to do | Skill |
| --- | --- |
| Turn a question or dataset into an EP; assess novelty; revise its design, writing or concept figure | [`design-episode`](.agents/skills/design-episode/SKILL.md) |
| Implement or run an existing EP; repair, monitor, analyze and show its results | [`run-episode`](.agents/skills/run-episode/SKILL.md) |
| Explicitly manage formal BR campaign state, Society review, reward or registered confirmation | [`manage-research-campaign`](.agents/skills/manage-research-campaign/SKILL.md) |

The first two are the everyday research workflows. The third is optional,
not a prerequisite or the next step after every experiment. Designing an EP
does not launch it, and requesting a status report does not authorize a repair
or another run. Scientific critique can inform either everyday workflow without
starting a formal campaign.

The former `br-autoresearch-episode` is now `run-episode`;
`brain-autoresearch-loop` is now `manage-research-campaign`.
`design-episode` adds the previously missing study-authoring entrypoint.
Old names are not duplicate discoverable skills. Existing episode contracts,
run identities and frozen campaign protocols are unchanged by this rename.

## How the loops are designed

Standalone episode work is the default: an explicit scientist request starts
work under the named episode's contract. Local exploration and evaluation do
not require Society review or canonical registration. Formal Society review,
reward, confirmation, and shared-claim promotion are a separate workflow used
when explicitly requested, with authority checked before the relevant action.

For an episode adopting the adaptive-search protocol, development proceeds
inside one authorized round:

```text
authorized round + frozen policy
    -> branch coverage
    -> hypothesis -> development trial -> falsifier -> successor
    -> incumbent/challenger decisions until a valid stop
    -> configuration lock
    -> episode-managed locked held-out evaluation
    -> finding, refutation, limitation, or incomplete search
```

Development trials remain local evidence. They do not individually become
accepted findings or authorize a new scientific round.

### 1. Start with a question

An idea can start with a question, a dataset, or a permitted prior observation.
Use [design-episode](.agents/skills/design-episode/SKILL.md) to establish the
closest prior work, useful scientific distinction, data-supported comparison,
and aligned study narrative/figure. It keeps exploratory choices open without
inventing universal pilot sizes or candidate caps.

An executable episode has `GOAL.md`, `DATASETS.md`, and one search policy,
normally `SEARCH_POLICY.yaml` (EP02 uses JSON). Use
[run-episode](.agents/skills/run-episode/SKILL.md) once execution is requested.
[AGENTS.md](AGENTS.md) defines workspace and data boundaries, targeted checks,
execution records, and canonical authority.
Scientific payloads remain read-only; source documentation can be maintained
within those boundaries. Checks and held-out evaluation occur when required
by the active scientific stage.

The Goal can be specific:

> Do condition-description features reduce held-out-dataset map error beyond a
> task-family prototype under a dataset-grouped split?

It can also be broad:

> I want to know which aspects of an experimental condition predict brain-map
> geometry across studies, and where that relationship stops generalizing.

Brain Researcher can help turn either form into a testable research question.
The scientist still decides whether the rewritten Goal is the right question.

### 2. Generate hypotheses before searching for a result

The agent first looks at the phenomenon, the available data, and what is
already known. It then proposes competing explanations, not just different
ways to obtain a significant result.

For each candidate it should state:

- the proposed explanation;
- what observation would support or weaken it;
- a falsifier;
- the main confounds and negative controls;
- whether the required data are actually available; and
- the approximate compute and data cost.

When an episode adopts the ordinary bounded-search policy, the runner can
generate two to eight candidates and carry forward at most two that are ready
to test. That candidate set and comparison plan are recorded before outcomes
that decide among them are read. These numerical limits do not apply to the
adaptive-search lane unless its episode policy adopts them; that lane permits
outcome-adaptive successors inside its frozen grammar and budget.

The agent may use the following as idea sources:

- the current [Research Landscape](https://brain-researcher.com/tbsci-9d4f41e7358c#landscape);
- the current [Self-evolvable Questions](https://brain-researcher.com/tbsci-9d4f41e7358c#questions);
- findings and unresolved questions from earlier episodes;
- Brain Researcher's 15 research routes; and
- optional OOD hypothesis generation from the Brain Researcher knowledge
  graph.

Landscape and Questions entries are useful seeds, not required inputs. OOD
generation looks for an unusual connection in the knowledge graph and turns it
into a hypothesis, a small discriminating test, a falsifier, and controls. An
OOD idea is still only a candidate, not a finding.

### 3. Request canonical review when needed

When the scientist requests formal Society review, the frozen candidate and
its evidence enter Brain Researcher's review workflow. Independent critique,
adversarial challenge, and synthesis inform the scientist's decision. The
panel cannot reward a direction, authorize confirmation, accept a claim, or
update the Landscape itself. A local candidate does not automatically start
this workflow.

[SOCIETY.md](SOCIETY.md) describes the review design and its limits;
[AGENTS.md](AGENTS.md) defines when to use canonical operations. Whether a
larger panel improves scientific review remains an empirical campaign question.

### 4. Run the experiment and turn the result into findings

Only a scientist-approved experiment is run as confirmation. Finishing the
computation does not automatically make the hypothesis true.

For a registered confirmation episode that freezes the scientific-learning
closeout policy, the server reviews the terminal result and can produce a
scientific review and report. Other runs stop at their terminal facts until a
separate review is requested. When the scientific-learning path is used, the
episode records:

- what the experiment actually tested;
- what was supported, refuted, or still unresolved;
- the strongest remaining alternative explanation;
- where the result is expected to generalize and where it may fail;
- failures, deviations, and resource use; and
- the next question suggested by the evidence.

The closeout is still non-promoting: it does not automatically create a
ClaimCard or change the Landscape. Reviewed outputs can motivate a successor
episode with a different test. Any shared finding, Questions update, or
Landscape transition still needs its own persisted review and transition
authority.

This is what we mean by **self-evolving**: the evidence from one episode changes
which questions and experiments are proposed next. It does not mean that an
agent can approve its own claim or silently launch the next experiment.

## The 15 Brain Researcher research routes

The routes are general ways to reformulate a problem. They are not 15 fixed
analysis pipelines.

1. audit an assumption and pivot if it is wrong;
2. replace an operator, model component, or representation;
3. redesign a fixed part of the data-generating process;
4. design a controlled comparison that separates competing explanations;
5. put different datasets or measurements into a shared representation;
6. rewrite the question as a more solvable object;
7. create a self-supervised learning signal from the data;
8. build known scientific structure into the method;
9. connect methods that are algebraically equivalent;
10. split heterogeneous data into parts that need different treatment;
11. split a large problem into smaller problems and delegate them;
12. turn a discrete search into a continuous one;
13. adapt a model by conditioning instead of retraining it;
14. measure a limit first, then try to exceed it; and
15. design a self-supervised objective for the property we want to measure.

An episode may use one, several, or none of these routes.

## Data already available on Sherlock

The first version of the campaign will mainly explore data that have already
been processed on Sherlock.

As of 2026-09-23, the main shared starting points are:

| Data | Count | Path |
| --- | ---: | --- |
| OpenNeuro FitLins `ds*` result directories | 55 | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses` |
| OpenNeuro fMRIPrep directories | 9 | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep` |
| HCP-YA derivatives | not yet inventoried here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP-YA-BIDS` |
| HCP connectivity data | not yet inventoried here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP1200_PTN` |

These counts mean that the directories exist. Each episode still checks the
task, subjects, events, contrasts, confounds, files, access rules, and which
data must remain untouched for confirmation.

See [DATA_CATALOG.md](DATA_CATALOG.md) for the shared releases, versions,
storage locations, and access notes currently available on Sherlock.

## What someone needs to submit

The episode contract and initial workspace are:

```text
episodeNN_short_name/
|-- GOAL.md
|-- DATASETS.md
|-- SEARCH_POLICY.yaml (or SEARCH_POLICY.json)
|-- inputs/README.md
`-- outputs/README.md
```

`GOAL.md` says what you want to know. It can be specific, broad, or include
possible analysis methods. Brain Researcher can help rephrase it.

`DATASETS.md` says which data you want to use, where they are, what is already
known, and what still needs to be checked. Imaging data do not go into Git.

The search policy freezes the admissible grammar, objectives, branch coverage,
budgets, falsifiers, stopping rule, and one-shot audit boundary.

## ASTRA experiment program

Maintain `outputs/astra/v0.0.14/astra.yaml` as the episode's full experiment
program: one root analysis with an `analyses` entry for each independently
runnable or evaluable experiment already required by its contracts. Inputs
and outputs express dependencies. Attempts, job IDs, failures, artifacts, and
next actions belong in `outputs/experiment_log.md`, keyed by analysis ID.
Follow the [ASTRA authoring reference](.agents/skills/run-episode/references/astra.md)
through [run-episode](.agents/skills/run-episode/SKILL.md).

The legacy `bin/astra-milestone` adapter emits a lossy single-analysis plan or
terminal projection. It is not a replacement for the full program and must
not overwrite it. ASTRA records do not approve execution, accept claims,
record reward, or update the Landscape. An authoring or validation failure
leaves ASTRA compliance open without blocking otherwise-authorized work.

To propose a new episode for campaign inclusion, submit its contract for
review through a pull request when that Git work is requested. Proposal review
is separate from starting already-authorized standalone work.

If you have another dataset you want to explore, tell me what it is and what
question you want to ask. If its access rules allow it, I can help put it on
Sherlock.

## Repository record

Each formal episode has one current Goal, dataset contract, and search policy.
Once candidate-scoring adaptive execution starts, its scored trials,
challenger decisions, and execution failures belong in the ordered trial log
and current output workspace; acquisition and preliminary readiness need only
the lean decision record required by the scientific question. Preserve frozen
paths, active writers, and protected history. Mark obsolete navigation as
historical; moving or deleting artifacts requires a separately scoped change.
Large imaging data and temporary compute files stay outside Git.

- [CAMPAIGN.md](CAMPAIGN.md) lists formal episode slots, their current status,
  immutable prior runs, and reserved IDs.
- [DATA_CATALOG.md](DATA_CATALOG.md) lists shared datasets available on
  Sherlock; episode-specific eligibility and audit boundaries remain governed
  by each episode's dataset contract.
- [EPISODE_REGISTRY.yaml](EPISODE_REGISTRY.yaml) separates stable episode IDs,
  historical prior identities, direct physical paths, and current policy
  references. The current EP01 is `episode01_narps_deep_search`; the old NARPS
  runs are frozen in the [historical prior archive](_examples/historical_prior_records/).
- [Adaptive-search protocol](ADAPTIVE_SEARCH_PROTOCOL.md) and
  [controller interface](ADAPTIVE_CONTROLLER_INTERFACE.md) define the search,
  append-only lineage, and one-shot audit boundary. A design becomes realized
  evidence only when its episode records the corresponding run artifacts.
- [Unnumbered examples](_examples/README.md) retain reusable design provenance;
  the NARPS deep-search example is retired and is not the current EP01
  scientific contract.
- [LANDSCAPE.md](LANDSCAPE.md) summarizes the research areas we are exploring.
- [QUESTIONS.md](QUESTIONS.md) contains possible questions for future episodes.
- [Foundation-20 crosswalk](portfolios/foundation20/CROSSWALK.md) records direct,
  bounded, partial, adjacent, and missing topic coverage; its exposure ledger
  prevents shared data from being mislabeled as independent confirmation.
- [SOCIETY.md](SOCIETY.md) explains the multi-agent review and its limits.
- [AGENTS.md](AGENTS.md) contains the instructions used by agents running here.
