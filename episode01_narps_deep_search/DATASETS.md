# EP01 data roles: NARPS cognitive-estimand audit

## Current status

EP01 is at scientific-design stage. The former smoothing and derivative-
sufficiency directions were retired before execution. No behavioral model,
BOLD model, team-map projection, or scientific score has been run under the
current question.

The study uses three evidence layers from one public experiment plus synthetic
recovery tests. They are complementary, not independent replications.

## Role table

| Resource | Scientific role | What it can establish |
| --- | --- | --- |
| Exact-design simulations | Behavioral and synthetic-coordinate recovery | Whether the proposed latent mechanisms and numerical map coordinates are identifiable at the actual design and noise level |
| NARPS original behavioral events | Mechanism decomposition | Choice-and-RT evidence for valuation asymmetry and value-independent rejection bias |
| NARPS fMRIPrep BOLD/confounds | Controlled fMRI factorial | How gain/loss maps move when decision-state and response blocks are included under one fixed pipeline |
| NARPS 70-team unthresholded group maps | Ecological analysis layer | Whether public team maps load differently on the frozen estimand axes |
| NARPS team method reports | Operationalization layer | Whether declared first-level choices are associated with those shape coordinates beyond the frozen sample, group-model, and map-technical controls |

No live sibling-episode output is an EP01 input. The historical NARPS episode
records are exposed lineage only and count as no new evidence.

## Participant-level source: OpenNeuro `ds001734`

The declared release is OpenNeuro `ds001734` version 1.0.5
(`10.18112/openneuro.ds001734.v1.0.5`). It contains 108 included participants:
54 equal-indifference (EI) and 54 equal-range (ER), each with four runs of 64
trials. All 256 gain-by-loss combinations were presented once per participant,
for 27,648 total trials before exclusions.

The task records four simultaneous response categories: strongly accept,
weakly accept, weakly reject, and strongly reject. The gamble remained visible
until response or a four-second deadline. Approximately 0.7% of trials were
reported as no-response in the data descriptor.

### Established locations

| Component | Location | Current availability |
| --- | --- | --- |
| BIDS metadata and original events | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/input/ds001734` | All 108 participants x 4 original event tables present |
| Common fMRIPrep derivatives | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep/ds001734/derivatives` | All 432 MNI 2-mm preprocessed BOLD series, confounds, run masks, and QC reports present |
| Existing standardized FitLins analysis | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds001734/task-MGT` | One benchmark pipeline with run/subject gain, loss, gain-minus-loss, all-trial, and RT products |

EP01 uses each `*_events_ORIGINAL.tsv` as the behavioral source. The sibling
`events.tsv` files are analysis-specific transformations and are not the event
source for this study.

The primary fMRI analysis starts from the complete fMRIPrep 21.0.2 series, not
from the partially materialized raw BOLD collection or the older fMRIPrep 1.1.4
annex links. Raw BOLD currently has 323 of 432 payloads materialized; no current
EP01 claim requires the missing 109 raw payloads.

The existing FitLins products are a benchmark only. They are not substituted
for the runwise `S00/S10/S01/S11/R11` fits because those models define the
scientific comparison.

### Required participant-level fields

The behavioral layer requires, for every retained trial:

- participant, task version, run, and trial order;
- gain and positive loss magnitude;
- original four-category response;
- binary accept/reject sign;
- response time; and
- deadline/no-response status.

The BOLD layer additionally requires original onset, BOLD series, confounds,
mask, censoring information, and the frozen runwise design. Display-side and
finger-mapping claims are out of scope unless the original task code supplies
participant-level mappings; `participants.tsv` does not contain them.

## Task-version and schedule structure

ER gains and losses both span 5--20. EI gains span 10--40 while losses span
5--20. EI and ER are between-participant task versions. They are modeled as
partially pooled strata and a transport stress test, not as independent
replications. No causal claim is made without evidence that assignment was
random.

Each task version uses eight optimized full-session sequence/onset templates,
counterbalanced across participants. Participant remains the biological
sampling unit. Schedule-template clustering and leave-one-template-out
sensitivities guard against a result driven by one or two sequences.

The common gain range 10--20 provides a prespecified sensitivity analysis. It
does not turn the between-participant context comparison into a within-person
experiment.

## Many-team result source

The fixed public result release is NARPS v2.0.1 from Zenodo record 3634120,
licensed CC BY 4.0. The immutable local source is:

```text
/oak/stanford/groups/russpold/data/br_autoresearch_data/
  narps_results/zenodo-3634120-v2.0.1/source/results.tgz
```

The archive already contains:

- concatenated unthresholded team statistic maps for hypotheses 1--9;
- team identifiers and NeuroVault collection mappings;
- corrected decision and confidence tables;
- rich team metadata, including first-level independent variables, RT
  modeling, movement modeling, preprocessing, software, statistic type,
  inference, smoothing estimates, and—where reported—participant counts,
  exclusions, higher-level covariates, and group-model choices; and
- public vmPFC, ventral-striatum, and amygdala masks in the threshold-
  simulation materials, whose exact original-hypothesis provenance must be
  qualified before any regional summary; and
- the original NARPS harmonization and generic map-similarity products.

EP01 does not claim novelty for sign rectification, t-to-z conversion,
resampling, smoothness estimation, map clustering, consensus maps, or generic
associations between software/smoothing and team decisions. Those were already
part of the original analysis.

For cognitive decomposition, statistic-map signs must be restored so a
positive statistic denotes a positive underlying coefficient for increasing
positive gain or loss magnitude. This is a coefficient-direction convention,
not a claim that `t`/`z` maps are beta maps or share effect units. The original
"rectified" convention orients maps toward each directed regional hypothesis;
it cannot be treated as a common loss-estimand direction.

The team results are group-level repeated analyses drawn from the same NARPS
participant pool, with team-specific exclusions and sometimes incomplete
sample-identity reporting. They support a map-shape and method-description
audit. They do not support:

- 70-pipeline participant ranking;
- pipeline-specific individual brain-behavior correlations;
- independent biological replication across teams; or
- a causal effect of any self-selected analysis choice.

Participant count, exclusions, available sample-identity summaries, higher-
level covariates/design, and group estimator therefore enter the frozen
technical-control set alongside mask, smoothness, statistic, software, and
preprocessing fields. Unreported fields receive explicit missingness
indicators. If exact sample identity or group-model details remain unavailable,
the method analysis cannot isolate cognitive specification from unrecorded
sample-composition differences and is labeled accordingly.

Subject COPE/VARCOPE images were optional in the original instructions and are
available for only a small subset of teams. Reproducing many pipelines is not
part of the primary EP01 contract.

## Synthetic recovery sources

Behavioral simulations reproduce the exact participant count, EI/ER offer
matrices, four-run allocation, four-second deadline, and observed missingness.
They vary valuation asymmetry, starting bias, drift criterion, urgency,
nonlinear utility, and contaminant responses under the fixed candidate family.

fMRI design simulations use the actual onsets and response times but no task
BOLD outcomes. They estimate design efficiency, collinearity, and variance
inflation for `S00`, `S10`, `S01`, `S11`, and `R11`.

Team-projection simulations form linear combinations of the frozen, base-
estimand-matched conditional-slope and specification-displacement templates
under known coordinates, smoothness, scale, masks, and noise. They test only
whether joint projection can recover the generating numerical coordinates
before public team maps are described.

## Explicit non-inputs

The following are outside the current evidence set:

- questionnaires, resting-state fMRI, diffusion MRI, and eye-gaze data, which
  were collected in the original project but are not part of the public
  `ds001734` release used here;
- local shared `first_level/switch`, `first_level/pupil`, `stp_bias.csv`, or
  other collaborator analyses;
- NARPS Open Pipelines subject-level reruns;
- thresholded team maps as primary measurements; and
- `ds000005` as a mandatory replication.

The small `ds000005` mixed-gamble sample may later test a narrow controlled-map
transport claim, but it cannot replicate the 70-team ecology and is not
required for the present paper.

## Data-use and inference boundary

All sources are public research data. Provisioned source payloads remain
read-only. Durable derived reports belong under `outputs/`; large runwise fits
and simulations belong in episode-specific scratch.

The NARPS observations and team maps have been extensively exposed. Run
splitting provides honest internal prediction and nonoverlapping
brain-behavior estimation; it does not create fresh confirmation. Any positive
result remains a reanalysis of one experiment and its associated analyst
ecology.
