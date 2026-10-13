# EP06 data plan

## Manuscript role and cross-episode access order

EP06 is the recording-domain component of EP07's proposed history-and-
population-components paper, not an independent LFP-to-spikes publication.
This is a writing decision, not merged execution authorization. Preserve the
session and whole-trial roles below. EP07's chronological source/target and
within-target trial roles differ, so an EP06 neural fit or score may expose
spikes still closed in EP07.

On this corpus, perform EP06 neural fitting/scoring only after EP07's full
freeze and single primary opening, unless a scientist approves a compatible
joint-role design prospectively before relevant scores. No sibling outcomes,
candidate feedback, or fitted state may enter EP07 development. EP06 must
still freeze its own final procedure before its own one-shot reserved test.
This ordering creates neither fresh outcomes nor independent replication.

The scientist adopted the joint eleven-session amendment on 2026-10-10.
`Mihili_CO_20140306.mat` is excluded in both regions for failed native-count
validity. The seven development and four reserved sessions below are now the
operative roster. This decision neither opens neural values nor resets budgets.

## What data are needed?

The intended source is the public Dryad release for Gallego-Carracedo et al.
(2022), DOI `10.5061/dryad.xd2547dkt`. Provider records describe 24 MATLAB
session files plus a README. An existing shared read-only payload is available:
`/oak/stanford/groups/russpold/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source`.
The October 6 metadata inventory inspected all 12 named paired sessions and
3,090 trials: array headers, guide tables, dates, event metadata and shared
trial-struct positions. It reported no structural problems and read no neural
or velocity values. All source trials declare 30-ms bins and success label
`R`. No separate `trial_id`/`trialID` exists; shared MATLAB trial-struct
position is the explicit structural pairing key, not an invented provider ID.
Neural eligibility, rank and usable feature support remain
unqualified. Historical metadata completeness does not validate spike counts;
the failed March 6 session is now excluded rather than repaired or replaced.
See [source handoff](outputs/source_handoff.md). No new source copy is required.

EP06 needs the sessions in which M1 and PMd were recorded simultaneously during
the same reaching task. For each included session, the analysis needs paired
behavior, LFP features, spikes, event times, reach directions, electrode and
unit identities, and the guide that gives each stored LFP feature its biological
meaning.

The released LFP matrices contain already processed features rather than raw
voltage. Raw phase, new filtering, phase–amplitude coupling, coherence,
traveling-wave, and re-referencing analyses are therefore outside this episode.

## Recording blocks and their roles

### Public online resources — checked October 6, 2026

- [Official Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.xd2547dkt): the public inventory includes the twelve named paired-session files within the 24-session release and a README; March 6 remains listed at 476.64 MB. No correction notice was visible. The existing local v5 payload is already available, so this is source documentation rather than a download requirement.
- [Author repository README](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/master/README.md): makes TrialData an explicit dependency for the published analysis examples. Public code availability does not establish that EP06's new residual/projector/transfer models are implemented or qualified.
- [Author per-band CCA source](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/master/MainAnalysis/fCCA.m): defines electrode intersection from the unit/LFP guides, spike square-root transformation, 0.05-width smoothing, PCA and band-specific CCA. These operations document the source method, not a lossless inverse for March 6 or permission to change EP06's training-only residual targets.

This lookup supplies public documentation for the existing recordings; it
does not authenticate repaired native counts, usable 15-electrode/rank
support, an eligible third paired animal or compatible joint outcome roles.
The [detailed public-source handoff](outputs/source_handoff.md#public-source-check--october-6-2026-pacific)
retains retrieval limits. EP07's freeze/opening dependency and EP06's reserved
session rules remain active. No neural array or model outcome was opened.

| Block | Reported recordings | Role in EP06 |
| --- | --- | --- |
| Mihili | 6 reported simultaneous M1/PMd sessions; March 6 excluded in both regions | Three development sessions and the unchanged two reserved dates; all five retained sessions still require scientific support |
| Chewie-L | 6 sessions with simultaneous M1 and PMd | Four development sessions and two reserved final-test sessions, if all six pass structural checks |
| Chewie-R | 4 M1-only sessions from a separate implant | Sensitivity analysis only; this is the same animal as Chewie-L |
| Han | 5 area-2 sessions | Recording-domain sensitivity only |
| Lando | 3 area-2 sessions | Recording-domain sensitivity only |
| Future third animal | Simultaneous M1 and PMd under a compatible task | Required for an external animal-transfer test; no such source is currently identified |

Sessions, regions, electrodes, bands, trials, time bins, and latent coordinates
are repeated measurements. They are not additional animals.

## Readable, outcome-blind session roles

Retain the original acquisition-date order, with provider session name as the
tie breaker. The reviewed exclusion removes only original Mihili position 5;
do not renumber the remaining sessions or change the reserved dates:

| Animal | Development and selection | Reserved final test |
| --- | --- | --- |
| Mihili | original positions 1, 2, and 4 | original positions 3 and 6 |
| Chewie-L | positions 1, 2, 4, and 5 | positions 3 and 6 |

The completed metadata inventory authenticates the names and dates below.
The role column incorporates the adopted eleven-session amendment. It is not
an assertion that neural eligibility or reader separation has passed.

| Animal | Position | Session name | Role |
| --- | ---: | --- | --- |
| Mihili | 1 | `Mihili_CO_20140217.mat` | development |
| Mihili | 2 | `Mihili_CO_20140218.mat` | development |
| Mihili | 3 | `Mihili_CO_20140303.mat` | reserved test |
| Mihili | 4 | `Mihili_CO_20140304.mat` | development |
| Mihili | 5 | `Mihili_CO_20140306.mat` | excluded in both regions: native-count validity failure |
| Mihili | 6 | `Mihili_CO_20140307.mat` | reserved test |
| Chewie-L | 1 | `Chewie_CO_20160929.mat` | development |
| Chewie-L | 2 | `Chewie_CO_20161005.mat` | development |
| Chewie-L | 3 | `Chewie_CO_20161006.mat` | reserved test |
| Chewie-L | 4 | `Chewie_CO_20161007.mat` | development |
| Chewie-L | 5 | `Chewie_CO_20161014.mat` | development |
| Chewie-L | 6 | `Chewie_CO_20161021.mat` | reserved test |

Do not change a session's role because of its neural result. The reviewed
[adopted amendment](outputs/paper_plan.md#october-10-eleven-session-amendment-adopted)
permits the three/four development support above. March 6 was originally
Mihili's fourth development session, not a fifth development session.
Further loss among the five retained Mihili or six Chewie sessions requires a
new scientific review before scoring. No replacement, source transformation
or automatic support reduction is permitted.

Positions 3 and 6 are **held out across the acquisition dates**. They are not
both later than the development sessions, so the study must not describe this
split as a simple train-early/test-late design.

## What must be checked before scoring?

Record only the source facts needed for this analysis:

1. the Dryad release, local read-only location, license, and reference analysis
   code actually used;
2. the animal, implant, session, array, region, trial, event, direction,
   electrode, unit, and LFP-feature tables;
3. the links showing that the M1 and PMd fields use the same trials, behavior,
   events, and time base;
4. the movement interval, trial-success field, direction label, velocity,
   spikes, LFP features, and their guides;
5. enough paired trials, all registered LFP bands, 15 usable physical
   electrodes per region, and the prespecified minimum spike-population rank;
6. the completed session-role table above; and
7. small known-answer tests for indexing, held-out prediction, projector
   orthogonality/completeness, source-only weight provenance, the
   component-energy gate, profile construction, and each required shuffle
   control.

Every transformation that sees neural values—including scaling, reliability
weights, spike-population axes, band weighting, and any across-session
alignment—must be fit only on the permitted training data.

For a cross-animal prediction, “permitted” is role-specific: target training
may fit axes, scaling, and bandwise base predictors, while all
frequency-combining weights come from source-animal development sessions and
remain frozen in the target animal.

## Exposure and independence

Related historical work inspected outcomes from this Dryad release. Treat the
entire current release as outcome-exposed. The seven development sessions can
support exploratory model selection; the four reserved sessions can test that
one final procedure was applied without further adaptation. They are not fresh
biological confirmation.

Before future fits, record each exact session/trial identity, its EP06 and
EP07 roles, permitted reader, actual opening and freeze dates, and information
transfers in the shared exposure record. Reserved Mihili March 3/Chewie October
6 are EP07 historical-training dates; reserved March 7/October 21 are EP07
target dates with distinct trial roles. EP07 therefore uses outcomes from
EP06-reserved dates. Preserve the EP07 full-freeze/primary-opening prerequisite
and the procedural reader-boundary disclosure; a ledger grants no access and
does not make EP06 reserved results independent confirmation.

A future external test needs a separately sealed third animal with simultaneous
M1 and PMd recordings, a compatible task, enough sessions and trials, and the
same interpretable LFP feature set. The original two-animal rule must be fixed
before those outcomes are seen. Additional Chewie sessions or a second Chewie
implant do not count as a new animal.

## Data for the explanatory follow-up

The primary profile always uses the single-trial spike residual after removing
a direction-by-time mean learned on training trials. The explanatory follow-up
is separate: it starts from the unresidualized spike response and asks which of
three fixed projections carries the regional-weight advantage.

Within every development or reserved session, assign whole trials within each
reach direction to three roles before fitting neural quantities:

| Trial role | Use |
| --- | --- |
| Training | Fit the session's population axes, scaling, primary residualization template, and bandwise base predictors under the frozen recipe |
| Calibration | Estimate the component-profile split-half reliability and distance from the source profile; no component score or combining-weight fit is allowed |
| Evaluation | Apply the frozen projectors and score the frozen weight rules once; these trials do not update an axis, base predictor, weight, boundary, or threshold |

The October11 metadata-supported development convention sets training=floor(n/2),
calibration=floor((n-training)/2), and evaluation=remainder within each direction.
The shared M1/PMd assignment uses local PCG64 SeedSequence
[6006101,animal_index,original_session_position,direction]. Require at least
7/4/4 trials per direction in training/calibration/evaluation; the existing
seven-session metadata has observed minima8/4/5. Preserve all original trial
positions and export each role in source order. Alternate assigned calibration
rows in that order within each direction for its two halves. Reserved-session
outcomes remain closed. The full prospective direction/window/electrode rules
are in [the selection record](outputs/experiment_log.md#2026-10-11-0002-pacific--source_session_support--development-role-conventions-selected-from-existing-metadata)
and `SEARCH_POLICY.yaml`; neural support is not established by these metadata.

Training freezes the population axes and the following trial-space projectors:

| Projector | Frozen subspace |
| --- | --- |
| `P_C` | Direction-invariant time effects |
| `P_D` | Sum-to-zero direction-by-time contrasts after removing `P_C` |
| `P_E` | The remaining within-direction, within-time trial activity |

The inner product gives every direction equal total weight, trials equal weight
within direction, and analyzed time bins equal weight. Under that inner product,
`P_C` and `P_D` are orthogonal and `P_E = I - P_C - P_D`; therefore
`P_C + P_D + P_E = I` on the frozen-axis analysis space. The projectors depend
on direction/time labels and the training-defined population axes, not on
evaluation spikes. Training fixes the
direction set, time grid, and contrast coding; evaluation uses only its
preassigned direction labels and time bins to apply those operators.

The [component specification](outputs/component_details.md) makes this
role-specific construction explicit. Evaluation means are computed only as
part of projecting observed targets for scoring; project each frozen prediction
using its own means. Do not copy observed target means into a prediction or
fit a new axis, contrast or target combining weight. `P_E Y_eval` removes an
evaluation cell mean; the primary training-template residual does not, so these
are separate targets under the existing roles.

For component `k`, the held-out target is `P_k Y_eval`, and the prediction is
`P_k Yhat_eval`. Score it as
`1 - sum||P_k Y_eval - P_k Yhat_eval||^2 / sum||P_k Y_eval||^2` under the same
direction-balanced norm. The denominator is identical for correct, pooled, and
swapped rules. This is a fraction-of-component-energy prediction score, not
explained variance. The energy gate uses the denominator divided by total
direction-balanced evaluation weight, so it is not inflated by more trials. A
positive minimum for that normalized energy is fixed from development and
known-answer synthetic tests before final evaluation. A nonfinite quantity or
energy below that gate is unscorable and unresolved.

Source-animal development data alone fit component-specific M1, PMd, and pooled
frequency-combining weights. Swapped is exactly the M1/PMd permutation. Target
training trials may fit axes, scaling, and bandwise base predictors, but may not
refit a combining weight. For each scoreable component, calculate
`G_region = score(correct) - max(score(pooled), score(swapped))` separately in
M1 and PMd. Only sign-consistent regional gains receive an equal-region mean.
Preservation requires a mean of at least 0.01 and positive gains in both
regions; two nonpositive gains are failure; opposite signs or a smaller
positive mean are unresolved.

The data needed for the follow-up are:

| Source | Scientific use | Condition before use |
| --- | --- | --- |
| Seven development sessions: three Mihili and four Chewie | Run the outer component-selection and support-prediction assessment | Three Mihili target folds nominate on the other two sessions; four Chewie folds nominate on the other three; only that prefold family and its frozen support prediction are scored in the left-out session |
| Source-development bandwise predictions | Fit M1, PMd, and pooled component-specific combining weights | No target session may update these weights; swapped only permutes M1 and PMd |
| Target-session bandwise predictions | Compare correct, pooled, and swapped weights | Same evaluation trials, projectors, denominator, electrodes, axes, model, band support, and base predictions for all three rules |
| Calibration trials | Measure profile reliability and source-profile distance before an evaluation score is seen | Split-half rule, feature ordering, distance scale, and direction balancing are fixed |
| Four reserved sessions across acquisition dates | Make one frozen prediction and one held-out score per session | Write all predictions before opening any evaluation trials; open all four together |
| Recording-quality metadata | Challenge the component-specific account | Use a fixed metadata-only comparator; do not expand it after seeing a failure |
| Future sealed sessions or a third animal | Estimate the support boundary with useful precision | Required before treating the boundary as a stand-alone transferable result |

For a calibration profile, reliability is the Spearman–Brown-corrected
correlation between two fixed, direction-balanced calibration halves. A
nonfinite reliability forces an abstention. Profile distance is the
root-mean-square bandwise difference from the applicable source-animal regional
prototype, divided band by band by the development between-session scale. A
zero or nonfinite bandwise scale forces an abstention. For a cross-fitted
development prediction, the other six development sessions
provide the reference median and scale; all seven provide them for a reserved
prediction. Standardization subtracts the reference median and divides by
1.4826 times the median absolute deviation. If either scale is zero, the rule
abstains. The single support index is standardized reliability minus
standardized distance. Pool the reference sessions directly: two Mihili plus
four Chewie when Mihili is left out, and three plus three when Chewie is left
out. Do not substitute animal-weighted medians. No coefficient is fit to the
seven session outcomes.

In each outer fold, all three components must be numerically scoreable in both
regions of the other two Mihili or three Chewie target development sessions; an unscoreable rival
leaves the full comparison unresolved, with no nominee. Among measured
components, only candidates whose regional gains agree in sign in every
nomination session are eligible. Choose the largest median equal-region gain
among eligible candidates; top medians within 0.01 are tied, and an empty
eligible set gives no nominee. A measured mixed-sign rival remains visible
but does not veto another eligible candidate. This is the best eligible
joint-region account, not a global numerical winner over mixed-sign rivals.
The left-out evaluation trials score only a valid
prefold nominee; the other two evaluation components stay unopened for the
fold. After the three Mihili or four Chewie prospective folds are recorded, the previously unopened
development components may be scored for a final nomination using all retained
target sessions; they cannot alter the outer-fold record. Final medians use the
sessions on which all three candidates are numerically scoreable in both
regions: all three Mihili sessions and at least three of four Chewie sessions,
including every available common session. Sign consistency is
candidate-specific across that common set; negative or mixed-sign rival
measurements are not missing data. This prospective 2026-10-02 amendment
changes nomination eligibility, not trial roles or numerical margins.
The same family must be nominated in all three Mihili folds and at least
three of four Chewie folds, with no 0.01-margin tie or left-out M1/PMd sign
contradiction. Missing folds never establish agreement. A known failure makes
selection unstable; missing required evidence remains unresolved. Each final
nominee must match its direction's stable outer-fold family. Both
transfer directions must make the same final untied nomination before reserved
outcomes are opened.

A scoreable component is preserved only when both regional gains are positive
and their equal-region mean is at least 0.01. Two nonpositive regional gains are
a failure; mixed signs or a smaller positive mean are unresolved. A support
index at or above 0.25 predicts preservation, at or below -0.25 predicts
failure, and the middle is an abstention. These margins and the positive
component-energy gate are fixed before evaluation and must pass known-answer
sensitivity checks. The within-corpus prediction needs at least three
non-abstaining reserved predictions with evaluable outcomes and no wrong
prediction; unresolved outcomes cannot count as correct. Observing only one
outcome class cannot establish the support boundary.

M1 and PMd use different arrays in this release. Matching observed metadata can
rule out measured recording differences, but it cannot remove an unmeasured
region-linked hardware effect. The strongest permitted claim is therefore about
the measured recording domains.

The public Dryad catalogue and source article indicate that the release contains
trial-level behavior, simultaneous spikes and LFPs, and dated repeated sessions,
so the projection analysis can be attempted in principle. EP06 has not yet
verified those fields in the files it will actually use. If trial identities,
dates, or enough trials per direction for training, calibration, and evaluation
are missing, the deeper prediction cannot be tested from this release.

Four reserved sessions cannot estimate a cross-session boundary precisely and
may not include clear examples on both sides. Report exact session outcomes,
whole-trial uncertainty stratified by direction with paired regions,
session resampling within animal, equal-weight animal-direction summaries, and
leave-one-session influence. Unless additional sealed sessions supply both
outcome classes with useful precision, EP06 is a planned component of EP07, not
an independent publication. The same decision follows if a focused novelty
check finds that the component claim is already established.

## Current readiness

Status: **revise before any scored analysis**.

- The Dryad recordings are available at the shared source above; EP06's
  source-specific structural and role qualification remains incomplete.
- The completed metadata inventory establishes session dates, trial-struct
  pairing, native bin size, success codes, event metadata and guide/header
  structure. Neural-value validity and all scientific feature meanings are
  not thereby established.
- Exact session names above incorporate the adopted March 6 exclusion;
  numerical eligibility and final trial roles remain open for retained dates.
- The four reserved sessions are not yet isolated from development scoring.
- The analysis code has not passed the targeted indexing, leakage,
  held-out-prediction, and shuffle tests.
- The training/calibration/evaluation trial counts, three projectors,
  minimum-energy gate, and calibration-visible support index have not been
  qualified on the files.
- The manuscript ownership decision is now EP07 main line with EP06 as its
  recording-domain component. Exact component-specific source/supplement/code
  overlap remains a scientific limit, not a claim of guaranteed originality.
- No suitable third-animal source has been identified or sealed in the named
  handoff. That limits the separately planned external transfer claim; it is
  not a prerequisite for the already bounded two-animal EP07 component.

## Storage and compute

Inputs remain read-only. Durable code, session tables, predictions, and reports
belong in this episode. Temporary arrays belong in
`$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints/`. Expensive or
sustained computation must run through Slurm.
