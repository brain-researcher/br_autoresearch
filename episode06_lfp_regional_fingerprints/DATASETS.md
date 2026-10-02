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

## What data are needed?

The intended source is the public Dryad release for Gallego-Carracedo et al.
(2022), DOI `10.5061/dryad.xd2547dkt`. Provider records describe 24 MATLAB
session files plus a README. The current episode does not yet have a verified
read-only copy of those recordings.

EP06 needs the sessions in which M1 and PMd were recorded simultaneously during
the same reaching task. For each included session, the analysis needs paired
behavior, LFP features, spikes, event times, reach directions, electrode and
unit identities, and the guide that gives each stored LFP feature its biological
meaning.

The released LFP matrices contain already processed features rather than raw
voltage. Raw phase, new filtering, phase–amplitude coupling, coherence,
traveling-wave, and re-referencing analyses are therefore outside this episode.

## Recording blocks and their roles

| Block | Reported recordings | Role in EP06 |
| --- | --- | --- |
| Mihili | 6 sessions with simultaneous M1 and PMd | Four development sessions and two reserved final-test sessions, if all six pass structural checks |
| Chewie-L | 6 sessions with simultaneous M1 and PMd | Four development sessions and two reserved final-test sessions, if all six pass structural checks |
| Chewie-R | 4 M1-only sessions from a separate implant | Sensitivity analysis only; this is the same animal as Chewie-L |
| Han | 5 area-2 sessions | Recording-domain sensitivity only |
| Lando | 3 area-2 sessions | Recording-domain sensitivity only |
| Future third animal | Simultaneous M1 and PMd under a compatible task | Required for an external animal-transfer test; no such source is currently identified |

Sessions, regions, electrodes, bands, trials, time bins, and latent coordinates
are repeated measurements. They are not additional animals.

## Readable, outcome-blind session roles

After checking only structural eligibility, order the six sessions within each
primary animal by acquisition date. If two dates tie, use the provider session
name to break the tie. Assign roles by position:

| Animal | Development and selection | Reserved final test |
| --- | --- | --- |
| Mihili | positions 1, 2, 4, and 5 | positions 3 and 6 |
| Chewie-L | positions 1, 2, 4, and 5 | positions 3 and 6 |

Before any regional prediction score is inspected, replace the position labels
below with the actual provider session names:

| Animal | Position | Session name | Role |
| --- | ---: | --- | --- |
| Mihili | 1 | not yet available | development |
| Mihili | 2 | not yet available | development |
| Mihili | 3 | not yet available | reserved test |
| Mihili | 4 | not yet available | development |
| Mihili | 5 | not yet available | development |
| Mihili | 6 | not yet available | reserved test |
| Chewie-L | 1 | not yet available | development |
| Chewie-L | 2 | not yet available | development |
| Chewie-L | 3 | not yet available | reserved test |
| Chewie-L | 4 | not yet available | development |
| Chewie-L | 5 | not yet available | development |
| Chewie-L | 6 | not yet available | reserved test |

Do not change a session's role because of its neural result. If either animal
has fewer than six structurally eligible paired sessions, stop before regional
scoring and review whether the scientific question is still identifiable.

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
entire current release as outcome-exposed. The eight development sessions can
support exploratory model selection; the four reserved sessions can test that
one final procedure was applied without further adaptation. They are not fresh
biological confirmation.

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

The exact fractions are fixed after checking the available trial counts and
before regional outcomes are inspected. Every direction must contribute to all
three roles; otherwise that session is structurally ineligible.

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
| Eight development sessions | Run the outer component-selection and support-prediction audit | In each transfer direction, the other three target sessions nominate one family; only that prefold family and its frozen support prediction are scored in the left-out session |
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
development prediction, the other seven development sessions
provide the reference median and scale; all eight provide them for a reserved
prediction. Standardization subtracts the reference median and divides by
1.4826 times the median absolute deviation. If either scale is zero, the rule
abstains. The single support index is standardized reliability minus
standardized distance. No coefficient is fit to the eight session outcomes.

In each outer fold, all three components must be numerically scoreable in both
regions of the other three target development sessions; an unscoreable rival
leaves the full comparison unresolved, with no nominee. Among measured
components, only candidates whose regional gains agree in sign in every
nomination session are eligible. Choose the largest median equal-region gain
among eligible candidates; top medians within 0.01 are tied, and an empty
eligible set gives no nominee. A measured mixed-sign rival remains visible
but does not veto another eligible candidate. This is the best eligible
joint-region account, not a global numerical winner over mixed-sign rivals.
The left-out evaluation trials score only a valid
prefold nominee; the other two evaluation components stay unopened for the
fold. After the four prospective folds are recorded, the previously unopened
development components may be scored for a final nomination using all four
target sessions; they cannot alter the outer-fold record. Final medians use the
sessions on which all three candidates are numerically scoreable in both
regions, with at least three of four required. Sign consistency is
candidate-specific across that common set; negative or mixed-sign rival
measurements are not missing data. This prospective 2026-10-02 amendment
changes nomination eligibility, not trial roles or numerical margins.
The same family must be nominated in at least
three of four folds, with no 0.01-margin tie or M1/PMd sign contradiction.
Otherwise the selection is unstable and there is no component claim. Both
transfer directions must make the same final untied nomination before reserved
outcomes are opened.

A scoreable component is preserved only when both regional gains are positive
and their equal-region mean is at least 0.01. Two nonpositive regional gains are
a failure; mixed signs or a smaller positive mean are unresolved. A support
index at or above 0.25 predicts preservation, at or below -0.25 predicts
failure, and the middle is an abstention. These margins and the positive
component-energy gate are fixed before evaluation and must pass known-answer
sensitivity checks. The within-corpus prediction needs at least three
non-abstaining reserved predictions and no wrong prediction; observing only one
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
whole-trial uncertainty, equal-weight animal-direction summaries, and
leave-one-session influence. Unless additional sealed sessions supply both
outcome classes with useful precision, EP06 is a planned component of EP07, not
an independent publication. The same decision follows if a focused novelty
check finds that the component claim is already established.

## Current readiness

Status: **revise before any scored analysis**.

- The Dryad recordings are not available in `inputs/`.
- Animal, session, region, trial, event, electrode, unit, and feature meanings
  have not been verified from the files to be used.
- The explicit session names have not been entered in the role table.
- The four reserved sessions are not yet isolated from development scoring.
- The analysis code has not passed the targeted indexing, leakage,
  held-out-prediction, and shuffle tests.
- The training/calibration/evaluation trial counts, three projectors,
  minimum-energy gate, and calibration-visible support index have not been
  qualified on the files.
- The manuscript ownership decision is now EP07 main line with EP06 as its
  recording-domain component. Exact component-specific source/supplement/code
  overlap remains a scientific limit, not a claim of guaranteed originality.
- No suitable third-animal source has been identified or sealed.

## Storage and compute

Inputs remain read-only. Durable code, session tables, predictions, and reports
belong in this episode. Temporary arrays belong in
`$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints/`. Expensive or
sustained computation must run through Slurm.
