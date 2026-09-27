# EP06 data plan

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
7. small known-answer tests for indexing, held-out prediction, profile
   construction, and each required shuffle control.

Every transformation that sees neural values—including scaling, reliability
weights, spike-population axes, band weighting, and any across-session
alignment—must be fit only on the permitted training data.

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

The primary analysis asks whether the complete M1–PMd profile transfers. If it
does, the follow-up asks whether the difference has a prediction consequence.
It uses:

| Source | Scientific use | Condition before use |
| --- | --- | --- |
| Eight development sessions | Learn the M1, PMd, and pooled frequency-weight rules | Fix the first result and the weight-construction rule before scoring the follow-up |
| Bandwise held-out predictions in the other animal | Compare correct-region, pooled, and deliberately swapped weights on identical predictions | Same trials, electrodes, latent dimensions, model family, and band support for all three rules |
| Four reserved sessions | Test the final classifier and the prespecified prediction consequence together | Open all four once; do not inspect one result before deciding whether to compute the other |
| Recording-quality metadata | Test whether array quality or missingness explains the region label | Use a fixed metadata-only comparison and matched support |
| Future third animal | Test the unchanged rule beyond Mihili and Chewie | Plan separately and keep outcomes sealed until the rule is final |

If the M1 and PMd frequency weights are effectively the same, or the target
bandwise predictions are so similar that the three rules cannot differ, label
the consequence inconclusive. Do not interpret that situation as proof that a
pooled rule is biologically sufficient.

M1 and PMd use different arrays in this release. Matching observed metadata can
rule out measured recording differences, but it cannot remove an unmeasured
region-linked hardware effect. The strongest permitted claim is therefore about
the measured recording domains.

## Current readiness

Status: **revise before any scored analysis**.

- The Dryad recordings are not available in `inputs/`.
- Animal, session, region, trial, event, electrode, unit, and feature meanings
  have not been verified from the files to be used.
- The explicit session names have not been entered in the role table.
- The four reserved sessions are not yet isolated from development scoring.
- The analysis code has not passed the targeted indexing, leakage,
  held-out-prediction, and shuffle tests.
- No suitable third-animal source has been identified or sealed.

## Storage and compute

Inputs remain read-only. Durable code, session tables, predictions, and reports
belong in this episode. Temporary arrays belong in
`$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints/`. Expensive or
sustained computation must run through Slurm.
