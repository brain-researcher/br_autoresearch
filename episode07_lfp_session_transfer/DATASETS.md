# EP07 dataset contract

This episode follows [the common adaptive protocol](../ADAPTIVE_SEARCH_PROTOCOL.md).
Downloading the public release does not make this episode development-ready.
The mixed source archive is acquired and verified, but the content-addressed,
role-filtered handoffs and their access-control receipts have not yet been
created.

## Acquired source and identity

The source is Dryad DOI `10.5061/dryad.xd2547dkt`, metadata version 7 and
file-bearing version 5. The verified local acquisition contains 24 session
`.mat` files plus one README: 25 objects and 9,571,571,204 bytes in total.
Provider SHA-256 identities, local SHA-256 verification, the provider metadata,
and the CC0-1.0 license receipt are present in the external source pack. The
publication analysis reference is BeNeuroLab commit
`cbda8e2e6106f5eb5ff98e18a689c595179ac5db`; it is provenance, not by itself a
reproducible runtime or permission to import earlier analysis outcomes.

These verified source bytes remain in a builder-only vault outside the episode.
They are not a candidate-readable input. The builder must independently bind
the source manifest, reader code, dependencies, container, and every derived
object by digest before a role-safe pack can satisfy this contract.

## Frozen primary corpus and whole-day roles

The primary corpus is the twelve simultaneous M1/PMd recording days from
Mihili and Chewie-L. Whole recording days are assigned forward in time. M1 and
PMd from the same day always have the same role.

| Animal | `source_only` recording days | `target_only` recording days |
| --- | --- | --- |
| Mihili | 2014-02-17, 2014-02-18, 2014-03-03 | 2014-03-04, 2014-03-06, 2014-03-07 |
| Chewie-L | 2016-09-29, 2016-10-05, 2016-10-06 | 2016-10-07, 2016-10-14, 2016-10-21 |

This gives six unique source days and six unique target days, or twelve
day-by-region data units in each role. Transfer is permitted only from M1
to M1 or PMd to PMd within the same animal and implant. No trial from a
`target_only` day may enter a source fit for any other target. A target's
calibration trials are its only target outcomes available for fitting.

Chewie-R is a separate-implant sensitivity and is not a third animal. It does
not enter primary search, selection, inference, or trial counts. Any later
Chewie-R analysis must use a separately frozen sensitivity manifest and cannot
change the selected policy. Area 2, incomplete days, and all other release
sessions are outside the EP07 claim. If either primary animal lacks all six
eligible days, stop for redesign before scoring outcomes; do not substitute a
different day, implant, or animal.

## Structural eligibility and target-trial roles

The frozen planning contract expects every primary day to contain eight
directions, at least 17 complete trials per direction, native 30-ms bins, and
support for both prespecified windows:

- M1 execution: `[movement onset + 150 ms, movement onset + 450 ms)`, exactly
  10 native bins; and
- PMd preparation: `[go cue - 450 ms, go cue)`, exactly 15 native bins.

These values are feasibility assumptions, not episode evidence. During episode
bootstrap and before any score is released, the episode builder must verify
them from authenticated bytes and lock an exact support summary containing
session, direction, eligible-trial, event, bin, region, channel, and
target-neuron counts. No interpolation or conversion to 50-ms bins is
permitted.

A whole behavioral trial is eligible only when `result == R`, `bin_size ==
0.03` seconds, its stable source identity, animal, implant, day, simultaneous
M1/PMd records, and one of eight authenticated reach directions join, its
required event indices are finite integers, both frozen native windows fit,
the declared matrix/guide shapes match, all required LFP/spike-window values
are finite, and spike counts are nonnegative integers. A failure in either
stratum excludes the whole simultaneous trial before hashing. This fixed rule
may inspect identities, structure, and validity only; it may not inspect
variability, magnitude, residuals, or candidate/comparator scores. All fitted
statistics, feature transforms, PCA/alignment state, residual means, and model
caches are downstream of role assignment and may use only their permitted
roles.

Direction labels are the nearest circular multiple of `pi/4` with error at
most `0.01` radians. Convert the provider's one-based MATLAB event indices to
zero-based once; for zero-based movement index `m` and go-cue index `g`, take
the Python time-axis slices `M1[m+5:m+15]` and `PMd[g-15:g]`.

The scored response is the provider-released native-bin spike count unchanged.
For each target day and region, include every authenticated unit-guide entry
whose identity and order are identical across all eligible trials and whose
spike row is present throughout every required window. Canonically order this
roster by the unit-guide tuple and freeze it from guide values and array shapes
before role assignment or any target spike value is available to a model. Do
not select neurons from calibration, development, or audit values; do not drop
low- or zero-variance neurons. A guide mismatch makes the day ineligible, and a
zero joint scoring denominator is a technical failure. Transfer and both
comparators use exactly this same roster and native count scale.

Construct an analogous outcome-blind source-neuron roster independently for
each source day and region: every eligible source trial must carry the same
authenticated unit-guide tuples in the same canonical order and the required
spike rows/windows must be structurally present. Do not intersect or match
neurons across days. Each source PCA uses only its own source-day roster; only
the resulting behavior-anchored latent coordinates may be aligned across days.

Within every `target_only` day and direction, order whole trials by a
deterministic hash of the committed split seed and canonical trial ID, then
set:

```text
n_calibration = floor(0.2*n + 0.5)
n_development = floor((n - n_calibration)/2)
n_audit       = n - n_calibration - n_development
```

Require `n >= 15` complete trials for each of the eight directions. The
contractual minimum is therefore 3/6/6 calibration/development/audit trials
per day and direction. Across the six target days, that is at least 144/288/288
unique whole-trial assignments. If the outcome-blind observation of `n >= 17`
is reproduced under the final reader, the realized lower bound becomes
144/336/336. These are unique behavioral trials: simultaneous M1 and PMd use
the same role assignment and must not be counted as independent trials.

Role assignment is immutable once its manifest is signed. A direction with
fewer than 15 eligible trials, a failed simultaneous-region join, or a window
with insufficient native bins makes the day ineligible and triggers the
six-day stop rule above; it cannot be repaired by changing rounding, dropping
the direction, borrowing trials, or redefining the window.

## Frozen provider-preprocessing exception

The release contains provider-preprocessed LFP feature matrices, not raw
voltage. In particular, session-wide centering and zero-phase
filtering/smoothing cannot be reconstructed independently inside the EP07
trial roles. Treat the released representation as one frozen,
target-label-blind provider exception shared identically by the transfer and
both comparator streams. EP07 may not re-filter, re-reference, infer raw
phase, or claim a raw-signal inductive benchmark. No additional
session-wide statistic may be fit across calibration, development, or audit
roles.

Consequently, a positive result concerns predictive information in this
released feature representation. It cannot show that an equivalent method
would transfer from independently processed raw voltage.

## Episode build and provenance

The episode builder may read the mixed `.mat` archive during bootstrap. It must
use a deterministic reader whose source, dependency lock, and tests are
recorded. The build receipt must record, for every derived object,
the source-file SHA-256, source variable path, canonical ID derivation,
MATLAB-to-analysis indexing conversion, event convention, eligibility
predicate, array shape and dtype, output SHA-256, reader commit, environment
digest, and build time. Synthetic fixtures must cover MATLAB indexing, event
boundaries, trial joins, role filtering, and native-bin slicing.

The builder must also produce:

1. an immutable 25-object provider/local source manifest and license receipt;
2. a structural inventory and trusted reproduction of the support report;
3. canonical animal, implant, day, region, trial, direction, event, channel,
   feature, and target-neuron identities;
4. the signed whole-day and target-trial role manifests;
5. a shared EP05--EP08 exposure ledger identifying prior human, agent, code,
   cache, and outcome access; and
6. byte manifests for each handoff described below.

The feature manifest must bind the exact `lfp_guide` column meanings and dtype,
the provider code for LMP, the complete band-code-to-eight-class map, canonical
class order, channel/electrode identity fields, and duplicate, missing, or
unknown-code rejection rules. It must prove that the guide is identical where
the contract requires it across trials within a day. Shape or dtype alone is
not enough to infer feature semantics, and numeric electrode IDs are never
matched across days.

No historical derived array, fitted transform, selected setting, score, or
winner may be copied into a handoff. A reader or eligibility change after any
development label has been scored invalidates all downstream packs and
requires a new versioned build before further search.

## Exact role-safe episode views

The episode must materialize or deterministically address role-specific views
and record their hashes. Candidate fitting code must not receive labels from a
forbidden role. OS-level permission separation, encryption, separate service
identities, and an external broker are optional implementation choices rather
than launch prerequisites.

### Candidate development handoff

The candidate worker may read only:

- `source_only` LFP feature arrays, spike targets, permitted behavioral
  covariates, events, directions, stable trial IDs, and structural metadata;
- target-calibration LFP features, spike targets, permitted covariates,
  events, directions, stable trial IDs, and calibration role labels;
- target-development LFP features and permitted covariates keyed only by
  opaque prediction IDs, with no spike targets or source trial IDs from which
  hidden roles can be recovered;
- frozen per-day target-neuron scoring rosters, feature/channel guides,
  window/bin specifications, schemas, split-support summaries, approved
  reader interfaces, and synthetic fixtures; and
- the candidate code/environment, trial configuration, budget state, and
  prior scalar feedback authorized by the controller.

It must not contain development spike targets; any audit ID, feature, target,
role, support summary, or metric; mixed source `.mat` files; other release
days; sibling episode data; prior winners or scores; evaluator code with
embedded outcomes; or fitted state that used a forbidden role.

### Development-evaluator view

The episode development evaluator may read the signed role and
neuron-roster manifests, the opaque-ID-to-development-target map, development
targets and validity masks, submitted predictions, the frozen R2 scorer,
fixed comparator implementations, matched-opportunity budget ledger, and the
scalar-feedback/diagnostic allowlist. It may also read the candidate pack
through a read-only evaluator mount when it must reproduce a configuration.
It returns only authorized aggregate scores and diagnostics; it never returns
trial-, bin-, neuron-, direction-, or session-level outcomes from which hidden
targets can be reconstructed. Development data are never mounted in the
candidate process.

### Held-out evaluation view

The audit payload is a separately encrypted or permission-sealed pack
containing the target-audit LFP features, permitted covariates, opaque IDs,
spike targets, validity masks, role map, target-neuron rosters, and complete
scoring metadata for all six target days and both strata. Candidate workers
receive no audit IDs, counts beyond the locked support summary, features,
labels, or partial results during search.

After a valid configuration-lock record, the episode's one-open evaluation
path loads the locked executable/environment, fitted-state recipe, and
held-out view. It runs the locked transfer policy and fixed comparators,
performs the prespecified whole-trial inference, and emits only the allowed
terminal report and access record. It may not expose partial target/session
results, permit refitting or successor proposals, or reopen the held-out view
after a score-bearing execution. An exact retry is allowed only for a
documented infrastructure failure that released no score and preserved every
input and lock hash.

## Episode execution boundary

Candidate fitting functions must receive only the declared candidate view,
configuration, controller messages, and job-local scratch. The episode must
test its role-aware loader against direct forbidden-role requests and record
the result during bootstrap. A separate Unix identity, network namespace,
signed denial receipt, and broker are not required. Because this is a
same-user logical boundary, the final report must state that limitation rather
than claiming a cryptographic confidentiality seal.

## Exposure and fixed-corpus claim

The Dryad release was outcome-exposed during earlier campaign work, including
EP05. New role manifests and a clean execution boundary can prevent additional
adaptive leakage but cannot make this corpus independent or unexposed. Before
the held-out stage, the exposure record must conservatively record all prior access,
shared code, cached features, human knowledge, and any irreducible uncertainty.

The held-out result is therefore a locked internal whole-trial evaluation on
six fixed target days from two fixed animals. It supports, at most, the added value of
source-day information for future trials from these same target days after
target calibration, separately within animal, implant, and region. It does
not support independent confirmation, population inference, prediction on a
wholly unseen day, zero-shot transfer, cross-animal transfer, cross-implant
transfer, cross-region transfer, raw-voltage performance, or clinical
generalization.

## Provisioning status and storage

The numeric execution bundle is defined jointly by
`NUMERIC_OPERATOR_GRAMMAR.md`, `ANCHOR_RUNTIME_CONTRACT.yaml`, and
`NUMERIC_SEARCH_CONTRACT.yaml`. Dataset and handoff builders must bind the
exact hashes of all three; none may infer a different roster, fit visibility,
shape fallback, or trial-count rule from the raw release.

Acquisition is complete. Structural eligibility, numeric-value/guide checks,
role-specific views, exposure recording, controller qualification, runtime
profiling, and held-out evaluation remain. They are opening stages of the
launched episode and may produce a technical terminal; they are not external
readiness or launch gates.

Large immutable bytes remain outside Git. Transient computation belongs in a
dedicated `$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/`
allocation.
Only manifests, policies, ledgers, hashes, receipts, locks, and final reports
are durable episode outputs.
