# EP08 data: what the pilot can and cannot teach us

EP08 uses the Foundation LFP source derived from Dryad DOI
`10.5061/dryad.xd2547dkt` (metadata version 7, file-bearing version 5). The
source contains motor-cortex LFP features, population spiking, reach events,
and electrode information needed to study post-pilot acquisition choices.

The required source pack is not currently present in this episode. No neural
score has been computed. Before analysis, the source must be acquired and the
scientifically necessary session, electrode, event, feature, and trial support
must be confirmed.

## Primary animals and sessions

The intended primary cohort is six Mihili and six Chewie-L M1 execution
sessions. If the data support the design, four whole sessions per animal are
used to develop one acquisition policy and two whole sessions per animal are
reserved for the final internal test.

Whole-session separation matters because a policy should work in a new
session, not merely on new trials drawn from a session that helped train it.
The exact session assignment must be saved before any policy-discriminating
score is examined.

Chewie-R is another implant in Chewie. It may test sensitivity to implant
identity but cannot stand in for an independent third animal.

## What every session must contain

Each retained session needs:

- stable animal, implant, session, trial, and physical-electrode identities;
- successful center-out reaches in the same eight directions;
- movement timing for the fixed 150--450 ms post-movement window;
- physical-electrode membership and geometry, with missing locations stated;
- LMP, 100--200 Hz power, and 200--400 Hz power with verified meanings;
- one population-spike target set chosen without reference to selector
  performance; and
- enough separate trials for the pilot, calibration choices, and untouched
  evaluation at every supported budget;
- enough remaining trials in every direction to branch from common replay
  states and evaluate more than one possible next trial; and
- enough eligible electrodes to form same-size, quality-matched swap sets that
  differ in pilot-predicted redundancy or complementarity.

The common pilot uses exactly two non-evaluation trials from each of the eight
directions, for 16 trials total, recorded on all eligible electrodes. If this
support is absent, the study stops for redesign before electrode or trial value
is calculated.

Every retained session must also support the complete grid:

```text
electrodes retained:             4, 8, or 16
additional post-pilot trials:   32, 64, or 128
```

A full-resource fit is kept as a ceiling check. A difficult grid cell is not
removed after its score is known.

## Data roles within a session

The trial pool is divided into three nonoverlapping parts before policy scores
are examined:

1. **Pilot:** the same 16 all-electrode reaches for every method.
2. **Acquisition pool:** later calibration trials from which a method may
   request one reach direction at a time.
3. **Untouched evaluation:** trials used only after model choices are final.

Within each direction, acquisition trials are placed in one ordinary seeded
order before any method is scored. A direction choice reveals only the next
trial in that order. This lets the analysis walk through recorded data as if
trials arrived one at a time while preventing a method from selecting a
particularly favorable future trial.

That primary order is not enough to explain adaptive trial value. Before any
neural score is examined, the explanation analysis also fixes several ordinary
replay orders and a common set of balanced replay states. Candidate checkpoints
are after 16, 32, 64, and 96 additional trials; the final list includes only
checkpoints supported by every retained primary session and is frozen after
availability checks. At each state, every legal direction is branched using
the same untouched evaluation trials. A conclusion cannot depend on one lucky
future trial or one chosen order.

This retrospective construction assumes that recorded trials within a
direction are exchangeable enough for the stated question. It does not create
a causal experiment in which a person or animal was asked to perform a new
movement online.

## What a policy may know at each decision

Before choosing electrodes, a method may use:

- static electrode geometry and quality information; and
- summaries from the common 16-trial all-electrode pilot.

The retained electrode set is then final for that session. During later trial
selection, the method may also use the selected-electrode LFP and spike target
from calibration trials already acquired. It cannot use future acquisition
trials or any evaluation spike target.

Learned electrode-value labels for one development session must come from a
model trained without that session. The final rule for the four untouched
sessions is fitted once using the eight development sessions and applied
without session-specific model selection.

The final test data are kept separate from candidate fitting. Candidate methods
receive only the pilot, acquired calibration data, and aggregate development
feedback allowed by the study. Once the global method and analysis are final,
all four untouched sessions are scored together. Partial results do not return
to model development.

## What must be retained to explain a successful policy

For each development session, preserve:

- physical electrode identities, geometry, and pilot-time reliability,
  missingness, line-noise, stationarity, repeatability, and redundancy
  summaries;
- every pilot-only electrode score and retained set at 4, 8, and 16
  electrodes;
- the simple and conditional-value retained sets, plus one-for-one swap sets
  matched on pilot reliability, artifact rate, and missingness;
- the ordered calibration pools and the information available before every
  direction choice;
- the predicted value of each legal direction, the chosen direction, and the
  calibration improvement observed afterward at every prespecified replay
  state and order;
- common-decoder predictions needed for electrode-removal and alternative
  next-direction comparisons on untouched trials; and
- the four performances at every budget: simple/balanced,
  simple/adaptive, conditional-value/balanced, and
  conditional-value/adaptive;
- the complete nine-cell performance surface, full-resource ceiling,
  repeated-random distribution, and every failure or missing-geometry
  fallback.

These records support two explanatory predictions:

1. **Conditional electrode value:** for a fixed retained set, how much
   untouched-trial loss increases when one electrode is removed.
2. **Next-direction value:** at a fixed acquisition state, how much loss
   decreases after the next trial from each possible reach direction is added.

Neither value may be returned to the policy in the session being evaluated.
They are outcomes used to test predictions made from the pilot or pre-choice
state, not rewards available when the original choice was made.

They also support the decisive coupling test. At the same replay state, the
calibration trials are held fixed while the retained electrode set changes.
The analysis asks whether a quality-matched swap changes the predicted and
observed ordering of next-direction value. If the ordering stays stable and
the four-way performance comparison is additive, the data support separate
electrode and trial methods rather than a joint acquisition principle.

## Evidence strength

The source outcomes have been used previously. New session and trial
separation can prevent additional adaptive leakage, but cannot make this public
corpus independent evidence.

| Data source | Scientific role | Claim limit |
| --- | --- | --- |
| Eight development sessions | Choose the acquisition policy and develop electrode/trial explanations using whole-session cross-fitting | Internal development evidence |
| Four untouched primary sessions | Test the original matched-budget policy once | Internal held-session evidence, not fresh confirmation of a later explanation |
| Chewie-R | Test sensitivity to another implant in the same animal | Not independent-animal evidence |
| Newly collected or sequestered sessions from a third animal | Test the chosen electrode-value and next-direction predictions and the joint policy | Required for an external-animal acquisition claim |

The current source contains preprocessed LFP features rather than a prospective
hardware power trace. EP08 can study retention of already recorded electrodes
and retrospective calibration choices. It cannot infer surgical placement,
amplifier power, wireless bandwidth, or real-time behavioral burden without
additional measurements.

## Readiness before neural scoring

The following checks remain because they directly affect the scientific claim:

- acquire the Foundation LFP source pack;
- confirm the twelve primary sessions and their independence;
- identify duplicate, derived, or overlapping sessions, including relevant
  EP05 use;
- verify eight directions, the 16-trial pilot, and support for all nine budget
  cells in every retained session;
- verify that every retained session supports common replay checkpoints,
  multiple prespecified within-direction orders, and legal next trials for all
  eight directions at those checkpoints;
- verify physical-electrode membership and state which sessions have usable
  geometry;
- verify that quality-matched one-for-one electrode swaps can be formed at
  each retained electrode count without using later neural outcomes;
- verify movement events, LFP feature meanings, and population-spike targets;
- save the development-session, held-session, pilot, acquisition-pool, and
  evaluation assignments before policy scores; and
- demonstrate once that future acquisition trials and evaluation spike targets
  cannot influence an earlier action.

If a primary animal cannot provide four development and two held sessions with
the complete supported grid, revise the design before scoring rather than
silently dropping sessions or budget cells.

If the nine-cell policy grid is supported but the repeated branch or
quality-matched swap requirements are not, EP08 may still test the primary
four-way policy comparison. It cannot claim that next-trial value is learnable
or that electrode and trial selection form a joint principle.

Large source files remain outside Git. Temporary computation belongs in
`$SCRATCH/br_autoresearch/episode08_lfp_electrode_trial_policy/`; only compact
data descriptions, session/trial assignments, policy results, and reports
belong in this episode directory.
