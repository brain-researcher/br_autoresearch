# EP08 dataset contract

This episode follows [the common adaptive protocol](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Real source and primary cohort

Use the authenticated Foundation LFP asset pack derived from Dryad DOI
`10.5061/dryad.xd2547dkt`, metadata version 7 and file-bearing version 5, with
the associated BeNeuroLab code pinned at
`cbda8e2e6106f5eb5ff98e18a689c595179ac5db`.

The primary cohort is six Mihili and six Chewie-L M1 sessions. Chewie-R is an
implant sensitivity, not an independent animal replication. Planning notes
suggest roughly 46–64 usable physical electrodes in primary sessions, but
exact counts, coordinates, unit intersections, and trial support must be
regenerated from authenticated bytes.

## Exposure and evidence status

The source outcomes were used previously. Prior scores, selected channels,
scaling surfaces, caches, and human observations must be listed in
`exposure_ledger.json` and cannot initialize this episode.
New held-session and within-session role manifests create a procedural
firewall for the search agent, but they do not make the public corpus an
independent confirmation source.

Current data can support only an **internal held-session audit**. A third
animal with compatible M1 LFP, population spiking, electrode geometry, events,
and acquisition metadata—collected or sequestered before policy selection—is
missing and is mandatory for an external-animal claim.

## Required records

The immutable input pack must provide:

- source/provider hashes, stable animal/implant/session/trial/channel IDs,
  reuse terms, code and environment pins;
- successful center-out reach events, direction labels, exact movement epoch,
  sampling and filter support;
- physical-electrode membership and geometry, with missing-coordinate flags;
- LMP, 100–200-Hz and 200–400-Hz feature identities plus raw provenance;
- one structurally eligible fixed neuron roster per session and spike outcome
  construction independent of selector performance; and
- synthetic sequential-acquisition fixtures and a clean replay interface.

The contract must identify exactly eight common reach directions, reserve two
hash-selected non-evaluation trials per direction for the fixed 16-trial
all-electrode pilot, and freeze disjoint calibration-acquisition and evaluation
pools. If that common direction/pilot support does not exist, stop before
neural utility is calculated rather than redefining the pilot after outcomes.

The audit must determine whether each primary animal supports at least four
development and two sealed sessions and whether every retained session
supports the common `{4,8,16} × {32,64,128}` post-pilot grid plus a
full-resource ceiling check. If not, stop before neural utility is calculated
or freeze a revised contract first. Do not drop a budget cell after seeing its
score.

## Firewall and sequential replay

Assign whole sessions from structural identifiers before outcome-derived
utility exists. Within every session, freeze pilot, acquisition-pool, and
evaluation trials separately. Within each direction, acquisition-pool trials
are hash-ordered under a frozen seed; an allocator chooses a stratum and
receives only its next queued trial. This assumes retrospective
within-direction exchangeability and does not identify a causal advantage for
real-time trial scheduling.

At policy time, the only permitted state is static geometry/QC for all
eligible electrodes; all-electrode pilot summaries; the locked retained set;
and, after each acquired calibration trial, selected-electrode LFP plus that
trial's calibration target. The evaluator computes nested-CV calibration-loss
reward and owns all future acquisition records and evaluation spikes. No
evaluation target can generate a ranker label, action, or bandit reward.

Development marginal-utility labels must be produced with whole-session
cross-fitting: the label/ranker used for one development session is trained on
the other development sessions. The audit ranker is fitted once on all eight
development sessions and transferred unchanged. Electrode retention closes
after the pilot; later addition and historical-trial backfill are invalid.

Store sealed-session mappings and evaluation outcomes under a
permission-separated audit path. Candidate jobs receive opaque IDs and scalar
development scores. The action ledger must record time, available information,
electrode-lock event, queued-trial position, action, nested-CV reward, pilot
and incremental cost, missing-geometry fallback, tie-break, and serialized
state hash so a later replay can prove there was no future information,
backfill, or cache leakage.

## Data needed to explain why the policy works

The primary audit can show that a policy wins at matched electrode and trial
counts. The [paper plan](outputs/paper_plan.md) proposes a later mechanism
round asking whether the pilot identifies nonredundant electrodes and
high-value calibration trials. That round is separate: it cannot change the
primary grid, policy, decoder, session split, or terminal result.

For each development session, preserve the following records without exposing
evaluation targets to the policy:

- physical-electrode identity and geometry, pilot-time artifact, missingness,
  line-noise, stationarity, repeatability, and pairwise redundancy summaries;
- every pilot-only selector score and the exact retained set at `E = 4, 8, 16`;
- the ordered acquisition queue, policy state before each action, predicted
  value of every legal reach-direction action, chosen action, and trusted
  calibration reward afterward;
- fixed-decoder predictions needed for prespecified electrode-removal and
  next-trial replay comparisons on the untouched evaluation set; and
- the nine grid-cell scores, full-resource ceiling, random-seed distribution,
  and failure/fallback record for every session.

These records support cross-fitted mechanism development only. Held-out
conditional electrode value is an evaluation label: for a fixed retained set,
it is the change in evaluation loss when one retained electrode is omitted and
the frozen decoder procedure is replayed. It may test a pilot-only prediction,
but it may not be returned to the selector, used as an audit ranker label, or
used to replace an electrode. The analogous next-trial value is computed by a
trusted replay on a frozen evaluation set; it cannot become a policy reward in
the session being evaluated.

### Evidence roles for the acquisition principle

| Source | Allowed role | Claim limit |
| --- | --- | --- |
| Eight development sessions | Cross-fit pilot scores, conditional-value predictions, and electrode-versus-trial sensitivity summaries | Mechanism development in an exposed corpus |
| Four primary audit sessions | Primary one-shot policy evaluation only; generic outputs declared before opening may be described but cannot test a mechanism selected afterward | Consumed internal held-session evidence, not mechanism confirmation |
| Chewie-R | Separately frozen implant sensitivity | Not an independent animal and cannot promote the policy |
| A newly collected or sequestered third animal | Test the frozen pilot-only predictions and joint policy | Required for an external-animal acquisition claim |

This chooses one unambiguous timeline. The explanatory mechanism is selected
after the primary conclusion, using cross-fitted development sessions only.
Therefore no result from the four consumed primary audit sessions can promote,
select, or confirm it. Its first confirmatory evaluation must come from the new
sealed external source described below.

A fresh external pack must contain whole-session roles, physical geometry,
the same authenticated LFP feature meanings or a frozen crosswalk, eight reach
directions, enough trials for the pilot and full grid, a structural neuron
roster, and a sequential replay interface. It must be sequestered before the
mechanism rule and all margins are selected. The policy may be refit on that
animal's declared development sessions only if the external contract says so;
it cannot tune on external audit sessions.

The released source contains preprocessed LFP features rather than a complete
prospective hardware power trace. EP08 can study post-pilot retention of
recorded physical electrodes and retrospective calibration replay. It cannot
infer surgical placement, amplifier power savings, wireless bandwidth, or the
real-time burden of requesting a behavior without additional measurements and
a separate cost model. Electrode count and trial count therefore remain two
axes; no conversion between them is implied.

## Missing blockers

The authenticated asset pack, exact geometry coverage, prospective
development/audit session manifest, within-session trial roles, sequential
replay evaluator, and permission-separated audit store are not yet established.
They block neural scoring or audit opening, not explicit task startup. The
absent third animal blocks external generalization but need not block an
explicitly internal episode.

Large inputs stay outside Git or under immutable read-only references.
Transient caches belong in
`$SCRATCH/br_autoresearch/episode08_lfp_electrode_trial_policy/`; durable
artifacts are limited to manifests, action/trial ledgers, lock bundles, and
reports.
