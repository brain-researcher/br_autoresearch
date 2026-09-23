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
pools. If that common direction/pilot support does not exist, the episode stops
at readiness rather than redefining the pilot after outcomes.

The audit must determine whether each primary animal supports at least four
development and two sealed sessions and whether every retained session
supports the common `{4,8,16} × {32,64,128}` post-pilot grid plus a
full-resource ceiling check. If not, stop at readiness or freeze a revised
contract before any neural utility is calculated. Do not drop a budget cell
after seeing its score.

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

## Missing blockers

The authenticated asset pack, exact geometry coverage, prospective
development/audit session manifest, within-session trial roles, sequential
replay evaluator, permission-separated audit store, and canonical bindings
are not yet established and block launch. The absent third animal blocks
external generalization but need not block an explicitly internal episode.

Large inputs stay outside Git or under immutable read-only references.
Transient caches belong in
`$SCRATCH/br_autoresearch/episode08_lfp_electrode_trial_policy/`; durable
artifacts are limited to manifests, action/trial ledgers, lock bundles, and
reports.
