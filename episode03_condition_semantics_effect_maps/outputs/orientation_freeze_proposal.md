# EP03 orientation and negative-control rules — frozen 2026-10-16

**Status: frozen on the scientist's instruction of 2026-10-16 (Pacific); recorded 2026-10-17.** All four “To freeze” items are adopted as written: R1, R4, R5 and R6 with precedence, the one-sided-factor rule and role dictionary; P/I readings; eligibility screens; and the trial-split negative-control identity, replicate-order fallback and route-selection rule.

- `outputs/orientation_freeze_proposal.md`
- `outputs/runs/source_handoff_qualification/20261016-orientation-proposal/`: `condition_roles.json`, `event_edit_review.json`, `orientation_by_map.tsv`, `route_manifest.tsv`, `manifest_summary.json` (build job 48189996).
- `outputs/code/build_route_manifest.py` (rule engine).

Any change now requires an explicit scientist amendment recorded before affected outcomes are used. The family taxonomy, negative-control pass threshold and normalisation details remain separate freezes required before target access for the family-prototype baseline.
The first permitted target load is the fitted-N lineage check: BART first, then all routes subject to the registered BART gate.
No map value, header, result PNG, group report or score was opened to make it.
One read-only critique of an earlier draft (text only) led to the amendments recorded
in the [log](experiment_log.md#2026-10-16-1837-pdt--proposal-critique-amended-rebuild-selected).

Evidence:
- [route manifest](runs/source_handoff_qualification/20261016-orientation-proposal/route_manifest.tsv)
- [per-map orientation](runs/source_handoff_qualification/20261016-orientation-proposal/orientation_by_map.tsv)
- [condition roles](runs/source_handoff_qualification/20261016-orientation-proposal/condition_roles.json)
- [event-edit review](runs/source_handoff_qualification/20261016-orientation-proposal/event_edit_review.json)
- census [job 48185964](runs/source_handoff_qualification/20261016-route-census-48185964/) and build job 48189996

## What is oriented

The unit is each Dataset map's run-level contrast **as it ran**: the model embedded
in the FitLins report committed with the map, applied to the events after the
producer's edits. Definitions are authoritative. Producer contrast names and
filenames never set a sign; five names contradict their definitions (for example,
ds000009 `attendneg` is +1 on attend-*neutral*). Each condition carries
methods-only factor levels in the role table, which is also the frozen
standardization dictionary. Every level has an evidence code: S documented
(sidecar, README, paper or episode record), L explicit label, P producer relabelling,
I interpretation. P and I readings are assumptions.

## Rules

**Screens.** A contrast whose weights cancel on every regressor is ineligible
(1 map). A definition identical to another map's within the same route and session
is ineligible as a duplicate (2 maps). The alphabetically first contrast name is
kept; this is a deterministic tie-break, not a sign decision. Other defects are
flagged without changing the sign:
- regressors never HRF-convolved (4 maps);
- names that contradict definitions (5);
- fewer than 3 contributors (57 maps, all N=2);
- producer timing assumptions (ds003545, ds003858, ds004920).

**Single-sign contrasts.**
- **R1 Event vs implicit baseline** (175 maps): modelled event(s) minus the
  unmodelled time, with s = sign of the weights. This is a declared coordinate
  convention, not a claim that the baseline is a matched control. The unmodelled
  time may contain inter-trial fixation or other states, which the description
  must state where documented. An explicitly modelled rest or fixation event is
  oriented like any other event.
- **R6 Response-time-duration regressor** (17): a unit-amplitude epoch lasting the
  response time estimates an event response over time on task, not a duration
  slope. It is oriented like R1, with s = sign of the weight.
- **R7 Parametric regressor** (31): higher minus lower value of the documented
  quantity, with s = sign × direction. A signed quantity needs a documented
  positive pole; otherwise it is unresolved (1 map: ds002006 `DELTAVALUE`). The
  construction counts as resolved only if verified from events (E) or set by an
  executed transformation (X). Producer code (C) or an original column (O) leaves
  it assumed.

**Comparisons** (weights sum to zero; positive set P, negative set N). Every
rule works in both directions: if N rather than P holds the higher level, s = −1.
Reversing the weights must give −s, and all 628 maps pass that check. Factors are
scanned in precedence order.
- At the first level where some factor separates P from N cleanly, all clean
  factors at that level must agree, or the map is unresolved.
- A factor whose levels are mixed across P and N blocks lower levels.
- A factor defined on one side only is skipped. If it sits above the deciding
  level, the result is marked assumed (12 maps).
1. **R2 Task minus control:** the documented control level is on one side only.
   Controls are designated per route from documented roles:
   - control, rest and fixation conditions;
   - scrambled or Fourier images, tones, false photographs;
   - go trials in stop-signal tasks;
   - neutral stimuli or cues, look/observe instructions;
   - literal sentences, faceless stimuli, non-food images, computer opponents.
   No label is a control by itself.
2. **R3 Higher minus lower** on one documented ordinal manipulation: conflict,
   load, difficulty, magnitude or intensity, nutrient or calorie load, body size,
   lexicality (symbols < pseudowords < words), the item's exposure count (repeated
   over novel presentations, studied over unstudied items), and documented
   familiarity (famous or familiar over unfamiliar). Recognition *responses* are
   judgements, not exposure counts, so they fall under R5.
3. **R4 Valence:** more positive minus less positive. Outcome valence
   (gain/win/reward over loss/punishment), stimulus or goal valence, and
   performance outcome (correct, successful or hit over error, failed or miss) are
   separate factors at this level. If they disagree, the map is unresolved.
4. **R5 Declared category order** (106 maps): with no asymmetry above, the
   canonical comparison is the alphabetically first standardized label minus the
   later one. A side with several labels is oriented only if all its labels sort
   before, or all after, the other side's. This is a coordinate convention, not a
   claim about the categories; self vs friend, recognition responses and trial
   phases fall here.

**Status vocabulary.**
- `resolved_frozen`: fixed before any map access (BART, 9 maps).
- `resolved_goal_rule`: R2, R3 or R7 on documented roles or construction.
- `assumed_goal_rule`: the same rules resting on P/I roles, C/O constructions or a one-sided higher factor.
- `proposed_rule`: R1, R4, R5 or R6, which need adoption. The `_assumed_roles` variant also rests on an assumption.

## What the rules give

| Status | All 628 maps | Included accessions (31; 322 maps) |
| --- | ---: | ---: |
| resolved (frozen + GOAL rule) | 9 + 199 | 9 + 97 |
| assumed, GOAL rule | 51 | 33 |
| proposed rule (+ assumed) | 323 + 22 | 155 + 8 |
| unresolved | 21 | 19 |
| ineligible | 3 | 1 |

By rule:
- R2: 192 maps (189 with s=+1).
- R3: 37 maps (27 with s=+1).
- R4: 47 maps (43 with s=+1).
- R5: 106 maps (42 with s=+1, 64 with s=−1).
- R1: 175 maps; R6: 17 maps.
- R7: 30 oriented, plus 1 unresolved.

The engine reproduces all nine frozen BART decisions. Under the proposal the four
open BART maps become:
- `pumps` +1 and `allpumps` +1 (R1);
- `explodevcash` −1 (R4: cash-out over explosion, as documented by Schonberg et al. 2012);
- `rt` +1 (R6).

Unresolved maps:
- ds005267: undocumented CND codes (11 maps);
- ds000229 `paperfig3a`/`paperfig3b`: interleaved load levels;
- ds000114 `fingerlipsvfoot` (both sessions): interleaved labels;
- ds001297 `controlvoddball`/`friendvoddball`: the oddball role is unknown;
- ds004056 `action*govnogo`: undocumented go/no-go roles;
- ds003789 `oldvsnew`: mixed item status;
- ds002006 `valuedifference`: the sign of a signed value difference is unknown.

## Assumptions fixed now, with their checks

These readings are adopted before any outcome they affect. Checks are recorded
here; their decision thresholds stay open until the inference specification is
frozen. No sensitivity sign or exclusion may be chosen by prototype loss or gain.
- **P/I roles** (32 maps in 11 routes):
  - ds002872: high/low and texture/vibration, as relabelled by the producer;
  - ds000212: decoded moral-condition codes;
  - ds001229: catch trials as controls;
  - ds001297: control identities as unfamiliar;
  - ds003481 `sar`: speech acts;
  - ds000170: later over earlier block;
  - ds000114: response present over absent.

  Sensitivity: report the primary comparison with these maps excluded, and once
  with each reading's sign reversed.
- **C/O constructions** (30 maps in 9 routes): parametric directions in ds000005,
  ds000009, ds001715, ds002006, ds003831 and ds004556. Each direction is +1 in every
  committed version. Before loading, recompute ds000009 from OpenNeuro events and
  compare with the fitted design columns.
- **R5 convention** (106 maps): report family support and prototype loss with R5
  maps excluded as a sensitivity analysis.
- **Definition and data defects** (decision needed at freeze):
  - exclude the null and duplicate maps;
  - recommend excluding the 4 maps on unconvolved regressors and the 57 N=2 maps (df=1);
  - hold ds003545 (40 maps) out of the primary set until its onset timing is
    settled. Job 48188582 found that every run's events end before the trimmed
    span (e.g., 360 s vs 364 s; 710 s vs 714 s), so the 4-volume shift is neither
    confirmed nor excluded.

## Negative-control contrast

The contract asks for a contrast whose two condition labels are scientifically
exchangeable within each family. Exchanging those labels must leave no information
for the procedure to exploit.

**Preferred identity: within-run trial split.** In each selected route, take the
condition that the rules make the task side of the route's first R2 contrast, or
failing that the event of its first R1 contrast. Split that condition's trials
within every run into two halves, A and B. Assignment is random and balanced in
time: trials are paired consecutively by onset, and a fixed seed —
SHA-256(`ep03-negctl|accession|subject|run|pair`) — decides which member of each
pair is A. Refit the executed run-level model with A and B as separate regressors,
everything else unchanged. Then carry A − B through the same Subject and Dataset
nodes and normalise like every target. A and B are the same condition, randomly
divided within the same runs, so the labels are scientifically exchangeable rather
than only over random draws.
- **Exchange unit:** the trial assignment. "Exchanging labels" means redrawing
  the split, never negating a finished map, which canonical orientation would undo.
- **Use:** run the complete procedure on these targets within each frozen family.
  The pre-search negative-control rule must find no positive group-weighted gain
  over the family prototype; a zero expectation does not guarantee a non-positive
  observed gain.
- **Leakage:** each split shares its accession group and subjects with the source
  maps, so grouped folds keep them together.
- **Scope:** this control tests fitting to noise, not leakage that rides on real
  task signal. That remains the job of the within-family permutation, the
  metadata-only model and duplicate exclusion.
- **Cost:** refitting needs the preprocessed BOLD and confounds of the selected
  routes, and runs after the freeze.

**Fallback: replicate-order contrast.** Where refitting is not possible, use the
paired difference between the first two runs or sessions, with a seeded per-subject
order, built from effect maps already written by the map commits. It needs no refit,
and 66 of 96 routes support it, including routes from all 31 included accessions.
Its labels are exchangeable only over random assignments: practice or session
effects enter as random-sign noise in the frozen sample.

**Not adopted:** existing acquisition or session replicates used as label pairs
(ds003858 multiband variants, ds003831 runs 1/2, ds000114 test/retest). Their
labels are not shown to be exchangeable, and they cover too few families.

## Adopted at the freeze (2026-10-16)

Adopted as written, in one decision:
1. R1, R4, R5 and R6 as written, including the precedence, the one-sided-factor
   rule and the role dictionary;
2. the P/I readings above, or mark those maps unresolved;
3. the eligibility screens for null, duplicate, unconvolved, N=2 and ds003545 maps;
4. the negative-control identity (trial split, with replicate order as fallback)
   and its route-selection rule.

The family taxonomy, the negative-control pass threshold and normalisation details
belong to the separate methods-only taxonomy and inference freezes, and must also
precede target access. Then the first target load can run: the BART df/recompute
lineage check, then the family-prototype baseline.
