# Metadata-only source chronology qualification

The coordinator prospectively adopted this mapping for implementation on
2026-10-04 at 21:49 UTC. Execution requires coordinator review and focused
tests. It changes no image eligibility, source role or outcome boundary.

Inputs are the fixed attempt003 run/presentation identities and B/C/D header
report, completed retained001 event projection/report, and completed
source_qualification/attempt002 report and image-qualification table. The
existing source loader replays all 49,980 row identities and 833 retained runs.
All retained runs participate in chronology, including presentations of
ineligible images. Absent participant/image combinations are not invented and
are not invalid ordinals. Eligible images separately require both repeat
halves in all four participants. There are no new downloads.

## Within-image ordering and repeat halves

The pinned provider
[schema](https://github.com/courtois-neuromod/things/blob/b17d3697171824c198ebc6f15fe7ceee9a1f526f/task-things_events.json)
defines repetition as cumulative presentations including the current one.
The provider's
[cleanup code](https://github.com/courtois-neuromod/things/blob/b17d3697171824c198ebc6f15fe7ceee9a1f526f/code/cleanup/clean_events.py)
maintains that count over its reconstructed full history before exporting
retained runs. Discarded `b` sessions and interrupted runs can therefore create
gaps in the retained source ordinals.

Group actually observed rows by participant and exact provider image path.
Require positive, unique source repetition ordinals, sort by that ordinal and
assign zero-based retained rank. For eligible images only, assign
`repeat_half = 1 + retained_rank % 2`. The parity of the source repetition
number is not the half rule: retained ordinals 1 and 3 still enter different
halves. Within each run the ordinal order must agree with `source_order`.
Unknown, duplicate or contradictory ordinals remain unresolved; no tie-break,
exclusion or substitution is permitted. Ineligible images retain their
metadata rank where known, but receive no analysis repeat-half label.

## Relative global acquisition sequence

The same cleanup code sets `trials_since_lastrep = i - old_i`, where these
indices refer to its reconstructed full history. For adjacent retained
occurrences of one image whose source ordinals are exactly consecutive,
the current row's known positive trial gap gives:

```
start(current_run) - start(previous_run)
    = trials_since_lastrep(current)
      + source_order(previous) - source_order(current)
```

Do not create this edge across a source-ordinal gap: the current gap then
refers to an omitted predecessor. Missing trial gaps produce no edge. A
same-run edge must imply zero offset. Solve the resulting weighted graph
using exact integer differences; every retained run is a node. Check all
edges/cycles, require nonoverlapping 60-trial run intervals within connected
components, and check every available image's ordinal order against the
reconstructed row positions. Contradictory components have no qualified
coordinates. Multiple components have no relative global order; lexical IDs
may identify components but cannot break chronological ties.

For a fully qualified participant, the graph has one consistent component;
subtract its minimum run start to choose an arbitrary zero origin. Sort
retained rows by these reconstructed positions to obtain retained acquisition
rank. Provider-history gaps remain gaps in the reconstructed positions.
These positions are neither elapsed time nor a claim that every actual or
discarded exposure is present in the EP17 roster.

The pinned
[QC notes](https://github.com/courtois-neuromod/things/blob/b17d3697171824c198ebc6f15fe7ceee9a1f526f/code/cleanup/qc_notes.md)
specify sub-01/session-17 run order **1, 2, 3, 5, 6, 4**. The reconstructed
order must agree. This known exception is a validation constraint, not a
license to fill other missing offsets or assume numerical run order. No
private scan dates, clock fields, behavioral responses, neural values,
pixels or model outputs are accessed.

## Outputs, completion and bounds

`presentation_chronology.csv` preserves the permitted event projection and
B/C/D coordinates for every observed row, adding eligibility, retained repeat
rank, eligible-image half, ordinal status, component identity, qualified
component-relative position, qualified global source-history position and
retained acquisition rank. `run_chronology.csv` retains all 833 run identities
with component/offset qualification. `chronology_report.json` distinguishes
eligible-image half qualification, global chronology and combined completion.
It reports skipped ordinal gaps, missing lag edges, component/issue counts
and bounded diagnostic coordinates. No roles, folds, blocks or neural
handoffs are produced.

Disconnected or contradictory evidence is a stopped scientific
qualification, with diagnostic metadata preserved and completion false.
Valid repeat halves can be separately qualified even when global order is
unresolved. An interrupted attempt may leave partial CSVs; a completed
terminal report is required for any complete mapping claim. Earlier attempts
and the adopted eligibility universe remain unchanged.

Run with one CPU, 4 GiB, no GPU; 120-second internal deadline, 180-second
external envelope and 10-second kill grace. Each existing input file is
bounded to 32 MiB. No retry, network request or successor is automatic.

```bash
timeout -k 10s 180s python3.12 outputs/tools/qualify_source_chronology.py \
  --out outputs/runs/source_chronology/attempt001
```

Focused fixtures live in `outputs/tests/test_source_chronology.py` and run
only in the coordinator-owned allocation. The graph assumptions are tested
without opening source payloads, outcome values or private timestamps.
