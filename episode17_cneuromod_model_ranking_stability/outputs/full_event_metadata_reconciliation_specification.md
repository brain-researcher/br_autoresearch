# Full retained-roster event metadata reconciliation

**Adopted by root, 2026-10-04 at21:00UTC, after code review and28 passing
compute fixtures.** The completed three-file pilot
supports the next source-preparation step. This document specifies two bounded
commands: exact annex-manifest preparation, then reconciliation of the retained
roster. The coordinator has selected a single bounded 40-minute batch wrapper
that invokes the second command only after the first exits successfully. The
full command validates exact roster coverage and the prespecified byte caps
before any new event-body request; it stops if those conditions fail.
No eligibility, repeat-completeness, role, fold, scientific threshold or model
choice is added by these commands.

## Existing evidence and selected population

`outputs/runs/released_event_reconciliation/pilot001/reconciliation_report.json`
records completion in **2.556249348 seconds**, with all **180 rows** matched,
zero mismatches/errors and **127,775 verified event bytes**. The existing 16
source-free fixtures passed in the coordinator's compute allocation. Unknown
first-repetition delay metadata was retained, not filled in. The pilot's
verified event-identity evidence and 180 projected rows are reused; its three
event bodies were never retained and are not downloaded again.

The completed `source_metadata/attempt003` report and retained-run CSV select:

| Participant | Retained runs | Retained presentations |
| --- | ---: | ---: |
| sub-01 | 213 | 12,780 |
| sub-02 | 213 | 12,780 |
| sub-03 | 213 | 12,780 |
| sub-06 | 194 | 11,640 |
| Total | 833 | 49,980 |

All runs retain 60 presentations. Subtracting the pilot leaves **830 new event
objects / 49,800 new rows**. The runlist is the existing provider-derived
retained roster, including its ordering and original omissions. No additional
run is included or excluded after inspecting events. A mismatch stops the
attempt; it never becomes a reason to discard a difficult run.

The exact source chain, CC0 attribution and metadata projection remain those
in [the pilot specification](event_metadata_reconciliation_specification.md):
release `0f22b6c5eca4001e86fa936f3b6c2569a0179970`, fmriprep gitlink
`5155ba0d43836a4451141b90b5a70507f41205cf`, and released events commit
`b17d3697171824c198ebc6f15fe7ceee9a1f526f`. The nested stimulus metadata commit
`f8f5162f79fb2e9fb0d8cbb8bac610db11a54180` is the same source used by attempt
003. There is no new dataset or moving-branch selection.

## Stage 1: resolve the exact retained annex objects

GitHub tree metadata identifies paths, file modes and Git blob identities; it
does not itself contain annex object keys. `prepare_released_event_manifest.py`
therefore performs this sequence:

1. Read only the completed attempt 003 metadata report and `run_metadata.csv`.
   Require the exact four-participant, 833-run roster and its retained ordering.
2. Fetch the pinned `things` root Git tree, then the four selected participant
   trees recursively. These are repository metadata, not event/neural bodies.
   Reject a truncated tree or missing retained path.
3. Require every selected event entry to be a `120000` symlink of at most
   512 bytes. Reject ordinary file blobs before requesting their body: an
   ordinary event blob could contain behavioral columns.
4. For the three pilot entries, reconstruct their already-known pointer text
   and compare it with the pinned tree's existing blob identity; make no
   additional pointer or event request for them.
5. For the remaining 830 paths, read only the bounded symlink text at the exact
   commit. Check its declared length and existing Git blob identity. Accept
   only the explicit `../../../.git/annex/objects/<two-part-directory>/<key>/<key>`
   grammar with a repeated `MD5E-s<size>--<md5>.tsv` key. Extract the directory,
   event size and existing annex digest. Do not follow the symlink.

The Git blob check ties the fetched pointer to the pinned tree; the annex
digest later ties its selected event to that pointer. These are existing
source identities used to avoid changing the release during acquisition,
not an added general provenance or scientific qualification requirement.

This stage has **835 requests at most**: five tree calls and 830 small pointer
calls, one at a time. Tree response caps are 512 KiB for the root and 8 MiB per
participant; each pointer body is capped at 512 bytes, with one additional byte
allowed solely to detect oversize. It requests no event bodies. Every request
has a 20-second timeout, no redirects, no authentication, and no automatic
retry. The stage uses **1 CPU / 4 GiB / 1080 seconds internally / 1140 seconds
outer envelope**. Its aggregate response cap is under 35 MiB. A deadline,
denial, truncation, missing path or invalid pointer leaves an incomplete
manifest with already-resolved entries and a fixed error code.

Prepared command, to be selected and run by the coordinator on compute:

```bash
python3 outputs/tools/prepare_released_event_manifest.py \
  --out outputs/runs/released_event_manifest/attempt001
```

The single `annex_manifest.json` reports exact selected run keys, original
source paths, expected bytes and digests, pilot reuse flags, aggregate/new byte
counts and maximum object size. The full
stage refuses a partial manifest, missing/reordered/extra roster entry, changed
pilot identity or unexpected path.

## Stage 2: reconcile all retained presentations

`reconcile_retained_events.py` reads the complete manifest, the existing
presentation identities, and the completed pilot report/projection. It verifies
that the 180 reused rows still match the selected retained coordinates and
that the pilot finished with verified annex identities and no mismatches.

For each of the 830 new runs, the same pilot transfer routine obtains the exact
public-CONP object with the existing annex length/MD5 checks. Mixed TSV bytes
remain in builder memory. The same column projector and reconciliation
function compare identities, provider trial order and the fixed
`int((onset - 3.0) / 1.49)` transformation. The only shared-function extension
is explicit participant/session arguments; the pilot defaults are unchanged.
The full stage never reads neural arrays, image pixels, behavioral values for
analysis, candidate features, recognition performance or audit scores.

The output retains allowed schedule metadata and legitimate unknown values.
Released-source repetition and the retained-design presentation count remain
distinct; neither is forced to equal the other. Schedule anomalies are
reported, not used to exclude runs, alter identity mappings or assign roles.

The first measured pilot gives a conservative all-in estimate of
`2.556249348 / 3 = 0.852083116 seconds/run`; 830 remaining objects project to
**707.228986 seconds** before additional full-roster bookkeeping. The prepared
full-stage envelope is **1 CPU / 4 GiB / 1080 seconds internally / 1140 seconds
externally**, with one worker and no GPU. Thus the internal bound has about
373 seconds beyond that projection. A slow network still stops at the bound;
there is no automatic renewal or retry.

Provisional transport limits are **128 KiB per event** and **64 MiB across all
833 selected events**, including reused pilot bytes. The manifest reports the
exact sizes before any new event is requested. If either limit is exceeded,
the coordinator must revise the resource selection before full acquisition;
no oversized event is silently omitted. The full runner verifies these limits
and permits only one request for each nonpilot object.

Prepared command, requiring a complete manifest that passes the fixed checks:

```bash
python3 outputs/tools/reconcile_retained_events.py \
  --manifest outputs/runs/released_event_manifest/attempt001/annex_manifest.json \
  --out outputs/runs/released_event_reconciliation/retained001
```

The projected `event_metadata.csv` is written incrementally in retained order.
A stopped attempt can contain partial allowed metadata, including a run that
reported an identity/onset mismatch. That projection is diagnostic and **must
not be consumed downstream as a completed source**. The final
`reconciliation_report.json` is written after the bounded run and must assert
`retained_roster_event_reconciliation_complete=true` before completion is
claimed. A missing terminal report, stopped state, mismatch, failed transfer
or incomplete row count means the stage is incomplete even if a CSV exists.
The runner reports every completed/reused run and the first stopped run; it
does not silently continue after failure.

A pass establishes event/design reconciliation only for the **833 retained
runs**, not for every acquired or excluded source event. It leaves
`all_source_events_reconciled=false`, `source_roles_complete=false`,
`eligible_720_concept_universe_verified=false`, and all role/fold/scoring/audit
flags false. Prospective repeat-completeness, taxonomy/balance choices and
materialized handoffs remain necessary. The full projection is builder metadata
and creates no empirical model result or new scientific null.

## Verification and ownership

The new source-free fixtures cover pinned-pointer changes, direct-event-blob
refusal before fetch, truncated/missing tree entries, exact roster selection,
pilot reuse without GETs, incomplete/oversized manifest rejection, participant
and session checks, mismatch retention, and partial-output publication without
a completion claim. Existing pilot fixtures cover transfer size/digest and
redirect/authentication-denial handling. Neither test set contacts the network
or opens source inputs.

Run both sets in the coordinator-owned compute allocation:

```bash
python3 -m unittest discover -s outputs/tests -p 'test_*events.py' -v
```

The selected batch recipe is
`outputs/tools/run_retained_event_reconciliation.sbatch`: one CPU, 4 GiB,
`normal`, `CPU_SKU:7502`, 40 minutes, no GPU and no requeue. It loads the already
qualified `devel` and `python/3.12.1` modules; these stages use only the Python
standard library. Each command retains its 1080-second internal deadline and
a 1140-second external timeout with a 10-second termination grace. Two maximum
external stages including those graces total 2300 seconds, below the
2400-second allocation. No time is padded with sleeps.

The coordinator creates the submission log directory and supplies the paths
via expanded environment variables, since SBATCH directives cannot expand
`$OAK`. The intended submission, following recorded selection and compute-side
fixture/shell validation, is:

```bash
EP17_ROOT="$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"
sbatch \
  --output="$EP17_ROOT/outputs/runs/released_event_reconciliation/submission001/slurm_%j.out" \
  --error="$EP17_ROOT/outputs/runs/released_event_reconciliation/submission001/slurm_%j.err" \
  "$EP17_ROOT/outputs/tools/run_retained_event_reconciliation.sbatch"
```

The coordinator owns selection, execution, resource accounting, log/navigation
updates and the ASTRA program. The original source, metadata attempt 003 and
pilot001 remain unchanged. Fresh attempts preserve all earlier successes and
failures; no new background service, automatic retry or agent renewal is created.
