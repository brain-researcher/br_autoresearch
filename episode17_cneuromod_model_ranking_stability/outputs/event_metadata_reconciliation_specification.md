# Released-event metadata reconciliation: bounded pilot

**Adopted by root, 2026-10-04 at20:44 UTC, after root and independent static
review; compute tests and source execution remain pending.** This is an executable precursor within
the existing `source_metadata` analysis, selected under the scientist's resume
instruction. It does not assign roles, complete `source_roles`, change any
scientific rule, or authorize neural/model scoring. The coordinator owns the
execution log, ASTRA program, actual allocation and subsequent launch decision.

## Scientific purpose and source identity

The completed metadata attempt 003 reconstructed GLMsingle presentation order
from sparse design coordinates, the runlist and the pinned stimulus-identity
table. The acquired release left `THINGS/fmriprep` unpopulated, so actual
released-event identity and schedule reconciliation remained incomplete.
Recover the released metadata needed for that reconciliation. This is neither
a replacement dataset nor a repetition-exclusion decision.

The acquired local release's Git metadata establishes:

1. CNeuroMod-THINGS release commit
   `0f22b6c5eca4001e86fa936f3b6c2569a0179970`;
2. its `THINGS/fmriprep` gitlink
   `5155ba0d43836a4451141b90b5a70507f41205cf`;
3. that commit's `sourcedata/things` gitlink
   `b17d3697171824c198ebc6f15fe7ceee9a1f526f` in the official
   [courtois-neuromod/things repository](https://github.com/courtois-neuromod/things/tree/b17d3697171824c198ebc6f15fe7ceee9a1f526f).

The [nested submodule definition](https://raw.githubusercontent.com/courtois-neuromod/things.fmriprep/5155ba0d43836a4451141b90b5a70507f41205cf/.gitmodules)
and the exact GitHub contents response identify the source. The official
`things` git-annex `remote.log` advertises the public CONP HTTP mirror:

`https://sftp.conp.ca/users/cneuromod/ria-conp/515/1ad3c-321e-4111-abf0-95fee0dcbb90/annex/objects/`

No S3 credentials, authentication changes or private remote are used. Before
selection, a HEAD request for the first exact CONP object returned HTTP 200
with Content-Length 41,689. No event body was fetched during preparation.

The [provider schema](https://raw.githubusercontent.com/courtois-neuromod/things/b17d3697171824c198ebc6f15fe7ceee9a1f526f/task-things_events.json)
defines identifiers, trial order, onset and repetition/schedule columns. The
[provider processing documentation](https://raw.githubusercontent.com/courtois-neuromod/things/b17d3697171824c198ebc6f15fe7ceee9a1f526f/code/README.md)
distinguishes released, de-identified events from private scan dates and logs.
Only released events are selected. The CNeuroMod CC0 attribution and existing
source-use record are retained; this task does not use image pixels.

The pinned `things` commit's `stimuli` gitlink is
`f8f5162f79fb2e9fb0d8cbb8bac610db11a54180`, exactly the stimulus-metadata source
already used by attempt 003. The schema names `category_nr` as provider
categories 1–720, `exemplar_nr` as 1–6, and `things_category_nr` as THINGS
categories 1–1854. The same named columns are copied directly from that
pinned `image_paths_fmri.csv` by attempt 003; no zero/one-base adjustment or
new concept crosswalk is introduced.

## Fixed pilot and resource envelope

Exactly the first retained `sub-01`, session 01, runs 01–03 are selected.
Each path is `sub-01/ses-01/func/sub-01_ses-01_task-things_run-XX_events.tsv`
at the pinned `things` commit. The exact annex pointers already inspected are:

| Run | Annex directory | Bytes | Existing annex MD5 |
| --- | --- | ---: | --- |
| 01 | `W5/2J` | 41,689 | `6bd1c67c8fb1c35739090756ca5827a3` |
| 02 | `08/W9` | 42,672 | `b89843ef07db10d7e6c5f30f94249834` |
| 03 | `66/24` | 43,414 | `32b057f64d9622449095c6f147166736` |

Each CONP path appends
`<directory>/MD5E-s<bytes>--<md5>.tsv/MD5E-s<bytes>--<md5>.tsv`.
Total expected event content is **127,775 bytes**; the reader permits only one
extra byte per object to detect oversize responses, at most **127,778 bytes**
over the three response-body reads. Matching the provider's length and
existing annex digest verifies transfer identity; no new provenance system is
introduced.

The coordinator will use **1 CPU, 4 GiB, no GPU, 600-second outer envelope**.
The runner has a **540-second internal deadline**, one transfer/worker at a
time, and **20-second timeout per request**. Each object has one GET, no
automatic retry, and no redirects. A missing file, authentication response,
transfer/identity failure or metadata mismatch stops the pilot. Historical
attempts and original sources are untouched. Do not respond by selecting a
different version, changing scientific eligibility or silently retrying.
The custom HTTP error callbacks close redirect streams before reading a body;
they do not invoke urllib's ordinary redirect-body consumption. Other HTTP
denials are closed without reading their bodies or attempting authentication.

## Builder boundary and output projection

Raw TSV bytes mix metadata and behavioral responses. They are fetched only
inside the trusted-builder Slurm process, kept in memory and never exported
as raw files, log snippets or exceptions. Decoding a TSV row necessarily
receives its mixed bytes; the boundary is an explicit column projection,
not a claim that behavioral bytes were absent or that same-UID users are
isolated by the operating system.

The projection selects only:

- subject/session/run IDs, `TrialNumber`, and provider `order`;
- image path/basename and provider/THINGS category and exemplar identifiers;
- stimulus onset and duration;
- repetition number, `atypical`, `not_for_memory`, `delay_days`,
  `delay_seconds`, and `trials_since_lastrep`.

It excludes answers, response times, accuracy, confidence, keypresses,
eye-tracking metrics, free-text correction logs and scan-clock fields. No
neural, anatomical-mask, image-pixel, checkpoint or model-feature array is
opened. Absent or provider-declared unknown optional schedule values remain
unknown, with an explicit coordinate in the report. Invalid core identity
metadata is a failure rather than an inferred replacement.

The already-completed `outputs/runs/source_metadata/attempt003/` report,
`run_metadata.csv` and `presentation_identity.csv` are the sole local data
inputs. The pilot verifies their first-session three-run roster and 180
presentation coordinates before fetching events.

For each source row, compare its image basename and canonical path suffix,
THINGS/provider category IDs, exemplar ID, trial number, provider order and
onset-derived TR with the corresponding fixed retained-design row. The
provider schema explicitly defines `TrialNumber` as the within-run trial
number 1–60 and `order` as image presentation order within the run starting at
zero. The pinned `code/cleanup/clean_events.py` uses `order` as `trial_num`,
inserts the subject number into `subject_id` and the session label into
`session_id`, and verifies numeric `run_id` against the filename. The parser
accepts those numeric identifiers with their documented `sub-`/`ses-`/`run-`
prefixes where applicable. This checks the source's declared semantics; the
pilot can reveal an inconsistency but may not redefine the columns to pass.
The 60-row expectation comes independently from each selected attempt 003
retained-run record and the provider's task/design documentation.

The
provider `GLMsingle_makedesign.py` uses exactly:

```python
int((onset_seconds - 3.0) / 1.49)
```

Python `int` truncates toward zero. Preserve that floating-point computation;
do not round, floor, infer an offset or optimize alignment. The image path may
have a source-directory prefix, but its canonical `images_fmri/<concept>/<image>`
suffix and basename must agree. A whole wrong participant/session/run is
refused, not dropped. All selected rows are preserved; none are excluded to
repair a mismatch.

Released-source `repetition` and the retained-design presentation count have
different scopes. Record both without asserting equality or selecting a new
repeat-completeness rule. Schedule flags do not exclude sessions. The existing
contract still requires a prospective repeat-completeness rule, taxonomy and
balance choices before eligibility or the 480/120/120 assignment.

## Commands and acceptance

After source-free fixtures and code review, the coordinator may run:

```bash
python3 outputs/tools/reconcile_released_events.py \
  --out outputs/runs/released_event_reconciliation/pilot001
```

The path must be a fresh direct child of the named episode output parent.
There is no argument that expands the participant/session/run selection.
Outputs are `event_metadata.csv` and `reconciliation_report.json`; only the
explicit projection is written. Reports retain per-run elapsed time, transfer
identity, mismatch coordinates and unknown schedule metadata. A failure exits
2 with no automatic successor. Passing exits 0 and asserts only
`selected_pilot_reconciliation_complete=true`; whole-source reconciliation,
source roles, eligibility and audit/scoring remain false.

The source-free fixtures in `outputs/tests/test_released_events.py` exercise
correct reconciliation, exact truncation, omitted rows/columns, wrong-run
refusal, order/identity mismatches, schedule unknowns, and suppression of
behavioral/closed-field sentinels. They also check transfer size/digest failures
with an in-memory response; no real network or source input is used by tests.
Redirect and HTTP-denial fixtures explicitly fail if an error-body read occurs.
Run these only in the coordinator-owned compute allocation:

```bash
python3 -m unittest discover -s outputs/tests -p 'test_released_events.py' -v
```

The full retained-roster follow-up is **unselected**. Its file list, aggregate
byte budget, resource projection and exact stop condition must use the measured
pilot. The approximately 833-run scope is not an authorization embedded in
this runner, and a pilot pass does not activate that download.
