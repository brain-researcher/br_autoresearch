# Experiment log

## 2026-10-17 — analysis_specification — Codex review, drafted fixes and re-verification (48224711, 48245676)

- At the user's instruction, Codex gpt-6-astra (max reasoning, text-only: code, spec and T11 summaries; no data, no T9) independently reviewed the build. It returned 16 findings.
- Accepted after checking each against the code: silent fallbacks for the normalization and readiness settings; races and missing coverage in the seal and manifest (now an anchored atomic `attemptNN` claim with a predefined restart rule, all runtime inputs hashed, and the T9 archive SHA-256 from the acquisition manifest); CLI checks that ran after sealing; a raw margin that could turn positive; the ignored unlabelled choice; emulator noise added at missing windows; the df-limit and intercept gaps; a second artifact threshold; ungated branches; unpersisted stages; unguarded injections.
- Rejected one finding: applying target readiness to the local-signal statistics (the policy uses source readiness there). CORAL keeps singular-value clipping, and the spec now says so.
- Codex drafted all the replacement code, 20 seal/freeze checks, the fixed-point artifact fence and the past-only SD guard. I reviewed it line by line, patched one over-strict metadata comparison, and applied it.
- [Checks](runs/analysis_specification/20261017-synthetic_checks08/tests.txt): 49 pass. [Calibration07](runs/analysis_specification/20261017-t11_calibration07_blocknorm/) gives the same conclusion: strict fails; screened recovers +0.181/+0.178 and −0.294 but still has 24%/33% non-estimable draws.
- The single 10× pre-artifact fence had collapsed day 444 (J −0.71); the fixed-point fence fixes it. I noticed this through a T11 score, so it is disclosed as post hoc.
- [Rehearsal04](runs/analysis_specification/20261017-t9_rehearsal04/): the NPZ and archive-reader runs give identical primary results. ASTRA validated in job 48245676.
- Compute: 48224711 (8 CPU) used 43,272 CPU-s (12.0 CPU-h) and timed out after 1.5 h while a permission prompt waited; the validation job was trivial. Next: unchanged, the scientist's decisions.

## 2026-10-17 — analysis_specification — T9 entrypoint rehearsed; records updated; two questions for the scientist (48169513)

- Final checks: 29 pass in 30 s ([tests.txt](runs/analysis_specification/20261016-synthetic_checks04/tests.txt)).
- [Rehearsal 03](runs/analysis_specification/20261016-t9_rehearsal03/) ran the identical sealed-evaluation code path on T11 with the development reader and forced branches. It finished in 2.5 min at 0.33 GB and returned `support_or_uncertainty_prevents_an_answer`.
- Rehearsals 01–02 exposed few-value scaler blow-ups in the tx_4_5, past-only and half-window paths, now fixed. The leakage control now shows +0.134 inflation, positive in every session.
- The exact T9 command, resources and prerequisite order are in the spec (`t9_evaluation`). `freeze_t9_evaluation.py` refuses until the T11 control passes and a development-search panel exists. Neither is true, so T9 stays sealed and unopened.
- Updated GOAL, DATASETS, policy `readiness`, ASTRA (new `t11_calibration` analysis, plus recipes for the synthetic checks and the T9 run) and the README.
- Compute today: attended allocation 48169513 (russpold, 8 CPU / 48 GiB) ran 15:32–00:21 PDT and was released: 253,792 CPU-s, 70.5 allocated CPU-hours for EP16. Much of it sat idle while an earlier turn was interrupted. ASTRA schema and semantic validation passed ([validation_20261017_job48169513.txt](astra/v0.0.14/validation_20261017_job48169513.txt)) using the pinned uvx validator with `linkml-runtime<1.12`.
- Next: the scientist decides block normalization and the readiness rule. Then rerun `run_t11_calibration.py`, run the T11 development search, freeze, and submit the T9 run once.

## 2026-10-17 — analysis_specification — T11 known-positive control fails under the frozen readiness rule (48169513)

`20261016-t11_calibration05_blocknorm` (supersedes 02–04, which are preserved). Under block normalization, pending the scientist:
- Values: λ_sbp 3.16, λ_tx_4_5 5.62, frozen-scaling λ 178; J_min = D_min = 0.07 (permutation-null q99 0.065, rounded up); raw margin −0.046 (= −0.15 × near F−B 0.302).
- Strict frozen rule: 4 sessions with J ≤ 0.07 leave the observed contrasts unresolved, and 96% of draws are non-estimable. **Control fails.**
- Screened diagnostic (not the frozen rule): adjusted 0.196, robust 0.492, fractional +0.66, raw −0.177. The +0.20 injection is recovered as +0.181 (adjusted) and +0.177 (robust); the −0.30 injection exactly. But 25% and 34% of draws stay non-estimable, so the bounds are infinite.
- [Figure](figures/ep16_t11_positive_control.png). Three-line summary:
  1. The estimator and the injections work.
  2. Readiness failures, not sensitivity, block the control: T11 has signal-free sessions, and many sessions sit near any J_min.
  3. Under the frozen rules a T9 null would be underidentified and T9 itself is likely unresolved.

## 2026-10-16 — analysis_specification — T11 local signal needs block normalization; frozen readiness leaves T11 unresolved (48169513)

- `t11_calibration02` (frozen source-only scaling): λ_sbp 178, median local J 0.027 (max 0.082), 25/64 sessions ≤ J_min 0.02, near-gap F−B −0.36. Every R contrast is unresolved.
- Development diagnostic: z-scoring each block with its own trials gives median J 0.33 (p10 0.13). Single channels correlate with the label up to |r| = 0.55. The signal is present, but sbp baselines shift between blocks; the provider applies block-wise normalization before decoding.
- `t11_calibration03_blocknorm`: λ_sbp 3.16, median J 0.41, near F−B +0.30. The weakest sessions (days 364, 226, 706, 1549: J −0.056 to 0.045) have no data-quality defect.
- The shifted-label null (q99 0.12) aliases the centre-out-back label sequence, so the J_min null is now a within-block permutation (post hoc, disclosed).
- Two items need the scientist (spec `proposed_pending_scientist`): block normalization as primary, and the readiness rule, under which any signal-free target leaves T11 (and likely T9) unresolved.
Next: `t11_calibration04_blocknorm`, running the positive control under the frozen strict rule and, as a labelled diagnostic, with readiness-failing sessions screened out before pairing.

## 2026-10-16 — analysis_specification — calibration attempt 01 failed numerically; axis and validity rules fixed (48169513)

`20261016-t11_calibration01` failed: λ went to the grid maximum, J reached −36,849, the raw contrast reached ~1e54, and the control stage hit an API error. No interpretable score was produced; the attempt is preserved.
Diagnosis on T11 features and electrode records:
- `electrode_id` is a file column number. Day 1549 lists the medial array first, giving a 384-channel axis. Utah-grid (array, x, y) is unique and consistent: 192 keys in every T11 and T9 session.
- Days 406 and 1057 have one dead 96-channel sbp array (SD ~1e-25).
- sbp window means reach 1e7 µV² (median ~40). 0.60% of live channel-windows exceed 10× the channel median, and the tail extends to 10^6×.
Rules fixed in the spec at 01:13 UTC: axis keyed by (array, x, y); dead channel = SD ≤ 1e-6 × session median SD; an sbp window above 10× the channel's session median is a missing channel feature (5×/20× change the flagged share by ≤0.1 point). No labels or scores were used.
Next: `20261016-t11_calibration02` with the corrected runner.

## 2026-10-16 — analysis_specification — synthetic checks pass; three amendments before any T11 comparison (48169513)

T11 packets ([packets.json](runs/analysis_specification/20261016-t11_packets01/packets.json), cursor at start only): 64/64 sessions resolve. 16 placeholder and 13 near-target-start trials were removed; 63/64 sessions have octant-complete first-32 subsets; 20/64 have configuration-balanced sides. The realized design equals the registered one (129/1,346 pairs, condition 13.8, retained energy 0.27, node share 0.056, passes).
Synthetic attempts 01–02 exposed three pipeline issues, fixed before any T11 neural score ([spec](config/analysis_specification.json) `amendments_before_T11_comparisons`, 01:08 UTC):
- Refitting n=64 ridge models on duplicated resampled trials biased bootstrap J by −0.044 (SD 0.052). The bootstrap now resamples evaluation trials only, with fits fixed.
- The repair population is the long pairs whose target has octant-complete first-32 subsets.
- Unfloored variance-matching E2 emulated sampling noise as degradation; added `E2_theta0_floor`.
Attempt 03: all 29 checks pass in 31 s ([tests.txt](runs/analysis_specification/20261016-synthetic_checks03/tests.txt)). Margins are unchanged. Two findings are recorded as interpretation notes: the adjusted contrast absorbs smooth drift, and the 0.05 recording-separation margin has low power (synthetic lower bound 0.008–0.038).
Next: T11 λ, J_min, raw margin and the known-positive control (`run_t11_calibration.py`, 9,999 draws).

## 2026-10-16 — analysis_specification — role seal tests pass; T11 feature extraction launched (48169513)

Seal and reader checks (`test_ep16_reader`, 3 tests) and the unchanged structural-reader tests (4) passed on sh04-04n23 in 0.034 s. They cover: refusal of T9 without a sealed token; refusal of T2/T6/T7/T8; refusal of a development archive whose file says T9; exact 30-bin means; per-channel invalid windows; windows crossing a block end.
Launched `extract_development_features.py --participant T11` for the 64 sessions of T11's passing P0 component (1 CPU). It writes transient arrays to `$SCRATCH/.../features/T11` and a count-only report to `runs/analysis_specification/20261016-t11_features01/`. No other participant's archive is opened.

## 2026-10-16 — analysis_specification — rules fixed before any neural read (allocation 48169513)

Wrote [analysis_specification.json](config/analysis_specification.json) at 00:18:38 UTC Oct 17, before any T11 or T9 neural or cursor value was read. Only structural metadata had been seen: T11 has 64 capable sessions with 2–8 kept blocks, T9 has 59 with 3–10.
It fixes: the role seal; the packet rules (whole-block 2-colouring with slot/octant constraints, center-configuration balance preferred); the ridge λ rule; J_min (0.99 quantile of the shifted-label null, D_min = J_min); raw margin = −0.15 × T11 near-gap mean of F−B; emulator E1/E2 (θ 0 and 0.5); recording margins 0.10/0.05; recovery as long-gap ratio of weighted sums; literal past-only history; complexity df ≤ n/2 and operator bounds [1/3, 3]; bootstrap with trials resampled inside whole-block strata; positive-control injections +0.20 and −0.30 with their pass rules; T11 search-stability rules.
The four T11-derived values stay null until computed. Disclosed interpretation: whole blocks are kept as strata and not resampled as units (reason in the spec).
Next: role seal, reader and packet code; synthetic checks; T11 feature extraction.

## 2026-10-16 — case_series_design — T9 exposure disposition final

The scientist confirmed in this session that they have no detailed prior knowledge of T9's cross-session decoding or tuning-stability results. With no verified participant-specific cross-session exposure, T9 is confirmed as the single evaluation case; its yield and within-session exposure stays disclosed ([review](../DATASETS.md#case-series-roles-and-t9-exposure-review)).
Protections continue until the evaluation finishes:
- no EP16 worker opens the article's results figures, the provider `data/` files or notebooks, or results from T9 publications;
- only the sealed step reads T9 neural and cursor fields.

Next: the role seal and the remaining analysis-definition work. Neither needs another scientist decision.

## 2026-10-16 — case_series_design — T11-to-T9 case series adopted

The scientist adopted the T11-to-T9 case series in this session and asked for a redesign record, a T9 exposure review and the list of remaining analysis-definition work. This replaces the three-held-out claim, whose Oct 6 disposition is retained; it was chosen after outcome-free support results only. The record is in [GOAL](../GOAL.md#adopted-case-series-amendment--october-16-2026), [DATASETS](../DATASETS.md#case-series-roles-and-t9-exposure-review), the policy block `case_series_amendment_2026_10_16`, `config/analysis_definition.json` and the paper plan.
- Feasibility check (allocation 48167579, 186 CPU-s): with each person's capable sessions median-split, both T9 halves are rank-deficient (69/144 and 69/89 pairs); T11's early half is rank-deficient and its late half fails retained energy (0.075). Within-person replication is therefore unavailable.
- Attempt 01 failed on a subset KeyError and is preserved; the fix guards missing participants and changes no code path when all participants are present.
- The T9 exposure review is drafted: no verified cross-session exposure. Next: the scientist confirms the exposure review and the proposed elements, then work proceeds to the analysis definition.

## 2026-10-16 — structural_support — support re-run under adopted rules (48155786)

`20261016-source_fact_support01` on sh02-06n01 (85 s, 260 MB RSS) kept all 729 sessions; P0 found 334 single-schedule sessions and excluded 314 mixed ones. Figure: [ep16_source_fact_support.png](figures/ep16_source_fact_support.png).
- P0: T9 is held-out eligible (59 capable radial-8 sessions, 153/909 pairs, condition 14.9, retained 0.192, node share 0.065). T11 is supported (64 capable, 129/1346 pairs).
- P0 failures: T2, T6 and T10 are rank-deficient; T5 fails on node share (0.31); T7 and T8 fail the information gates (retained 0.030/0.040).
- S1–S10: only T9 is ever eligible; T5 passes only under S7; T11 fails under S2. The ≥4 development + 3 held-out design is unsupported in every variant.

The source-fact gap is closed and the disposition stands. Next step: the scientist decides on the T11-to-T9 case series. Compute today: attended allocation 48155786 (russpold, 2 CPU / 8 GiB) ran 13:41–14:45 PDT and was released, 7,726 CPU-s = 2.15 CPU-hours for EP16. ASTRA schema/semantic validation of the edited program is pending; I found no validator in an EP16-owned environment.

## 2026-10-16 — structural_support — classification-only check (48155786)

Ran `20261016-source_fact_classification01` (8.3 s, 172 MB RSS); it computed no pair or gate values.
- Precision: 0 pairs between 1e-12 and 1e-4 RMS, so the merge rule is threshold-insensitive; 29 sessions merged one or more roundoff duplicates, including T10 days 67, 112 and 130.
- Transitions: the non-permitted fraction is 0 in 3,151 radial blocks, ≤5% in 21 and 5–20% in 11 (all T6); none exceeds 20%.
- T11 days 367, 576, 632, 905 and 1017 have only center↔outer edges, plus one repeat edge on day 905.
- The finite-sampling check flagged 23 blocks. After seeing these counts I added the check's consequence: flagged blocks are distinct layouts. This affects only five capable T11 sessions. The unchecked alternative is S10 (post hoc).

## 2026-10-16 — structural_support — rules fixed before application (allocation 48155786)

Adopted [source_fact_rules.json](config/source_fact_rules.json) at 21:05 UTC (file write time), before any pair, rank or gate value under it. I had seen only layout metadata and the Oct 3 report.
Primary P0 combines nine rules:
- precision merge ≤1e-9 RMS;
- intended-target slots within 1% of ring radius or lattice spacing;
- layout-only families: radial-8 (oval allowed), radial-4 and n×n grid (n=5–10) are size-independent; Fitts/random and unclassified layouts are excluded;
- placeholder go periods removed;
- center↔outer edges with ≤5% exceptions;
- one template per session, so mixed sessions are excluded;
- missing slots treated as finite sampling;
- the packet-coverage necessary condition;
- the unchanged adopted geometry and component gates.

Sensitivities S1–S9 and their data checks are prespecified in that file. Code: `code/apply_source_fact_support.py`; six focused synthetic checks passed on sh02-06n01 in 0.043 s. Next: a classification-only run, then the support run.

## 2026-10-16 — structural_support — source facts recovered; reopening authorized

The scientist's 2026-10-16 standing authority says missing source facts are not stopping points. The scientist explicitly asked, through the coordinating session, that EP16 be handled this way. This reopens the support assessment under stated assumptions; roles, estimands and held-out access are unchanged.

Sources read:
- the release README and SOURCE.md;
- the preprint supplement methods (PMC12236888);
- pinned array-paper code (no `data/`, no notebooks);
- methods sections of PMC3715131, 3775656, 4805425, 5823702, 6685768, 5319839, 5371026, 5239755, 4765319, 8218873 and 12089531.

Item classifications are in [schedule_definition.md](schedule_definition.md#october-16-source-facts-and-adopted-assumptions). No provider was contacted.

Exposure disclosure (types only, no values):
- A search summary and a PMC page summary returned abstract-level results: pooled yield; dSNR > 1 in 11 of 14 arrays; pooled tuning stable over about one month; early yield decline for S2, S3 and T2.
- Supplement 2.6 and code config record that T2, T3 and T11-lateral were excluded from tuning stability for inconsistent dSNR < 1. This is within-session exposure, including for candidate T2.
- No participant-specific EP16 cross-session result was seen.

## 2026-10-06 — structural_support — scientist-adopted support-limited closeout

The scientist replied to the outstanding-judgment recommendations:
"i agree with your judgement; EP20 is fine with simulation because we dont have real data".
For EP16, this adopts the original three-held-out-participant claim's
registered disposition **support_or_uncertainty_prevents_an_answer**.
This is a support limitation, not a neural result, biological null or
successful three-person adjudication. EP20's separate simulation permission
does not select EP16 synthetic work.

Retain the completed 729-session structural inventory and all prior analysis
preparation. Only T11 and T9 pass the current conditional numerical design,
and neither is fully qualified; this does not supply four supported
development and three supported held-out participants in the unchanged 6/3
roles. No neural comparison or held-out neural result has been opened.
Current failure is not a proof that authentic new metadata can never change
support. Reopening requires concrete authentic metadata or a prospectively
adopted redesign that changes the support basis, followed by the applicable
fixed support and exposure/role qualification before neural scores.

No unchanged support rerun, role replacement, single-case study, simulation
or neural successor is selected. The optional T11/T9 case-series draft and
its missing analysis/access/resource choices remain unadopted. Updated only
the local contract, plan, ASTRA description and navigation; no payload, test,
job or scheduler action occurred, and no canonical BR state changed.
Historical proposal wording below predates this acceptance.

## 2026-10-06 — structural and analysis preparation reconciled into data readiness

Updated GOAL/DATASETS from the existing all-session structural, conditional
design and analysis-definition records. They now distinguish the completed
729-session inspection and three passing arithmetic checks from session
annotation/exposure qualification, inadequate current participant support and
unfinished neural analysis. The schedule handoff's stale “not yet executed”
check wording now links the existing passing run. No role, threshold, source,
policy, access rule, proposal adoption or job changed.

## 2026-10-06 Pacific — public provider field definitions added

Added the verified Dryad source-field documentation and pinned provider
metadata code to the [existing schedule handoff](schedule_definition.md#october-6-public-source-update--documented-fields-and-unresolved-session-annotations).
Public definitions cover binning, units, task-coordinate positions and index
bounds; they do not supply a newly authenticated per-session task-size,
controller-version or transition-rule annotation. The bounded review's lack
of such a finding is not proof that original provider records do not exist.
The current conditional support results and exposure limitations remain
unchanged; no role change or single-case redesign was adopted.

Only user-requested documentation edits and bounded readback occurred on
sh04-ln04 with no allocation. No neural/cursor/impedance values, held-out
outcome, test, import, scheduler call, provider contact or new job was involved.

## 2026-10-06 — critique revision: support interpretation and unadopted closure

The existing paper plan now distinguishes the six/three role allocation from
the four-supported-development/three-supported-held-out requirement. The
conditional T11/T9 passes are not full eligibility. Rejected an unsupported
monotonicity claim: authentic session/edge exclusions can change conditioning,
information, leverage and connected components, so current failure is not
universal source infeasibility. Proposed the registered support-limited
disposition with explicit reopening conditions and an optional T11-development,
T9-evaluation case-series draft. Both recommendations remain unadopted.
No source, neural outcome, scheduler state or test was read/run; no policy,
role, margin, resource allowance or access boundary changed.

## 2026-10-06 05:29 UTC — analysis_definition: focused checks pass

All three new score/window/source-scaling checks passed in 0.047 seconds in
root-owned allocation **46721202** on **sh04-18n32**, one CPU / 2 GiB / no GPU,
with all six numerical thread caps one. Evidence:
`runs/analysis_definition_checks001/tests.txt`. Changed ASTRA schema and
semantic validation also passed:
`astra/v0.0.14/validation_20261006_46721202.json`.

The allocation subsequently completed at 05:33:09 UTC, **COMPLETED 0:0**;
its **373 allocated CPU-seconds = 0.1036111111 CPU-hours** are charged once
to EP06, with no duplicate EP16 charge. The compute shell has exited and no
reservation remains.
These small checks verify arithmetic and fixed-axis missingness behavior;
they are neither neural findings nor completion of the scientific calibration
requirements. Source size/layout facts, current cohort support, exposure
history, remaining margins/model/inference choices and reader provisioning
remain unresolved. No source-support rerun or neural successor is selected.

## 2026-10-06 05:26 UTC — analysis_definition: aggregate checks selected

Independent arithmetic and scientific-contract static reviews passed the
declared score/window/scaling implementation. Root selects its three focused
checks and changed ASTRA validation within one attended 1 CPU / 2 GiB /
maximum 30-minute allocation, aggregated with EP06's substantive fixed
source-metadata inventory. The parent is charged once to EP06, without a
duplicate EP16 charge or scientific trial. Only EP16 code/configuration and
small algebra fixtures are inputs; no source payload, role opening, cohort
pruning or repeated source-support run. Six numerical thread caps are one.
Root owns scheduling and the actual compute shell. The submitted allocation
is **46721202**, initially queued; runtime pending.

2026-10-02 — Concept figure: generated/edited and visually inspected [ep16_question_imagegen-v2.png](ep16_question_imagegen-v2.png) with built-in image generation; exact prompts in [ep16_question_imagegen-v2-prompt.md](ep16_question_imagegen-v2-prompt.md). Replaced intention/representational-cause wording with the direction proxy, target-local versus transport comparison, label-blind emulator and 32-label repair; no score bars. Earlier assets retained; no analysis, outcome access or scientific-rule change.

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md): Compared the proposed contribution with NoMAD and PRI-T; added a concrete prior-work table and changed the coexisting-result branch from causal mechanisms to operational signatures. No data were opened or scored and no experimental choice was changed.

## Current entry

- Stage: scientific design and readability revision
- Question: does a mapping trained in one BrainGate session suffer an extra
  loss after at least 180 days, and if so, what explains it?
- Work completed: the question, three diagnostic branches, controls, budgets,
  and paper plan have been stated in reader-facing form.
- Data analysis: not started
- Held-out participant results examined: none
- Scientific result: none

## Next entry should record

The next entry should describe the target-schedule and near/long pair-support
check. It should state which participants meet the structural requirements and
why, without reporting neural scores.

Future entries should be chronological and distinguish a completed scientific
comparison from a software repair or duplicate run.

## 2026-09-30 — component-design refinement

- Work: added `component_details.md` and linked it from GOAL, DATASETS and
  `paper_plan.md`; two parallel read-only reviews informed the root-authored
  design. No earlier entry was replaced.
- Components: matched target-data exposure and local learning curves;
  recording-summary-fitted emulation with transformed-source evaluation and
  matched-random degradation; explicit signed 32-label recovery quantities.
- Scientific choices still open: time exposure for literal past-only methods;
  raw-score units/octant support; channel-quality ordering and gain/noise
  assumptions; reproduction errors and recovery aggregation; existing numeric
  thresholds. SEARCH_POLICY values were not changed.
- ASTRA: the required path was absent. Added
  `astra/v0.0.14/astra.yaml` as a design-only worklist for the existing contract.
  Transform fitting (`recording_transform`) has no benchmark-score input;
  reproduction is evaluated separately (`recording_emulation`).
- Validation: the current interpreter has no YAML, ASTRA or Brain Researcher
  package, and no ASTRA validator command was found. Pinned 0.0.14 schema and
  semantic validation remain open; no package was installed for this revision.
- Execution: none; no Slurm/process ID. No neural payload or held-out outcome
  was read. No Git mutation was performed.
- Next design action, `analysis_definition`: resolve the specific choices in
  `component_details.md` before treating this worklist as executable.
  The existing later structural action, `structural_support`, remains the
  outcome-blind schedule/pair-support assessment when separately authorized.

## 2026-10-03 — structural_support: source and selective-reader preparation

The scientist authorized resuming feasible episode work and completing jobs;
the coordinator separately assigned EP16 source/role preparation. The next
substantive stage remains the outcome-blind target-schedule and near/long
support assessment. No participant roles, model choices or scientific margins
have been changed.

Read the exact public-release `SOURCE.md`, provider README and existing
acquisition metadata under
`$OAK/data/br_autoresearch_data/braingate_long_term_array_performance/dryad-x0k6djj1h-v6`.
The source statement identifies CC0 public deidentified clinical-trial data
with no personally identifiable information. Existing verification covers
all 24 source files / 84,729,840,092 bytes and all 23 archives. Those passing
acquisition checks are reused, not repeated. Data roles were not assigned by
ingestion. The provider README identifies exact block/trial index semantics
(zero-based starts, exclusive ends), array/electrode fields and target arrays.

The structural reader must avoid neural values, cursor trajectories, success,
yield, impedance and model scores. `electrodes` contains impedance alongside
identity fields, so loading the complete struct is not an acceptable shortcut.
Only one target coordinate per trial belongs in the structural view. All
unassigned participants remain closed beyond those authorized structural fields.

Selected initial attempt:
`outputs/runs/structural_support/20261003-format01`, using
`outputs/code/inspect_structural_format.py`. It reads only the first 128-byte
MAT header of one session in **T10**, which is already assigned to development
by the contract, to choose a field-selective extractor. It does not deserialize
any variable or extract a complete archive. Request one CPU, up to 256 MiB,
at most five minutes inside the coordinator's substantive allocation.

Observed on job **46425841**, `sh03-06n50.int`, at 19:00:42 UTC:
[format report](runs/structural_support/20261003-format01/structural_format_report.json)
identifies MATLAB Level 5 in
`decoding/T10/T10_day_104_decoding.mat` (35,429,020 uncompressed bytes).
Exactly 128 header bytes were read and no variables deserialized. The archive
mode was `0660`; this is not a permission seal.

The data-location registry reports that the shared source remains writable;
integrity verification is distinct from a permission seal. No source permission
was changed. The seal and separate role views remain prerequisites for a
protected final evaluation, while this explicitly allowed structural preparation
can proceed. The six/three assignment, actual schedule/pair support and existing
pre-scoring scientific choices remain unresolved.

Selected next attempt: `outputs/runs/structural_support/20261003-fields01`,
the first same T10 session, one CPU, 2 GiB, five minutes within the coordinator's
allocation. The [field-selective reader](code/mat5_structural.py) follows the
[MathWorks Level-5 format specification](https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf).
Before source execution, run the focused `test_mat5_structural.py` boundary
tests on compute: compressed/uncompressed fields, skipped complex forbidden
fields, selected-field rejection, target shape, and both byte-order tags.
Then execute `extract_structural_metadata.py --participants T10 --max-sessions 1`.
Opaque MAT bytes are temporarily spooled one session at a time into an unlinked
private scratch file; no raw MAT or archive copy is retained in the episode.
Two passes decode only participant/day/bin size, array/electrode identity and
grid coordinates, block/trial indices and one target point at each trial start.
Neural, cursor and impedance values remain undecoded. The retained JSONL is the
permitted structural view. Whole-block 64/64 capacity is a necessary support
condition, not final packet eligibility; no roles or schedule tolerances are
chosen by this reader.

Observed on coordinator job **46425841**: all four focused boundary tests
passed in 0.010 s, and
[the representative extraction](runs/structural_support/20261003-fields01/structural_metadata_report.json)
passed in 0.168096 s at 19:08:35 UTC. T10 day 104 has four blocks and 405
declared whole trials; its permitted view contains exactly 405 target points
and nine distinct target coordinates. Whole-block 64/64 capacity is available.
Decoded paths exactly match the whitelist; neural and cursor variables were
skipped. No impedance field was present in this session's electrode struct;
the focused fixture separately verifies that a present impedance field is
skipped. No neural/cursor/impedance values were decoded. This is structural
source evidence, not a neural comparison or participant-role decision.
[Official ASTRA validation](astra/v0.0.14/validation_20261003_fields.json)
passed schema and semantics for the representative-stage program.

Selected next attempt: `outputs/runs/structural_support/20261003-fields_all01`,
one serial pass over the nine exact decoding archives for T2, T3, T5, T6, T7,
T8, T9, T10 and T11, omitting `--max-sessions`. Source verification records
79,428,705,009 bytes across those archives; its historical archive-listing
times total 673.819 s. Together with the representative reader measurement,
this supports a 25-minute bound on one CPU with 2 GiB RAM within the current
allocation. Only one opaque session member exists temporarily in private
scratch at a time. The 14 yield archives remain unopened. No passing field
tests will be repeated. The permitted metadata view is extracted for every
available session; role assignment, schedule tolerance, exclusion decisions
and outcome exposure review remain separate and unfinished. Exact command is
the `structural_fields` recipe, with a 1,500-second external execution bound.
If it cannot finish within that bound, preserve partial metadata and diagnose
before continuation; do not substitute a reduced scientific cohort.

Prepared the same stage's next metadata-only description at
`outputs/runs/structural_support/20261003-support01`, conditional on complete
all-nine extraction. `describe_structural_support.py` consumes only the
retained structural JSONL, describes exact raw target layouts and within-block
transition counts, and counts near/long pairs among sessions with the same
participant/array set and whole-block capacity. These pair counts deliberately
ignore schedule restrictions and therefore are upper bounds, not qualified
support. An upper bound below six can prove insufficient support; a count above
six cannot establish eligibility or fixed-effect identifiability. No coordinate
tolerance, task-size assumption, equivalence class, scientific threshold or
participant role is selected. Request one CPU, 2 GiB and up to two minutes in
the same substantive allocation; no additional source payload is opened.

Observed **46425841** completion at 19:17:41 UTC:
[all-nine extraction](runs/structural_support/20261003-fields_all01/structural_metadata_report.json)
retained allowed structural metadata for all **729 sessions**, including all
196 T11 sessions, in **362.486045 s**. It traversed 79,607,082,343 opaque MAT
member bytes and retained **328,548 target points**, exactly one per declared
trial. No neural, cursor or impedance values were decoded; source archives
were unchanged. The nine compressed archive bytes and opaque uncompressed
member bytes are different quantities. The current ASTRA program passed
[official schema and semantic validation](astra/v0.0.14/validation_20261003_all_fields.json).

The coordinator then completed the
[structural support description](runs/structural_support/20261003-support01/structural_support_description.json)
in the same allocation. **T3 cannot be held out under the existing support
rule**: only six of eight sessions have whole-block 64/64 capacity, yielding
at most **four near and five long pairs** before schedule restrictions. Both
fall below the required six. The other eight participants exceed these loose
upper bounds; none is yet qualified by schedule or fixed-effect support.

This is not whole-episode infeasibility. `SEARCH_POLICY.yaml`,
`development_requirements.development_participants_with_sufficient_data_minimum`,
requires **four** supported development participants among the six assigned
development roles. The authorized branch can retain T3 in development with
its support failure recorded, retain initial development T5/T10/T11, and choose
three held-out participants from T2/T6/T7/T8/T9 if the frozen full structural
and exposure rules permit. All nine identities and the six/three allocation
are retained; no unsupported eight-person amendment or participant replacement
is made. Roles have not yet been assigned. Next substantive work is to fix the
outcome-blind schedule tolerance/eligibility implementation, qualify actual
matched schedule graphs and rank, complete the exposure review, and apply the
existing ordered participant-selection rule. Neural methods and thresholds,
required synthetic checks and protected role views still precede scoring.

## 2026-10-03 — structural_support: coordinate matching and conditional rank

Selected the next metadata-only step under the continuing execution authority:
`outputs/code/describe_schedule_design.py`, consuming only
`outputs/runs/structural_support/20261003-fields_all01/structural_sessions.jsonl`.
The prospective [schedule definition](schedule_definition.md) fixes allowed
translation/positive-isotropic-scale normalization and an explicit float64
roundoff formula, and defines weighted source/target design diagnostics. It
does not infer physical coordinate units or task-size independence. The source
README lists radial, grid and Fitts variations but supplies no reliable task
identity/target-size field. Those semantics and observed-transition compatibility
remain unresolved; candidate geometry graphs are not qualified schedule groups.
The source archives and all existing code/results are preserved.

Preselected command for coordinator-owned compute, one CPU / 2 GiB / at most
five minutes, with BLAS/OpenMP threads set to one:

```sh
python3 outputs/code/describe_schedule_design.py \
  --structural-jsonl outputs/runs/structural_support/20261003-fields_all01/structural_sessions.jsonl \
  --output-dir outputs/runs/structural_support/20261003-schedule_design01
```

This performs arithmetic on permitted structural metadata only. Output is
`schedule_design_report.json`: geometry groups, all candidate chronological
near/long edges summarized by connected-component design rank, condition,
residual long-gap information and session influence, plus block coverage and
transition counts. No score, neural/cursor/impedance read, trial selection,
participant assignment, scientific threshold application or eligibility claim
is authorized by this step. The recipe is ready but has not yet been run.

Bounded public code-documentation follow-up identified associated source commit
`93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834`. Its `array_paper/metadata.py` supplies
participant/array labels, not the needed session/block task-size classification;
the inspected `metrics/decoding_snr.py` target/trial API also supplies no such
identity. Exact links and inspection limits are in `schedule_definition.md`.
No public result CSV, notebook output, neural/cursor value or new participant
outcome was opened. The required task-class evidence remains unestablished,
not verified absent from every provider source.

Observed coordinator job **46425841** at **19:34:45 UTC**:
`20261003-schedule_design01/schedule_design_report.json` completed in
**2.081683 s**, describing all **729 sessions / 355 geometry groups**. No roles,
trials, neural/cursor/impedance values or scientific eligibility were opened or
selected. Official `validation_20261003_schedule_design.json` passed ASTRA
schema and semantics. This attempt is preserved with NumPy centroid/RMS sums;
the tolerance was a roundoff-scale descriptor, not a formal error bound or a
provider-resolution allowance.

Seven conditional components had at least six pairs in each gap class:
T11 (102 near / 845 long, rank103/103), T2 (21/30,22/23),
T5 (27/54,30/30), T7 (13/10,15/19 and 26/21,28/28),
T8 (87/239,58/58), T9 (153/909,115/115). The T2 and five-coordinate T7
components had no identifiable residual long-gap effect. These are conditional
geometry results, not a participant-selection rule. This historical summary
used a six/six component filter that the contract does not require; see the
aggregation clarification below. T6/T10 had no component with both six-pair
minima under this strict geometry description, but that is not itself a
development-support gate. Source precision,
size dependence and transition eligibility are still unqualified. No failed
component is removed to obtain a preferred support pattern.

Selected precision-only successor, preserving attempt01: use `math.fsum` for
centroid and RMS squared-radius accumulation, retain the same descriptive
tolerance formula and all comparisons, and explicitly label the formula's
limited meaning in the report. No generic test expansion is needed. Run the
same command/resources with
`--output-dir outputs/runs/structural_support/20261003-schedule_design02`.
This second attempt is selected but not yet executed; it remains unqualified
and cannot assign participants or relax a scientific rule.

Observed precision-only successor on **46425841** at **19:37:50 UTC**:
[attempt02 report](runs/structural_support/20261003-schedule_design02/schedule_design_report.json)
completed in **1.689202 s**, again **729 sessions / 355 geometry groups**.
`centroid_rms_accumulation=math.fsum`; the report explicitly identifies its
tolerance as a descriptor, not a formal error bound or source-quantization
allowance. The seven components meeting both pair-count minima and their rank
outcomes are unchanged from attempt01. Official
[ASTRA validation](astra/v0.0.14/validation_20261003_schedule_design_attempt002.json)
passed schema and semantics. All roles/trial-selection/qualification/value-access
flags remain false. The coordinator explicitly closed the attended allocation
at19:38 UTC; no further compute is scheduled by this worker.

The five people with at least one conditional full-rank component are **not
five fully qualified participants**, and this is not terminal source
infeasibility. Their size/transition/stability rules and scientific numerical
support gates remain open; conversely the descriptor is intentionally stricter
than any future source-documented quantization allowance. No tolerance is fit
to obtain a desired number of participants.

Exact source target dtype remains **unestablished** by retained evidence.
The initial report read only the128-byte MATLAB Level5 container header.
`mat5_structural.py` reads each selected target with its actual MAT primitive
dtype internally, but writes it into `np.empty(..., dtype=float)` and its trace
does not retain the primitive tag/dtype. JSON numbers therefore cannot certify
original single/double precision. Provider code at pinned commit
`93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834`,
`array_paper/batch_summary/generate_decoding_summary.py:103–104`, similarly
casts positions to `dtype=float` after `loadmat`; it is analysis code, not an
export precision guarantee. The inspected public code-tree paths contained no
named exporter/preprocessor or task dictionary. No raw MAT values were reopened.

Next substantive source step, in a separately coordinated allocation if needed:
record only original `target_position` MATLAB array class, numeric primitive
tag, dimensions and dtype from selected source headers, then obtain provider
export/task-controller documentation for coordinate generation/quantization
and session/block task-size classification. A storage dtype alone would still
not prove the original task resolution or size independence. Retain all roles
and scoring closed; no source processing command has been selected for this
future allocation.

## 2026-10-03 — structural_support: target-storage header reader prepared

Prepared `outputs/code/inspect_target_storage.py` for the fixed nine decoding
archives and all **729 expected sessions**. This is a new, separate header-only
entrypoint; existing readers/results are unchanged. It records only each
`target_position` variable's MATLAB array class, primitive tag, shape, byte
order/dtype and declared body length. No target, neural, cursor or impedance
numeric value is decoded. A compressed matrix is inflated only far enough to
read its metadata/tag; its remaining compressed bytes are skipped opaquely.
Only one unlinked temporary source member exists at a time and is removed on
normal completion/failure. No source permission or file is changed.

Two focused synthetic boundary checks are prepared in
`outputs/code/test_target_storage.py`: both float precisions/byte orders with
guarded numeric bodies, and compressed metadata inflation stopping before
numeric bodies. These verify the new reader's actual access boundary; the
previous reader suite is not repeated. Neither new check nor reader has run.

Selected future compute bundle, subject to the coordinator providing a new
attended allocation: **1 CPU, 2 GiB RAM, 10 minutes**, one worker/thread. The
previous all-nine extraction traversed79,607,082,343 opaque MAT member bytes in
362.486s; this equivalent serial archive pass reads fewer decoded fields, so
the ten-minute estimate includes reasonable I/O margin. Temporary disk needs
one largest MAT member, not the full archive set; the reader reports the actual
peak member size. No new archive copy or final role view is retained. A timeout
or failed header/count check preserves the partial JSONL for diagnosis.

```sh
python3 -m unittest discover -s outputs/code -p test_target_storage.py
python3 outputs/code/inspect_target_storage.py \
  --source-root "$OAK/data/br_autoresearch_data/braingate_long_term_array_performance/dryad-x0k6djj1h-v6" \
  --scratch-dir "$SCRATCH/br_autoresearch/episode16_braingate_cross_session_failure_modes/target-storage" \
  --output-dir outputs/runs/structural_support/20261003-target_storage01
```

Run the second command only if the first passes. The allocation is currently
closed; **no execution or new job is implied by this selection**. A successful
storage census would establish serialization metadata only. It cannot identify
task-controller resolution, task-size dependence, legitimate physical jitter,
or participant eligibility, and cannot relax tolerance to achieve desired
support. All roles/scoring remain closed.

Coordinator execution update: a new attended allocation **46427768** on
`sh02-01n18.int` provides2CPUs/4GiB/30minutes for the combined EP16/EP18 bundle.
EP16 uses one worker/thread. Both focused header-access tests passed, and the
all-nine census is running at the selected attempt01 path. Its transient root
is `$L_SCRATCH_JOB/ep16-target-storage`, overriding the proposed shared-Scratch
example above. Root verified158GiB node-local free space, more than the entire
79.61GB opaque-member total, so an additional source scan to learn the largest
member was unnecessary. The historical archive verification lists success/time,
not member sizes; the exact maximum remains unknown until the census records it.
An independent read-only review confirmed metadata-only inflation and opaque
numeric-body skipping; no source or test edits were needed. MATLAB small-data
tags can physically embed up to four body bytes, so the general guarantee is
zero numeric values decoded, rather than a claim that no numeric byte could
ever reside in a tag. Final census output/validation are pending.

Observed completion on **46427768 / sh02-01n18.int**,
**2026-10-03 19:59:29 UTC**:
[target storage report](runs/structural_support/20261003-target_storage01/target_storage_report.json)
completed successfully in **404.087567 s**. All nine exact participant counts
match, totaling **729 sessions**. Every target matrix is **mxDOUBLE_CLASS /
little-endian float64 (`<f8`)**. The metadata-only per-session records are
`target_storage_headers.jsonl`. There were zero errors, zero decoded numeric
values, no modified source archives and no retained raw member copies.

The single-worker pass copied **79,607,082,343 opaque MAT bytes** through
unlinked node-local temporary files. The actual largest single temporary MAT
was **577,212,492 bytes**; temporary files were under
`$L_SCRATCH_JOB/ep16-target-storage`, with158GiB free verified by the coordinator
before execution. The overall attended allocation is2CPUs/4GiB/30minutes;
EP16's selected limit was1CPU/2GiB/10minutes and its observed walltime was6m44s.
These are allocation/bound and file-size facts, not a measured peak-RSS claim.
Durable metadata remains in the episode's Oak attempt directory. The coordinator
completed [official ASTRA schema and semantic validation](astra/v0.0.14/validation_20261003_target_storage.json)
successfully in job 46427768. No extra source processing is selected.

Storage dtype is now resolved; **task-controller precision is not**. A double
can contain values generated earlier at lower resolution, rounded by an export
step, or changed by genuine task-layout variation. The result neither supplies
a source-quantization allowance nor authorizes changing coordinate tolerance.
It also supplies no task-size classification. The five conditional full-rank
people remain neither a qualified cohort nor proof of infeasibility.

The concrete same-contract route is recorded in
[schedule_definition.md](schedule_definition.md#remaining-source-and-support-route):
obtain an outcome-free participant/day/block crosswalk to documented task and
layout/size semantics plus coordinate-generation/export precision; qualify
the prospective geometry/coverage/transition rule from that evidence; then
rebuild all chronological matched graphs, apply the prospectively specified
rank/information/leverage gates and the existing exposure/role-selection rule.
T3 remains ineligible for held-out assignment under its unconditional pair
upper bounds, while six development roles with at least four supported people
remain the requirement. No roles, packets, neural scores or revised claims are
selected by the current census.

### 2026-10-03 — Clarified schedule-group counts versus component identifiability

Metadata-only contract review found that the earlier six/six per-component
summary was stricter than the frozen gate. `SEARCH_POLICY.yaml` lines 28–30,
`GOAL.md` lines 183–185 and `DATASETS.md` lines 108–113 require six near and six
long pairs **per held-out participant in a common schedule group**. The policy
lines 97–113 and datasets component rule separately require each contributing
connected component to contain both lag classes and pass its own constrained
rank and prospectively fixed numerical gates. Counts can accumulate across
qualified components of the same schedule group. Contrasts remain separately
fitted within components and aggregated with equal schedule-group weight and
equal component weight within each group; pooling unrelated groups or pooling
coefficients to repair failed component identifiability is not authorized.
The held-out count threshold is not silently added to development eligibility.

The existing descriptor code did not implement the overconservative summary
filter and needs no repair for this issue. Bounded reading of all emitted
attempt02 components finds six both-lag full-rank components in five people:
T11 `geometry_0034` (102 near / 845 long, rank 103/103) and `geometry_0037`
(2 near / 39 long, rank 17/17), T5 `geometry_0149` (27/54, rank 30/30), T7
`geometry_0273` (26/21, rank 28/28), T8 `geometry_0281` (87/239, rank 58/58),
and T9 `geometry_0309` (153/909, rank 115/115). The additional T11 component
does not change the descriptive person count. Only two geometry groups have
multiple components: T11 `geometry_0032` has near-only components with 3 and 5
pairs; T2 `geometry_0104` has a two-pair long-only component and a six-pair
near-only component. Combining their counts cannot make either component's
own adjusted long-minus-near contrast identifiable.

This corrects the interpretation without changing code, coordinate tolerance,
scientific thresholds, role assignment or scoring access. The descriptor's
geometry groups still lack source-supported task/size/transition qualification;
its roundoff tolerance is not a controller-precision allowance. Five
conditionally full-rank people therefore remain neither a qualified cohort
nor a terminal seven-person-support failure. The outcome-free source crosswalk
and full qualified rule described in `schedule_definition.md` remain the next
same-contract step.

### 2026-10-03 — structural_support observed-schedule diagnostic selected

The coordinator selected a bounded diagnostic on the **existing structural
JSONL only**, implemented by `outputs/code/describe_observed_schedules.py`.
It records every declared trial exactly once with original row identity and
start/end indices, including declared blocks with no trials. It emits aligned
block target sets, target recurrence and actual within-block directed
transition identities/counts. It describes within-session and aligned-session
support differences while keeping absent sampled edges distinct from a proven
forbidden-transition conflict. Only one target point per trial is available,
so within-trial stability is explicitly not observable. Size dependence stays
unresolved, not inferred from shape, recurrence or a generic missing field.

Alignment reuses the previously recorded roundoff descriptor only. The newly
proposed scientific tolerance and conditioning/information/leverage limits
are **not activated**, and the script makes no support/role/trial-selection
decision. Exact target distinctions remain; normalized minimum/nearest-neighbor
spacing summaries make near-duplicate behavior reviewable without merging or
rounding points. No source archives or neural/cursor/impedance values are read.

Selected command for the coordinator's attended allocation **46427768** if
sufficient time remains: **1 CPU, 2 GiB, at most 5 minutes**, all numerical
threads set to one. The earlier all-session geometry pass took 1.69 seconds;
this retains extra sequences and writes several metadata JSONL files, so the
five-minute bound allows ample metadata I/O margin. Expected durable output
is tens of MB, with no raw scratch/source copy required.

```sh
python3 outputs/code/describe_observed_schedules.py \
  --structural-jsonl outputs/runs/structural_support/20261003-fields_all01/structural_sessions.jsonl \
  --output-dir outputs/runs/structural_support/20261003-observed_schedule01
```

The output directory is exclusive-create; failures preserve partial metadata
and require diagnosis before a new attempt. Durable files are
`observed_schedule_report.json`, `observed_sessions.jsonl`,
`observed_blocks.jsonl` and `aligned_session_pairs.jsonl`. This is selected,
not yet executed by this worker. The changed ASTRA object should be validated
by the coordinator after the step.

The prior documentary requirement for a complete task/controller/source-
precision crosswalk was overbroad. `DATASETS.md` explicitly uses observed
schedules because task identity and sizes are incomplete. Full names/controller
versions are useful supplementary evidence, not universal gates. Prospective
geometric/support numerical choices remain delegated before scores. A narrow
session/block annotation identifying schedules whose definition requires
missing target size may still be needed; radial-looking coordinates cannot
prove size independence. No threshold, role, contract or protected access
changed.

Observed completion in **46427768 / sh02-01n18.int**, at
**2026-10-03 20:11:32 UTC**, took **7.835988 seconds**:
[compact report](runs/structural_support/20261003-observed_schedule01/observed_schedule_report.json).
All **729 sessions, 328,548 declared trials and 5,150 blocks** were retained.
There are **6,957 roundoff-aligned session pairs**; **785** have different
observed transition supports and **1,795** have different observed block-target
set families. These pair counts include all aligned day gaps, not just eligible
near/long edges. They are sampled-support differences, not proven forbidden-
transition conflicts, unstable layouts or participant eligibility failures.
All activation/selection/access flags remain false. The updated
[official ASTRA validation](astra/v0.0.14/validation_20261003_observed_schedule.json)
passed schema and semantics. The coordinator closed the attended allocation
at **20:13:27 UTC**; no compute session remains for worker tests or science.

Illustrative structural records distinguish a stable observed point set from
insufficient recurrence without assigning task names. T10 day62 has190 declared
trials, nine exact target points, seven blocks all covering those points and
one observed transition hub. T10 day49 has675 trials,587 exact target points
and14 distinct block target sets; its minimum normalized spacing is0.001402.
Neither record establishes target-size dependence or within-trial stability.

The diagnostic also identifies actual near-duplicate coordinate ambiguity:
T10 day67 has831 exact points and minimum normalized spacing6.539e-17
(roundoff descriptor2.214e-14); day112 has70 points/minimum8.459e-17;
day130 has26 points/minimum8.867e-17. These are retained exact distinctions,
not proof of serialization error or physical jitter. The proposed raw minimum-
spacing cap would become smaller than ordinary normalized-coordinate arithmetic
resolution in such cases. They must be disclosed as unresolved target
distinctness, without merging coordinates, adding a tolerance floor or switching
spacing statistics to achieve more support.

The coordinator accepted the independently reviewed direction of prospective
limits, with the corrections recorded in
[the exact inactive proposal](config/schedule_support_proposal.json).
The angular argument is only for moving a target relative to a **fixed cursor
under the identical transform**; it bounds no actual cross-session label
differences. The node influence gate uses incident **residual long-indicator
squared information share**, rather than full-hat mass/rank: the latter can
conceal domination of this particular contrast by one session and remains
descriptive. Condition100, residual norm0.10 and retained information0.10 are
operational pre-score limits, not power or coverage guarantees. Thresholds have
not been applied, and frozen policy files are unchanged.

The proposal explicitly exposes one boundary interpretation for review:
equality is unresolved, following the policy's numerical-margin rule. With a
strict node-share cap below0.25 and shares summing to2, at least nine sessions
are necessary; an inclusive cap would permit eight only if all shares equal
0.25. No support is computed under either interpretation. Actual eligibility
still requires defensible size/transition/layout classification; metadata
unknowns are not automatically passed.

### 2026-10-03 — structural_support prospective numerical rules adopted before application

At **20:22:11 UTC**, following the coordinator's explicit instruction and
independent mathematical review, the geometric and numerical support values
were adopted under existing pre-score authority. The unchanged inactive
`schedule_support_proposal.json` is preserved. The active specification is
[schedule_support.json](config/schedule_support.json); only the corresponding
previously-null policy tolerance, condition, residual-information and node-
influence fields were filled, with precise metric references. No other margin,
participant rule, gap, model, scoring access or budget changed.

Passing requires geometry error below the declared symmetric spacing bound,
complete pairwise compatibility rather than connected-chain grouping, no
unresolved near-duplicate target distinction, full constrained rank, weighted
column-normalized condition **<100**, weighted residual long-indicator norm
**>0.10**, retained weighted long-indicator energy **>0.10**, and maximum
session incident residual-squared information share **<0.25**. Equality is
unresolved. The shares sum to2, so the last strict condition requires at least
nine session nodes; it does not guarantee power, stable bootstrap coverage or
scientific eligibility. Full-hat mass/rank remains descriptive. These values
were fixed before applying them and will not be swept against cohort support.

The next implementation applies these rules to the existing structural JSONL
only and reports designs **conditionally**, even while block/transition/size
qualification is pending. Nontransitive compatibility makes the entire
connected geometry component unresolved; no greedy clique selection is allowed.
Observed block-set or transition-support differences remain distinct from
proven conflicts. A known size-dependent schedule with missing size must be
excluded; unknown size dependence cannot auto-pass. No numerical application,
role assignment, neural decoding or new compute has occurred at this entry.

Prepared entrypoint: `outputs/code/apply_conditional_schedule_support.py`.
It reads only the existing structural JSONL and active configuration. Exact
unique target cardinality and all declared trials remain represented. It
evaluates all geometry pairs within participant/physical-array strata; all
connected compatibility components must be cliques. It evaluates fixed-effect
designs on all chronological near/long edges with whole-block capacity inside
each unambiguous geometry group, without discarding failed components. The
numerical output remains conditional and every final-eligibility flag stays
false while size/transition/exposure questions remain unresolved.

Six focused checks were prepared for the new behavior: allowed versus
forbidden transforms; preserved near-duplicate ambiguity; rejection of an
entire nontransitive matching component; a known-answer weighted residual and
contrast-specific node influence example; and unresolved geometric and support
threshold equality.
They have not been run. This is not another generic runner-check sequence.

Selected future bundle for a **new suitable substantive coordinator allocation**:
**1 CPU / 2 GiB / at most 5 minutes**, numerical threads1. Expected work is
seconds to under one minute: one metadata read, per-layout distance summaries,
within-participant geometry pairs and modest fixed-effect SVDs, comparable to
the earlier1.69-second design descriptor and7.84-second sequence diagnostic.
Do not submit a separate checking-only job or pad runtime, and do not enter
the running EP01 allocations. No current attended allocation is available.

```sh
python3 -m unittest discover -s outputs/code -p test_conditional_schedule_support.py
python3 outputs/code/apply_conditional_schedule_support.py \
  --structural-jsonl outputs/runs/structural_support/20261003-fields_all01/structural_sessions.jsonl \
  --config outputs/config/schedule_support.json \
  --output-dir outputs/runs/structural_support/20261003-conditional_support01
```

Run the second command only if focused checks pass. Output creation is exclusive;
preserve any failed attempt before diagnosis/retry. The coordinator should
validate the changed ASTRA object in the same allocation. This entry selects
the future commands, not a launch or scientific result.

Independent read-only review by `/root/inventory_ep01_07` found no concrete
launch-blocking defect in the active configuration, application source or six
focused checks. The review confirmed preserved raw target distinctions,
unambiguous strict-spacing bijections, whole-component nontransitivity handling,
target-equal weights, sum-zero Helmert effects, residualized long-information
shares, strict gates and conditional-only reporting. The reviewer executed
no tests and opened no structural or neural payload. Compute checks and
application remain pending the next coordinator-owned substantive allocation.

### 2026-10-03 — structural_support public-method exposure disclosure

During narrow size-dependence methods retrieval, the independent reviewer
`/root/inventory_ep08_14` called public `web.find(pattern='Fitts')` on
<https://pmc.ncbi.nlm.nih.gov/articles/PMC12236888/>. The returned excerpt was
broader than intended and included participant-specific summaries. The worker
stopped further article reads and disclosed identities/types only, without
numeric results, rankings or outcome-based eligibility advice:

- Yield summaries: A1, S2, S3, T1, T2, T3, T5, T6, T7, T8, T9, T10, T11.
- Within-session decoding summaries: T3, T6, T7, T9.
- Generic prior online cursor-control statements: S1, S2, S3, A1, T1.
- No participant-specific EP16 cross-session transport outcome was identified.

This exposure belongs in the prospective participant-specific exposure review;
the reviewer cannot be treated as an untouched future evaluator. The coordinator
and implementation worker received no numerical outcomes. No participant is
automatically relabelled from this disclosure, and no role has been assigned.

Narrow schema/code review still found no inspected participant/day/block size
annotation. Provider go epochs may arise from target changes **and/or task-state
signals**; center transitions therefore do not recover the full capture rule.
Pinned provider `array_paper/metrics/utils.py` describes cursor-coordinate scales
and distance-exclusion overrides, not target sizes or a Fitts classifier.
These quantities cannot fill in missing target widths. The residual source
question is a size-dependence disposition for otherwise contributing schedules,
not complete task/controller provenance and not proof that all schedules depend
on size.

### 2026-10-03 — structural_support conditional numerical application completed

The coordinator ran the selected bundle in attended allocation **46429138**,
node **sh03-06n50.int**, PTY88461. All **six focused checks passed** in 0.003s.
The [conditional support report](runs/structural_support/20261003-conditional_support01/conditional_support_report.json)
completed at **20:33:50 UTC** in **2.821248 seconds**, retaining all
**729 sessions / 328,548 declared trials**. It formed **368 geometry groups**
and evaluated **32 conditional chronological designs**. There were **zero
nontransitive compatibility components** and **two conditional numerical
passes**. No source archive/neural/cursor/impedance value was opened; no pair
was pruned, trial selected or role assigned. Adopted thresholds were unchanged.
[Official ASTRA schema and semantic validation](astra/v0.0.14/validation_20261003_conditional_support.json)
passed. No further EP16 compute is selected.

The two passing components are:

| Participant / group | Near / long pairs | Rank | Column-normalized condition | Residual norm | Retained energy | Maximum contrast-node share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T11 / geometry_0049 | 102 / 845 | 103/103 | 13.1060 | 0.233265 | 0.267146 | 0.144655 |
| T9 / geometry_0322 | 153 / 909 | 115/115 | 14.8701 | 0.203748 | 0.192361 | 0.064845 |

These are passing **conditional components**, not two eligible or supported
participants. T11 is already development; T9 is the only person in the current
held-out candidate pool with a passing conditional component. Current numerical
support therefore does not authorize the three-person held-out comparison or
the four-supported-development full search.

Other both-lag full-rank components fail prospectively fixed gates: T5
`geometry_0164` has node share 0.307586; T7 `geometry_0286` has residual
norm 0.085772, retained energy 0.029559 and node share 0.905012; T8
`geometry_0294` has norm 0.099357, energy 0.039489 and node share 0.601335;
T11 `geometry_0052` has norm 0.081696 and node share 0.840961. Remaining
both-lag components in T2, T6, T7 and T11 fail full constrained rank. In these
rank-failed cases the reported tiny residual and its normalized node shares
are numerical remnants, not scientific influence estimates. No near miss
justifies a relaxed threshold, alternative spacing rule or smaller preferred
pair set.

Targeted unresolved metadata now has a concrete scope:

- The exact **53 T11** and **59 T9** contributing session days are retained
  in their passing component records. Source size-dependence dispositions are
  needed for those otherwise contributing participant/day/block schedules,
  and for any other schedule proposed to contribute under a fixed rule.
  Full controller/task-name provenance is not required.
- Every contributing T11 session and 12 contributing T9 sessions have at least
  one nonempty block that does not cover the full session target set. The T9 days are 195,
  273, 329, 336, 406, 413, 426, 681, 686, 689, 693, 707. These are specific
  observed coverage differences, not proof of unstable layouts; the all-block
  criterion is sufficient evidence of compatibility, not a claim that all its
  failures establish a changed protocol. A defensible coverage/layout rule must
  resolve that sampling-versus-layout distinction without dropping inconvenient
  blocks to restore support.
- All contributing sessions within each passing component share the same
  pooled positive transition-edge support under the target bijection. T11's
  geometry-group-level transition differences occur on days 367, 576, 632, 905,
  1017, which are outside its 53-node conditional design because they lack the
  existing whole-block capacity. Thus the broader group flag must not be
  mistaken for a transition disagreement among the passing component's
  actual nodes. Positive support agreement is observed compatibility, not
  authenticated task identity or evidence of forbidden unobserved edges.
- Previously disclosed near-duplicate target distinctions and participant
  exposure review remain unresolved; neither may be silently reclassified
  to increase support.

**Decision under the current conditional panel:** stop before role selection,
neural scoring and the planned held-out claim. The no-pair-pruning rule and
requirement that every claim-contributing component pass prohibit deleting
failed components/edges, selecting a favorable subclique or relaxing a gate
to manufacture a qualifying cohort. This is **insufficient current conditional
support**, not proof that every defensibly qualified observed-schedule panel
under this source is impossible. Independent size/layout/transition evidence
may legitimately determine eligibility and thereby change the applicable
graphs; any such rule must be justified independently of rescuing these
support counts, fixed before application and applied consistently with all
exclusions disclosed. No such successor rule or computation is selected here.

## 2026-10-06 05:17 UTC — analysis_definition: prospective score, quality and exposure preparation

Under the coordinator's new preparation assignment, selected the concrete
definitions in `schedule_definition.md` and `config/analysis_definition.json`:
first 30 bins at declared trial start, summed 2D loss, all-eight-octant weighting,
complete-window/source-only scaling, direct availability/finite-rate order,
packet target coverage and evidence-based treatment of ambiguous block/edge
support. Adopted the existing .10/-.15/.50 normalized effect anchors in policy;
geometry/support limits, roles and the current failed support dispositions are
unchanged. Unanchored margins and remaining model/uncertainty choices are
explicit internal work, not missing provider information.

Recorded the nine-participant existing exposure history and the concrete review
procedure without claiming a completed clean-history review or automatically
moving yield/within-session-exposed candidates. No publications/results were
reopened. `code/analysis_primitives.py` consumes the selected numerical config;
three focused score/missingness tests are prepared in
`code/test_analysis_primitives.py` but **not executed**. No generic admission
gate or neural reader was added. Independent numerical/interface review passed;
it checked the three consequential arithmetic/access semantics without executing code or
adopting scientific choices on behalf of the coordinator. The existing ASTRA
`analysis_definition` now consumes the partial config as an explicit input;
the future complete specification still requires the remaining real work.
Official validation of this small ASTRA edit remains pending in compute.

Proposed coordinator check, inside an existing suitable substantive compute
allocation: from `outputs/code`,
`timeout --signal=TERM --kill-after=10s 45s python -m unittest -v test_analysis_primitives`.
One CPU, at most 1 GiB, all BLAS/OpenMP thread counts 1; use the existing
Python 3.12/NumPy stack. Do not submit a standalone sub-ten-minute job. These
three checks replace no scientific qualification and do not expand simulation
work. No job, source read, test execution, support rerun, role assignment,
neural score, held-out access or compute charge occurred for this preparation.
