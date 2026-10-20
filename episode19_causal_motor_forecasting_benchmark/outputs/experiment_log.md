# Experiment log

## 2026-10-17 18:03 Pacific — source_views_and_timing — fixed-label gap/H diagnostic pre-declared

Post hoc follow-up to job 48281685 (standing engineering authority), declared before any output: [rules](source_timing_handoff.md#october-17-fixed-label-gaph-diagnostic--pre-declared-before-execution). Codex gpt-6-astra (max, text only) drafted the design and code in one round; reviewed here.
It tests the 48281685 mixture on the fixed D2 labels, v5 k=5 onsets and closed PRD features (15 development people). Each tail event is split after its last deep (median r ≤ θ3) internal non-qualifying run ≥ 80 ms; λ* = L − R* is diagnostic only. Classes: B (H ≥ 288), A (λ* < 288), C (λ* ≥ 288).
Weakens (a), premature R joined across deep pauses: f_A CI upper < 0.5 or P95(λ*) CI lower ≥ 288. This needs KPG power: synthetic deep-pause-joined events must be classed A, CI lower > 0.8. Support also needs KPB/KPA specificity (continuous shapes split, CI upper < 0.2).
Weakens (b), a person-concentrated minority with H ≥ 288: both T2 (within-tail clustering) and T3 (run-half reproducibility) non-significant, with σ=0.75 plasmode power ≥ 0.80 and a passed σ=0 null check. T2's outcome was largely anticipated from the counts already released.
k, gates, acceptance and the reference estimator are unchanged; v5 stays UNDERIDENTIFIED; provider values are read only after the decision closes.
Next: a health-only smoke (sub-02/15/22, scratch), then `sbatch outputs/code/run_gap_h_diagnostic.sbatch` (russpold, AVX-512, 8 CPU, 32 GB, 1 h).

## 2026-10-17 14:41 Pacific — source_views_and_timing — premature-R diagnostic O4 (unresolved); control precision limits the linear near gate

Job 48281685 (sh04-04n24, 110 s, exit 0) closed every stage. The exact first-32 regression passed for all 15 people. [Result](source_timing_handoff.md#result-job-48281685) · [figure](figures/ep19_v5_followup_diagnostics_20261017.png).
PRD (pre-declared): O4 UNRESOLVED in 14/15 people (sub-04 KPA-uncovered). Every known-positive check passed (KPB ΔH 162 [142,182] ms; KPA ΔH 0, ΔC_E −0.75). ΔH = 19 [6,82] ms is neither ≥ 100 nor inside ±50. ΔC_E = −0.149 [−0.292,−0.029] falls between the continuity and discontinuity thresholds. No drift signature.
CPC (pre-declared): linear near mean 0.893 [0.884,0.902] with all 2,545 backgrounds (v5's 32: 0.888). CP1 near: 1 below (sub-15 linear), 15 across, 29 above. Observed = timing-error-implied near fraction in all 45 gated cells. dev_linear_q50 near 0.866 (13 people); q10 strata uncovered (negative rise rates). Sub-02 sham 0.051 [0.026,0.078] (v5's 0.070 retained).
Post hoc: tail R→L intervals are intermittent (occupancy 0.74 vs 0.96; internal quiet run median 88 ms) and fall after R (66% vs 32%). 54/289 tail events reach half the later amplitude ≥ 288 ms before L (sub-06/20/14). Provider post-lock: t½ − AccStartIndex is +32/+28 ms (tail/reference), against R −260/−16.
Strongest surviving alternative (post hoc): a mixture of premature R (intermittent low-amplitude activity) in most tail events and a late label for substantial movement in some. Next: pre-declare a fixed-label gap/H diagnostic; estimator resolution versus gate logic, a v6 reference and p95 versus worst case go to the daily batch. Codex round 3 drafted the interpretation; reviewed here.

## 2026-10-17 14:24 Pacific — source_views_and_timing — diagnostics smoke passed; full run submitted

Smoke 48281137 (health-only, sub-02/15/22, sh04-04n24, 35 s, scratch): all four stages closed. The exact first-32 regression passed for all three people. Only engineering counts were written: KP0/KPB/KPA estimates supported in 28–32 of 32 backgrounds per person (sub-22: 4 unarmed backgrounds); every determinate real event was supported.
Full run under the unchanged rules: job 48281685, `sbatch outputs/code/run_v5_diagnostics.sbatch` (russpold, AVX-512 nodes, 8 CPU, 32 GB, 1 h).

## 2026-10-17 14:20 Pacific — source_views_and_timing — diagnostics smoke: CPU-dependent regression failure fixed

Smoke 48280333 (health-only, sub-02/15/22, sh03-06n06, 28 s): both PRD stages closed; the CPC first-32 regression then stopped on `direction_y` (sub-02). No feature, synthetic or gate value was written.
Cause (engineering): v5 ran on an AVX-512 node (sh04-04n23, CPU_GEN:SIE), while the smoke node is AVX2-only (RME); the BLAS dot behind the direction norm differs in the last bits. Check job 48280993 on an AVX-512 SKX node reproduced all 14,784 sub-02 first-32 rows exactly.
Fix: `run_v5_diagnostics.sbatch` now requests `-C CPU_GEN:SKX|CPU_GEN:SIE`; the regression stays exact, and the rules are unchanged. Next: rerun the smoke.

## 2026-10-17 14:12 Pacific — source_views_and_timing — premature-R diagnostic and control-precision check pre-declared

Post hoc follow-up to job 48261284 (standing engineering authority), declared before any output: [rules](source_timing_handoff.md#october-17-premature-r-diagnostic-and-control-precision-check--pre-declared-before-execution). Codex gpt-6-astra (max, text only) drafted the design and code in 2 rounds; reviewed here (one fix: O3 also needs the KPA check).
PRD, on the fixed D2 labels and v5 k=5 onsets (15 development people): within-person tail (lag ≥ 288 ms) vs reference events on H = L − t½ (t½: the departure first reaches half the amplitude of F = [L, L+100)), sensor-frame continuity C_E, pre-R drift and quiet-anchor behaviour.
Weakens premature R (H_A): O1, ΔH ≥ 100 ms with CI > 0 and continuity not worse. Weakens a detector miss of substantial movement (H_B): O2/O3, ΔH CI inside ±50 ms plus discontinuity/drift (O2) or continuity (O3).
Each statement needs its synthetic known-positive check (KPB: slow continuous onset; KPA: orthogonal drift, then the reach) on a decision set of ≥ 12 people.
CPC: the six v5 families and gates unchanged on all 2,545 eligible backgrounds (the first 32 must reproduce v5 field by field), plus four descriptive development-slope strata. Sub-02's v5 sham/anchor failures are retained; k, gates and acceptance are unchanged.
Provider AccStartIndex is read only after both decisions close. Next: a health-only smoke (sub-02/15/22, scratch), then `sbatch outputs/code/run_v5_diagnostics.sbatch` (russpold, 8 CPU, 32 GB, 1 h).

## 2026-10-17 11:20 Pacific — source_views_and_timing — detector v2 / raw-onset v5: UNDERIDENTIFIED; no candidate frozen

- **Summary — outcome:** Calibration/diagnostic evidence only (15 development people, no EEG). D2 and D1 are UNDERIDENTIFIED: 8/15 people passed the k=5 controls, and the classification disagreed across k.
- **Summary — timing:** D2's k=5 p95 bounds are [324,324] ms (+12=336); D1's are [360,360] (+12=372); v1's were [696,696]. The numeric FAIL classifications are descriptive.
- **Summary — status:** No candidate is frozen. The v1 labels/exposure (48219799) stay frozen and self-paced primary scoring stays blocked; unfrozen D2/D1 labels are in `runs/source_views_and_timing/detector_v2_job_48261284/`.
- Job 48261284 (83 s, exit 0): both regressions PASS. c=1, LOW/HIGH 0.027917/0.053088 g; both candidates are noise-admissible. D2 has 3,964 onsets and 2.153 h at risk (v1: 4,189 and 2.612 h).
- Post-lock: D2 L−AccStartIndex is 60/208 ms median/p95 (v1: 228/564). [Result](source_timing_handoff.md#result-job-48261284) · [figure](figures/ep19_detector_v2_acceptance_20261017.png) (rendered by job 48262428 with the revised plot script).
- **Alternative (post hoc):** R is premature because low-amplitude activity or drift joins the reach. In the 291 tail events R−provider has median −260 ms, against −15 elsewhere; this is suggestive but conditioned on large L−R.
- **Next (post hoc):** declare a fixed-label continuity/drift diagnostic and same-margin control-precision checks, keeping all people and gates. Changing k or the gates needs the scientist (daily batch).
- Codex gpt-6-astra (max, text only; round 3) drafted these records; they were reviewed here.

## 2026-10-17 10:52 Pacific — source_views_and_timing — v5 smoke and controls check; full run under v5 unchanged

Smoke 48260443 (sub-02/15/22, health-only, 72 s, scratch): both regressions PASS (D1 with v1 thresholds reproduces 4,432 cues and
every label array of 48219799; the v4-mode estimator reproduces every v4 estimate). Settings: c = 1, LOW 0.027917, HIGH 0.053088 g; D2 and D1 admissible.
Event health only: 2/691 unresolved at k = 5, 0 following bouts. No lag values were read.
Controls-only check 48260978 (all 15, scratch; synthetic, no events): 8/15 pass at k = 5. Fails:
the linear τ5 = 64 ms near band (0.80–0.90) in 7 people; sub-02 shams 7.0% plus 2 unanchorable backgrounds; sub-03's fast band 0.93.
So the all-person control gate will fail and v5 will return UNDERIDENTIFIED. v5 is not revised after this post hoc check;
the full run stays as declared, for labels/exposure, the pre-declared outcome and descriptive lags. Submitting `sbatch outputs/code/run_detector_v2_acceptance.sbatch`.
Job 48261201 failed in 10 s with no stage output: under `set -u`, bash 4.2 rejects empty-array expansion. The sbatch was fixed (engineering only) and resubmitted as job 48261284.

## 2026-10-17 10:44 Pacific — source_views_and_timing — self-paced detector v2 and raw-onset test v5 pre-declared

Post hoc follow-up to job 48230171 (standing engineering authority), written before any v2 output: [§1.3](component_details.md#13-self-paced-detector-v2-october-17-pre-declared-before-any-v2-output), [v5 rules](source_timing_handoff.md#october-17-self-paced-detector-v2-and-raw-onset-test-v5--pre-declared-before-execution). Codex gpt-6-astra (max, text only) drafted it in 2 rounds; reviewed here.
Thresholds: the v1 quantile rule, applied to the support-complete final 480 ms of the frozen still windows (expected c = 1: LOW 0.0279, HIGH 0.0531 g). Guards: per-person crossings ≤ 5%, pooled entry/at-risk ≥ 0.90, each person at-risk ≥ 0.75.
Candidates: D2 labels the LOW crossing whose run reaches a confirmed HIGH crossing (pending from LOW); D1 keeps the v1 rules. Order D2 → D1; freeze the first admissible `BELOW_EMPIRICAL_P95_ONLY`.
v5 control labels are detector-independent: L = T_det,5 + target, on the 4-ms clock. Gated families: linear/quadratic (τ5 = 64 ms) and continuity-verified pulse+growth; slow and near-threshold families are descriptive. Near targets 288–316 ms, every 4 ms.
Lags are signed (an anchored following bout ≤ L+60 ms counts); aligned shams sit at 4 positions per background. The v4 gates, k = 3/5/8, the >5% unresolved rule and the 12 ms are kept.
Stage order: regressions → settings → controls closed → labels → decisions closed → provider. Control failure weakens identification; FAIL weakens the empirical timing argument; BELOW only weakens the large-lag alternative (no worst-case or EEG clearance).
Next: smoke (sub-02/15/22, health-only), then the full `sbatch outputs/code/run_detector_v2_acceptance.sbatch` (8 CPU, 32 GB, 1 h).

## 2026-10-17 10:17 Pacific — source_views_and_timing — noise inspection: v1 threshold scale was set by unscreened history

Job 48257954 (sh03-06n05, 22 s, exit 0): [run](runs/source_views_and_timing/detector_v2_noise_job_48257954/pasteback.txt). Every frozen count was reproduced:
3,661 windows (1,901/1,760), v1 LOW/HIGH, v4 θ3/5/8 and 2,545 backgrounds. Pooled fit X has q99.9 0.0851 g over full 1-s windows
but 0.0279 g over the support-complete final 480 ms. 86% of windows with X > 0.03 g exceed it only in samples whose
500-ms baseline lies before the screened window, and all 24 support-complete exceedances fail a 20-ms block screen.
Under the v1 quantile rule on corrected support: c = 1 gives LOW/HIGH 0.0279/0.0531 g, with 0/1,901 fit and 0/1,760 validation windows crossing.
Fit backgrounds then give pooled entry 0.965 and at-risk 0.963 (D1) / 0.955 (D2); sub-04 is lowest (0.84/0.79).
With c = 0.5/0.75, false crossings run 0.72/0.13 per minute (v1 validation: 0.10). Next: Codex round 2 drafts the declarations and code.

## 2026-10-17 10:11 Pacific — source_views_and_timing — noise-only inspection before the self-paced detector redesign

Selected under standing engineering authority (post hoc follow-up to job 48230171). It is meant to set the redesign's
threshold and availability rules from development still noise before any redesigned-detector output.
Code: [`inspect_detector_v2_noise.py`](code/inspect_detector_v2_noise.py), drafted by Codex gpt-6-astra (max, text only), reviewed here.
Scope: the 15 development people. Whole runs are loaded through `sp_load`, but computations use only pre-start still windows
(the final 480 ms of support) and v4 background segments. It computes no event crossing, onset, label, provider value or EEG.
Probes: quantile/MAD scales × c ∈ {0.5…4}, D1/D2 noise replays, and plateau feasibility. `srun -p russpold`, 8 CPU, 32 GB, 1 h.

## 2026-10-17 01:31 Pacific — source_views_and_timing — self-paced raw-onset lag test result

- Ran job 48230171: 47 s, 15 development people, 155 runs, 4,189 accepted onsets; 3,661 stationary windows reproduced exactly. Calibration/diagnostic evidence only.
- Full-run v4 included post hoc rounds 3–4 after two smoke runs, including 48227088; exposed sub-02/sub-15/sub-22 remain included.
- Outcome: UNDERIDENTIFIED; 0/15 people passed all primary controls. Pooled k=5 sham alarms: 11.3%; no control-passing participant subset.
- k=5 p95 lag bounds: 696–696 ms (+12=708); 9 unresolved; 1,868/4,180 determinate totals ≥300 ms. k=3/8 bounds+12: 808/628 ms. The 188-ms proxy is challenged; timing failure is not validated.
- Post hoc pooled k=5 controls: 9,001 determinate usable injections; 95.0% within ±20 ms; near/long classification 91.7%/99.9%. Conditional accuracy does not override failed controls.
- Post-lock provider comparison: k=5 R−provider median/p95 −16/60 ms; L−provider 228/564 ms. Shared accelerometry limits independence.
- Next (post hoc): predeclare a global detector redesign from all 15 development people and corrected acceptance controls—feasible targets, primary-bout-aligned shams, verified 1-kHz calibration; exclude held-out tuning/normalization initialization and retain original artifacts.
- Keep self-paced primary scoring blocked; WAY development non-neural baseline selection can proceed. One daily scientist batch: p95 versus worst-case criterion, minimum meaningful primary EEG gain and other unset margins.

## 2026-10-17 01:19 Pacific — source_views_and_timing — raw-onset test rules revised (v4); full run launched

Post hoc revision (Codex gpt-6-astra, max) after smoke 48227088. Quiet = no qualifying 40-ms window; controls on 2.8-s
pre-start backgrounds (final 1.9 s still); provider ticks rounded. Rules: [handoff](source_timing_handoff.md#october-17-self-paced-raw-onset-lag-test--pre-declared-before-execution).
v4 smoke 48229838 (3 people, scratch, rule-forced UNDERIDENTIFIED): 153–224 eligible backgrounds/person; 0.6% censored (k = 5);
determinate lag p95 612/520 ms (k = 5/8). Controls fail the gate: shams 9–31% (k = 5), and sub-15 1-kHz calibration leaves 14
cells uncovered. Post-lock: R − AccStartIndex median −7.5 ms. Full run under v4 unchanged:
`sbatch outputs/code/run_raw_onset_test.sbatch` (8 CPU, 32 GB, 1 h).

## 2026-10-17 01:15 Pacific — source_views_and_timing — raw-onset smoke run exposes two design flaws

Smoke job 48227088 (sub-02, sub-15, sub-22; scratch; 11 s) ran stages 1–3, then Stage 4 failed parsing float provider ticks
(523376.00000000006). Engineering findings: (1) at most 3 still 4.5-s pre-tone control backgrounds per person (≥20 needed);
(2) the "every sample ≤ θ for 200 ms" quiet anchor left 38% of onsets censored at k = 5 (θ 0.013–0.018 g).
Either would make the full run UNDERIDENTIFIED regardless of the labels. Seen outcome (subset, rule-forced underidentified): determinate
lags p95 692 ms (k = 5) and 520 ms (k = 8), 37% ≥ 288 ms. Next: Codex revises (post hoc, disclosed) with shorter backgrounds whose
still part is ≥ 1.9 s, quiet defined as no qualifying 40-ms window, and rounded provider ticks. Then the full run; the three smoke
people stay in it.

## 2026-10-17 01:03 Pacific — source_views_and_timing — self-paced raw-onset lag test pre-declared

Tests the alternative that self-paced labels lag sensor-detectable movement. Full rules: [handoff](source_timing_handoff.md#october-17-self-paced-raw-onset-lag-test--pre-declared-before-execution);
Codex gpt-6-astra (max) drafted the design and code (3 rounds), reviewed here. Statistic: native |a − mean[t−520, t−20)|; threshold
person median + 5·MAD-SD of frozen still windows (k = 3/8 sensitivity); onset R = start of the bout reaching the label (≥80%/40 ms,
bouts split by ≥200 ms quiet, 200-ms quiet anchor); lag = L − R. Controls: per-person noiseless-calibrated linear/quadratic/pulse ramps
on real 4.5-s still backgrounds plus shams, truth = detectable onset. Decision: controls fail, >5% unresolved, a p95 interval
straddling 288 ms or k disagreement → UNDERIDENTIFIED; else p95 lower + 12 ≥ 300 → FAIL, upper + 12 < 300 → BELOW (not a worst-case
proof). Provider columns are read only after the decision is written. Smoke (sub-02, sub-15, sub-22; scratch) first, then sbatch.

## 2026-10-17 00:18 Pacific — source_views_and_timing — development detector freeze complete (calibration)

Job 48219799 (102 s, exit 0; replay = `WayRiskStream` on 84/84 series): [run](runs/source_views_and_timing/detector_freeze_job_48219799/detector_freeze_summary.json).
WAY 2,030/2,712 cues armed, all with onsets, 0.206 h at risk; self-paced 4,195/4,432 armed, 4,189 onsets,
2.612 h. 300–600-ms positives/scorable decisions: 3,527/14,847 and 24,005/187,972. Crossing term p95 67 / 176 ms
→ pre-declared bounds WAY 247 ms, self-paced 188 ms (< 300); p99 terms 283/280; max terms 352/612; per-event
≥ 300 ms: 7/2,030 and 36/4,189. Post hoc (Codex review): the p95 gate does not prove GOAL's worst case. Strongest
alternative: self-paced labels follow AccStartIndex by median 228 ms (p95 564; a third ≥ 288), i.e. a 576-ms bound.
Next: pre-declare and run a provider-blind raw-onset lag test for self-paced labels; WAY non-neural baseline selection
can start. Daily question: may a p95-term budget stand for "worst case"? Session compute 0.80 CPU-h (11 steps).

## 2026-10-17 00:14 Pacific — source_views_and_timing — independent review fixes; freeze re-run selected

First full run 48195622 (125 s, exit 0; replay = `WayRiskStream` on 84/84 series) is kept but superseded.
A Codex gpt-6-astra (max) code review (text only) found 12 implementation defects and no rule change. P1:
a tone could be confirmed before its bridging burst was seen (self-paced cues up to 100 ms early); a gap
cut follow-up one sample early; a burst unresolved at a segment end did not limit follow-up; no guards for
an unresolved fit or nonfinite TRIG. P2: 0.25-s confirmation was 248 ms at 250 Hz; float schedule edges;
WAY crossing support short by ≤2 ms; starts in invalid segments missing from cue counts; derivative binding
not one-to-one. P3: rejected-crossing count after the cutoff. All fixed; same rules re-run below.

## 2026-10-16 19:25 Pacific — source_views_and_timing — full development detector freeze launched

Smoke job 48195238 (WAY P2; self-paced sub-15, sub-22; scratch only, 2-person global thresholds, not
frozen) ran in 76 s: the generic replay matched `WayRiskStream` on 7/7 series (0 state, onset or label
mismatches); the sub-15 burst at 221.77 s was rejected (0.099 s) with adjacent tones classified; tone
onsets sit 32 ms after provider TrialStartIndex (median). No rule changed. Full run: all 12 WAY and 15
self-paced development people, `srun -p russpold`, 8 CPU, 48 GB, 45 min → job 48195622 (exit 0, 125 s;
superseded by 48219799 after review fixes; kept in `runs/source_views_and_timing/detector_freeze_job_48195622/`).

## 2026-10-16 19:15 Pacific — source_views_and_timing — detector rules fixed before any detector output

Noise job 48191618 (and TRIG cue probes 48193388/48194320) showed heavy-tailed rest noise (0.1–3% of
WAY ticks > median + 6 MAD-SD), 174–201/226 position-still WAY windows per person, and noisy-TRIG
runs (sub-11/ses-01) where 20× tone detection fails. Rules, now frozen
([§1.2](component_details.md#12-detector-freeze-rules-october-16-written-before-any-detector-output)):
quantile scale σ_q = q99.9 − median of fit still noise; low/high = median + c·σ_q / + 2c·σ_q, with c the
smallest of 1–4 where ≤ max(1, 1%) of fit windows confirm a crossing; WAY per person (S1–S7), self-paced
global; TRIG 100-ms envelope > 4× causal reference, 0.25-s tone confirmation, schedule-based classes;
crossing term = pooled p95 of the backward supra-low run plus 20-ms support. Not yet run on any labels.

## 2026-10-16 18:50 Pacific — source_views_and_timing — noise-only inspection before the detector rules

Selected to set the detector-rule constants from development noise only: WAY P1–P12 S1–S7 pre-cue
windows [LED rise − 1 s, rise) (EMG/speed summaries, position spread and quantum, object/force rest
values); self-paced 1 s before start tones, TRIG tone bursts and accelerometer axis variance. It
computes no crossing, onset, label, HandStart or AccStartIndex. Code:
[`inspect_detector_noise.py`](code/inspect_detector_noise.py) with readers
[`detector_readers.py`](code/detector_readers.py) and [`detector_summaries.py`](code/detector_summaries.py).
`srun -p russpold`, 8 CPU, 32 GB, 30 min → job 48191618, exit 0 in 29 s,
[`detector_noise_job_48191618/`](runs/source_views_and_timing/detector_noise_job_48191618/noise_inspection.json).

## 2026-10-16 18:40 Pacific — source_views_and_timing — all eleven open source facts classified

Jobs 48183086 (COMPLETED 0:0, 59 s) plus post hoc 48184379/48184852. **Resolved:** WAY window start = HS
sample `500·trial_start_time − 1000` in 2,712/2,712 windows (bit-identical kin/EEG/EMG); LED >0.5 = lit in
84/84 files; positions cm (lift 6.24; weight-series cue 1–3 s after digits >15 cm in 96.1%, post hoc split);
forces N (1.72/3.45/6.71 N for 165/330/660 g); surface codes 1/2/3 = silk/suede/sandpaper. Self-paced:
sub-03/ses-02 `DataFile` is a stale copy line; no invalid development run (Table 1 count error); 1 g =
1.4465×10⁶ µV. **Assumed with sensitivity plans:** WAY filter/drift bound 220 ms < 300 ms, actiCHamp
filter, absolute TRIG latency, unmarked gaps (31 ambiguous cycles). Pre-declared misses are listed in the
[classification](source_timing_handoff.md#october-16-source-fact-classification-current-supersedes-the-open-items-below).
EP19 compute today 0.255 allocated CPU-h (15 short steps). Next: development detector/label freeze (see README).

## 2026-10-16 17:50 Pacific — source_views_and_timing — three post hoc follow-ups selected after job 48183086

Selected after reading the 48183086 results, so labelled post hoc: (1) WAY FASTRAK
velocity spectra during reaches (update rate, possible interpolation look-ahead,
prompted by lift-off lags of −24 ms at p10); (2) WAY env condition-code timing at
the LED rise; (3) self-paced 60-Hz mains-phase continuity, because cue-playback
jitter (p1–p99 −111/+200 ms) made the pre-declared ±60-ms interval rule
insensitive and 49 short cycles (~−120 ms) were uncompensated. Code:
[`source_fact_followups.py`](code/source_fact_followups.py). Same data scope; no new data roles.

## 2026-10-16 15:45 Pacific — source_views_and_timing — source-fact checks selected (scientist decision of Oct 16)

Selected under the October 16 rule that missing source facts are resolved from
papers, code and permitted data, or adopted as stated assumptions; no provider
contact. Fourteen checks (W1–W9 WAY, S1–S5 self-paced) and their decision rules
are pre-declared in [the handoff](source_timing_handoff.md#october-16-pre-declared-source-fact-checks-written-before-the-data-were-read).
Data: WAY P1–P12 series 1–7 (HS view + WS S1–S7), 155 self-paced development
runs and the sub-03/ses-02 derivative. Closed: WAY S8–S9, `AllLifts`, the eight
held-out people, EEG models. One-file smoke runs passed (48182660, 48182890);
full run submitted as job 48183086 (russpold, 8 CPU, 24 GB, 1 h).

## 2026-10-12 11:21 Pacific — reference_panel — final-refit companion checks and ASTRA passed

At 18:21:06 UTC, the retained
[companion status](runs/reference_panel/finding_a_companion001/status.txt)
records parent **EP21/job 47373800**, `empirical_access=false`,
`checks_exit=0`, `astra_exit=0` and final `status=closed`.
The [test receipt](runs/reference_panel/finding_a_companion001/checks.txt)
reports **three tests passed in 0.184 seconds**, all from
`test_reference_final_refit.FinalReferenceRefitTests`:

- `test_complete_cohorts_exact_offset_join_three_seed_dispatch_and_probability_ensemble`
- `test_each_actual_backend_requires_explicit_final_stage_before_consuming_batches`
- `test_incomplete_or_wrong_role_offsets_refuse_before_fit_and_failures_never_retry`

The [ASTRA receipt](runs/reference_panel/finding_a_companion001/astra_validation.json)
records both schema and semantic validation passed. These are the three
separately selected final-refit methods, distinct from the complete-source
development-driver methods that passed in 5.558 seconds on October 11. Both
sets of passing evidence and all prior attempts remain preserved; the
earlier descriptions of these final-refit checks as pending are historical.

The coordinator observed the substantive EP21 first-readout parent RUNNING
on sh04-04n21 at 18:20:41 UTC after the companion closed. Accounting observed
at 20:19:04 UTC subsequently recorded the parent cancelled under account
UID 467774 at 20:13:27 UTC, with 6872 elapsed seconds and 3 allocated CPUs.
The actual parent charge is **5.726667 allocated CPU-hours**
(`6872 × 3 / 3600`), counted once in EP21 against its 216 CPU-hour maximum;
the serialized companion adds no separate allocation or EP19 charge.
The later parent cancellation does not invalidate these separately closed
three checks or ASTRA validation. See the
[EP21 cancellation and accounting record](../../episode21_wang_neural_code_conversion_reproduction/outputs/experiment_log.md#2026-10-12-1313-pacific--finding_a_feature_decoding--first-readout-cancelled-at-the-scientists-instruction-before-outcome-exposure).
That EP21 record establishes a deliberate scientist-directed stop; no system
fault or retry is implied.

This closes the selected software checks, not an empirical refit, source
release or forecasting result. Source timing/calibration, authentic run
bindings, genuine conditions, qualified onset labels and calibration
prefixes, prospective margins, enforced reader access and production resource
admission remain unresolved. No EP19 scientific payload was opened, and no
new experiment or successor is selected by this completion.

## 2026-10-11 23:54 Pacific — reference_panel — complete-source driver checks passed in closed CPU step

All three new development-driver integration methods passed in
**5.558seconds**, and changed ASTRA schema/semantic validation passed, in
attended step **47375558.0** on **sh02-15n09.int**. Slurm independently
reported the actual step allocation as1CPU/512MiB/noGPU/fourminutes. The
compute shell verified its own host/job/step identity before testing. The
conservative sum of all current processes owned by the scientist on that
node was11093440KiB RSS before tests, below even the single16GiB parent
allowance; this observation included both owned training processes rather
than attributing a historical aggregate peak to current free memory.

The [three methods](runs/reference_panel/development_driver001/checks.txt)
exercise complete72/5-fold execution through the existing seed loop,
prediction/calibration and scorer using instrumented fitted models. They
check common condition rows/labels/offsets, release of prior-fold EEG models,
late-fold input failures before any fit, and fit/budget failures without
retry or partial scoring. No numerical backend fit, source payload or
scientific result is established by these software fixtures. No failing
method or production repair occurred; the verified implementation remained
unchanged. Only the test module's obsolete UNRUN docstring was subsequently
removed, with no executable change.

The test interpreter used13.73wall seconds,10.79CPU seconds and370144KiB
peakRSS; [ASTRA validation](runs/reference_panel/development_driver001/astra_validation.json)
used3.92wall seconds and74572KiB peakRSS. Both exits and the overall
verification exit are zero in the
[closed status](runs/reference_panel/development_driver001/status.txt).
The shell exited at06:54:06UTC. At06:54:37UTC, Slurm confirmed the step
**COMPLETED0:0**,93elapsed seconds at1CPU and14.078CPU seconds. This step is
included once in EP21 parent47375558's existing reservation and eventual
actual charge; do not add its overlapping93seconds as another allocation.
Local temporary storage was6bytes and durable receipts7447bytes before final
status lines, each below the selected16MiB bound. No package was installed.

The same scheduler cycle confirmed both EP21 content parents still RUNNING;
their inspected progress had advanced to674/1024 and496/1024 respectively.
Their wrappers, grants, model bindings, limits and scheduled companions are
unchanged. No open testing shell remains. This closes verification of the
fixed-recipe development composition; real-data timing/calibration, genuine
conditions, run bindings, labels/prefix qualification, margins, readers and
production resource admission remain unresolved. The separate three changed
final-refit methods still wait in47373800 and were not rerun or replaced.

## 2026-10-11 23:52 Pacific — reference_panel — bounded independent CPU verification step selected

Select an attended CPU-only Slurm step inside task-owned substantive EP21
parent47375558, which the coordinator confirmed RUNNING onsh02-15n09 with
2CPUs/16GiB/oneGPU and more than33hours of allocation time remaining. The
unchanged training process uses one numerical thread and its own15GiB RSS
ceiling. Independent source review found no process-exclusivity rule or worker
pool in that training path. This is a separately selected concurrent software
step, not a modification of its frozen wrapper, grant, training or companion
selection. Total active numerical threads may not exceed the two allocated
CPUs. All parent cost remains charged once to EP21; no additional allocation
or increase to its96CPUh/48GPUh reservation is selected.

The step is bounded to1CPU/512MiB/fourminutes/noGPU/one numerical thread.
Before any tests, verify actual compute shell/job identity and current owned
training-process memory; a historical Slurm peak is not headroom evidence.
Do not execute if combined resource headroom is insufficient. Use child-local
environment and temporary paths, at most16MiB local and16MiB durable receipts,
and exactly the three new driver checks plus changed ASTRA validation. Prior
same-import panel verification used363280KiB peak RSS; this is supporting
evidence, not a guarantee for the new composition. Required failures stop
without retry or weakening the bounds. No scientific payload is opened and
the separate pending final-refit checks in47373800 are not rerun or replaced.
Intended receipts: `runs/reference_panel/development_driver001/`.

## 2026-10-11 23:49 Pacific — reference_panel — complete-source fixed-recipe driver implemented and reviewed

`code/reference_development_driver.py` now connects the existing fitting,
prediction/calibration and scoring components for one registered EEG recipe.
Caller-supplied fold inputs contain the already fitted baseline, original
prepared fitting participants, four condition-window maps, common semantics
and validation inputs. All72 WAY or five self-paced folds, shared fitting
rows/labels/offsets and independent scored suffix labels bind before any EEG
fit. The unchanged three-seed fitter executes every condition; each fold's
models are released after its predictions. Only a complete-source score is
returned, with original fit reports and calibration evidence. Baselines are
not refitted and failures do not trigger retries or partial feedback.

Root and independent source review found no actionable defect. The three
focused integration methods in `tests/test_reference_development_driver.py`
exercise the original seed loop, predictor, calibration and scorer using
instrumented fitted models. They cover complete-roster execution/model
release, late-fold input refusals before any fit, and fit/budget failures
without scoring a subset. **All three tests and changed ASTRA validation
remain UNRUN.** No allocation, process step, empirical input access or new
compute charge occurred. The existing `reference_panel` ASTRA inputs/output
and component description now include this composition; testing remains to
be coordinated in a suitable substantive allocation, without changing the
already submitted EP21 jobs or their fixed companion selections.

This completes the source implementation of fixed-recipe development
orchestration, not its runtime verification or an empirical experiment.
Authentic conditions and source timing/calibration, unresolved run bindings,
qualified labels/prefixes, scientific margins, source readers and production
resource admission remain required. Supplied inputs/baselines and completed
fold predictions remain resident; classical fitting needs an outer process
limit. The separately selected three final-refit checks in47373800 remain
pending and are not replaced by these new methods.

## 2026-10-11 23:43 Pacific — reference_panel — complete-source fixed-recipe execution selected

Select the missing composition between the existing development-fold EEG
fitter, matched prediction/calibration and complete-source scorer. One call
will execute one registered EEG recipe against already fitted, frozen
baseline results for every canonical development fold: all72 WAY folds or
all five self-paced folds. Fit all four existing EEG conditions from their
three original seeds, with shared fitting rows, labels and baseline offsets;
score only after every required fold completes. This is864 WAY or60
self-paced EEG fits per recipe, not a new search allowance. Preserve the
existing natural-prevalence pooling, participant weights, calibration and
failure rules. No condition construction, scientific default, recipe selection,
source reader, empirical release or partial feedback is selected.

The implementation owner is limited to new
`code/reference_development_driver.py` and its focused test module. Existing
fitters, prediction/scoring modules and all submitted EP21 runtime remain
unchanged. The coordinator owns documentation and ASTRA; a separate reviewer
will inspect the full fold/condition scope and failure behavior. New checks
are **UNRUN** and no new allocation or compute charge is selected here.
Verification will use a suitable coordinator-owned compute opportunity;
there is no login-node testing. Production memory/runtime, authentic source
timing/calibration, run bindings, qualified labels/prefixes, genuine control
construction and empirical access remain unresolved. The separate previously
selected final-refit checks remain pending in parent47373800.

## 2026-10-11 17:12 Pacific — reference_panel — complete-source scoring composition verified

Implemented `code/reference_development_panel.py` for one fixed registered
baseline/EEG recipe across every canonical development fold: all72 WAY folds
or all five self-paced folds, with12 or15 complete participants respectively.
It binds the independently supplied scored-row roster and all three label
arrays before aggregation, requires every original matched condition and
unchanged recipe, and reuses `forecast_scoring` for all three horizons.
Natural-prevalence pooling within people, equal-person weighting and the
participant-specific strongest surrogate remain unchanged. Self-paced
calibrated32 and zero-label outputs remain separate with full calibration
records and explicit identity fallbacks; WAY's two keys identify the same
unadapted prediction. Missing primary support refuses the source; missing
diagnostic-horizon support remains explicitly unresolved without dropping a
person, matching the existing development convention. No model fitting,
source release, margin, recipe selection or biological finding follows.

Root and independent source review found no actionable defects. Both new
files remained fixed during the attended compute run. On **sh04-18n32.int**,
job **47434740**,1CPU/4GiB/10minutes/noGPU/one numerical thread, the
[two focused methods](runs/reference_panel/development_panel001/checks.txt)
passed in **0.500 seconds**. Their independent scalar oracle checks all
three horizons, row pooling versus unequal participant sizes, matched
comparisons, alignment, calibration records and diagnostic support. Refusal
subcases cover missing/duplicate folds, role/recipe/condition changes,
row/label mismatches, nonfinite predictions, missing primary support and
cancellation. The test interpreter used5.20wall seconds,4.55CPU seconds and
363280KiB peakRSS. Unchanged component tests were not rerun.

[ASTRA schema and semantic validation](runs/reference_panel/development_panel001/astra_validation.json)
passed in2.90wall seconds. Both exits are zero in the
[closed status](runs/reference_panel/development_panel001/status.txt), and
the compute shell exited at00:12:12UTC. At00:12:48UTC, Slurm confirmed
**COMPLETED0:0**,77 allocated seconds at1CPU: **0.021389 allocatedCPUh**
charged once to EP19, within the1/6CPUh reservation. Preserve all prior charges;
no EP21 parent is charged for this session. The node-local test directory was
10bytes and durable receipts8621bytes before the final closure lines, within
the selected64MiB/16MiB bounds. No package was installed and no empirical
payload was read.

This closes the per-recipe, complete-source scoring connection. Production
memory/runtime remain unmeasured, with supplied fold arrays and the assembled
panel resident together. Genuine conditions, authenticated source clocks and
calibration, qualified labels/prefixes, unresolved run bindings, margins,
reader admission and empirical fitting orchestration remain prerequisites.
The separate three changed final-refit methods remain pending in **47373800**;
this evidence does not replace them. Submitted EP21 runtime is unchanged.

## 2026-10-11 17:07 Pacific — reference_panel — complete-source development scoring composition selected

Close the explicit connection between `reference_development_prediction` and
the existing `forecast_scoring` without opening empirical inputs. Select one
array-only composition for a fixed registered baseline/EEG recipe: all72 WAY
validation folds or all five self-paced folds, every canonical participant,
the caller's complete expected row identities and labels, and every original
development condition. Reject incomplete/duplicate folds, changed recipes or
row/label mismatches. Preserve natural-prevalence within-person pooling,
equal-person weighting and separate calibrated/no-label results through the
unchanged scorer. Return continuous scores/increments only; no margin, winner,
claim, reader permission, source correction or scientific amendment follows.

Implementation owns only new `outputs/code/reference_development_panel.py`
and focused `outputs/tests/test_reference_development_panel.py`; existing
submitted EP21 runtime and EP19 companions remain unchanged. Coordinator owns
the component description, ASTRA and execution records. Independent review
will check the full registered fold/condition scope. Required verification is
limited to the new composition's numerical pooling/alignment behavior and
refusal of incomplete or mismatched panels, plus changed ASTRA validation.
No new simulation study or repeated qualification of unchanged components is
selected. Tests are UNRUN. A separate attended compute session, if needed,
is bounded to1CPU/4GiB/10minutes, one numerical thread, no GPU,16MiB durable
receipts and64MiB node-local temporary space; actual Slurm binding and charge
will be recorded before/after execution. Source timing/calibration, genuine
conditions, qualified labels/prefixes, margins and empirical access remain
unresolved and are not supplied by this software work.

The two new files are implemented and frozen for verification. Root source
review found no defect; independent source review is in progress. Requested
the selected attended `dev` allocation as **47434740** (1CPU/4GiB/10minutes,
no GPU). Its live `srun --pty` handle initially reports queued/waiting for
resources. No test or scientific import has run on the login node. Intended
receipt directory: `runs/reference_panel/development_panel001/`. The maximum
reservation is1/6 allocatedCPUh; record the actual charge when it closes.

## 2026-10-11 09:16 Pacific — reference_panel — development prediction checks and ASTRA passed

At 16:16:48 UTC, the coordinator observed EP21 parent **47375558** RUNNING
on **sh02-15n09**, elapsed 8:54. Its independent EP19 software companion has
already closed successfully. The [check output](runs/reference_panel/cuda_roster005_prediction_companion001/development_prediction.txt)
records all **three `test_reference_development_prediction` methods passed
in 0.239 seconds**: both-source offset/seed/calibration composition, explicit
failure and cancellation handling, and role/condition/seed/row-identity
refusals. The [ASTRA receipt](runs/reference_panel/cuda_roster005_prediction_companion001/astra_validation.json)
records successful schema and semantic validation. The
[companion status](runs/reference_panel/cuda_roster005_prediction_companion001/status.txt)
records `development_prediction_exit=0`, `astra_exit=0`, `status=closed`,
`empirical_access=false` and `parent_charge_once=true`. Independent bounded
receipt review confirmed the three-method count and both successful exits.

These results validate the selected array-only prediction/calibration bridge;
they do not establish complete forecasting-pipeline validation or empirical
support. No EP19 source release, real-data fit, forecast score or held-out
evaluation occurred. Keep the companion within EP21's parent charge once;
the parent remains active at this observation, so no terminal parent cost is
inferred from the check duration.

The separate three changed `test_reference_final_refit` methods and their
companion in **47373800** remain pending successful EP21 training47370327.
This result does not stand in for those unrun methods. Authentic source
clocks and physical calibration, unresolved source/run-role bindings,
qualified natural-grid labels and complete32-event prefix/guard support,
actual surrogate construction, frozen margins and remaining empirical
orchestration still block scientific execution. Preserve the earlier dated
entries as their observation-time history; no runtime, grant or scientific
contract changed.

## 2026-10-11 05:46 Pacific — reference_panel — matched development prediction and calibration bridge implemented

The new `code/reference_development_prediction.py` connects existing fitted
baseline/EEG ensembles to one complete canonical validation fold, consuming
prepared arrays only. It retains both sources, all four original development
EEG conditions and three separate seeds per model. Frozen uncalibrated
baseline probability means supply identical evaluation offsets to all EEG
branches. Each completed model's seed probabilities are averaged before
separate self-paced participant/model calibration; prefix and guarded suffix
reset independently and one suffix evaluation supplies both zero-label and
calibrated outputs. WAY remains unadapted. No guard predictions, fitting,
condition construction, scores, selection or incomplete-source feedback occur.

Independent source review identified and corrected one cross-condition
binding gap: the existing bandpower FIR coefficient tuples now match across
real/zero/shift/phase as well as across seeds. The corresponding subcase fits
inside the existing identity test; no filter or scientific default changed.
Final independent review passed with no remaining findings. Exactly three
`test_reference_development_prediction` methods remain **unrun**.

Select those three methods and changed ASTRA validation as a separate
software companion in EP21's newly selected original CUDA pair5 parent.
Use `runs/reference_panel/cuda_roster005_prediction_companion001/`, the
qualified Python3.12 module environment, one numerical thread and175seconds
plus5seconds kill grace. EP21 charges its parent once; the companion shares
the actual StartTime+169200-second clock bounded by EndTime minus60. Ordinary
EP19 check/validation failure does not gate EP21; SIGKILL or exhausted parent
clock stops the parent. The parent was submitted as **47375558** at05:46:56Pacific.
No EP19 payload access or fit is selected.

The existing refit implementation/tests in pending parent47373800 remain
unchanged. Physical timing, valid natural-grid labels, actual surrogate
construction, complete32-event prefix/guard qualification, margins and
empirical orchestration remain unresolved. This bridge returns internal
arrays with explicit qualification/fallback limits; it does not establish
complete-source performance or a scientific result. See the
[component convention](component_details.md#development-fold-matched-prediction-and-calibration).

## 2026-10-11 05:32 Pacific — reference_panel — independent companion parent submitted

EP21's separately charged first original readout parent was submitted as
**47373800**, dependent on successful47370327. Its source-reviewed batch
includes the selected three changed EP19 reference-refit methods and ASTRA
validation, with separate records under `runs/reference_panel/finding_a_companion001/`.
They remain **unrun**. No EP19 empirical access, source release or new fit is
authorized; the companion retains its independent failure interpretation
and the shared absolute parent clock/accounting recorded at05:30.

## 2026-10-11 05:30 Pacific — reference_panel — changed ensemble checks selected as independent EP21 companion

The three changed `test_reference_final_refit` methods and changed ASTRA
validation are selected as a serialized software companion in EP21's first
original Finding A readout parent, which depends on successful47370327.
Use `runs/reference_panel/finding_a_companion001/` for separate checks,
validation and status records. No source payload, fitting or scoring is
admitted. Ordinary EP19 check/validation failure does not gate the EP21
readout; allocation exhaustion still stops the parent. EP21 alone accounts
for the3CPU/16GiB/72h parent and its216CPUh maximum; there is no separate
EP19 resource charge or deadline reset. All work shares the parent's actual
StartTime+255600-second clock, bounded by EndTime minus60seconds. The
companion remains **unrun** pending that dependency and actual admission.
The five unchanged backend checks from47360846 remain reusable.

## 2026-10-11 05:19 Pacific — reference_panel — development-fold ensemble composition implemented

The original reference search needs a three-seed EEG fit inside each existing
development fitting fold. The selected baseline already supplies same-fold
training offsets and all five EEG backends already accept those scopes, but
`code/reference_final_refit.py` previously composed only final refitting.
It now also exposes `fit_development_reference_ensemble`, sharing the existing
identity join, family dispatch, fitting loop and ensemble construction with
the unchanged public final-refit entrypoint.

One development call requires all five WAY fitting series or all twelve
self-paced fitting participants, uses only the registered recipe menu and
seeds1901/1902/1903, and binds the original or factorial adapter to its
condition. It excludes the validation series/people, WAY feedback series7,
final scopes and final-only surrogate conditions. Baseline offsets remain
uncalibrated training predictions from the same frozen fitting scope, never
out-of-fold evidence. Required failures propagate without retry or a partial
ensemble. The caller still supplies authentic condition-specific windows,
training-only preprocessing and any factorial recipe selection.

Author review and independent root source review found no actionable defect.
The existing three methods in `tests/test_reference_final_refit.py` now cover
both sources and both stages, exact offset joins, scope refusals, seed failure
and cancellation. **The changed methods are unrun.** The unchanged numerical
backends and five scope/report checks passed under47360846 and remain reusable;
the earlier eight-check result does not validate this later wrapper change.
The changed `reference_panel` ASTRA objects also await pinned validation.

No empirical fit, source access, score, new grant or scheduler submission was
performed. Timing/calibration, qualified labels, frozen margins and remaining
empirical orchestration still prevent scientific execution. Run the three
changed methods and ASTRA serially in a subsequently selected substantive
allocation, preserving module PYTHONPATH and the parent's deadline/accounting.
This is original pipeline implementation, not activation of a revised study.

## 2026-10-11 02:37 Pacific — reference_panel — eight focused checks and ASTRA passed

The selected CPU-only EP19 child of EP21 parent **47360846** completed on
**sh04-01n01**. The [check output](runs/reference_panel/cuda_companion001/checks.txt)
records **8 tests passed in 0.034 seconds**: all three new final-reference
composition methods and the five selected classical/GRU/TCN/EEGNet/ensemble
scope and report methods. Existing numerical backend evidence is reused;
this is not a claim that every forecasting-pipeline test ran or passed.
The [ASTRA receipt](runs/reference_panel/cuda_companion001/astra_validation.json)
records successful schema and semantic validation, and
[status.txt](runs/reference_panel/cuda_companion001/status.txt) records
`checks_exit=0`, `astra_exit=0`, `empirical_access=false`, `status=closed`.

The coordinator reports the parent's separate GPU measurement is still live.
Its independent EP21 collection child exited 139; that separate-process
failure does not invalidate the completed EP19 checks and ASTRA receipt.
Charge the parent once to EP21's separate CUDA resource grant, with no
duplicate EP19 cost and no EP19 rerun selected.

These checks establish the prepared final-refit composition and affected
interfaces only. No empirical data, model fit, source release, forecast score,
calibration or held-out evaluation was produced. Authentic source timing,
physical calibration, qualified labels, frozen selections/margins and
complete empirical orchestration remain unresolved. The prior 02:30
pending-validation entry is historical; its scientific and resource limits
remain unchanged.

## 2026-10-11 02:30 Pacific — reference_panel — final-refit implementation reviewed; compute validation pending

The selected array-only `code/reference_final_refit.py` composition and three
tests in `tests/test_reference_final_refit.py` are implemented. The affected
`classical_forecast_backend.py`, `gru_forecast_backend.py`,
`tcn_forecast_backend.py`, `eegnet_forecast_backend.py` and
`ensemble_forecast_backend.py` now admit the explicit canonical final-refit
stage while retaining development as the default. Independent source review
found no actionable defect. No project imports or tests ran on the login node.

One call joins all supplied rows to frozen baseline training offsets by exact
identity and fits one registered recipe and already constructed condition at
seeds 1901/1902/1903. It returns actual fit reports and the existing frozen
probability-mean ensemble. Required failures propagate without retries or a
partial result; the returned scope completion is explicitly not full-panel
completion. WAY final fitting remains one person's complete series 1–7,
repeated separately for all 12 people in eventual orchestration; self-paced
final fitting remains all 15 development people. Offsets are uncalibrated
training predictions, not OOF estimates. Numerical models are unchanged.

ASTRA `reference_panel.reference_final_refit_checks` now names the wrapper,
all five affected backend/ensemble files, existing offset/recipe dependencies
and the focused 1-CPU, 2-GiB, 115-second command. It runs the three new methods
plus the five changed scope/report methods only; existing numerical fitting
evidence is reused. Those **eight methods and changed ASTRA validation are
unrun**, to be executed serially in the coordinator's substantive allocation.
The selected companion output location is
`runs/reference_panel/cuda_companion001/{checks.txt,astra_validation.json,status.txt}`;
it is a planned location, not existing evidence. The coordinator will use the
established pinned ASTRA 0.0.14 schema/semantic validation command on the
changed program after the focused checks.
No dedicated short job, empirical fit, reader release or held-out evaluation
has been launched by this implementation or documentation work.

**Companion submission accepted:** EP21 parent Slurm **47360846**, requested
2CPU/16GiB/1GPU/two hours, includes these eight methods and the changed EP19
ASTRA validation serially. The selected115-second CPU-only check child uses
one numerical thread and no empirical data; charge the parent once to EP21's
separate CUDA resource grant. Submission is not a test pass or forecast result.

Caller-owned source/resource admission, final recipe freeze, authentic matched
condition construction and empirical orchestration remain required. Classical
fitting needs an outer process limit in addition to the callback; production
memory/runtime remain unmeasured. The existing source-clock, calibration,
label qualification and scientific-margin gaps are unchanged. No forecast
result, scientific support or new empirical authorization follows.

## 2026-10-11 02:19 Pacific — reference_panel — final EEG refit and ensemble composition selected

The original final fitting cohorts and baseline-offset convention already
exist, but all five EEG-family backends and their ensemble still accept only
development folds. This prevents a frozen selected reference from being
refit on its actual prescribed final fitting role. Select a narrow explicit
final-refit entry path while retaining development as the default: one WAY
participant's complete series1–7 or all15 self-paced development participants,
always disjoint from held-out data and using the existing canonical scope.

Add an array-only `fit_final_reference_ensemble` composition for one frozen
registered recipe and one already constructed matched condition. Join frozen
baseline training offsets by exact row identity, fit exactly the three
registered seeds with unchanged models/optimization, and return their actual
fit reports and frozen ensemble. No model/seed/horizon search, calibration,
surrogate construction, new timing assumption, scientific margin or empirical
data access is selected. Final recipe freeze and source/resource admission
remain caller responsibilities. Real/zero/surrogate and factorial orchestration
remain distinct required work.

The implementation worker owns the five affected backend/ensemble files,
the new `reference_final_refit.py` composition and focused checks. An independent
worker reviews API and role preservation; root owns records and later compute.
Reuse existing numerical tests and add only scope/cohort/offset/dispatch checks
needed by this change. No test-only job, signal read, model fit, held-out
evaluation or change to the live EP21 measurement is selected. Missing timing,
physical calibration, label qualification and frozen empirical selections
continue to limit actual forecasting; this repair does not resolve them.

## 2026-10-11 01:52 Pacific — source_views_and_timing — 155 metadata rows complete; three competing bindings

Parent47357234's independent EP19 task completed successfully in its explicit
one-CPU step on sh03-06n05. Four focused reader methods passed in0.189seconds,
the metadata process completed in0.507209seconds, and ASTRA schema/semantic
validation passed. Evidence is in `runs/source_views_and_timing/column_stripe_job_47357234/`
and `column_stripe_job_47357234_metadata/development_stream_layout.json`.
All155 frozen development triplets were inspected using binary stat-only and
NewSegment-only metadata; no signal arrays, labels, held-out or WAY data opened.

The declared publisher filename mapping supports **152 file associations**.
It resolves the three earlier header-reference skips and three legacy marker
references for metadata purposes while preserving all nine original defects.
The remaining **sub-03/session02/runs0009,0010,0011** each name the extant
**run0008 EEG** in their marker DataFile declaration. These are competing
authentic references, not merely absent legacy names; their marker association
remains unresolved. No affected run is excluded and no trace, synchronization,
calibration, valid-run or forecast claim follows from accepted filename mapping.

The EP21 parent subsequently closed with its own pre-measurement fixture-file
cap failure, which does not invalidate this completed independent metadata task.
The whole parent26seconds×3CPUs is charged once to EP21; no EP19 duplicate cost.
No EP19 rerun is selected. Further work needs resolution of those three
competing associations plus the existing physical calibration/filter-delay,
onset uncertainty, WAY clock and valid-run questions.

Focused review of the already captured sub-03/session02/run0008–0011 records
finds distinct nominal frame counts144580/144645/144555/144515 and distinct
NewSegment starts11:23:28.518207,11:35:26.187849,11:51:51.095835 and12:04:02.445661
on2023-09-29, timezone unspecified. Thus whole EEG files and the retained
NewSegment records are not identical; copied/overlapping signal content and
editing history remain untested. Repeated legacy DataFile text is plausible,
not established. Keep same-file frame/layout metadata while withholding the
three absolute marker-clock associations. A targeted source correction is
needed if those absolute starts are required; the conflict does not invalidate
unrelated layout records or justify another extraction. No new payload was read.

## 2026-10-11 01:43 Pacific — source_views_and_timing — documentary triplet mapping selected

The publisher's Data Records section explicitly defines matching participant,
session and run stems for corresponding EEG, header and marker files:
https://www.nature.com/articles/s41597-025-06039-9 . Combined with the frozen
development export's exact unique 155 triplets / 465 original members, this
supports an explicit `publisher_same_entity_filename_contract` mapping for
metadata processing. Select that narrow reader repair under the scientist's
standing continuation authority. Preserve all original embedded references,
their verification status, the three header-reference skips and six marker
binding discrepancies. Do not silently rewrite source files or claim a clean
BIDS validation. Any reference known to bind another extant identified run
remains a competing binding requiring focused resolution.

Read-only BR request `ext_review_dir_2ff95d154683` supplied external-review
criteria, not a server verdict or canonical approval. Independent local review
applied those criteria: frozen subject/manifest coverage passes; embedded
reference discrepancies and missing physiological/clock QC remain flags.
Decision impact: these nine filename cases no longer inherently require a
provider response, while trace identity, synchronization, calibration, filter
delay, gap-free acquisition, 240-versus-238 eligibility and forecast validity
remain unestablished. Filename correspondence does not establish those facts
for either these nine or the other 146 runs.

Select the existing bounded metadata-only reader over the same 155 development
entities after focused changed-code checks, retaining stat-only binary access,
header/frame arithmetic and NewSegment-only marker extraction. No signal
values, trial labels, held-out people, WAY assets or model fitting are opened.
The checks and at-most-75-second metadata process will share the separately
selected EP21 two-hour parent serially, with the parent charged once to EP21;
no separate EP19 job or duplicate CPU charge. Implementation/review and actual
parent admission remain pending; this entry is selection, not completion.

**01:49 submission:** the separately funded EP21 parent is **47357234**.
Independent source review passed; four focused reader methods precede the
bounded metadata continuation in an explicit one-CPU Slurm step. Its outputs
will be `runs/source_views_and_timing/column_stripe_job_47357234/` for checks
and the separate direct-child `column_stripe_job_47357234_metadata/` for the
fresh report. No additional EP19 CPU charge and no completion claim at submission.

## 2026-10-11 01:36 Pacific — source_views_and_timing — stream metadata complete with nine unresolved bindings

Job **47355564 completed, exit0:0**. The final
[retry02 report](runs/source_views_and_timing/job_47355564_stream_layout_retry02/development_stream_layout.json)
contains all155 development runs and all fifteen development participants.
Of these,152 had both declared source files checked; the three unresolved
header references received no binary or marker access. All152 checked binary
sizes are integral FLOAT32 sample frames, with141 runs at39 channels/250Hz
and11 at67 channels/1,000Hz. Their nominal stored sample durations range
from388.46 to675.98seconds and sum to87,058.74seconds; this is not certified
continuous elapsed acquisition time or eligible stillness exposure.

All152 inspected marker files contain one NewSegment at sample1 with a
syntactically valid timestamp whose timezone is unspecified. This records
declared starts, not proof of gap-free recording. Six of these markers have
unverified `DataFile` binding: sub-03/session02/runs0009–0011,
sub-15/session02/run0016, sub-20/session01/run0003, and
sub-21/session02/run0009. Their marker facts are recorded but are not accepted
as aligned segment evidence for the corresponding EEG until that binding is
resolved. Together with the three original-header conflicts, **nine runs
retain unresolved metadata**. None was scientifically excluded or rebound.

The final extraction took0.139186seconds and exited zero. Both repair paths
and intermediate reports remain preserved in the
[allocation evidence directory](runs/source_views_and_timing/job_47355564_stream_layout_checks/).
The final three stream-layout tests passed in0.065seconds. The four previously
unrun suites also passed: completed-return6/0.020s, risk-stream6/0.072s,
forecast-label5/0.003s, and factorial-uncertainty4/1.137s. These are **24 unique
software checks**, not24 empirical trials or scientific recoveries. Changed
EP19 ASTRA schema and semantic validation passed; unchanged validation was
not repeated after the signature-only code repair. The EP02 companion's two
checks and ASTRA also passed in this parent, with evidence in its own episode.

Scheduler accounting at the final observation reports start01:28:59Pacific,
end01:36:17Pacific, elapsed438seconds, one allocated CPU, TotalCPU16.113seconds,
step MaxRSS100,664K. Charge the full **438/3600 = 0.121666667 allocated CPU-hours
once to EP19**, including the EP02 companion; do not add the extern record or
a second EP02 charge. The one-third-hour reservation is released. The attended
shell exited normally; no process, job, renewal or successor remains active.

Next source work needs authenticated mapping for the nine affected runs,
physical accelerometer conversion/axis calibration and acquisition/filter
delay evidence, alongside the unresolved WAY clock-origin/calibration facts.
No signal sample, stimulus label, held-out recording or empirical forecast was
opened. Calibration, the full below300ms timing bound and scoring readiness
remain false. Do not rerun completed extraction or fixtures without new
evidence or a concrete defect.

## 2026-10-11 01:26 Pacific — source_views_and_timing — development stream layout selected

Select a bounded metadata report for the existing 155 self-paced development
runs, using their completed header/view declarations, binary file sizes and
only `NewSegment` marker positions/timestamps. Reuse the completed acquisition
header extraction and its passing checks. Before metadata access, require the
exact 155-run declaration and all fifteen development people; do not inspect
held-out paths, archives, WAY arrays or signal samples. Do not emit or use
stimulus/event descriptions. This newly selects NewSegment metadata access;
the earlier header-only extraction did not authorize marker access.

Implement file-size/sample-count arithmetic only for source-documented binary
formats and preserve unsupported cases. Segment markers describe recorded
boundaries, not proof that every acquisition gap is known. The report can
resolve run dimensions and declared segmentation without authenticating
accelerometer units, historical filter/acquisition delays, valid-run eligibility
or the WAY clock origin. It does not freeze onset thresholds, change the
238-versus-240 source-run eligibility issue, open EEG scores or qualify the
strictly-below-300-ms timing requirement.

One source worker owns the extractor and focused fixtures; root owns review,
records and compute. Implementation is selected; no process, job or actual
metadata read has started. Existing budget history remains; no search clock
is reset. A bounded joint execution will be recorded separately if ready.

Select one attended integration allocation, 1 CPU / 7 GiB / 20 minutes,
combining the final source review, the three new stream-layout fixtures,
changed EP19 ASTRA validation and the selected 155-run metadata extraction.
EP02's two new reader fixtures and changed ASTRA validation may run serially
in the same session with their own evidence directory; no actual EP02 pack
access. Root owns the PTY and all tests. No standalone fixture batch jobs,
background service, renewal or synthetic research sweep is selected. Do not
repeat the completed acquisition extraction or unrelated passing suites.
Reserve at most one third allocated CPU-hour, charged once to EP19, without
duplicating a charge to EP02. Record the actual job/time after admission and
release unused allocation promptly after this attended work finishes.

Submitted attended job **47355564** (`ep19-layout`); the initial scheduler
response is pending for resources. No compute host has yet been verified and
no tests or metadata extraction have run. Do not submit a duplicate.

At08:29UTC, root verified `sh04-04n22.int`, parent47355564, one CPU and
7,168MiB in the attended shell. EP02's two companion checks subsequently
passed after a local PYTHONPATH correction; its changed ASTRA also passed
after removing an unsupported shell placeholder. Those diagnostics belong
to EP02's own run directory. The exact EP19 metadata scope remains unchanged.

The arithmetic and marker interpretation follow the vendor's
[BVCDF specification](https://www.fieldtriptoolbox.org/assets/pdf/BrainVisionCoreFileFormat_1.0_2018-08-02.pdf)
and [marker documentation](https://pressrelease.brainproducts.com/markers/):
the supported explicit formats are multiplexed binary FLOAT32 (four bytes)
or INT16 (two bytes), and sample positions are one-based. The optional
NewSegment timestamp has twenty digits through microseconds; no time zone
is assigned here. These format facts do not authenticate the historical
sensor calibration or acquisition response. Unsupported metadata remains
unresolved rather than being silently converted.

The three new fixtures passed in0.047seconds and changed ASTRA schema and
semantic validation passed. Independent source review also passed. The first
empirical metadata attempt then refused before any development file stat/open:
three saved headers contain old internal filenames that do not match their
materialized triplets. The bounded diagnostic reads only the two previously
completed metadata reports. Evidence is preserved in
`runs/source_views_and_timing/job_47355564_stream_layout_checks/`, including
`extraction.txt` (exit2) and `triplet_reference_diagnostic.json`.

The mismatches are sub-09/session02/run0009 (marker reference),
sub-18/session02/run0006 (data reference), and sub-21/session01/run0003
(marker reference). Do not infer aliases, edit source headers or drop these
runs from the cohort. Correct the extractor to retain all155 declared rows
and mark those three unresolved without opening either of their source files.
Continue metadata inspection only for the152 triplets whose references bind.
This implements the already-selected unsupported-metadata reporting rule; it
does not amend scientific eligibility. Update the existing binding fixture,
review the repair independently and rerun only the affected three-method suite
before the bounded retry. Preserve the original attempt and checks.

Retry01 completed at08:33:58UTC in0.422666seconds, retaining155 rows and
checking152 binary/marker pairs. All152 binary sizes divide exactly into
declared FLOAT32 frames:141 runs at39 channels/250Hz and11 at67 channels/
1,000Hz. However, all markers remained unresolved because the parser accepted
`Brain Vision` signatures while the inspected Recorder first line uses
`BrainVision Data Exchange Marker File Version 1.0`. The diagnosis read only
the first160-byte-bounded line of the first reference-bound development marker;
no event was printed. Preserve retry01's report and logs as intermediate evidence.

Select the narrow signature spelling correction after primary-reader/source
verification, preserving version1 and the existing section/field rules.
Extend the existing marker fixture, obtain source review, and rerun the
affected three-method suite before retry02. No data roles, original-reference
handling, scientific thresholds, event access or metadata scope change.

The maintained [MNE BrainVision reader](https://github.com/mne-tools/mne-python/blob/main/mne/io/brainvision/brainvision.py#L407-L421)
explicitly accepts the `Brain ?Vision` spelling variant. The repair adds only
the observed exact version1 string; it does not copy the broader version
acceptance of that reader. Independent source review passed. The same three
fixture methods passed in0.065seconds and retry02 completed all155 metadata
rows. The current report is in
`runs/source_views_and_timing/job_47355564_stream_layout_retry02/`;
the two earlier attempts remain preserved. No further source reread is selected.

### Same-allocation completion of already implemented component checks

With the metadata extraction complete and more than ten minutes remaining,
select the four existing, previously unrun suites from the current ASTRA
program: `test_way_completed_return.py` (6), `test_way_risk_stream.py` (6),
`test_forecast_labels.py` (5) and `test_forecast_uncertainty.py` (4). Run them
serially in this same attended parent, at most30seconds per suite, using
array/file-free fixtures and preserving module paths. These21 checks were
already implemented and independently source-reviewed; no new fixtures,
scientific simulations, source reads or model fits are added. Their analysis
owners are `source_views_and_timing`, `past_only_controls` and
`joint_final_evaluation`; use the existing named recipes and preserve prior
checks. No new job, reservation or duplicate charge. Save evidence alongside
this allocation's other checks and stop on any diagnosed failure requiring
more than the existing allowance.

## 2026-10-10 22:44 Pacific — non_neural_selection — nine refit and scope checks passed

EP21 parent **47329203 completed, exit 0:0**. Its companion ran all five
`test_baseline_final_refit` methods and the four named non-neural, boosted,
temporal and ensemble scope/offset checks: **nine methods passed in 0.263
seconds**. Evidence is in `runs/scoring_composition_checks_job_47329203/`.
The current EP19 ASTRA program also passed schema and semantic validation.
This includes the program's latest uncertainty entry, but does not execute
that reducer's four still-unrun tests or other later suites outside the tail.

Charge the allocation once to EP21, with no duplicate EP19 spending. These
fixtures qualify the final-refit and training-only offset interfaces; they are
not empirical forecasts, physical-unit calibration or source-timing validation.
The authentic timing/calibration and source-view integration requirements remain.
No new EP19 empirical fit, evaluation, job or automatic successor is selected.

## 2026-10-10 — joint_final_evaluation selects registered factorial-equivalence reducer

Implement the missing array-only uncertainty reducer for the policy's fixed
factorial-equivalence family: two held-out sources, five reference families,
two horizons and two endpoints (Q11 minus original full reference, and
Q11 minus Q10 minus Q01 plus Q00). Preserve all forty endpoints and complete
registered cohorts: twelve WAY participants and eight self-paced participants.
Participant point scores come from an admitted caller; this library reads no
source, prediction or held-out file and grants no evaluation access.

Use 10,000 whole-participant bootstrap draws independently within each source,
sharing each source's draws across its endpoints. Center the bootstrap means
on the observed equal-participant means and use one maximum absolute statistic
over all forty endpoints for simultaneous two-sided 95% intervals. Make the
random seed an explicit caller input to be frozen before evaluation; do not
choose it from outcomes. Use the upper empirical quantile (NumPy method
`higher`) as the prospective finite-bootstrap implementation convention.
Accept only externally frozen positive equivalence margins; absent margins
remain unresolved. Interval containment alone does not establish the full
temporal explanation, which has additional registered conditions.

One worker owns the reducer and focused tests, another reviews its statistical
scope and code, and root owns integration and records. No empirical read,
threshold choice, test execution or job is selected. New tests stay outside
EP21 job47329203's unchanged companion tail. The existing point scorer and
passing checks remain unchanged; source qualification and final access are
still unresolved.

Implementation and independent source review are complete, with no concrete
defect found. The reducer consumes explicit participant/family/horizon/condition
axes, refuses incomplete cohorts or nonfinite scores, and forms paired
participant contrasts before resampling. Two PCG64 streams are spawned from
the supplied seed in WAY/self-paced order. Processing 256 draws at a time
bounds intermediate arrays while retaining all 10,000 joint maxima. Results
include forty named intervals, separate full-reference and interaction
equivalence calls, and an explicit statement that the full temporal explanation
was not evaluated. Non-containment means equivalence is not established, not
that non-equivalence has been demonstrated.

The four focused methods independently compare the scalar bootstrap and upper
order statistic, exercise shared within-source draws and the joint-source
maximum, preserve missing/exact-boundary margin semantics, and reject incomplete
or misbound panels. All four remain **unrun**. The changed ASTRA program records
the component and a bounded future test recipe; its validation is pending.
No new suite was added to47329203, and no empirical input, forecast, inference,
resource reservation or job was created by this work.

## 2026-10-09 — public wrist-unit clue narrows the source question

Independent source inspection and coordinator verification of the pinned
provider utility identified a downstream wrist-position unit clue, now linked
with its inference limits in the
[source handoff](source_timing_handoff.md#october-9-downstream-wrist-position-unit-clue).
The note narrows the original export/calibration question; it selects no
unit conversion, clock correction, detector change or empirical access.
The existing two-second discrepancy and full timing/calibration prerequisites
remain unresolved. No test or job was launched.

## 2026-10-09 — source_views_and_timing selects WAY streaming risk integration

The completed-return predicate and forecast-label projection are implemented,
but the existing conditional detector is not the required causal state
reducer. Select a separate WAY-only scalar stream implementation using the
existing starting recipe: native 500-Hz ticks, 25-tick decision spacing,
20-tick confirmation, and 250 preceding quiet ticks for requalification.
Inputs are caller-qualified eight-path causal summaries, fixed low/high
thresholds, authenticated start/end events, exact-start ReturnSupport and
explicit reset identity. This does not estimate thresholds, parse raw cues,
convert physical units or qualify source timing; the starting recipe's
empirical/freeze status and the old pilot's readiness flags remain unchanged.

First arming requires supported return at the start itself. Failed/unknown
start support cannot accumulate into a later first arming in that cue.
A current high crossing suspends a same-tick grid prediction immediately.
Track all primary paths while pending AND during fresh stillness
requalification after rejected crossings. The old conditional pilot skipped
new crossings during that latter interval, which could omit an onset from
the labels of earlier at-risk decisions. Any accepted reach consumes the
cue's single reach; quiet holding cannot rearm it. A later valid reach requires
an observed end, a new start and supported return.

Preserve high-at-or-above initiation, below-low rejection, all-at-or-below-low
quietness and earliest accepted crossing time. Earlier/tied unresolved
crossings conservatively delay establishing the first onset's identity;
known_tick records that establishment, not a replacement onset timestamp.
No different per-path confirmation times are invented. Requalification uses
fresh preceding stillness, not pre-crossing history.
The exact fresh window is [t-250,t): when reject-all occurs at r, erase earlier
quiet history but allow the current r sample, if all quiet, to be its first
sample. Earliest rearming is then r+250, with r itself still suspended; a new
crossing at the rearming tick again suspends that decision immediately.

Immutable snapshots retain
the original grid decision, cue scope, pending facts and as-of label evidence;
later confirmations/rejections cannot restore suspended rows or revise saved
prefix evidence. Once the first onset is established, later pending paths do
not truncate otherwise valid follow-up merely because prediction is suspended.

Cue end and a sensor/history gap close the admitted evidence interval before
that tick's confirmation. Missing ticks are not bridged; after a gap use a new
reset identity. Simultaneous start/end or a new start without the preceding
cue's observed end are refused, not interpreted as an invented return phase.
A small join to forecast_labels binds the same recording/reset AND cue scope
and uses only the evidence snapshot's own exclusive endpoint; it cannot reuse
later adjudication for an earlier prefix cutoff. Self-paced detector rules,
source qualification and empirical label/forecast execution remain separate.

One worker owns code/way_risk_stream.py, another its focused sequential-trace
checks, and a third reviews the source. Root owns records and compute. No raw
payload, replay, import, test, allocation or scientific claim is selected.
New checks stay outside the unchanged EP21 companion; passing numerical cores
are reused. The full strictly-below-300-ms timing bound, cue/physical calibration
and eligible source preparation remain unresolved.

Implementation is complete in [way_risk_stream.py](code/way_risk_stream.py),
with six focused methods in
[test_way_risk_stream.py](tests/test_way_risk_stream.py). The separate test
owner, independent reviewer and coordinator completed bounded source review
without finding a concrete defect. Checks cover immutable identical prefixes
with diverging futures, competing paths, failed/unknown starts, exact
requalification and movement during it, one-reach/end/gap boundaries, and
event labels with separately advancing follow-up coverage. These are software
fixtures, not physiological simulations or empirical detector qualification.

The implementation also refuses supplied return support that contradicts a
previously observed reach/end interval. A declared history gap takes precedence
over a simultaneous end; both close evidence before current-tick confirmation.
Stream tokens bind snapshot lineage only and do not provide data authorization
or reader isolation. The reducer retains at most eight path records and no
snapshot history; callers own snapshot retention. Per-step work is bounded by
the eight paths. No signal reader, calibration or empirical prefix selection
is provided.

All six methods and changed ASTRA validation remain **unrun**. The future
recipe is one CPU / 1 GiB, 30 seconds plus five seconds termination grace,
inside a substantive allocation. No separate short job has been selected.
Short scalar traces dominate fixture work; numerical-library imports are
expected to dominate memory, but runtime and RSS remain unmeasured. EP19 now
has 26 pending methods: nine final-refit/scope checks in the unchanged EP21
proposal and seventeen completed-return/label/stream checks outside it. Source
timing/calibration, cue interpretation, the self-paced detector, source-to-label
integration, first-32-movement prefix selection and the full timing bound
remain unresolved. No empirical labels, forecast, scientific finding or Slurm
job were produced by this implementation.

## 2026-10-09 18:50 Pacific — past_only_controls selects forecast-label projection

Select the missing pure, per-decision label projection from the registered
50-ms grid and 19-category censoring rules. A caller supplies an immutable
original risk decision, the earliest accepted onset and the time its identity
became known (including resolution of earlier competing crossings), and
continuous admitted onset-status follow-up within the same recording/reset
identity. The helper does not read signals or construct/qualify those facts.
Use integer ticks with an explicitly supplied timebase divisible by 20 and
grid origin; do not round timestamps or repair clocks. Event bins are
left-closed/right-open through 900 ms; category 18 means no onset in [0,900),
and -1 is censoring. Only complete 50-ms bins contribute negative evidence.
Keep zero-information at-risk decisions, and retain non-at-risk states without
labels; later adjudication cannot restore suspended decisions.

Observation, administrative and optional calibration-prefix endpoints are
exclusive. The caller must supply the earliest unresolved onset-status bound
as of the requested endpoint, not the later full-record resolution. An onset
not known before that endpoint cannot enter the label: its unresolved bound
censors negative evidence, and a missing required bound is refused. Earlier
unresolved crossings can precede the eventually accepted crossing. Decisions
at/after an explicit prefix endpoint are refused. Selecting the first 32
complete movements and constructing the as-of evidence remain upstream.
Scoring follow-up coverage is separate from training censoring bins, including
for an observed early event; it never inherits complete horizon eligibility
from the event alone. No new scientific threshold or empirical label is chosen.

Specifically, observed_stop_tick is a hard admitted source/evidence boundary;
its minimum with label_stop_tick is the knowledge cut. unknown_from_tick
separately clips continuous status coverage. Compare onset known_tick with
the knowledge cut, not that earlier coverage stop. An onset at 300 ms known
by 395 ms within an admitted 400-ms interval survives later unknown status
beginning at 390 ms: category 6, seven complete coverage bins, no 600-ms
scoring eligibility. Unknown status at/before the accepted crossing instead
censors it. This distinction does not extend admitted evidence access.

One worker owns code/forecast_labels.py, a second owns
tests/test_forecast_labels.py, and the third reviews source. Root owns these
records and any compute selection. No allocation, source reader, empirical
replay, physiological simulation or new claim is selected. Focused checks
will remain unrun until coordinated substantive compute is available and are
outside the unchanged EP21 companion proposal. The existing conditional
detector's scientific-label and online-risk qualification flags stay false.

Implementation is complete in [forecast_labels.py](code/forecast_labels.py),
with five focused methods in
[test_forecast_labels.py](tests/test_forecast_labels.py). Root and independent
source reviews found no concrete defect; the separate test owner also checked
the final API. Checks cover exact interval edges and integer clocks, meaningful
integration with the existing censoring loss and horizon scorer, incomplete
follow-up after an observed event, earlier unresolved crossings and prefix
knowledge limits, unchanged non-at-risk decisions and reset identity binding.
The helper uses scalar arithmetic, allocates no data arrays and performs no I/O.

All five checks and changed ASTRA validation remain **unrun**. The future
recipe is one CPU / 1 GiB, 30 seconds plus five seconds termination grace,
within a substantive allocation; no separate short job or allocation has
been selected. The test arrays are tiny; existing NumPy/SciPy imports dominate
the expected footprint, and actual runtime/RSS remain unmeasured. EP19 now
has 20 pending methods: nine final-refit/scope checks in the unchanged EP21
proposal, and eleven completed-return/label checks outside it. Authentic
clocks/trim, sensor calibration/cue interpretation, a qualified full risk/onset
state machine, source-to-label integration, first-32-movement prefix selection
and the strictly-below-300-ms timing bound remain unresolved. No empirical
labels, forecast, new scientific finding or Slurm job were produced.

## 2026-10-09 18:35 Pacific — source_views_and_timing selects completed-return implementation

Select the missing array-only WAY completed-return predicate from the already
adopted component_details section 1.1. This is a stateless evaluation exactly
at an authenticated start cue on the native 500-Hz grid: the 250 preceding
ticks, excluding the start sample, must satisfy both 0.15-m digit clearances,
all eight existing low-threshold paths and their causal histories, supported
object position/motion and each released plate's separate XYZ tolerances.
References, tolerances, causal summaries, physical-unit conversion and cue
interpretation are supplied facts; no calibration, filter or threshold is
estimated here. Missing prerequisites or nonfinite required support give
unknown, with priority over a measurable veto. A complete observed violation
gives not_supported; a pass only gives supported for this predicate.

At genuine initial series entry no earlier end is invented. After a reach,
require prior_reach < authenticated_end < window_first < current_start.
The caller explicitly supplies phase and uninterrupted history boundaries;
a gap cannot silently become initial entry, and the full window must lie in
the current series and uninterrupted history. Evaluation must equal the
start tick, so later quietness cannot restore an earlier decision. One-reach
per cue, onset detection and the armed/pending state machine remain outside
this helper; the conditional detector and its readiness flags stay unchanged.

One worker owns code/way_completed_return.py, a second owns its focused
contract checks, and an independent reviewer checks the implementation. Root
owns the records and any compute selection. No source payload, calibration,
empirical replay, forecasting, allocation or new scientific claim is selected.
The checks and changed ASTRA validation remain unrun, outside the unchanged
pending EP21 companion. Authentic clocks/trim, calibration/cue interpretation
and the strictly-below-300-ms timing bound remain unresolved.

The worker completed [way_completed_return.py](code/way_completed_return.py)
and the separate check owner prepared six focused methods in
[test_way_completed_return.py](tests/test_way_completed_return.py).
Independent source review found no blocking mismatch with section 1.1.
Root reviewed the final source, including two focused corrections: off-start
calls return before touching sensor arrays (unknown before the start,
not_supported after it), and unrepresentable computed distances/departures
remain unknown. Unknown precedence applies to evaluations at the start.
All factual declarations default to missing/false; source units, calibration
and cue meanings cannot be inferred by constructing the configuration.

The six methods and changed ASTRA validation are **unrun**. The future recipe
uses one CPU / 1 GiB, a 30-second timeout plus five-second termination grace,
inside a substantive allocation; no separate short job is selected. The small
400-row fixture and intermediates are estimated below 1 MiB, excluding the
Python/NumPy runtime; actual time and memory have not been measured. All
numerical threads are limited to one. There are now 15 pending EP19 methods
(the prior nine final-refit/scope methods plus these six), with these six
outside the unchanged EP21 companion. No source recordings, timing correction,
detector labels, empirical fit or scientific result were produced.

## 2026-10-09 17:07 Pacific — non_neural_selection selects missing final-refit and offset composition

The qualified development selector and actual ensemble-cost callback do not
yet provide the registered final baseline refits or frozen EEG offsets.
Select that missing array-only production composition under the existing
source-specific fitting rules: WAY person-specific series1–7 final fits and
the self-paced final group fit on all15 development people, using the already
selected recipe and three-seed convention. Preserve fitting-versus-evaluation
roles and prohibit held-out labels entering baseline fitting. Reuse existing
backends and selection/calibration arithmetic. The offset convention is
clarified prospectively below.

A worker owns only the new refit/offset module and focused checks. Resolve
any underdetermined offset convention explicitly before implementation rather
than assuming it. No empirical arrays, held-out series/people, final model
publication, source-reader release or allocation is selected. Clock/trim,
units, timing uncertainty and participant support remain empirical gates.
Root owns records and any later compute selection. Existing passed suites
remain reusable; do not rerun them or launch a surrogate simulation campaign.

The initial planning interpretation proposed nested OOF offsets for EEG
training. Source review of component_details and the registry found no such
requirement: OOF refers to development validation/selection, while fusion
explicitly uses a frozen training-fit baseline. Select same-fitting-cohort
baseline predictions as training offsets and untouched-fold predictions for
evaluation. Both are the log of the same uncalibrated three-seed probability
mean, shared across all matched EEG conditions. Complete matched models receive
their own existing self-paced32-movement calibration afterward. Training
offsets are not called OOF or held-out predictions. Development scores following
baseline-recipe selection are adaptive selection evidence; they cannot replace
the untouched final series/people. This is a within-grammar implementation
choice, not a new nested estimator or permission to open final outcomes.

BR's read-only review router returned directive `ext_review_dir_30aaa3de8bd5`
at00:08:44 UTC: criteria about within-fold fitted transforms, leakage,
test-set selection and dependent folds. BR did not read local artifacts or
return a substantive verdict. Root and the worker applied those relevant
criteria to the source above; no canonical review/acceptance was claimed or
submitted. Permit the worker's narrow final-refit scope extension in the
four existing baseline/scope modules, with default development-only behavior
retained for other consumers. Tests and actual fits remain unselected.

The worker completed `code/baseline_final_refit.py`, five focused methods in
`tests/test_baseline_final_refit.py`, and the four coordinated scope/backend
edits. Root independently read the new composition, tests and scope changes;
no concrete source defect was found. The final API requires all12 WAY people
with series1–7 separately (36 seed fits) or all15 self-paced development people
jointly (three seed fits). A separate development-fitting-fold helper excludes
its outer validation groups. Outputs retain actual models and read-only
training log offsets; source qualification and empirical freeze are not claimed.

Five new methods plus four affected existing scope methods and changed ASTRA
validation remain **unrun**. The unactivated EP21 measurement proposal includes
this175+5-second companion; existing numerical backend and selector suites are
not repeated. No empirical fit or additional allocation has been selected.

## 2026-10-09 — non_neural_selection selects cost-callback companion in 47133348

After the EP20 full-preflight process returned its terminal timeout, select
the three prepared `test_baseline_ensemble_cost.py` checks with a 115-second
timeout plus five-second grace, then changed ASTRA validation under 30 seconds,
serially in the existing attended parent 47133348. Its shell was verified on
sh04-04n22.int with one CPU, 7 GiB and approximately fifteen minutes remaining.
Reuse the proven Python3.12.1/NumPy1.26.3/SciPy1.12.0/scikit-learn1.5.1/
PyTorch2.4.1 module environment in a subshell, all numerical threads one, no GPU.
Preserve evidence under `runs/ensemble_cost_checks_job_47133348/`.

No empirical payload, numerical fitting or scientific trial is selected.
Existing backend and driver checks remain valid. The shared parent stays charged
once to EP20 within its existing reservation; EP19 adds no allocation or duplicate
charge. Source qualification, production resource admission and authentic
timing/calibration remain unresolved. Results are unobserved at selection.

The serial companion returned test and ASTRA exits zero at 14:56 Pacific.
All **three checks passed in 0.486 seconds**, including the native tree-storage
fixture. See [tests](runs/ensemble_cost_checks_job_47133348/tests.txt),
[resources](runs/ensemble_cost_checks_job_47133348/resources.txt),
[status](runs/ensemble_cost_checks_job_47133348/status.txt) and
[ASTRA validation](runs/ensemble_cost_checks_job_47133348/astra_validation.json).
This closes the callback's unrun status and adds no failed execution event.
No empirical baseline fit or latency measurement occurred. Root closes the
shared parent after the selected EP20/EP19 work; actual accounting belongs once
to EP20 and is recorded in that episode's log.

Parent 47133348 subsequently completed at 14:57:11 Pacific and PTY 2073 closed.
Its 6,567 allocated CPU-seconds replace EP20's parent reservation once; this
EP19 companion adds no separate charge or outstanding reservation.

## 2026-10-09 — non_neural_selection implements actual ensemble cost measurement

Selected the missing cost callback consumed by the existing complete development
baseline driver. It measures the actual fitted three-seed ensemble on explicit,
label-free probes from its fitting fold; the same probes and order must be used
across all nine recipes. This is implementation work only: no source payload,
empirical fitting, validation outcome, new allocation or method freeze is opened.

The operational counting convention includes all fitted seed coefficients and
offsets plus fitted scaler means/scales. For boosted trees it counts each learned
internal feature choice and threshold and each leaf score, excluding topology
bookkeeping. The timing convention is the observed nearest-rank 95th percentile
of individual three-seed prediction plus probability-mean calls on prepared
features, under explicitly fixed warmup and repeat counts. Resets, raw feature
construction, calibration and I/O are excluded and disclosed. This does not
settle the separate end-to-end latency requirement or any scientific margin.
The existing driver retains every fold cost and takes separate maxima across
the full required fold set; its scoring and selection rules remain unchanged.

One agent owns the new callback, a second reviews its data-role and backend
integration, and a third prepares at most three focused checks without numerical
model fitting. Root owns shared records and compute. The EP20 parent 47133348
is occupied by its selected full-plan preflight; no companion command has been
sent to that busy shell. New code checks and changed ASTRA validation remain
unrun. Authentic timing/calibration, qualified source inputs and production
resource admission still prevent empirical baseline selection.

The [callback](code/baseline_ensemble_cost.py) is now implemented and both root
and independent source review found no blocking defect. Existing drivers and
backends are unchanged. The caller passes `BaselineEnsembleCostMeasurer` and
its `parameter_count_basis`/`latency_basis` to the existing driver. Owned probes
remain fixed across recipes; successful records retain per-seed count breakdowns
and every timed duration. The implementation resets state after success or
failure and does not return predictions. This source review is not runtime
validation. Actual CPU/thread settings, fitting-only empirical probes and pass
counts must be fixed before measurement. The ASTRA program now includes the
callback and its separate check recipe; schema/semantic validation is unrun.

The [three check methods](tests/test_baseline_ensemble_cost.py) are now prepared
and source-reviewed, still **UNRUN**. They populate existing backend storage
directly without fitting, including native small tree objects and frozen TCN
parameters, and instrument prediction/clock calls for the timing and reset
checks. The native scikit-learn tree-state fixture remains an unexecuted runtime
compatibility risk. The prepared command has a 115-second timeout plus five-second
termination grace, within the existing 120-second companion envelope. No new
process, scheduler job or parent resource charge has been selected.

## 2026-10-09 — non_neural_selection selects prepared baseline-driver companion

Select the existing five `test_development_baseline_driver.py` methods under
a120-second fail-fast cap, followed by30-second ASTRA validation, as a serial
companion in the coordinator's next EP20 full-dimensional cost allocation.
The parent is1CPU,7GiB,30minutes and charged once toEP20; EP19 adds no parent
reservation or duplicated cost. No empirical signal/outcome payload is read.

Reuse the documented successful EP19 module environment in an isolated shell:
Python3.12.1, NumPy1.26.3, SciPy1.12.0, scikit-learn1.5.1 and PyTorch2.4.1,
all numerical threads1 and no GPU. PyTorch is imported by an existing backend
even though the fixtures instrument fitting; the NumPy-only EP20 environment
is insufficient. No new software install or backend qualification is needed.
Root alone owns the compute PTY and scheduler. These checks are still unrun;
no job ID or empirical forecast is implied by this selection.

The parent submission subsequently returned **47131800**, pending resources at
12:42Pacific. Its eventual serial companion evidence will be retained under
`runs/baseline_driver_checks_job_47131800/`; checks remain unrun until the
coordinator verifies the actual compute shell and executes them.

At12:46Pacific the serial companion returned test and ASTRA exits0 in the
verified parent47131800 on`sh04-04n22.int`. All **five checks passed in1.135s**;
changed schema and semantic validation passed. See
[checks](runs/baseline_driver_checks_job_47131800/tests.txt),
[resources](runs/baseline_driver_checks_job_47131800/resources.txt) and
[ASTRA validation](runs/baseline_driver_checks_job_47131800/astra_validation.json).
The existing numerical backend results are reused. These close this driver's
unrun composition-check status; source timing/calibration, qualified production
rows/guards, actual model-cost measurement and empirical admission remain.
The parent remains attended and is charged once toEP20; no separate EP19 charge
or real forecasting result is implied.

Parent 47131800 subsequently completed at 12:57:09 Pacific and its compute
PTY closed. Its actual resource charge is recorded once in EP20's existing
execution log; these passing EP19 results need no rerun.

## 2026-10-09 06:10 Pacific — non_neural_selection complete grouped-development baseline driver implemented

Added [development_baseline_driver.py](code/development_baseline_driver.py)
and [five focused composition checks](tests/test_development_baseline_driver.py).
The driver connects the existing nine non-neural recipes, three fixed seeds,
canonical folds, probability ensemble, four-parameter calibration and scorer.
It preserves all 12 WAY people and 72 participant/series folds (1,944 fits),
or all 15 self-paced development people in five group folds (135 fits).
WAY OOF rows are pooled within person before equal-person selection. Self-paced
calibrated32 is the fixed primary procedure; no-label predictions share the
identical guarded suffix. No source clock, sensor scale, onset, prefix endpoint
or guard was inferred, and no signal or outcome payload was opened.

Root and an independent agent reviewed the source. The review identified that
boosted-tree size depends on the actual fit: a prior recipe-level numeric table
would not count the new fitted model. The implementation instead takes a
prospectively specified measurement callable for each actual three-seed
ensemble before validation, retaining all fold costs and using separate maxima
for count and latency. Resetting after that measurement preserves inference
state. The [component specification](component_details.md) records this
operational convention and the explicit upstream measurement responsibility.
Root also corrected a fixture's self-paced count from405 to135; production
already returned the correct nine-times-three-times-five total.

All five new checks remain **UNRUN**. They use instrumented existing interfaces
to verify complete grouping, unequal-series pooling, probability averaging,
prefix-only calibration, matched suffixes, follow-up support and tie-breaking;
they do not fit another simulation cohort or repeat the passing backend suites.
The changed [ASTRA program](astra/v0.0.14/astra.yaml) records the callable and a
120-second check recipe; schema/semantic validation also remains unrun.
No new scheduler job, current-job modification, resource reservation or
empirical forecast was selected. The live EP21 assembly recipe is unchanged.

Next: execute the five checks and changed ASTRA validation in the next suitable
coordinator-owned substantive allocation. Actual forecasting still requires
authentic timing/calibration, prepared natural-risk rows and prefix/guard
bindings, an implemented comparable cost measurement and production resource
admission. Final refitting, EEG offsets and the remaining controls are separate
unfinished work. The driver returns no partial-success ranking or method freeze.

## 2026-10-09 01:28 Pacific — past_only_controls surrogate feasibility review corrects covariance inference

A targeted contract/backend review established that the tangent model uses
separate older and recent-slot covariances. The earlier working assumption
that a whole-window coherent phase transform necessarily leaves this model's
features unchanged was too strong. Exact full-window covariance preservation
does not imply preservation after these two temporal crops. The
[component note](component_details.md) records the correction, its direct
`classical_forecast_backend.py::_slots` evidence and the primary mathematical
reference, [Prichard and Theiler1994, Section II](https://arxiv.org/pdf/comp-gas/9405002).
No empirical feature or response array was read to reach this conclusion.

At08:23:32UTC BR returned external advisory directive
`ext_review_dir_7c82e8423f66` for a self-contained question about causal all-pass
and delayed EEG controls. It returned generic scientific-review criteria with
`tailored_checks=[]`; it did not read Oak artifacts, review a fitted model,
issue a verdict, launch a run or provide canonical approval. The local review
used its relevant null/estimand, temporal-boundary, seed and leakage criteria.

The registered wording leaves phase construction open, but a causal all-pass
transform alone does not establish an alignment null. Carrying IIR state from
the32-event calibration prefix conflicts with the existing no-overlap rule;
resetting it avoids that route but changes finite startup behavior and still
needs a supported null interpretation. Delays require common row eligibility
and explicit no-wrap boundaries. The existing shift/null check and every
mandatory surrogate remain required. No all-pass replacement, changed row
mask, protocol amendment or new computation is adopted. Exact construction
remains an internal method decision, separate from missing source timing and
sensor-calibration facts; previous completed model/scoring checks remain valid.

## 2026-10-09 00:42 Pacific — past_only_controls all five horizon scoring checks and ASTRA pass

The [closed scoring log](runs/scoring_checks_job_47086577/tests.txt) records
**five checks passed in 0.029 seconds**, covering horizon boundaries, extreme
probabilities, full-follow-up eligibility, unequal series sizes with equal
participant weighting, complete scope/alignment and participant-specific
surrogate comparisons. Changed ASTRA schema and semantic validation passed;
both exits are zero in the accompanying status. Reuse these results.

The attended EP17 parent47086577 is still active; its entire actual cost will
be charged once to EP17 on closure. These results complete this scoring
component's checks, not source timing/calibration, production admission or an
empirical forecast. No candidate, held-out prediction or scientific decision
was computed.

## 2026-10-09 00:37 Pacific — past_only_controls horizon scoring implemented and checks selected

`code/forecast_scoring.py` now converts aligned 19-category log probabilities
to the registered near, strict and longer-horizon binary log scores, retaining
natural prevalence and full follow-up eligibility. It pools rows within each
person, then weights the complete registered participant cohort equally.
Matched real, non-neural, zero and every required surrogate condition share
exact row identities. The strongest surrogate is selected from complete
participant means separately at each horizon. This returns continuous scores;
it supplies no margins, bootstrap, decision, reader or empirical admission.

Select its **five new focused checks**, bounded to 60 seconds, and changed
ASTRA validation as serial children of attended EP17 allocation **47086577**
(one CPU, 7 GiB, two-hour parent). All numerical threads remain one; charge
the parent once to EP17. Tests remain unrun at this selection. Existing passing
model and ensemble checks are reused, and no source or held-out value is opened.

## 2026-10-09 00:27 Pacific — past_only_controls all five ensemble checks and ASTRA pass

The [closed test log](runs/ensemble_checks_job_47085635/tests.txt) records all
**five ensemble integration checks passed in 0.112 seconds**. The
[status](runs/ensemble_checks_job_47085635/status.txt) has `tests_exit=0` and
`astra_exit=0`; changed ASTRA schema and semantic validation passed. Reuse
these checks and the prior 17 passing backend checks; no repeat is selected.

The coordinator observed shared parent **47085635** terminal at 07:26 UTC:
43 seconds, one CPU, `FAILED` / `2:0`. Its final EP17 private-copy child failed
a directory-mode check before neural access, after these checks and EP20's
complete generation. That independent failure does not invalidate EP19's
closed checks. Charge the entire 43 CPU-seconds once to EP20, without another
EP19 charge. This is implementation evidence; no empirical forecast, source
clock/calibration qualification or reader release is established.

## 2026-10-09 00:19 Pacific — past_only_controls ensemble checks accepted in parent 47085635

EP20 parent **47085635** was accepted at the coordinator's **07:19:42 UTC**
observation. It includes the five selected ensemble checks and changed ASTRA
validation, with results to `runs/ensemble_checks_job_47085635/`. Startup and
test results remain unobserved. The parent is charged once to EP20. No
empirical EP19 data or forecast is included.

## 2026-10-09 00:17 Pacific — past_only_controls ensemble checks selected in EP20 generation parent

The coordinator selects the five new ensemble integration checks and changed
EP19 ASTRA validation as serial, optional children of the forthcoming EP20
direct-surface generation job: one CPU, at most 7 GiB in the parent, 60 seconds
for the checks and 30 seconds for validation. The parent is charged once to
EP20, with no duplicate EP19 charge. No job has yet been accepted for this
selection. Earlier 17 passing EEG backend checks are reused. This opens no
empirical signal, forecast, prefix calibration data or held-out outcome.

## 2026-10-09 00:02 Pacific — past_only_controls three-seed ensemble and calibration integration prepared

The new `code/ensemble_forecast_backend.py` joins all three fixed seed models
and averages their normalized probabilities before the existing four-parameter
calibration. It preserves explicit row alignment, prefix censoring and natural
prevalence, and generates paired no-label/calibrated suffix predictions from
one reset encoder evaluation. Missing or incomplete seed fits remain unresolved.
Classical models now retain their already-validated fitting scope; their
numerical fitting rules and prior results are unchanged. The adapter reuses
the existing ensemble and calibration primitives rather than defining a new
statistical procedure. The exact method is recorded in `component_details.md`.

Coordinator source review corrected reliance on supplied scope metadata and
added equality of the retained bandpower FIRs. Canonical WAY fold IDs already
include participant identity; this interface change is not evidence that prior
fits mixed participants. Five focused checks now include actual tiny classical
fits, probability-versus-logit averaging, calibration order, alignment, reset
behavior and identity fallback. **All five remain unrun**, as does validation
of the changed ASTRA program. Reuse the earlier 17 passing EEG backend checks.
The new checks are bounded to 60 seconds inside a suitable substantive
allocation; no separate short job is submitted by this preparation.

No empirical data were opened and no forecast was run. Source clock/calibration,
production data integration, prefix/guard eligibility and remaining controls
still prevent empirical forecasting; the new adapter does not certify them.

## 2026-10-08 23:32 Pacific — past_only_controls — all seventeen EEG backend checks and ASTRA pass

The [closed test output](runs/gru_component_checks_job_47081340/tests.txt)
records **17 tests passed in 4.885 seconds**: six GRU, five EEG TCN and six
compact EEGNet checks. The retained directory name covers all three families.
Its [status](runs/gru_component_checks_job_47081340/status.txt) records
`tests_exit=0`, `empirical_execution=false` and the parent charge assigned once
to EP21. No test failed or skipped. These include the tiny fixed 100-epoch
fits, censored likelihood/offset/participant-weight checks, complete-window and
role refusals, causal state/phase boundaries, matched capacity and masked-input
invariance, frozen inference and budget/RNG handling.

The changed EP19 ASTRA program passed both schema and semantic validation in
the same component child; its direct
[validation receipt](../../episode21_wang_neural_code_conversion_reproduction/outputs/decoder_training/parallel003_checks_job_47081340/ep19_astra_validation.json)
is stored with the EP21 wrapper's component records. This closes the pending
GRU, TCN and EEGNet implementation checks and changed ASTRA validation in the
earlier entries. No repeat is selected.

The coordinator observed physical EP21 parent **47081340**, work48, **RUNNING**
on `sh04-04n21` at **2026-10-09 06:32:19 UTC**. The one-CPU / 7-GiB / eight-hour
parent continues independent real-data decoder work; it is not terminal.
The **4.885-second test-suite runtime is included in that parent charge**, not
a separate allocation cost, and the parent's final actual cost is not inferred.

These supplied-array checks establish the selected backend behavior, not EEG
forecasting performance or complete production controls. Source timing and
physical calibration, raw feature construction, production reader/memory
qualification, final refits and empirical family/control/Q11 integration remain
unresolved. No source or held-out payload was opened and no empirical forecast
or scientific-stage activation follows from these passing component checks.

## 2026-10-08 23:26 Pacific — development_specification / past_only_controls — compact causal EEGNet implemented and reviewed

The new [EEGNet backend](code/eegnet_forecast_backend.py) implements the
registered 4/8 temporal filters and depth multiplier 2. The coordinator chose
the remaining operator details prospectively: linear temporal/spatial
convolutions, ELU/pool4/dropout, linear depthwise/pointwise temporal
convolutions, ELU/pool8/dropout, and two slot means followed by the shared
19-output linear correction. Biases are enabled; no normalization, weight
constraints or additional head dropout is introduced. The complete adopted
method is recorded in [component details](component_details.md#baselines-and-matched-comparison-interface).

Independent source review found no blocking defect in the exact 128-ms raw-grid
and 64-ms post-pool4-grid kernels, causal boundary arithmetic, full-window
right-edge pooling phases across factorial resets, masked-slot invariance,
canonical development roles/windows, unchanged offsets/participant weights
or fixed training. This is source review only. The six focused tests include
an independent finite-sum operator comparison and one tiny five-example
125-sample actual 100-epoch fit; **all six remain unrun**. The preceding
23:20 observation that the named backend file was absent is now historical.

Added `past_only_controls.eegnet_backend_checks` and its explicit dependencies
to ASTRA. The coordinator selects **all 17 EEG backend checks** (GRU six,
TCN five, EEGNet six) together in the next EP21 `work_0048` replacement's
`full_workers_parallel_003.sbatch`. They share the existing optional
**120-second test-suite limit**, within the unchanged **420-second outer
component-check limit** and **one-CPU / 7-GiB / eight-hour parent**. The changed
EP19 ASTRA validation shares that outer child with a 30-second limit. This
selection adds no separate allocation or repeated earlier baseline checks;
the full physical parent charge remains recorded once by EP21.

The selected report location retains its legacy name:
`outputs/runs/gru_component_checks_job_ID/{status,tests}.txt`, with all 17
backend checks recorded there. The replacement is **not yet submitted** and
all selected tests/changed ASTRA validation remain pending. Parent startup
also requires its separately selected EP21 helper/integration checks; this
EP19 entry grants no dispatch or recovery authority. Root owns the wrapper,
submission and resource accounting.

The backend consumes supplied development arrays and materializes its fitting
fold. Source timing/calibration, raw feature extraction, production-memory
qualification, final refits and empirical family/control/Q11 integration
remain unresolved. No forecast result, source payload read, held-out access
or scientific-stage activation is implied by implementation or this selection.

## 2026-10-08 23:20 Pacific — development_specification / past_only_controls — existing EEG TCN handoff reconciled

The existing [EEG TCN backend](code/tcn_forecast_backend.py) and its
[five-test module](tests/test_tcn_forecast_backend.py), prepared on October 7,
were missing from this execution handoff. Bounded source inspection confirms
both files exist and describes their selected method in
[component details](component_details.md#baselines-and-matched-comparison-interface).
The coordinator's prior implementation handoff records independent source
review. This documentation reconciliation changes neither source nor model
choices and makes no new runtime-validation claim.

The registered 16/32-channel encoder has five two-convolution residual blocks
with dilations 1/2/4/8/16, fixed left-zero boundaries and no normalization.
Original mode encodes the complete supplied window before averaging the older
and recent-200-ms hidden positions; factorial mode separately encodes the two
reconstructed/masked slots using the same weights. Both concatenate two means
into the same 19-output correction head. Existing baseline offsets, censored
likelihood, global participant weights, three seeds and the fixed 100-epoch
CPU training schedule are preserved. The backend materializes the fitting
windows; it does not qualify production memory or source admission.

All **five TCN tests remain unrun**. The six prepared GRU checks, including the
shared likelihood arithmetic used by TCN, also remain pending in the current
record. Added `past_only_controls.tcn_backend_checks` to ASTRA with explicit
source/helper dependencies and a 120-second test-command bound plus five-second
termination grace, using one CPU and a 2-GiB child allowance in a suitable
substantive allocation. This is a prepared recipe, **not a submission or an
addition to any already-selected worker wrapper**. Root owns compute selection.
Validation of the changed ASTRA objects is deferred to that compute allocation.

The exact path `code/eegnet_forecast_backend.py` does not exist at this
inspection; no completed EEGNet backend is claimed. The method record also
corrects the historical non-neural TCN statement to its already-passing
October 7 evidence, without repeating those checks.

Source calibration/timing, raw feature integration, production readers,
remaining model/control integration and Q11 equivalence remain unresolved.
No empirical forecast, source payload read, held-out access, new scientific
stage, test execution or scheduler action occurred in this handoff repair.

## 2026-10-07 18:35 Pacific — development_specification / past_only_controls — GRU correction implemented

The selected [one-direction GRU backend](code/gru_forecast_backend.py) is
implemented and independently reviewed by a second agent, with no concrete
source-level defect identified. It retains the registered 16/32 hidden units,
three seeds, 100 epochs, AdamW, fixed 19-category non-neural log offsets and
equal total fitting weight per participant. The implementation does not infer
sensor units, clocks or labels and does not refit the shared voltage scale
after condition masking.

Under the user's delegated implementation judgment, the coordinator selected
the last hidden state for each slot and zero initialization at each complete
window. Original mode carries the older state through the 200-ms cutoff;
factorial mode resets the recent slot. Both use the same encoder and
concatenated two-state head, dropout 0.25 and 19 correction logits. These fixed
choices preserve the selected family and capacity without adding a search
dimension. The [component specification](component_details.md#baselines-and-matched-comparison-interface)
records the training and inference details.

The [six focused tests](tests/test_gru_forecast_backend.py) are prepared but
**unrun**. They cover loss/gradient agreement, fixed offsets and person weights,
causal state boundaries, masked-block invariance, development-fold and complete
past-window admission, and one tiny exact 100-epoch fit. The ASTRA program now
describes `past_only_controls.gru_backend_checks`; validation of this changed
program also awaits compute. No separate short testing job is requested.
Run the six checks and changed ASTRA validation in the next suitable bounded
substantive allocation; existing busy workers are not reused concurrently.
The unsubmitted EP21 `full_workers_parallel_002.sbatch` now selects these six
checks for a 120-second serial child in the next eligible work48 allocation,
using the existing PyTorch 2.4.1 module. The changed EP19 and EP21 ASTRA
programs each have a 30-second validation bound. All optional checks and
setup share a 420-second outer limit; one parent is charged once to EP21.
Ordinary optional failures are recorded; observed storage or log-write errors
stop the affected parent. No allocation has been submitted for this selection.

Source timing/calibration, production reader/memory qualification, remaining
family/control integration and Q11 equivalence remain unresolved. This change
has produced no empirical forecast, opened no held-out data and activated no
new scientific stage. Earlier passing boosted and temporal checks remain valid.

## 2026-10-07 05:53 Pacific — past_only_controls — boosted and temporal checks pass

All **ten** selected checks passed in **5.863 s** inside physical job
**46891123** (EP21 array **46891123_6**) on `sh04-04n24.int`: five for the
boosted hazard backend and five for the temporal hazard backend. The
[test output](runs/component_checks_job_46891123/tests.txt) covers conditional
censoring masks, participant weights and roles, unresolved-head/budget
refusals, complete past-window causality, training-only scaling and frozen
inference. It also includes the tiny actual fixed tree-schedule fit and the
tiny complete 100-epoch temporal fit. These supplied-fixture checks establish
the selected implementation behavior; they do not establish empirical
forecasting performance, onset validity or production memory feasibility.

The [changed ASTRA program validation](runs/component_checks_job_46891123/astra_validation.json)
records schema and semantic **PASS**, including
`past_only_controls.boosted_backend_checks` and
`past_only_controls.temporal_backend_checks`. The
[wrapper record](runs/component_checks_job_46891123/wrapper.txt) verifies the
compute host, one CPU, 7 GiB and Python 3.12.1 with NumPy 1.26.3,
SciPy 1.12.0 and scikit-learn 1.5.1. The
[closed component status](runs/component_checks_job_46891123/status.txt)
records runtime, test and ASTRA exits zero and `empirical_execution=false`.
The coordinator observed the parent running at **12:53:42 UTC**, following
its **05:52:16 Pacific** start; this closes the component checks only.
The entire parent is charged once to EP21, with no separate EP19 charge.

This result supersedes the then-unrun checks and ASTRA statuses in the
preparation entries below. No observed failure requires a repair or repeat.
Source units/calibration, exporter clock and filter delay, the full onset
uncertainty bound, raw feature extraction, production reader/memory
qualification, final refits and remaining model/control integration are
still unfinished. Required data separation, calibration procedures, numeric
margins and warning rules remain in force. No empirical source or held-out
payload was read by these checks, no forecast was scored, and no next-stage
access or scientific claim is activated. Continue the unresolved source and
empirical-pipeline work while reusing these passing component results.

## 2026-10-07 03:05 Pacific — development_specification temporal baseline implementation prepared

The missing registered small non-neural TCN is implemented in
[`temporal_hazard_backend.py`](code/temporal_hazard_backend.py), with five
focused checks in [`test_temporal_hazard_backend.py`](tests/test_temporal_hazard_backend.py).
**No test, scientific import or empirical fit has run for this implementation.**
The coordinator selected the remaining operator/training details under the
user's delegated pre-score judgment; the registered three weight decays,
three seeds, width 16, five dilation blocks, 18 hazards and 100 epochs are
unchanged. The complete selected recipe is in
[component details](component_details.md#baselines-and-matched-comparison-interface).

The input is a caller-admitted complete regular two-second **`[t−2,t)`**
window, preserving the existing `PastWindow` convention. Every sequence row
must be available by its own sample time; prepared summaries may be available
through `t`. No rate or acquisition mapping is inferred. Left-zero convolution
extension applies only outside this complete finite window, with no right
padding, missing-history/value imputation or 125-sample eligibility minimum.
Both convolutions in each block use ReLU and dropout; the residual sum uses
ReLU. The final readout averages all window hidden positions and concatenates
the separately standardized summary vector. This is a selected finite-window
operator, not a claim that pre-window physiology was zero.

Fitting uses channelwise sample weights `example_weight/window_samples`,
separate weighted summary scaling, and global natural-prevalence/person
weights in the censored objective. Each minibatch uses
`N/batch_size * sum(global_weight * NLL)`. Every conditional head must have
observed positive-weight intervals. Single-class observed heads remain in
the finite neural fit and are reported; all-unobserved heads are unresolved
before model construction. The boosted classifier's single-class refusal is
unchanged. There is no pseudo-event or non-neural probability floor.

CPU float32/native initialization, deterministic execution and the declared
seed govern the fit. AdamW explicitly uses learning rate 0.001, betas
`(0.9,0.999)`, epsilon `1e-8`, weight decay on every parameter including bias,
and `amsgrad/maximize/foreach/capturable/differentiable/fused=False`.
Uniform epoch permutations have a separate CPU generator with the same seed.
The schedule is exactly 100 epochs, batches up to 128 and norm clipping at 1;
nonfinite logits, objective, gradients, parameters or fitting scales refuse
the fit. A required caller resource/deadline callback runs immediately before
each optimizer step. The backend neither queries Slurm nor changes the
caller's one-thread limit; CPU RNG and deterministic settings are restored.
It freezes completed parameters and evaluation mode before supplying offsets.

The coordinator checked pinned official PyTorch 2.4.1 source for the selected
[AdamW arguments](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/optim/adamw.py)
and [native convolution initialization](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/nn/modules/conv.py).
These identify implementation semantics, **not the unobserved runtime version**;
actual versions remain to be recorded on compute. BR directive
`ext_review_dir_c132924f2f00` supplied only generic causality/group-weighting
criteria. It did not inspect or approve these artifacts, and no tailored BR
verdict is claimed.

The five tests cover likelihood/gradient and global batch weighting,
complete-window causality at 40 samples, grouped training-only scaling and
roles, unobserved-head/budget refusal, and **one five-example/four-sample fit
that executes all 100 epochs**. The latter also checks single-class reporting,
per-step callback ordering, explicit optimizer defaults, frozen normalized
19-category output and RNG restoration. All remain unrun; estimated runtime
is seconds to tens of seconds, not a measured performance promise. The
coordinator has now selected these with the earlier five unrun boosted checks
inside the existing **120-second EP19 component slot**, followed by the
existing 30-second ASTRA limit. Its child module setup now includes the
discovered PyTorch 2.4.1 module alongside NumPy/SciPy/scikit-learn. This is
EP19's ten-check portion of the shared EP02/EP15/EP19 **35-check** child,
still bounded by 540 seconds to TERM plus five seconds to KILL. No job has
been submitted for this selection; all ten EP19 checks remain unrun.
The parent remains 1 CPU, 7 GiB and eight hours, charged once to EP21 with
this EP19 work included, no duplicate charge or separate short allocation.
The coordinator owns the wrapper/submission; no empirical sibling inputs,
background service or automatic follow-on job is added. Selected command,
only in that coordinated substantive allocation with numerical threads one:

```bash
cd outputs/tests
timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v test_boosted_hazard_backend test_temporal_hazard_backend
```

ASTRA links the new code/checks and prospective baseline-selection input;
changed schema/semantic validation is pending. This array-only backend
materializes the full fitting fold's windows. Raw extraction, production
reader/memory qualification, final refits, empirical selection and source
units/timing/calibration are unfinished. No forecasting access, held-out
access, source payload read or scientific claim is added.

## 2026-10-07 02:49 Pacific — past_only_controls boosted checks selected for a shared remainder allocation

Select the **five new boosted-backend checks** and changed EP19 ASTRA
schema/semantic validation in the optional serial component child of the next
eligible EP21 exact-remainder `work_0007` / global array index **6**. The
coordinator owns its finite wrapper and submission. **No job has been
submitted for this selection; the five checks and ASTRA validation remain
unrun.** The unchanged sixteen older backend checks are not selected again.

The parent retains its existing **1 CPU, 7 GiB, zero GPUs and eight-hour**
allocation. The shared EP02/EP15/EP19 child is bounded by **540 seconds to
TERM plus five seconds to KILL**, including setup and all three serial scopes.
Within EP19, runtime-version reporting is capped at 10 seconds, the selected
five-test module at 120 seconds, and ASTRA at 30 seconds, each with five-second
kill grace. Numerical thread limits remain one; no separate allocation,
background service or empirical sibling input is introduced. The **whole
physical parent is charged once to EP21**, including this EP19 portion, with
no duplicate EP19 charge.

The child loads the discovered NumPy/SciPy/scikit-learn modules in its own
shell after clearing inherited Python paths. Actual interpreter/library
versions and exit statuses will be recorded under
`outputs/runs/component_checks_job_${SLURM_JOB_ID}/` as `wrapper.txt`,
`tests.txt`, `astra_validation.json` and `status.txt`. These paths describe
future outputs; no result directory or receipt was created by this selection.

Independent source review confirmed finite process-group containment through
the outer timeout and foreground nested timeouts. A concrete review finding
was corrected before launch: optional status-report writes now remain under
`set +e`, so their failure cannot terminate the parent before decoder work.
Module changes stay in the child, and the decoder retains its independent
allocation-EndTime-minus-300-seconds deadline. Optional component failure is
recorded and does not override the separate required EP21 decoder checks.

This is source-free implementation checking, not forecasting, detector
qualification, calibration or source-role release. No data/header access,
scientific import, test, scheduler query or compute execution occurred while
preparing this selection.

## 2026-10-07 02:41 Pacific — development_specification boosted baseline implementation prepared

The missing registered `gradient_boosted_discrete_hazard` comparator is now
implemented in [`boosted_hazard_backend.py`](code/boosted_hazard_backend.py)
over the existing prepared-context interface. It retains the three depths
2/3/4, **18 conditional logistic heads**, **200 trees per head**, learning rate
0.05 and seeds 1901/1902/1903 from the existing registry. No recipe count,
scientific margin, source role, onset label or sensor assumption changes.

The coordinator selected explicit implementation settings before outcomes:
scikit-learn binomial `GradientBoostingClassifier`, `init='zero'`, all samples
and features, Friedman-MSE greedy trees, minimum split size two/minimum leaf
size one, no class reweighting, no pruning, no subsampling, no warm start and
no validation early stopping. Each head uses the declared seed. Raw
`decision_function` logits feed the existing stable hazard-to-19-category
transform; no clipped `predict_proba` output or new non-neural probability
floor enters that transform. These API semantics are supported by the
[official scikit-learn 1.5 documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html),
checked by the coordinator. The coordinator discovered
`py-scikit-learn/1.5.1_py312` through the site module system; nothing was installed
or imported in this login shell. Actual runtime versions remain to be recorded
when execution occurs.

Only observed conditional risk intervals enter each tree head, with the
unchanged fitting-person weights and natural prevalence. Unknown future
intervals are excluded. A required head with no observations or only one
class leaves the **whole fit unresolved before any tree is fitted**; no
pseudo-event, replacement head, floor or cohort reduction is supplied.
The model preserves canonical development-fold admission, fitting-only
standardization, feature order, latest feature-availability time, reset
boundaries and WAY participant-specific weights. Its frozen 19-category
outputs can supply the existing EEG correction offsets; baseline selection
and raw feature construction are still separate work.

Five focused checks are prepared in
[`test_boosted_hazard_backend.py`](tests/test_boosted_hazard_backend.py): exact
censoring-likelihood masks, retained conditional weights, unresolved heads,
one 21-row full 18-head/200-tree fit, and source/fold/past-only interface
binding. The coordinator's independent static review found no blocking issue;
the bounded source/document whitespace check is clean. **All five remain
unrun.** The previous sixteen backend checks need no repeat because their
sources are unchanged. Proposed bounded command in an already
authorized substantive allocation, with NumPy/SciPy/scikit-learn supplied by
the module environment and one thread:

```bash
cd outputs/tests
timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v test_boosted_hazard_backend
```

The ASTRA program now links this actual backend and its focused checks under
the existing controls and baseline-selection analyses; changed-object
schema/semantic validation is pending compute. No generic simulation program,
new allocation, empirical fit, signal/header reread, scheduler query or
held-out access occurred. The causal TCN baseline, final-refit integration,
source-dependent feature extraction and the previously recorded timing,
calibration, onset and margin dependencies remain unfinished. Forecasting
readiness remains false.

## 2026-10-07 01:28 Pacific — source_views_and_timing manufacturer context added to actual headers

The new observed amplifier/filter settings enabled a bounded manufacturer-source
check. Its [interpretation and direct sources](source_timing_handoff.md#october-7-interpreting-the-observed-actichamp-acquisition-settings)
are now in the existing handoff and DATASETS: disabled optional filters do not
remove automatic anti-aliasing, and present-day AUX resolution cannot replace
the historical header scale. No signal or new outcome was accessed, numerical
correction selected, or timing/calibration condition declared satisfied.
The operating-manual route required login and was not pursued.

## 2026-10-07 01:22 Pacific — source_views_and_timing all development acquisition headers extracted

The selected acquisition reader completed in EP21 worker **46877911_10**,
physical allocation **46877919**, on **sh04-04n23.int**. The
[report](runs/source_views_and_timing/job_46877919_acquisition_metadata/acquisition_metadata.json)
records `complete_development_acquisition_metadata_extraction`, all **155**
headers from the original **15 development participants**, no remaining
members and **zero unresolved supported settings**. Its reported extraction
time is **0.306228986941278 seconds**; this excludes child setup and parent
decoder work. The [adjacent execution log](runs/source_views_and_timing/job_46877919_acquisition_metadata.txt)
records `acquisition_metadata_exit=0`. No fixture suite was repeated.

All headers record **BrainVision Recorder Professional V. 1.23.0003** and
**actiCHamp Base Unit (5002)**. The acquired configuration groups are:

| Headers | Development participants | Sampling rate | Recorded channels | Recorded channel high cutoff |
| ---: | --- | ---: | ---: | ---: |
| 144 | The 14 development participants other than sub-15 | 250 Hz | 39 | 69 Hz |
| 11 | sub-15 | 1,000 Hz | 67 | 280 Hz |

Every extracted channel row records low cutoff **DC** and notch **Off**;
every header records software filters **Disabled**. The resolution/unit tokens
are **0.0488281 µV** for electrode/EOG channels and the additional numeric-name
channels in the 1,000-Hz layout, and **0.298023 µV** for **TRIG, X, Y and Z**.
Those four auxiliary inputs consistently map to physical channels **65, 66,
67 and 68**, respectively. Other physical-channel routing is retained per
header and must not be replaced by one assumed cohort-wide mapping. Gradient
and offset tokens are absent, rather than measured as zero. The extraction
does not contain a reference-configuration field, so it does not newly verify
an EEG reference.

These are recorded acquisition settings, not an authenticated accelerometer
conversion to physical acceleration or a measured filter transfer function.
The report explicitly keeps `physical_acceleration_calibration_verified`,
`filter_transfer_or_delay_verified`, `timing_bound_verified` and
`forecast_scoring_ready` **false**. Filter group delay, the complete onset
uncertainty below 300 ms and the separate WAY raw-to-HS/WS origin/calibration
questions remain unresolved. Zero unresolved parser settings does not mean
these scientific metadata requirements are complete.

The report records no signal-array, marker/event, held-out-header or archive
access. The coordinator's **08:19:35 UTC** observation still had the parent
running; no terminal parent cost is claimed here. The entire allocation is
charged **once to EP21**, including this child, with **no additional EP19
physical job or duplicate charge**. This update inspected only the named
report/log, reader source and episode documents; it opened no raw header or
payload and ran no test, scientific import or scheduler command. No policy,
timing correction, calibration, reader release or forecasting result changes.

## 2026-10-06 23:49 Pacific — source_views_and_timing development acquisition extraction selected for EP21 refill

**Historical selection snapshot:** the extraction subsequently completed in
46877919, as recorded above. Its then-unrun status is superseded by that result.

The coordinator selected the existing
[`inspect_development_acquisition.py`](code/inspect_development_acquisition.py)
for the next eligible EP21 fresh refill, assignment index **10** (`work_0011`),
after the EP21 loss correction. The **155-header extraction remains unrun**:
no job ID, extraction result or completed header count is recorded by this
selection. No current job is changed and no separate allocation is requested.

The reader's fixed inputs are the completed metadata reports
`runs/source_views_and_timing/job_46425841_headers/development_headers.json`
and
`runs/source_views_and_timing/job_46425841_development_view/development_view.json`,
plus their exact 155 role-assigned self-paced development `.vhdr` files for
the original 15 development participants. Those text files remain beneath
`$SCRATCH/br_autoresearch/episode19_causal_motor_forecasting_benchmark/development_view_job_46425841/self_paced/development/`.
The reader admits the complete fixed path set before reading headers and opens
no archive, signal array, marker/event file, WAY file or held-out header. This
selection did not inspect or stat those source files.

Run from the exact EP19 directory with `python3.12 -u -B`, the existing reader,
`--max-seconds 60` and
`--output-dir outputs/runs/source_views_and_timing/job_${SLURM_JOB_ID}_acquisition_metadata`.
The output directory must be fresh and must not be pre-created. The intended
report, relative to `outputs/`, is
`runs/source_views_and_timing/job_${SLURM_JOB_ID}_acquisition_metadata/acquisition_metadata.json`.

The coordinator will enforce **one CPU** and `ulimit -v 1048576` (**1 GiB
virtual address space**) in a separate guarded child. The reader retains its
**60-second internal deadline**; the entire child, including setup, module
loading and logging, receives **70 seconds before TERM plus five seconds to
KILL**, at most 75 seconds. This is stricter than the older outer timeout in
the generic ASTRA recipe. The unchanged EP21 parent retains its eight-hour
worker limit and existing budget cap; its whole allocation cost is charged
**once to EP21**, without a separate EP19 reservation or duplicate charge.

Reuse the [four passing reader checks and ASTRA validation](#2026-10-06-2259-pacific--source_views_and_timing-acquisition-reader-checks-pass-in-ep21-job-46864089)
from job 46864089. The ASTRA program and reader are unchanged; no repeat fixture
run is selected. This extraction qualifies recorded acquisition metadata only.
It does not fit EEG, produce forecast scores, authenticate accelerometer
calibration, establish filter transfer/group delay, or satisfy the full onset
timing bound. Unknown or unsupported header fields remain unresolved.

## 2026-10-06 22:59 Pacific — source_views_and_timing acquisition-reader checks pass in EP21 job 46864089

All **four** prepared `test_development_acquisition` checks passed in
**0.017 s** inside coordinator-owned EP21 allocation **46864089**, one CPU /
7 GiB on **sh04-04n24.int**. The
[test output](runs/component_checks_job_46864089/test_development_acquisition.txt)
covers the two recorded header layouts, exact development-path selection,
held-out/reduced-cohort refusal and preservation of unknown or inconsistent
settings. These are controlled text/access-selection fixtures, not an
extraction result from the 155 actual development headers.

The [status receipt](runs/component_checks_job_46864089/status.txt) records
test exit zero and ASTRA exit zero. The
[ASTRA output](runs/component_checks_job_46864089/astra_validation.json)
explicitly reports successful schema and semantic validation of
`outputs/astra/v0.0.14/astra.yaml`. The reported test runtime excludes process
startup, ASTRA validation and the rest of the parent workload.

The prepared 155-header extraction did **not** run in this component check.
No source archive, empirical sensor/event payload, EEG fit, forecast score,
clock correction, calibration or access release is established by these
passes. The next substantive source action remains the existing bounded
development-header extraction in a suitable coordinated allocation; it can
reuse these passing checks. Acquisition/filter timing, physical calibration
and the complete onset uncertainty remain unresolved.

The coordinator reports the parent still running at this documentation
update. Its whole allocation cost is charged **once to EP21**, consistent with
the receipt; no separate EP19 charge or final parent runtime is recorded.
This documentation task read only the named check receipts and episode
records, with no source payload, import, test or scheduler operation.

## 2026-10-06 21:34 Pacific — source_views_and_timing acquisition metadata extraction prepared

A bounded inspection of two existing development `.vhdr` text files found
acquisition settings omitted by the completed header parser's output. The
sub-02/session-01/run-0001 header records a 69-Hz high cutoff at 250 Hz;
sub-15/session-01/run-0001 records 280 Hz at 1,000 Hz. These are individual
source-header observations, not yet a cohort-wide result. The inspected
amplifier tables also record physical auxiliary channel mappings, voltage
resolution, DC low cutoff, notch Off and software filters Disabled. See the
[source details and limitations](source_timing_handoff.md#development-acquisition-settings-omitted-by-the-earlier-header-report).

The new [`inspect_development_acquisition.py`](code/inspect_development_acquisition.py)
reads exactly the 155 development headers already bound by the completed
header and materialized-view reports. It retains recorded amplifier and filter
settings with source line numbers and leaves unsupported fields unknown.
The prior report identifies 144 recordings at 250 Hz and eleven at 1,000 Hz;
the new extraction has not yet established their filter settings. It opens
no archive, signal array, marker/event file or held-out participant header.
Existing reports and source files remain unchanged.

Coordinator static review of the reader, its dependencies and four focused
text/access-selection checks found no blocking issue. Those checks, the
155-header extraction and changed ASTRA validation are **unrun**. The selected
next action is to bundle them into a suitable substantive allocation: one CPU,
at most 1 GiB, 60-second internal extraction limit and 75-second outer limit,
writing a fresh direct child of `runs/source_views_and_timing/`. No separate
short job or additional resource reservation is selected.

Recorded cutoff settings do not establish filter transfer/group delay,
accelerometer calibration or the complete onset uncertainty bound. A bounded
official vendor search supplied no version-specific delay evidence; the
manufacturer manual index requires account access. No gated retrieval was
attempted, no clock correction or threshold was selected, and no forecast
score or scientific claim is produced by this preparation.

## 2026-10-06 — completed support and backend work reconciled into readiness

DATASETS, GOAL status and the paper-plan opening now include the completed
twelve-person conditional S1 support panel, preserved five unresolved cases,
fixed model/fold definitions and passing sixteen-backend-check result. The
public timing/force conventions are linked without asserting an authenticated
clock shift, physical conversion or onset bound. Remaining implementation and
nineteen decision fields are not mislabeled as nineteen new approval requests.
This records existing evidence only; no sensor read, test, job, scientific
threshold, role, access rule or proposal adoption occurred.

## 2026-10-06 15:49 Pacific — public event-generator findings added to the source handoff

At the scientist's request, recorded the verified primary-source findings in
the [existing timing handoff](source_timing_handoff.md#october-6-1549-pacific-published-event-generator-and-remaining-exporter-gap):
direct copying of the stored window start into `StartTime`, the separate
2-ms first-sample convention, and named force-sign/sensor conventions. The
provider branch and issue listings yielded no alternate exporter/correction.
These findings do not identify the upstream clock origin, authorize a
two-second correction, or supply physical calibration or the full onset bound.

The bounded search inspected public code/documentation and metadata only.
Incidental third-party method/result snippets appeared in search results;
their outcomes and settings were not used. No HS/WS signal/event array,
held-out data, test, import, scientific execution, scheduler operation or
external contact occurred. This addition changes documentation only; existing
recipes, margins, resources, roles and empirical-access state are unchanged.

## 2026-10-06 11:52 Pacific — prepared backend checks pass inside EP21 job 46801996

All 16 source-free classical/non-neural backend tests passed in **0.871 s**
inside the approved one-CPU / 7-GiB EP21 job on sh03-06n10. Both status exits
are zero, including EP19 ASTRA schema/semantic validation. Evidence is
`runs/classical_backend_checks_job_46801996/{tests.txt,status.txt,astra_validation.json}`.
The entire parent allocation is charged once to EP21, not again to EP19.
This verifies the prepared arithmetic/interfaces on fixed fixtures; it opens
no empirical sensor data and does not resolve acquisition timing, calibration,
eligible support or the remaining scientific decisions. No EP19 forecast ran.

## 2026-10-06 11:49 Pacific — prepared backend checks included in approved EP21 job

The scientist approved the prepared EP21 first-layer allocation, submitted
as **46801996**. Its previously prepared, separately logged EP19 prelude
selects the 16 source-free checks in `test_classical_forecast_backend` and
`test_non_neural_forecast_backend`, with a shared 120-second timeout plus
five-second grace and a separate 30-second optional ASTRA validation.
The parent remains one CPU / 7 GiB / 24 hours; charge its full actual time
once to EP21. This selects only arithmetic/interface checks, no sensor,
event or held-out payload, fit, timing correction, forecast or new scientific
threshold. Failure does not gate EP21's unrelated real-data fitting.
Job submission is observed; check execution/result is not yet observed.

## 2026-10-06 11:15 Pacific — provider importer and series-origin documentation

The resumed source investigation inspected the official utility repository's
directory/commit metadata, usage text, EEGLAB importer and relevant timestamp
code, plus the article's Data processing/Data Records definitions. The
[handoff records source URLs and interpretation](source_timing_handoff.md#october-6-1115-pacific-official-importer-and-series-origin-construction).
The importer adds no two-second correction after `WEEG_GetEventsInHS`.
The article documents removal of leading/trailing series samples but does not
give the export assignments or per-series trim counts. Thus a trimming-origin
difference and a trial/window-origin difference remain distinct possible
causes; no empirical correction or full onset bound is authenticated.

The useful next source is the original raw-to-HS/WS constructor or an
authenticated binding between its trimmed series zero and stored start field.
The current utility listing does not supply that constructor. No unknown raw
sensor unit is inferred from the EEG importer's 0.1 scale. No new data read,
timing test, detector fit or forecast is selected. This inspection makes no
claim of exhaustively searching all provider records.

The coordinator reports one BR read-only
`kg_method_condition_coverage_guard` call for `10.1038/sdata.2014.47`, requesting
task/preprocessing fields with persistence false; it returned
`publication_not_found`. That supplies no scientific evidence and was not
retried. Primary provider sources support the documentary findings above.

Exposure disclosure: an exact-term web search also returned incidental
third-party WAY methods/result snippets. Those analyses were not opened or
used for scientific choices; this reviewer is not described as having no
public-outcome exposure. No EP19 signal, event vector or held-out outcome was
read. Work was bounded documentation/code-text inspection on sh04-ln04 without
a Slurm allocation; public HTTP reads used finite response/time bounds. No
imports, tests, scientific computation, scheduler action or contact occurred.

## 2026-10-06 10:57 Pacific (PDT) — supplied review reconciled; timing and decision proposals unadopted

Revised `source_timing_handoff.md` from the policy, component definition and
existing permitted log evidence. All 34 P1 on/off crossings have the same
two-second discrepancy under the tested convention; that supports an origin
hypothesis, not an automatic correction or the full strictly-below-300-ms
timing bound. The recommended next clock test needs an explicit independently
supported candidate origin binding first. The 300–600-ms horizon is retained.

Source-unit conversion must be authentic; no physical scale is inferred from
signal magnitudes. Existing cue/return logic is distinguished from its missing
calibration and tolerances. Five unresolved S1 support dispositions remain in
the original twelve-person panel, not permanent exclusions or permission for
a favorable subset. Listed all nineteen null fields in the policy's margin
block, distinguishing effect criteria from numerical/timing/latency and
participant rules. Prioritize the primary real-minus-non-neural practical
gain for scientist judgment; remaining delegated specification work still
requires the appropriate scientific or numerical rationale.

Recommendations remain unadopted. No threshold, correction, units amendment,
data role, budget or forecast is activated. Current participant-routing code
remains repaired; its tests are still unexecuted. Work stayed in documentary
inspection/editing on sh04-ln04 without a Slurm allocation: no new payload,
closed outcome, import, test, scheduler action or external contact.

## 2026-10-06 08:19 UTC — logistic baseline model interface implemented, unexecuted

Implemented `code/non_neural_forecast_backend.py` under the existing three
source-specific logistic L2 recipes and seeds. It connects caller-prepared
non-neural feature vectors to the existing 18-head censored hazard fitter and
returns frozen 19-category log probabilities suitable as the classical EEG
models' baseline offsets. It preserves canonical development fitting folds,
equal total weight per person with natural point prevalence, fitting-only
feature standardization, fixed feature order, reset boundaries and visible
nonconvergence. No raw sensor statistics, imputation, cue mapping, sampling
rate, onset label, calibration or scientific threshold is invented.

Root and independent static review identified and repaired one routing gap:
source and boundary labels alone could not prevent using a WAY P1 model for
P2 after reset. Context vectors now carry explicit participant identity.
Fitting checks it against each example's identity; WAY prediction checks the
model's fitted person. The self-paced group model can still predict development
validation people outside its fitting fold, with source-role access owned by
the caller. Canonical scope checks precede consumption of caller batches.

Four focused tests in `tests/test_non_neural_forecast_backend.py` use prepared
vectors and mock the existing optimizer. They cover scope/recipe/weighting,
frozen probability compatibility, feature order/source/person/time/boundary
errors, unchanged scaling, and nonconverged-fit rejection without retry.
Independent static rereview passed. **No test, import, fit or modified ASTRA
validation ran.** These checks do not repeat numerical optimizer qualification
or create a physiological simulation program.

The two EP19 test modules now share the same proposed EP21 first-layer
allocation's **120-second ceiling plus five-second grace**, with separate
EP19 logs and non-gating status. Sixteen EP19 checks replace the earlier twelve;
the parent remains 1 CPU / 7 GiB / 24 hours, unapproved and unsubmitted, with
all eventual cost charged once to EP21. No separate allocation or reservation
was opened. Updated the ASTRA program and current navigation accordingly.

This closes a model-interface gap, not the empirical baseline pipeline.
Declared feature availability timestamps do not prove raw causal history or
the required 50/200/1000/2000-ms summaries; exact source extraction and missing-
history handling remain unfinished. Only development inner-fold fits are
implemented: final WAY series1–7 and all15-person self-paced refits, boosted
and TCN baseline families, baseline selection, remaining neural models and
scoring orchestration remain separate work. Timing/calibration, P1 clock
mapping, full cohort support and scientific margins remain unresolved. No
signal payload or protected outcome was accessed, and no empirical forecast
or positive neural finding is claimed.

## 2026-10-06 07:44 UTC — independent backend review and fitting-scope repair

Independent static review confirmed the existing recipe's censored
likelihood/gradient, weighted affine-invariant tangent features, fitting-only
scaling and original/factorial FIR boundaries. It identified one concrete
admission defect: a manually constructed fitting scope could name held-out
series while passing membership checks against its own contents. `fit` now
reconstructs the registered development fold and requires exact agreement
before feature construction or consuming any batches. The new twelfth check
rejects held-out WAY series, an added validation series and a held-out
self-paced substitution without accessing examples or features.

This repair changes no registered split, source role, model recipe or scientific
threshold. All twelve checks remain unexecuted; the same proposed substantive
EP21 parent and unchanged 120-second prelude cap cover them if approved. No
signal data or additional compute was accessed during the repair.

## 2026-10-06 07:37 UTC — classical reference backends implemented; execution pending

Implemented `code/classical_forecast_backend.py` under the unchanged section-5
recipes: the filter-bank elastic-net and affine-invariant tangent reference
families now have fitting and timestamped past-window prediction interfaces.
The code also implements the logistic-hazard censored likelihood, gradient and
L2 fitter. It retains equal participant weights, fitting-only feature scaling
and tangent references, explicit baseline offsets, all three seed probabilities,
whole-group fitting admission and visible nonconvergence. Rates and causal FIR
coefficients must be supplied explicitly; no source setting is inferred.

The coordinator and implementation agent inspected the code statically.
Eleven focused checks in `tests/test_classical_forecast_backend.py` are prepared
for a future authorized substantive allocation. They cover the censored
likelihood/gradients, missing groups, future samples and recording boundaries,
FIR support/reset, masked slots, feature geometry, probability averaging and
fit/prediction interfaces. **These new checks have not run.** They are small
arithmetic/interface fixtures, not a resumed physiological simulation program.
ASTRA now links the code and planned checks; its modified program has not yet
been validated. No compute allocation or additional CPU cost was incurred.

The coordinator subsequently placed these checks in the **proposed, unapproved**
EP21 first-layer job's bounded prelude: at most 120 seconds plus five-second
grace, with separate `runs/classical_backend_checks_job_<jobid>/` test/status
and ASTRA records. An EP19 check failure does not gate unrelated EP21 fitting.
If that one job is approved, its full parent cost is charged once to EP21;
there is no separate EP19 reservation or check-only allocation. This scheduling
preparation does not report that any check has passed.

This does not complete the forecasting pipeline. Logistic hazard still lacks
its source/context feature extractor and model wrapper; classical wrappers
require caller-admitted, common fitting-standardized 29-channel input and frozen
baseline offsets. Source preprocessing, onset/risk labels, grouped loaders,
surrogates, calibration/scoring orchestration and complete-fold feedback remain
external. Three neural reference backends and boosted hazard remain unfinished.
The sensor units, synchronization/calibration, P1 clock mapping, complete cohort
support and scientific margin blockers are unchanged. No signal payload,
empirical fit, EEG forecast, source pilot or held-out result was accessed.

## 2026-10-06 05:31 UTC — development_specification: finite recipes and grouped folds prepared

Implemented `model_recipe_registry.json`, `code/model_recipes.py` and three
focused tests in `tests/test_model_recipes.py`. The existing component note,
section 5, now defines all five reference families with exactly 12 configurations
and three seeds, all three source-specific non-neural families, the fixed
five-fold 15-person self-paced assignment, and participant-specific WAY
leave-one-whole-series-1–6 folds. Whole runs/state never cross roles; WAY7 and
8–9 and the eight self-paced held-out people remain outside inner fitting.
Natural prevalence is retained within each person before participant averaging.
All unresolved participants retain their original identity/fold.

Final primary and factorial enumeration requires one explicit development-frozen
recipe per each of the five families: 150 and 60 seed/condition identities,
respectively. It cannot silently repeat the configuration search. Development
enumerates 720 recipe/seed/condition identities before source/fold expansion;
these are implementation identities, not 720 new scientific search trials or
a launch request. All eight final surrogates and all four Q conditions remain.
The secondary menu retains the existing branches, one-operator follow-ups and
unchanged trial/resource caps. Source timing/units, P1 correction, detector
thresholds and missing sensor information have no invented default.

Independent static review of registry, interface and tests passed. The existing
ASTRA `development_specification` now references the prospective registry while
preserving its future complete-method dependency on real source qualification
and remaining margins/backends. No scientific scoring/backend implementation
is claimed complete. The source handoff distinguishes remaining internal code
and preprocessing work from missing acquisition/calibration evidence.

Root selected **only these three new tests and changed ASTRA validation** for
the remaining time after EP06 metadata in existing allocation **46721202** on
**sh04-18n32**, one CPU / 2 GiB / 30 minutes. From the EP19 episode directory:
`timeout --signal=TERM --kill-after=10s 45s python -m unittest discover -s outputs/tests -p test_model_recipes.py -v`.
The tests need only the standard library and less than 256 MiB. Root owns the
compute shell and official ASTRA validator, and will stop if insufficient
allocation time remains. The whole parent is charged once to EP06; no separate
EP19 allocation charge or checking-only job is added.

Observed completion: all3 tests passed in0.003s in
[`runs/model_recipe_checks001/tests.txt`](runs/model_recipe_checks001/tests.txt),
and official ASTRA schema/semantic validation exited0 in
[`validation_20261006_46721202.json`](astra/v0.0.14/validation_20261006_46721202.json).
Parent **46721202 completed0:0**, from05:26:56 to05:33:09UTC:373 seconds at
1CPU, **0.1036111111CPUh**, with MaxRSS343,768KiB. The whole cost is charged
once to EP06's source-check allowance, reservation released; EP19 adds no
charge or duplicate share. EP06's substantive metadata step completed all
12 sessions/3,090 records in229.198s without source outcomes. No compute
session remains active. These checks establish finite enumeration and fold
coverage only; they do not fit, qualify or score a forecasting model.

No signal payload, onset detector, repeated support panel, fitted forecast,
held-out access, public lookup or scheduler action occurred during preparation.

## 2026-10-06 00:37:50 UTC — publication-supplement inspection complete

Bounded text inspection of the existing PDF's exact pages 2, 3, 5 and 6 completes
the seven-page review; pages 1, 4 and 7 were read previously. Page 5 distinguishes
experimental chronology from released S8/S9/ST numbering and identifies Weight
6 as EEG-only `HS_P1_ST.mat`–`HS_P12_ST.mat`. Page 6 records participant-specific
weak EMG, breaks, spikes, flatlines and Polhemus interference. The
[handoff records the exact mapping and quality notes](source_timing_handoff.md#october-6-2026-all-seven-publication-supplement-pages-inspected).
No units, calibration, acquisition/filter delay, cue-edge or clock-origin
blocker is resolved; eligibility, frozen splits and the required <300-ms bound
are unchanged. No page of this publication supplement remains unread, but its
local `Information.pdf` name still does not authenticate a separately named
dataset document. No download, render, signal payload, test, scheduler action
or additional compute allocation was used. Only this log and the handoff changed.

## 2026-10-06 00:09 UTC — equipment dependencies and protocol cue direction documented

The coordinator's bounded read of supplement pages 4 and 7 identified FASTRAK,
ATI F/T DAQ and Brain Products equipment, and the red-light reach/hold/replace
instructions. Primary-manual review, with ATI independently delegated, records
exact [source sections and remaining configuration requirements](source_timing_handoff.md#october-6-2026-equipment-and-instruction-documentation).
FASTRAK units/filtering/sync and ATI calibration/bias/gain are configurable;
manufacturer timing or bandwidth specifications do not establish the study's
onset bound. Actual settings and P1's export-specific clock mapping remain
unresolved; no defaults or automatic two-second correction are adopted.

Equipment-manual sections were read through provider-indexed web results after
direct PDF retrieval redirected or exceeded the tool limit. The ATI component
revision is not authenticated as the study-era revision. Supplement pages
2/3/5/6 and scientific payloads remain unread in this action. No code, test,
detector setting, scheduler action or additional compute charge is added.

## 2026-10-06 00:02 UTC — publication supplement page-one extraction completed

Coordinator reports parent `46680856` COMPLETED with exit 0 and the bundled
`ep19_information_page01` action status 0. The saved
[`status.txt`](source_documents/way_information/20261005-page01/status.txt)
records `complete_document_page01_only`; `execution.txt` records
`sh04-04n24.int`, job `46680856`, one CPU and page 1 only. `pdfinfo.txt` confirms
seven pages and 193,612 bytes. The source URL/XML records authenticate the
publication supplement `sdata201447-s1.pdf`; its local `Information.pdf` name
does not authenticate a separate data-release document with that name.

The [extracted first page](source_documents/way_information/20261005-page01/page01.txt)
is only a table of contents. It points to equipment suppliers on page 4 and
instructions on page 7 but supplies no measurement or timing specification.
No other page or scientific payload was opened. The existing
[source/timing handoff](source_timing_handoff.md#october-6-2026-publication-supplement-page-1-reviewed)
now records these facts and the remaining calibration/onset gaps. No automatic
two-second correction, threshold, scientific result or follow-up read is
selected. The whole parent allocation is charged once under EP04; EP19 adds
no separate charge.

## 2026-10-05 23:39 UTC — original WAY supplementary document located; page-one read selected

The official [Europe PMC article XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4365902/fullTextXML)
identifies Supplementary File1 as`sdata201447-s1.pdf`,193612bytes. Its exact
[PMC open-data object](https://pmc-oa-opendata.s3.amazonaws.com/PMC4365902.1/sdata201447-s1.pdf)
returned HTTP200 with that content length. This is a newly authenticated public
document route, not a retry of the stopped Sherlock Springer-CDN route.

Select [one bounded document reader](code/read_way_information_page01.sh)
inside the already-reserved, dependency-held shared parent **46680856**, after
its separately scoped EP04 sensitivity and EP12 reporting actions. It uses
one CPU, the existing4GiB step, a90-second external cap plus10-second kill
grace, and the discovered`system poppler/0.47.0`module. Download only the
193612-byte public PDF with a20-second/250000-byte limit; retain it and source
links at`outputs/source_documents/way_information/20261005-page01/`and extract
**page1 only**. Participant/held-series note pages and scientific arrays are
not selected. No retry, alternate route, calibration or forecasting is included.
Preserve separate status; the parent allocation is charged once under EP04.

The accessible article methods additionally distinguish continuous-series
StartTime from event windows starting2seconds before cue onset. Reported
synchronization error is at most2ms after shared-sync-channel alignment; EEG
was acquired at5kHz with0.016–1000Hz hardware bandpass and downsampled to500Hz
with anti-alias filtering. These are provider method facts, not an authenticated
mapping of P1's observed2second discrepancy or a complete EP19 onset-error bound.
Raw position/force scale, signs/offsets, filter response and return calibration
remain unresolved. The pending page-one extraction is not yet a source result.

Exposure disclosure: two supplement-focused web searches returned incidental
snippets reporting outcomes from unrelated public WAY decoding analyses. The
reviewer did not open those analyses or use their findings/settings. Only the
original provider methods/XML and retained source documentation support this
review; no EP19 held-out signal, event row or array was opened. Public outcome
exposure is not described as absent.

## 2026-10-05 20:45 UTC — WAY return predicate specified, calibration still unresolved

The delegated [component definition](component_details.md#11-way-completed-return-predicate-parameterized-development-definition)
now names the position/force channels and fixes the return logic: both digits
clear the object by 15 cm, the existing 500-ms primary-path stillness holds,
object/force consistency holds, and an observed end precedes return and the
next known start. Initial entry and gaps have explicit handling. The 15-cm
design choice uses the provider's automatic-cue criterion as context, without
treating a new cue or retrospective `HandStop` as completed return.

Source-unit conversions, force signs/offsets, calibration references and
tolerances, cue interpretation and the full timing bound remain unresolved.
Unknown inputs prevent arming; no guessed values, altered onset target,
reduced participant panel, code execution or new job is selected. This closes
a Boolean/order specification gap, not empirical detector qualification.

## 2026-10-05 20:40 UTC — bounded provider-document review; return and timing unresolved

The [official WAY methods](https://www.nature.com/articles/sdata201447)
distinguish object replacement, release and hand return. Automatic cues require
digits away from the object, whereas surface-change trials use a researcher
button; a new cue alone therefore does not certify completed return. The
[official event utility](https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_FindEvents.m)
computes `LOC_HandStop` using a threshold from the latter half of the entire
trial, so that retrospective rule cannot be copied into the past-only state
machine. The methods document sampling, EEG anti-alias filtering and a
reported synchronization error of at most 2 ms, but the reviewed sources do
not establish recording-specific acquisition/filter response or the complete
strictly-below-300-ms onset uncertainty bound.

Supplementary File 1, `Information.pdf`, remains **uninspected**, not absent or
uninformative. The Nature supplemental link returned an internal retrieval
error; later Nature navigation failed with an authentication redirect/500 and
PMC presented a browser challenge. Root's bounded Figshare collection API and
title-metadata search found the 12 participant dataset records, no supplemental
record. An inferred Springer CDN HEAD request failed DNS on `sh04-ln04` at
20:36:32 UTC; that route stopped. BR's exact WAY-EEG-GAL lookup recovered the
official collection DOI `10.6084/m9.figshare.c.988376`; its registry guard's
`publication_not_found` is not evidence that provider methods are absent.

Causal return specification remains delegated development work under the
existing [component design](component_details.md), without a new approval gate.
Its exact sensor/unit/consistency rule and the acquisition, filter and crossing
timing bounds remain unresolved; support counts do not resolve them. No new
threshold, label, pilot or EEG forecasting step is activated. This closeout
adds no source retrieval, raw-array read, test, scientific computation or
scheduler action, and preserves all 12 support outcomes.

## 2026-10-05 20:21 UTC — original 12-person S1 support panel completed

Analysis `source_views_and_timing`. The coordinator confirms shared parent
`46660232` COMPLETED after 14 minutes 28 seconds on four CPUs and its EP19 step
`46660232.6` COMPLETED after 120 seconds. The saved
[remaining-panel report](runs/source_views_and_timing/20261005-remaining-S1-46660232/remaining_s1_support.json)
finished `2026-10-05T20:21:48.596994+00:00`, with
`complete_remaining10_conditional_support_panel` and `complete=true`.
Process elapsed time was **119.173767827 seconds**, CPU time **14.999046849
seconds**, and peak RSS **1,079,168 KiB** on `sh03-06n06.int`.

Bounded read-only review confirms exactly P3–P12, all S1 fitting files, with
every original manifest identity admitted before payload processing. All 12
preparation stages and all eight stages per participant returned. Each file
completed the original support-only function with unchanged settings, returned
its MAT load and restored scoped bindings. Signal handlers were restored. Every
file had zero nonfinite primary sensor values and zero unknown cue samples.
The saved function results and all participant rows are retained; there are no
technical failures or uncomputed participants in this remaining panel. The
completed run does not establish the cause of the earlier panel timeout.

Together with the unchanged
[P1 result](runs/source_views_and_timing/20261003-peripheral-pilot02/peripheral_pilot_report.json)
and [P2 result](runs/source_views_and_timing/20261005-P2-support-46658777/p2_support_trace.json),
the original fixed 12-person S1 support panel is now complete. The table copies
saved counts; no sensor calculation or new source read was performed during
review. “Sufficient” means only the existing conditional support-count rule:
at least eight stationary windows, at least four in each fixed parity, and at
least half of eligible windows stationary. No thresholds were fitted.

| Participant | Stationary / eligible windows | Fitting windows | Validation windows | Conditional support |
| --- | ---: | ---: | ---: | --- |
| P1 | 13 / 34 | 7 | 6 | Unresolved: below half |
| P2 | 27 / 28 | 14 | 13 | Sufficient; thresholds not fitted |
| P3 | 8 / 34 | 3 | 5 | Unresolved: below half and fewer than four fitting windows |
| P4 | 32 / 34 | 16 | 16 | Sufficient; thresholds not fitted |
| P5 | 23 / 28 | 10 | 13 | Sufficient; thresholds not fitted |
| P6 | 16 / 34 | 9 | 7 | Unresolved: below half |
| P7 | 25 / 34 | 9 | 16 | Sufficient; thresholds not fitted |
| P8 | 23 / 28 | 11 | 12 | Sufficient; thresholds not fitted |
| P9 | 11 / 34 | 5 | 6 | Unresolved: below half |
| P10 | 29 / 34 | 15 | 14 | Sufficient; thresholds not fitted |
| P11 | 24 / 28 | 11 | 13 | Sufficient; thresholds not fitted |
| P12 | 10 / 34 | 6 | 4 | Unresolved: below half |

Seven participants have sufficient conditional support and five remain
unresolved under the unchanged rule. P1's original negative support result and
the four additional insufficient-support results are preserved; no favorable
subset is selected or pooled for production thresholds. This is a description
of one fitting series per person and one operational rest-window recipe, not
a population success rate, proof of physiological rest, clinical finding, or
positive/negative EEG forecasting result. All primary sensor paths remain
required. The whole-HS loads incidentally decoded EEG and discarded it; there
was no EEG feature/score, onset/risk label, feedback or held-out access.

The selected support panel is finished. There is no automatic next pilot,
threshold relaxation or label generation. P2–P12 cue polarity remains
conditional; P1's duration-based support does not authenticate other files.
LED-off remains a return instruction, not completed return. Provider position
units/object-force semantics and the causal completed-return predicate still
require resolution, and the total acquisition/filter/crossing timing bound is
unknown. None of these requirements is waived by support counts. Forecasting
and scientific margins remain unactivated.

The coordinator records **0.9644444444 core-hours** for the entire shared
four-CPU parent once under EP04. This is a reference only: neither the EP19
step nor its process CPU seconds incur a second parent charge. The review
worker performed bounded JSON metadata inspection and these log/navigation
edits only; no raw array, rescoring, test or scheduler operation was performed.

## 2026-10-05 20:04 UTC — remaining original S1 support panel selected

Analysis `source_views_and_timing`. Root and independent source review accept
the prepared driver below. Select the exact remaining **P3–P12, S1** fitting
files, preserving completed P1/P2 without rereading them. The unchanged
support-only calculation runs sequentially in one CPU / 4 GiB with a
600-second whole-process deadline, external 620 seconds plus 10-second grace.
Every participant outcome remains present, including technical failures or
uncomputed trailing files. No threshold fitting, labels, EEG forecasting,
feedback or held-out access is selected. This completes the original support
question if all files return; it cannot authenticate cue polarity or timing.
The action follows the separately scoped EP04 RSA workers and EP03 source step
in one finite shared parent, charged once under EP04. Fresh EP19 output is
`runs/source_views_and_timing/20261005-remaining-S1-JOBID/`; launch ID and actual
result follow separately. There is no automatic retry or successor.

Shared submission returned **46660232** at **20:05 UTC**. The EP19 result
destination is `runs/source_views_and_timing/20261005-remaining-S1-46660232/`.
This is a submitted step, not completion of the remaining panel.

## 2026-10-05 20:00 UTC — P2 support completed; remaining fixed P3–P12 prepared

The coordinator confirms shared-parent step `46658777.1` completed in 18 seconds.
Its exact report is
`runs/source_views_and_timing/20261005-P2-support-46658777/p2_support_trace.json`,
finished `2026-10-05T19:55:06.374047+00:00`. Independent read-only review found
the exact manifest/source admitted, all ten recorded stages returned, original
settings unchanged, and binding/signal-handler restoration true. The original
support-only function returned 28 eligible windows, 27 stationary windows,
14 fitting windows and 13 validation windows. One window failed the
flexor-digitorum stationarity check; primary nonfinite values were zero. This
meets the unchanged conditional support-count rule. No noise threshold was
estimated or frozen, and no onset/risk label or EEG feature/score was made.

Process elapsed time was 16.564478405 seconds, CPU time 2.752130977 seconds,
and peak RSS 1,025,976 KiB for a 70,454,110-byte source. The recorded MAT load
took 0.840195212 seconds, sensor summaries 1.105193941 seconds, and stationarity
classification 0.012106698 seconds. Total elapsed includes stage publications
and other overhead. This successful P2 pass does not identify the operation
that stalled the historical eleven-file attempt. P2's cue polarity, return
completion and total onset uncertainty remain unqualified. P1's preserved
13/34 support result and P2's 27/28 result are both retained; neither is a
dataset-wide verdict or a reason to select a favorable subset.

The remaining original fixed panel is exactly P3–P12, each series 1. The
coordinator authorized preparing `code/run_way_remaining_s1_support.py`:
sequential calls to the unchanged `one_file` and `PilotSettings`, using the
completed P2 tracing bindings and atomic writer. Every exact fitting-role
manifest entry, destination and size is admitted before any selected payload.
P1/P2 reports are recorded as references, not reopened; their payloads are not
reread. All ten ordered participant outcomes remain present. File-specific
technical failures are retained while other fixed files proceed within the
same cap; a deadline stops the pass and leaves unfinished participants unknown.
Every file has a durable stage report, and already returned support survives
later interruption. No thresholds, labels, EEG features, feedback, held-out
access, parameter changes or automatic retries are selected. Whole-HS decoding
still incidentally reads EEG, then the original function discards it.

Prepared envelope: one CPU, 4 GiB, all six numerical thread variables one,
600 seconds for the whole process, external 620 seconds plus 10-second grace.
The ten-file runtime extrapolation from P2 is roughly three minutes; this is
a sizing estimate, not a completion guarantee. The 600-second cap and 4-GiB
memory request allow margin for file-size and publication variability. The
registered resource ceilings are unchanged. The coordinator owns the actual
shared-step selection, verifies its remaining time, and accounts the parent
once; this preparation makes no new allocation charge or completion claim.
Existing scientific checks are reused. No tests, imports, raw arrays or
scheduler actions were executed by the implementation worker.

Prepared command, not a launch record:

```sh
python3.12 -u -B outputs/code/run_way_remaining_s1_support.py \
  --development-view "$SCRATCH/br_autoresearch/episode19_causal_motor_forecasting_benchmark/development_view_job_46425841" \
  --output-dir "outputs/runs/source_views_and_timing/20261005-remaining-S1-${SLURM_JOB_ID}" \
  --max-seconds 600
```

The main output is `remaining_s1_support.json`, with separate
`P3_S1/support_trace.json` through `P12_S1/support_trace.json`. This records
the remaining support panel without promoting it to forecast-ready labels.

## 2026-10-05 19:52 UTC — shared setup stopped before EP19; corrected parent selected

EP04 parent 46657948 stopped at its CPU guard before any EP19 step. The
8-GiB request exceeded the scheduler's observed 8000-MiB per-CPU limit and
received two CPUs. Root selected a 7-GiB corrected parent with the same
P2/S1 scope, original functions and 180-second internal bound. No MAT read
or support attempt occurred. The failed parent's 0.0144444444 core-hours
are charged once under EP04, not again here.

Corrected shared parent **46658777** was submitted at 19:52 UTC. P2 output is
`runs/source_views_and_timing/20261005-P2-support-46658777/`.

## 2026-10-05 19:44 UTC — fixed P2/S1 support trace selected, not yet executed

The coordinator selected one fitting-only source pass, not a retry of the
eleven-file panel. The new `code/run_way_p2_support_trace.py` admits only
`HS_P2_S1.mat` from the completed `job_46425841_development_view` manifest and
its existing `way/fit_and_inner_selection` destination. It calls the unchanged
`check_way_rest_support_panel.one_file` with participant 2 and the existing
`PilotSettings`. Five EMG channels, three hand-motion paths, fixed 0.5 cue
decoding, one-second candidate windows, stationarity rules and `support_only=True`
are unchanged. No thresholds, onset/risk labels, EEG features, forecast scores,
feedback or held-out results are produced. The whole-HS MAT load incidentally
deserializes EEG before the existing function discards it; a nonreturned load
leaves the extent of that access unknown. P1 and other participant payloads are
not reopened. P2's cue polarity remains conditional, and this pass cannot
authenticate return completion or the total onset timing bound.

The earlier step `46430982.17` is terminal TIMEOUT, exit 0:9, 158 seconds,
1,007,468 KiB peak RSS and 2.614 CPU seconds. Its output directory has no saved
report, and the old implementation wrote only at the end. The underlying
stopped operation therefore remains unidentified; this is not evidence of
an input defect, OOM or a diagnosed filesystem fault. The new entrypoint saves
atomic pre-call and returned/interrupted stage checkpoints around source
admission, imports, MAT loading, channel selection, cue decoding, sensor
summaries and support classification. Stage entry is not proof of completed
I/O. Returned metadata/support is retained even if a later deadline prevents
completion. Scoped wrappers restore all original bindings and change neither
scientific arguments nor returned values. Old code and attempt directories
remain unchanged; there is no automatic retry or follow-on panel.

The clock and policy were checked before this selection record. Current policy
limits remain 2,000 CPU core-hours, 480 GPU hours, 240 wall-hours and 2 TiB of
scratch. The coordinator's historical shared-parent bound is 4.3933333 core-hours
for `46425841` (four CPUs, 3,346 seconds) and `46430982` (one CPU, 2,432 seconds);
these are already charged shared allocations, not new EP19 charges or a complete
episode usage ledger. The new pass is to run before EP04 RSA in the selected
shared one-CPU, 8-GiB, 40-minute parent, accounted once under EP04. Its own step
uses one CPU and at most 4 GiB, all six numerical thread variables equal to one,
an internal whole-process cap of 180 seconds, and external 190 seconds plus
10-second termination grace. No GPU, new environment, source copy or allocation
is created by this entrypoint. The coordinator must verify remaining parent time
before execution; no job ID or completion is claimed here.

Use the existing Python 3.12 / NumPy 1.26.3 / sklearn 1.5.1 module stack recorded
below (sklearn supplies SciPy), preserving its `PYTHONPATH`. The eight previously
passing scientific checks are reused; no new qualification study or test-only
job is selected. In the coordinator's verified compute shell:

```sh
python3 outputs/code/run_way_p2_support_trace.py \
  --development-view "$SCRATCH/br_autoresearch/episode19_causal_motor_forecasting_benchmark/development_view_job_46425841" \
  --output-dir "outputs/runs/source_views_and_timing/20261005-P2-support-${SLURM_JOB_ID}" \
  --max-seconds 180
```

The fresh output is `p2_support_trace.json`. Preparation used only bounded
documentation/source inspection and editing on the login node. No scientific
payload, import, test or scheduler action was performed by the implementation
worker. The coordinator owns actual execution, accounting and interpretation.

Root and independent static review passed. Shared EP04 parent **46657948** was
submitted at 19:49 UTC. The selected P2 step uses fresh directory
`runs/source_views_and_timing/20261005-P2-support-46657948/` and the bounds
above. No source return or scientific result is established by submission.
Allocation cost belongs once to the EP04 parent ledger.

## 2026-10-03 22:20 UTC — ASTRA identifier repair validated

The schema-only identifier repair (`completed_p1_peripheral_pilot`) passes
the pinned validator in attended allocation `46433944`; the fresh result is
`astra/v0.0.14/validation_20261003_pre_recovery_group02.json` (exit zero).
The prior failed validation is retained. No source computation was retried:
the expanded rest-support route remains stopped after its bounded timeout.

## 2026-10-03 21:54 UTC — `source_views_and_timing`: rest-support-panel01 bounded stop

The coordinator reports that the selected panel did not return output by its
90second internal deadline. Root sent Slurm INT to step46430982.17, and the
scheduler reported cancellation DUE TO TIME LIMIT at21:54:52UTC, at the2minute
step limit. The parent40minute allocation also reached its end around21:54:53.
The worker performed no retry, source probe or scheduler query. Preserve
`runs/source_views_and_timing/20261003-rest-support-panel01/` and any report
or logs it contains. The coordinator owns checking the selected report, if
present, and terminal accounting before a next action is chosen.

The prior approximately30second estimate was too low. Runtime cause and
source-processing progress are UNKNOWN pending that evidence; do not infer
an input defect, completed panel, dataset-wide failure or new scientific
result from the timeout. The90second internal signal did not produce a returned
result before Slurm stopped the step; its underlying cause remains unresolved.
No attempt budget expansion or resubmission is selected. The changed EP15 and
EP19 ASTRA validations did not run because the allocated CPU remained occupied;
earlier validations do not cover the new recipes. Code and both programs are
frozen for the coordinator's inspection.

At21:56:06UTC the attended PTY closed with exit137 after Slurm killed the
timed-out step and parent shell. No compute commands ran after expiry. The
selected output directory exists but `rest_support_panel.json` is absent;
there are no saved participant results to interpret or reuse. The corrected
test invocation passed all8 tests in0.021s before the source stage. The
underlying runtime cause and exact completed source reads remain unknown.
Stop this affected source-processing route pending concrete diagnosis; no
retry, alternate-node workaround, or wider source probe is selected.

Final accounting records step46430982.17 TIMEOUT after2m38s, totalCPU2.614s,
MaxRSS1,007,468KiB and exit0:9. This establishes that CPU computation did not
dominate the wait; it does not identify the blocked operation. The2GiB memory
allocation was not exhausted by the recorded peak. For a user-reviewed support
report if needed: on sh03-06n50.int,2026-10-03 approximately21:52–21:55UTC,
job46430982 step17 failed to return from a fitting-source MAT support pass
within its90s internal/2min Slurm bounds, used only2.614s CPU, and saved no
final report. No shared-system configuration was inspected or modified, and
the affected route is stopped. The user decides whether to send this symptom
to Sherlock support; no contact has been made on their behalf.

## 2026-10-03 — `source_views_and_timing`: fixed S1 support panel selected, unrun

The coordinator adopted the fixed S1 panel across all12 WAY participants.
The roles are by **series**: S1–S6 fitting, S7 feedback and S8–S9 held for
every participant. DATASETS.md, SEARCH_POLICY.yaml and the completed
development-view manifest agree. P7/P8/P9 are participant identities, not
feedback/held roles. Reuse the completed P1_S1 pilot02 report without reopening
its source; read only HS_P2_S1 through HS_P12_S1 from the explicit fitting
directory, checking all11 exact manifest identities before any payload.

Prepared `code/check_way_rest_support_panel.py`, a sequential support-only
reader. The11 files total590,292,150 bytes, largest78,698,851 bytes. P1's1.22s
and463,044KiB peak provide a bounded cost reference; doubled file sizes imply
roughly30seconds and approximately1GiB peak for this panel, not a guarantee.
Use1CPU/2GiB, a5minute envelope and default240second internal deadline,
shortenable to fit the current allocation. No new source extraction or
scratch copy is needed. Output is
`runs/source_views_and_timing/20261003-rest-support-panel01/rest_support_panel.json`.

All eight paths, fixed0.5 cue decoding,1s windows, stationarity ratios,
parity split and50% support requirement are unchanged. One new focused
fixture checks that the support-only branch retains counts but returns before
any noise-median/threshold computation, even when support is sufficient.
Existing detector fixtures are reused. No tests or source execution ran in
the worker shell. Every participant and every window failure is retained;
timeouts explicitly retain uncompleted participant rows. No successful subset
is pooled, selected or promoted. Whole classic-MAT HS loads incidentally decode
EEG before discarding it; there are no EEG features, forecast scores, onset/risk
labels, WS timestamps, feedback or held-out payloads. Only P1_S1 has the prior
duration-based polarity evidence; other cue interpretations stay conditional.
Return remains unknown and the total onset timing bound remains NULL.

The first coordinator test invocation failed before imports/source access
because `PYTHONPATH=outputs/code` replaced the module-provided NumPy path.
This is an invocation error, not a detector or data failure. Correct the
command to prepend the local code while preserving the existing environment:
`PYTHONPATH="outputs/code${PYTHONPATH:+:$PYTHONPATH}" python3 -m unittest discover -s outputs/tests -p test_peripheral_onset.py -v`.
The coordinator selected a90second panel deadline within a2minute step for
the remaining allocation. No source stage launched during the failed invocation.

## 2026-10-03 21:45 UTC — `source_views_and_timing`: peripheral pilot02 completed, noise support unresolved

The coordinator ran the fixed0.5 conditional pilot in Slurm46430982, step13,
on sh03-06n50. All seven focused fixtures passed. The selected HS_P1_S1
replay completed in1.220730seconds with463,044KiB peak RSS; evidence is
`runs/source_views_and_timing/20261003-peripheral-pilot02/peripheral_pilot_report.json`.
The whole classic-MAT HS structure was incidentally deserialized, including
EEG, which was discarded without computing EEG features or forecast scores.
No WS timestamps, feedback-series values or held-out source were read.

There are34 eligible candidate pre-cue windows and13 satisfy all eight paths'
fixed quarter-stationarity checks:7 noise-fit and6 noise-validation windows.
The13/34 fraction is38.24%, below the prospective50% support requirement.
The total and parity counts are sufficient; the fraction is the unresolved
criterion. Motion drift and EMG quarter-noise instability account for the
rejections; every rejected window and path reason is preserved in the report.
All primary peripheral values are finite, so this is not evidence of missing
channels. No noise threshold was estimated or frozen, and the detector did
not generate onset labels, risk exposure, predictions or scores.

This outcome describes one fitting series and one candidate rest recipe.
It does not establish WAY-wide absence of stationary support, physiological
rest, or any positive/negative EEG forecasting result. The50% rule, sensor
paths, stationarity ratios, window length and all prior settings remain
unchanged. No favorable subset is promoted. Completed-return semantics,
provider annotation origin and total onset uncertainty remain unresolved;
the combined timing bound stays NULL and known-return stays false.

Proposed next empirical step, not launched by this entry: apply exactly the
same candidate-window support diagnostic to the fixed first fitting series
of all12 WAY participants, retaining every participant and every failed
window. This explicitly defined cross-participant panel can distinguish a
single-series limitation from broader rest-window inadequacy without finding
a successful subset or relaxing thresholds. Higher=illuminated support is
currently authenticated only for P1_S1's interval durations; other files must
retain conditional cue interpretation. Report support counts and rejection
causes only, without selecting or pooling successful files for production
noise thresholds. A prospective causal return predicate still needs the
provider's position-unit and object/force semantics separately; no LED-off
edge or candidate quiet interval is silently declared completed return.

## 2026-10-03 — `source_views_and_timing`: conditional peripheral pilot02 selected, unrun

Root accepted the distinction between fitting-file polarity support and the
unresolved absolute metadata origin below. Selected the same HS_P1_S1 file for
another peripheral-only pilot, changing only cue decoding to the fixed raw
source-unit comparison `ParticipantLED >= 0.5`. The95-level distribution and
all34 matching illuminated-interval durations support this threshold for this
fitting file. It uses the current sample only, without whole-file amplitude
normalization, timestamp shifting, WS metadata or future LED-off inputs.
Nonfinite samples remain unknown, and a gap-to-high transition cannot invent a
new observed low-to-high cue. This does not authenticate every development
file or resolve the provider annotation clock origin.

All previous `PilotSettings` are unchanged: every one of the five EMG and three
hand paths;20ms trailing summaries;500ms past EMG centering;40ms confirmation;
500ms stillness; the1s candidate-noise windows, quarter-stationarity criteria,
even/odd fit-validation division, minimum8 windows with4 per division and
at least50% eligible support;3/6 scaled-MAD thresholds; and5% maximum held-noise
any-path crossing fraction. No favorable path/participant subset or new noise
threshold is selected. Add one focused decoder check for smooth transitions,
the exact0.5 boundary, missing values and invariance to an arbitrary future
amplitude; the existing six detector fixtures are retained.

The replay remains conditional: low LED denotes a return instruction, not
completed return. Existing rest/noise support cannot prove that the object was
released and the hand returned to its rest location. Reports retain
`known_end_return_phase_authenticated=false`, `online_risk_contract_qualified=false`,
`scientific_labels_usable=false` and a null total onset bound. The selected
file's duration-based polarity support is named separately from global timing
authentication. No EEG feature, forecast fit, score, feedback or held-out
payload is allowed.

Prepared command is `code/pilot_way_peripheral_onset.py` with the existing
development view and fresh output
`runs/source_views_and_timing/20261003-peripheral-pilot02`. Reuse1CPU/2GiB and
the10minute envelope/internal540second deadline. The worker performed no tests
or source execution. Root reviews the code before source rerun.

The next useful sensor issue after this pilot is a causal completed-return
predicate, separate from noise support and onset timing. The provider's
digit-to-object distance rule can inform a prospective rule once position
units and its all-sensor consistency are checked; no such return predicate or
its thresholds are activated here. Acquisition response and threshold-crossing
uncertainty still require separate bounds before any scientific forecast score.

## 2026-10-03 21:35 UTC — `source_views_and_timing`: cue comparison completed; absolute origin unresolved

Root ran `check_way_cue_polarity.py` in Slurm46430982, step10. All three
field-reader fixtures passed in0.001seconds. The comparison completed in
1.212392seconds with344,956KiB peak RSS. Evidence is
`runs/source_views_and_timing/20261003-cue-polarity01/cue_polarity_report.json`.
The reader decoded exactly102 LED metadata scalars for34 windows, skipping
129,374,432 bytes of unrequested matrix payloads without numeric waveform
decoding. The continuous HS load again incidentally deserialized EEG before
discarding it; no EEG feature or score was computed.

All34 upcrossings and34 downcrossings at0.5 are exactly2seconds earlier
(within floating-point precision) than the attempted
`trial_start_time + LEDon/LEDoff` convention. The comparison completed; it did
not pass absolute-clock authentication. The report and attempt are preserved.
No fitted2second correction is applied. Reinspection confirms that the
official `WEEG_GetEventsInHS.m` adds `StartTime` to relative event times and
`WEEG_FindEvents.m` copies `ws.win.trial_start_time` unchanged. The paper
describes the absolute trial start and separately the window's2second pre-cue
padding. No official correction instruction was found; this convention
inconsistency remains an explicit unresolved source issue.

Both on and off residuals have the identical2second discrepancy for every
window. Therefore the34 raw high durations `(down-up)` equal all34 official
illumination durations `(LEDoff-LEDon)` without fitting or shifting any origin.
This supports higher=illuminated polarity for this fitting file, while leaving
absolute annotation timing unresolved. Opposite-polarity errors vary across
trials rather than sharing this origin discrepancy. This interpretation does
not turn LED-off into completed return or establish the onset uncertainty
bound. No provider movement-event value was read or used.

## 2026-10-03 — `source_views_and_timing`: fitting-only LED polarity comparison selected, unrun

Following the completed LED diagnostic, root authorized preparation of a
narrow comparison against official LED metadata from the same fitting series.
The exact allowed source member is
`$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/ep19_public_sources/archives/way/P1.zip::WS_P1_S1.mat`,
whose existing directory inventory reports50,960,640 uncompressed bytes.
The only selected WS numeric fields are `ws.win.trial_start_time`, `.LEDon`
and `.LEDoff` for every window. This is source-semantic authentication, not
admission of provider-windowed signals or future LED-off information into
predictors. No AllLifts file, movement-event scalar, feedback member or held-out
payload is opened.

`code/read_way_cue_metadata.py` parses classic little-endian MAT element tags,
retains only those three scalar fields and skips all other matrix payloads by
their declared byte lengths. Unsupported structure/layout fails without a
full-load fallback. Nested/compressed byte materialization can include waveform
bytes; the report explicitly distinguishes that access from waveform numeric
array decoding. Internal decompression is capped at512MiB. Three focused
fixtures include unreadable waveform/event payloads to test the field boundary,
compressed and uncompressed input, and missing/truncated metadata.

`code/check_way_cue_polarity.py` compares every official
`trial_start_time + LEDon/LEDoff` pair with fixed0.5 up/down crossings in the
already-authorized continuous fitting `HS_P1_S1.mat`. Selecting0.5 for this
comparison follows the observed exact0/1 plateaus; no production decoder is
changed. The HS loader again incidentally deserializes EEG, then retains only
ParticipantLED. Preserve all windows in recorded order, require equal counts
for a complete comparison, and report signed/direct absolute errors for normal
and opposite polarity. Do not fit an offset, match a successful subset, export
the future cue table as predictors, or set a scientific timing pass threshold.
Even perfect agreement only establishes cue semantics for this fitting file;
known return, broader development eligibility, causal acquisition response and
the full onset uncertainty remain to qualify.

Run the three focused fixtures followed by the comparison in root's attended
allocation with1CPU/2GiB/4minutes; the comparison has an internal180second
deadline. Planned output is
`runs/source_views_and_timing/20261003-cue-polarity01/cue_polarity_report.json`.
No worker tests, source processing or scheduler action occurred. This new ASTRA
recipe awaits coordinator schema/semantic validation and does not depend on
changing any scientific margin.

## 2026-10-03 — `source_views_and_timing`: LED diagnostic completed

The coordinator ran `inspect_way_led_levels.py` in Slurm46430982, step7.
It completed in0.387706seconds at148,436KiB peak RSS. Evidence:
`runs/source_views_and_timing/20261003-led-diagnostic01/led_diagnostic.json`.
Of119,496 finite samples,57,496 equal0 and61,902 equal1; only98 differ from
those exact plateaus. The95 unique levels span−0.0017 to1.0009, including small
undershoot/overshoot and intermediate values. Every prespecified retrospective
threshold produces34 upcrossings and34 downcrossings. At the central threshold,
the median complete high/low comparison-run durations are3.527/3.252seconds;
the displayed first20 crossing times differ by at most one500Hz sample across
the three thresholds. All34 crossing-time sensitivity differences were not
exported by this diagnostic, so no all-edge timing bound is asserted.

The result supports a two-state recorded channel with non-exact transitions,
not a95-state task cue. It does not establish whether those transition samples
come from recording, synchronization interpolation or another acquisition
operation, and does not itself assign illuminated/off polarity. No onset,
noise threshold, return rule, risk state, EEG feature or forecast score was
computed. The existing exact-two-level pilot result remains preserved; the
decoder has not been changed. All total onset timing components previously
unknown remain unknown.

## 2026-10-03 21:29 UTC — `source_views_and_timing`: LED diagnostic selected, unrun

The coordinator requested diagnosis of the95-level ParticipantLED result before
any cue decoder changes. Primary source review found that SC/ZOOM sampled the
peripheral LED channel at500Hz; the paper describes5kHz-to-500Hz anti-alias
downsampling for EEG, not for ParticipantLED. Neither analog measurement noise
nor resampling is established as the cause of the95 levels. The official
`WEEG_FindEvents.m` copies `ws.win(L).LEDon` and `.LEDoff`; it does not decode
the continuous LED channel. `WEEG_GetEventsInHS.m` adds window offsets to
AllLifts event values. These code paths do not authenticate numerical polarity,
and no WS or AllLifts payload is opened for this diagnosis.
[Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/),
[official event code](https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_FindEvents.m),
[official continuous-time conversion](https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_GetEventsInHS.m).

The Methods also distinguishes two cue-generation routes: without a surface
change, cue onset follows digits being at least15cm from the object for1–3s;
after surface replacement, a researcher button starts a0–2s wait. This is
useful for a subsequent causal return rule, but does not certify every return
or the released channel polarity. The provider's `LOC_HandStop` uses the later
part of an entire trial to set its positional threshold, so it cannot be copied
as an online rearm rule. The total onset uncertainty remains unknown.

Prepared `code/inspect_way_led_levels.py` for the exact same fitting file
`way/fit_and_inner_selection/HS_P1_S1.mat`. Before reading its values, fixed
diagnostic summaries are: finite/missing counts; amplitude quantiles at
0,.001,.01,.1,.25,.5,.75,.9,.99,.999,1; exact value counts (at most256 levels);
the five largest adjacent value gaps; and comparison crossings at25%,50%,75%
of the full observed range. Report edge counts, first20 sample/time locations,
crossing-interval and complete above/below comparison-run durations. These
full-file amplitude thresholds are retrospective diagnostics, not prospective
scaling or selected cue thresholds. Above/below labels carry no on/off meaning.
No source waveform is emitted, and no onset, noise fitting, risk state or model
is computed. All existing onset/decoder code remains unchanged.

Classic nested MAT decoding necessarily materializes all modalities, including
EEG, before only ParticipantLED is retained for analysis; report this explicitly.
Use1CPU/2GiB and a3minute envelope in the existing attended allocation, with
an internal120second deadline. Planned output is
`runs/source_views_and_timing/20261003-led-diagnostic01/led_diagnostic.json`.
No tests, source processing, scheduler checks or submissions were run by this
worker. The prior ASTRA validation predates this new diagnostic recipe.

## 2026-10-03 21:22 UTC — peripheral pilot completed, cue unresolved

Root executed the selected HS_P1_S1 pilot in attended allocation 46430982,
step 3, on sh03-06n50 with one CPU and a ten-minute step bound. Six new
component tests passed. The pilot completed in 0.676 seconds with peak RSS
197,972 KiB and status `peripheral_pilot_unresolved_cue` (exit 0 means a
completed diagnostic, not scientific qualification). The participant LED
signal contains 95 finite levels, so the prospectively selected exact
two-level decoder could not define candidate cue segments. No noise
thresholds, accepted onsets, online risk labels or EEG forecasts were produced.
Eight primary peripheral paths were finite; the EMG/kinematic/misc sampling
rates were the expected 4000/500/500 Hz. Nested MAT decoding incidentally
deserialized the EEG array, which was discarded without feature/scoring use.
No feedback or held-out source was accessed.

Evidence: `runs/source_views_and_timing/20261003-peripheral-pilot01/peripheral_pilot_report.json`.
Numeric cue polarity and completed return-phase semantics remain
unauthenticated. The acquisition/filter/threshold onset bounds remain unknown,
so the required combined uncertainty below 300 ms is not qualified. A source-
grounded cue interpretation is needed before changing this diagnostic; this
result is neither a positive nor a negative forecasting finding.

## 2026-10-03 21:10 UTC — `source_views_and_timing`: conditional cue boundary repair before execution

Root review identified that two observed ParticipantLED levels cannot by
themselves establish numerical polarity or a completed end/return phase.
The [source methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/) describe
illumination as the reach instruction and extinguishing as the instruction to
replace the object and return the hand. They do not establish whether the
released higher numerical value means illuminated, and a lower level plus a
subsequent rise does not directly authenticate completed return. Those source
semantics remain unresolved; no raw values have been inspected for this pilot.

The unchanged selected numerical replay is now explicitly **conditional** on
the higher-level/start assumption. Report keys name candidate cue segments,
candidate sensor onsets, and conditional candidate risk exposure. Top-level and
replay fields set numerical-polarity authentication, known-end/return
authentication, online-risk qualification and scientific-label usability to
false. Pre-edge noise windows are likewise conditional, not known pre-cue rest.
This repairs the overstatement in the earlier selection entry; it does not
invent an end/return detector or relax the contract. All selected thresholds,
support criteria, fixed source, six tests, resource bounds and command stay
unchanged. Root may use the pilot for peripheral/timing preparation only.
Qualified forecast labels require source-supported cue polarity and an
executable known end/return rule before their later use.

## 2026-10-03 21:02 UTC — `source_views_and_timing`: selected peripheral onset pilot

Before any peripheral values are inspected, the coordinator selected one
substantive development-only pilot on the existing fitting view
`$SCRATCH/br_autoresearch/episode19_causal_motor_forecasting_benchmark/development_view_job_46425841/way/fit_and_inner_selection/HS_P1_S1.mat`.
The raw five-EMG and three hand-position paths remain required together:
P2 is index finger, P3 thumb, P4 wrist; P1 object and force channels cannot
replace them. The mapping and source alignment context are from the
[official WAY methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/).
This pre-score detector development is delegated by the operative contract;
it does not freeze scientific margins or change the target, roles or budget.

The selected implementation is `code/pilot_way_peripheral_onset.py`, using
`code/peripheral_onset.py`. Before source access the same run executes six
focused new detector tests: finite-window equivalence to the already-qualified
CausalFIR, future/constant-offset invariance, pending-state exclusion and one
reach per cue, earliest accepted crossing, stationary support, and gap/cue
handling. The existing eight passing forecasting-component tests are reused.
No test or pilot has yet run under this entry.

Prospective settings: native 4 kHz EMG is centered by the previous 500 ms,
then receives a trailing 20-ms RMS; native 500 Hz hand position receives a
backward difference and trailing 20-ms speed mean. EMG summaries enter each
500 Hz tick using only the native sample at that tick and earlier samples.
Unknown primary samples reset all sensor histories; no future padding,
resampling interpolation or session normalization is used. Exactly two finite
recorded ParticipantLED levels are required by this pilot; a multilevel cue
returns unresolved without silently choosing a new threshold. All paths use
40-ms confirmation and 500-ms entry stillness. The label is the earliest
accepted crossing, never a timestamp shifted by an assumed filter delay.
Pending predictions remain absent even when a crossing is rejected. One cue
can produce only one primary reach; known LED-off ends that cue's risk set.

The noise-support rule is selected before raw values. Every complete 1-s
window immediately preceding a raw LED rising edge is a candidate only if the
LED was low throughout. No quietest-window ranking is allowed. All three
position paths must have four-quarter mean displacement no greater than three
times within-quarter RMS dispersion. Each EMG path must have positive
within-quarter RMS, maximum/minimum RMS ratio at most two, and quarter-mean
drift no greater than three pooled within-quarter SD. These are operational
stationarity checks, not proof of physiological rest. Preserve every window
and rejection reason. Require eight passing windows, four at each cue parity,
and at least half of eligible windows passing. Even cue indices fit noise;
odd indices validate it. Only each window's final 480 ms supplies summaries,
so their complete 520-ms EMG history lies inside the candidate window.

Each path's thresholds are its fitting median plus respectively 3 and 6 times
normal-consistent MAD (1.4826 times raw MAD). Zero/nonfinite noise scale is
unresolved, with no epsilon floor. The unused parity must have at most 5% of
summary samples exceeding any high threshold. Failure produces an unsupported
pilot, not a negative neural result or a favorable sensor subset. Threshold
development uses this fitting series; its subsequent fixed-parameter replay
is not out-of-sample scientific evaluation. The future-cue window selection
has no role in the streaming risk-state logic.

Selected command, in coordinator-owned Slurm only, with numeric thread counts
one and the existing Python/NumPy/SciPy module stack:

```sh
python outputs/code/pilot_way_peripheral_onset.py \
  --development-view "$SCRATCH/br_autoresearch/episode19_causal_motor_forecasting_benchmark/development_view_job_46425841" \
  --output-dir outputs/runs/source_views_and_timing/20261003-peripheral-pilot01 \
  --max-seconds 540
```

Envelope: one CPU, 2 GiB, ten minutes, 540-s internal deadline, compact durable
JSON and test transcript only. The classic MAT loader incidentally decodes
the nested EEG array, then discards it before any peripheral calculation;
this is acknowledged development EEG-array access, not EEG feature/scoring
access. No waveform, provider event, forecast score, series-7 feedback or
held-out signal is reported/read by the pilot. It returns cue/noise support,
crossings and risk exposure only if supported, runtime/memory, and an honest
timing ledger. Published alignment of at most 2 ms is source-method evidence,
not newly measured alignment. Acquisition/filter-response and crossing
uncertainty remain unknown; total onset bound remains null. A long past
centering history is not itself a delay bound. This run cannot qualify the
strictly-below-300-ms total bound or promote candidate EEG scoring.

## 2026-10-03 — selected WAY schema diagnosis after structural adapter failure

The selected one-file structural read in job 46425841 deserialized the
permitted development `HS_P1_S1.mat` and failed with `KeyError: participantnum`.
No structural success report or subsequent ASTRA validation was produced.
The historical empty scratch output directory is preserved. This is a source
adapter defect, not an onset-timing or forecasting result.

Selected next bounded step: `code/inspect_way_schema.py` reads the same single
development file and emits field names, Python types and array shapes only.
It never emits scalar values (including identifying strings), signal values,
onsets or scores. Use one CPU, 4 GiB, at most ten minutes within the current
attended allocation, and fresh durable output
`runs/source_views_and_timing/job_46425841_way_schema_attempt002`.
An actual schema-supported adapter repair follows this observation; public
documentation alone did not authenticate the original assumed field name.

The schema read completed in job 46425841. The source has integer fields
`participant` and `series`; the five modality groups retain the documented
`names`, `samplingrate` and `sig` schema. No scalar or array values, including
the source `name` string, were emitted. The preserved observation is
`runs/source_views_and_timing/job_46425841_way_schema_attempt002/way_hs_schema.json`.
Corrected only the two participant/series lookup keys. Selected the same
one-file structural read in fresh scratch `way_structure_job_46425841_attempt003`,
with its compact report preserved to
`runs/source_views_and_timing/job_46425841_way_structure/`. Retain the same
one-CPU, four-GiB, ten-minute maximum; no scope or scoring change.

## 2026-10-03 — `source_views_and_timing`: selected structural source inventory

The scientist authorized feasible continuation across episodes. Full operative
contracts and the existing component design were read. The named read-only
`ep19_public_sources` quarantine contains12 WAY ZIP archives and one self-paced
ZIP archive; they are already acquired and will not be downloaded again.
Only immediate directory names and `QUARANTINE.md` were inspected on login.
No archive payload, signal, event value or held-out participant header was read.

The selected compute effect is `outputs/code/inventory_source_members.py`:
read ZIP central-directory metadata only, write an archive-member index and
aggregate role classification under a fresh episode scratch directory, and
report unresolved filenames without guessing their role. No member is
decompressed or extracted. WAY series1–6, series7 and series8–9 remain distinct;
the15 versus8 self-paced participant lists remain exactly the policy lists.
Mixed event files and unclassified members remain unavailable to candidate
code. AJILE12 is outside the primary study and is not inspected.

The coordinator owns the Slurm step: one CPU,2GiB and at most10 minutes.
This structural operation does not establish clock alignment, timing bounds,
valid onset rules, physical role separation or EEG-scoring readiness. Next
source preparation depends on the actual member layout and may read only
explicitly identified development text headers or source documentation.
Scientific margins stay unset and zero forecast trials/final openings occur.

Actual structural inventory completed at2026-10-03T19:00:03.231055Z on
`sh03-06n50.int`, allocation `46425841`. It found1265 members: WAY144
fit/inner-selection,24 series7 feedback,48 held-out and24 mixed/unresolved;
self-paced650 development,359 held-out and16 dataset-level/unresolved.
These are member counts, not runs or participants. The actual artifacts are
`outputs/runs/source_views_and_timing/job_46425841_inventory/`.
WAY uses `HS_P<n>_S<s>.mat` and `WS_P<n>_S<s>.mat`; each participant also has
`HS_P<n>_ST.mat` and `P<n>_AllLifts.mat`, left closed as unresolved/mixed.
Self-paced includes raw BrainVision files and separately derived MAT files;
the latter are not substitutes for the required continuous raw input.

Selected next effect: inspect only development-participant `.vhdr` text
headers in the self-paced archive, returning sampling/channel/layout metadata
and checking source-member references without opening EEG, markers or event
values. Copy only three dataset-level documentary members (`README.md`,
`task-reachingandgrasping_events.json`, `code/experiment/PlayStartEnd.m`) for
cue/annotation semantics review. No held-out header is read. The command is
`outputs/code/inspect_development_headers.py`, one CPU/2GiB/10min maximum in
the coordinator's allocation, with a fresh scratch output directory. This
does not yet certify raw clocks, channel data, timing bounds or valid-run
eligibility. Initial ASTRA validation was not executed because of a coordinator
executable-path typo (exit127); the corrected validator is separate from
scientific execution and pending verified output.

Under `past_only_controls`, selected additional implementation is the existing
19-category censored likelihood, provisional four-parameter calibration and
fixed-slot EEG construction from `component_details.md`. One-sided finite
filters require actual warm-up history and reset at gaps. The complementary
and older-slow parts share the prescribed voltage scale; masking precedes any
family feature/state update. The component module accepts no source paths.
Focused checks will compare analytic calibration gradients with independent
finite differences, distinguish first-bin events from zero-bin censoring,
check probability normalization and identity fallback, and mutate forbidden
input segments. These qualify components only, not the five reference models,
onset timing, known-signal recovery or numerical/scientific margins.

The selected component qualifier is
`outputs/code/qualify_forecast_components.py --output-root <fresh-scratch-directory>`:
eight focused checks, one numeric CPU/2GiB and at most10min in the attended
allocation. Source rows are absent; synthetic tensors exercise the same
19-category layout and three six-bin calibration groups required by both
acquired sources. Right-censored zero-bin rows have zero loss/gradient in the
whole-grid mean objective; first-bin observed events still contribute.
Future-sample and both forbidden-block tests check reconstructed slots only;
the tests do not claim that any unimplemented EEG model's state is qualified.

Actual development header inspection completed at2026-10-03T19:03:00.080995Z
in allocation `46425841`:155 raw BrainVision headers cover all15 development
people;144 are250Hz/39 channels and11 are1000Hz/67 channels. Every header
contains all29 scientific channels. The report and three documentary members
are preserved under `outputs/runs/source_views_and_timing/job_46425841_headers/`.
TRIG and X/Y/Z are explicitly named; auxiliary header units are amplifier
microvolts, not an established physical-acceleration calibration. The source
events dictionary confirms post-hoc onset extraction and ultimately selected
cup labels; these cannot become current predictor inputs. No EEG, marker,
event-value or held-out header was opened.

The first component qualifier attempt used an existing environment without
SciPy and could not import the test module. This is an environment failure,
not a failed equation. The coordinator selected the existing Python3.12,
NumPy1.26.3 and sklearn1.5.1 module stack (which supplies SciPy) for an unchanged
code retry in a fresh output directory. Both attempts are to be preserved.

Actual retry of the selected component checks **PASS**: eight tests in0.014s
under the existing Python3.12/NumPy1.26.3/sklearn1.5.1 module stack, on
allocation `46425841`. The failed import attempt is preserved at
`outputs/runs/implementation_qualification/job_46425841_attempt001/`; the
successful unchanged-code retry is at the corresponding `_attempt002/`.
Official ASTRA schema and semantic validation of the component/header recipes
also **PASS**, saved as `outputs/astra/v0.0.14/validation_20261003_components.json`.
The subsequently added development-view extraction recipe will receive its
own affected-program validation. The eight passes do not qualify an onset
detector, actual32-event packet, model state, signal score or scientific margin.

Selected next substantive source preparation is a trusted-builder extraction
using `outputs/code/prepare_development_views.py`. From the observed index,
the exact allowlist contains84 continuous WAY HS files for series1–7 and465
BrainVision members (155 raw triplets for15 development people):549 members,
9,038,196,400 uncompressed bytes. WAY series7 is placed separately from1–6.
All WS future-windowed records, HS_ST, AllLifts, derived MAT, event TSV and
held-out members are excluded. Streaming copy reads only selected development
payload bytes, without computing/displaying signal values. It writes a fresh
scratch view and a durable-copyable report; originals remain unchanged and no
held-out view is opened. Planned envelope is one CPU,2GiB and at most20min,
charged to source preparation, with9.1GB scratch. Read-only copied files are
not represented as OS-level evaluator isolation. The view is regenerable and
does not certify onset, timing or candidate-scoring readiness.

The exact dry member declaration is `outputs/development_member_plan.json`,
written from the already-observed central-directory index before extraction.
The builder requires this549-member/9,038,196,400-byte plan, rechecks each
member's development role and archive identity, and verifies every selected
central-directory size before decompressing any member. It writes the selected
plan alongside the new scratch view. These checks address the concrete mixed-
archive exclusion risk; no source hashes, signatures or generic receipt chain
is added.

After a complete development view, the selected direct WAY structural check
is `outputs/code/inspect_way_development_structure.py`. It reads only the
explicit fitting file `HS_P1_S1.mat` and checks internal participant/series identity,
modality names, declared sample rates, array dimensions and scientific montage
coverage. SciPy must deserialize nested development arrays to obtain these
fields; this is acknowledged signal-array access, not a metadata-only claim.
The script emits no raw values, event labels, onset estimates or forecast
scores. It excludes held-out files by construction. Envelope: one CPU,4GiB,
10min maximum for this single file. The full timing/synchronization bound remains
unverified even if these structural checks complete.

Actual development extraction completed successfully at 2026-10-03 19:17:32 UTC
in allocation `46425841`, as observed by the coordinator: 538.5855 seconds,
549 members and 9,038,196,400 bytes. The view contains 72 WAY fitting files,
12 WAY feedback files and 465 self-paced development files. No held-out,
provider-windowed or derived member was extracted. Development payload bytes
were read to stream-copy these files; source archives were not modified.
The regenerable view remains in episode scratch, with the report and selected
member declaration preserved and directly verified under
`outputs/runs/source_views_and_timing/job_46425841_development_view/`.
Extraction completion does not establish timing, onset validity, final evaluator
isolation or readiness for candidate scoring.

The selected one-file structural inspection completed at
2026-10-03T19:23:26.775100+00:00 on `sh03-06n50.int`, allocation `46425841`,
after the coordinator's recorded key-name diagnosis/repair. The actual MAT
uses `participant` and `series`; the first attempt's `participantnum` lookup
failed before a structural report. Its failure and separate schema-only
attempt remain preserved. The successful report is
`outputs/runs/source_views_and_timing/job_46425841_way_structure/way_development_structure.json`.
It covers only fitting file `HS_P1_S1.mat`: EEG is 119,496 by 32 at 500 Hz;
EMG is 955,969 by 5 at 4,000 Hz; kinematic, environment and miscellaneous
streams have 119,496 samples at 500 Hz and respectively 36, 2 and 6 channels.
All 29 scientific EEG channels are present and every modality's channel count
matches its named channels. Nominal durations around 239 seconds do not prove
alignment or an onset uncertainty bound. Nested development signal arrays were
deserialized; waveform values, subject initials, event values, onset estimates
and forecast scores were not emitted, and no held-out source was read.

Official schema and semantic validation of the final source/component program
**PASS**, as verified in
`outputs/astra/v0.0.14/validation_20261003_development_view.json`.
The source-preparation stage now has concrete raw development views and one
WAY structural example. It still needs development-only channel-to-sensor/cue
semantics, acquisition/filter timing and gap checks, executable stillness/onset
rules and their total uncertainty bound strictly below 300 ms. Detector settings,
model recipes and numerical/scientific margins remain pre-score work; no
candidate EEG score, valid forecast trial or final opening has occurred.

2026-10-02 — Concept figure: generated/edited and visually inspected [ep19_question_imagegen-v2.png](ep19_question_imagegen-v2.png) with built-in image generation; exact prompts in [ep19_question_imagegen-v2-prompt.md](ep19_question_imagegen-v2-prompt.md). Replaced the flat-peripheral assumption and pseudo winner bars; made older complement retention and all four input combinations explicit, with a hand-onset illustration. Earlier assets retained; no analysis, outcome access or scientific-rule change.

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md): Added direct contrasts with prior premovement prediction and cued-to-self-paced asynchronous detection. Clarified sensor-conditioned positive/null claims and retained model search as secondary. No neural data or held-out outcomes were opened and no model was run.

## Current entry

- Stage: component design refinement, 2026-09-30
- Primary question: does EEG add information about movement onset 300–600 ms
  ahead beyond cue, context, and past peripheral sensors?
- Secondary question: do the same model families change order between 0–300
  and 300–600 ms?
- Work completed: onset/risk states, exact EEG factorization, peripheral
  baseline inputs and the four-parameter calibration recipe are specified in
  `component_details.md` and reflected in the contracts and paper plan.
- Signal analysis: not started
- Held-out results examined: none
- Scientific result: none
- Active jobs/processes: none launched by this design task

## component_specification — design milestone

Three parallel, read-only internal reviews covered onset, block construction,
and baseline/calibration; the main agent alone edited the episode. Selected
detector and optimizer defaults remain development-checkable proposals.
Reviews clarified continued detection during pending states, adapters within
original family allowances, and first-bin events versus zero-bin censoring.
Public methods were consulted; no source signals or held-out outcomes were read.
No experiment, scheduler action or Git mutation was performed.

Artifacts: `component_details.md`, updated `../GOAL.md`, `../DATASETS.md`,
`../SEARCH_POLICY.yaml`, and `paper_plan.md`.

Program: `astra/v0.0.14/astra.yaml` now records the existing experiment
dependencies, not execution. The pinned ASTRA validator is unavailable on
PATH; schema/semantic validation remains open and is not an execution gate.
The focused reference/dependency review found no missing producer, cycle or
partial held-out output. Secondary factorial reports can record failure or
unavailability; their existence is not a primary-result success gate.

Scoped tracked-file and new-file whitespace checks passed. These document
checks are not runtime validation of onset, factorization or calibration.

Remaining work is implementation and development validation, not more evidence
from this documentation task. Scientific margins are unchanged and unset.

## Next entry should record

`source_views_and_timing`: after signal-access/execution authorization, prepare
role-separated views and check development clocks, channels, sampling, gaps,
sensor completeness, causal cue availability and the full onset timing bound.
Do not return held-out signals or candidate EEG scores from this step. Then
record the required method settings and past-only/calibration implementation
checks under `development_specification` and `past_only_controls`.

Future entries should distinguish scientific trials from software repairs and
duplicate runs.
