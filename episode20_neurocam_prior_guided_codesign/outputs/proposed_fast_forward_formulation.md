# ADOPTED 2026-10-12: a feasible full-size EP20 forward formulation

**Adoption, 2026-10-12 22:13 Pacific.** The scientist was asked to approve
Route C (with Route B as fallback and a 2026-10-18 checkpoint), template-level
validation in place of per-window replay digests, and per-record rather than
per-interval energy output. The scientist replied that compute limits could
be raised as needed. That reply
is recorded as approval of the recommended plan and as authority to raise
compute caps; see `scientist_amendment_20261012.fast_forward_route` in
SEARCH_POLICY.yaml. The text below is kept as the adopted specification; its
"unadopted" wording describes the draft as prepared.

**Status as drafted: unadopted proposal dated 2026-10-12; not a log entry, result or amendment.** It serves step 1
of the relaunch plan in the [2026-10-12 21:46 log entry](experiment_log.md) under
`scientist_amendment_20261012` (336-h clock restarted 2026-10-13T04:46Z; formal-run failures only,
6 of 12 used). Preparation was read-only: no project code, tests, Slurm or Git (one stray no-op
`python3 -c` print ran on the login node, touching no project code or data), and nothing opened under
`inputs/`, no world or noise array, no Rat3, final-model or outcome material. **Measured** = retained
in a cited log entry or report; **derived** = arithmetic on measured values; **estimate** = my judgment.

## 1. What the forward model must keep

- **Contract.** [GOAL](../GOAL.md), [DATASETS](../DATASETS.md) and [SEARCH_POLICY](../SEARCH_POLICY.yaml)
  fix the 64 × 64, 150-µm, one-TFT hardware, the published frontier and legal schedules. Contact and
  state templates (E_area/E_sublinear, R1–R3) give noise, impedance, settling and crosstalk from one
  circuit; coupled device mechanisms, accounted resources and paired worlds and draws stay. The
  amendment keeps model, parameters, schedules, noise, outputs, conditions, comparisons, thresholds,
  data roles and final test, and requires reproducing the reference on small arrays and short
  windows within stated tolerances before full-size use.
- **Selected recipe** ([conditional_forward_recipe.md](conditional_forward_recipe.md)). B2/2001 with
  69,848 states: 4,096 pads, 64 sources, 65,536 CPE branch states, 64 buffers, 64 filters, 24 holds.
  16-branch CPE (poles 2π·10⁻⁴–2π·10⁸ rad/s); TFT 2 µS on, 20 pS off; mutual capacitance 0.02 pF
  between pads, 0.2 pF between sources; gain-8 drive clipped at ±50 mV; 20-pF/1-mS holds; 2.5-µs
  slots, 0.5–2.0-µs apertures, 18-bit ±10-mV quantizer, pixels from words 3–5. Tolerances: 10 nV and
  1 pA on inputs, rtol 10⁻⁵, 10⁻⁸ V, 10⁻²⁰ C, 10⁻¹⁸ J.
- **Data-access stage.** This is `model_specification` development preparation: no fitted
  reference or candidate score exists, Rat3 is closed and the final models are not implemented. All
  proposed work uses only saved B2/2001, the retained attempt002 noise draw and numerical fixtures.

## 2. Measured facts

| Quantity | Measured value | Evidence |
| --- | --- | --- |
| Full record | 16 s = 4,000 × 4-ms cycles; 6.4 M 2.5-µs actions; 104 M conversions; 16.384 M pixel + 1.6 M marker observations; ≥ 18.13 GB of tables | recipe; log 10/9 10:10 |
| Hold preparation | 960.6 s and 968.5 s for 12.8 M holds and 38,400,001 work units | `conditional_preflight_cost_attempt001/002` |
| Whole-plan preflight | > 4,987.8 s and > 4,979.4 s, unfinished; completed windows unknown | same; log 10/9 16:48 |
| Constant-hold method | ≥ 1,347,979,725 holds, against a 40 M cap | log 10/9 10:26 |
| Solver intervals | 4 per slot (0.5, 0.75, 0.75, 0.5 µs) in both measured slots | component attempts 002–004 |
| Direct solver, per interval | slot 1: 4.49, 1.30, 0.72, 0.41 s; slot 2 (mux change): 0.94, 0.42, 0.47, 0.41 s | `conditional_component_cost_attempt004` |
| Accepted parts; band factorizations | 32, 8, 4, 2 \| 8, 2, 2, 2; 41, 11, 5, 2 \| 4, 2, 2, 2 | `component_comparison.json` |
| Contact-block GMRES, late intervals | 140–152 operator products, 0.53–0.57 s each | log, contact-block repair entry |
| Traced profile (GMRES) | previews 15.5 s, GMRES 9.95 s, G products 5.37 s, LU 0.008 s; noise setup 45.8 s | log, bounded-closeout entry |
| Memory; consumption | 1.13–1.18 GB peak RSS per attempt; 5.90 owned (10.20 conservative) of 4,000 CPU-h | reports; log 10/12 21:46 |

## 3. Where the time goes, and why

**Planning and preflight: ≥ 1.65 h measured, unfinished, before any electrical advance.**
[repeating_acquisition_schedule.py](tools/repeating_acquisition_schedule.py) replays one immutable
4-ms template 4,000 times. Even so, [streaming_array_acquisition.py](tools/streaming_array_acquisition.py)
re-checks and re-digests every window record by record: about 3,200 fragments, each with a
4,096-value field evaluation and a SHA-256; JSON canonicalization of ~26,000 conversions, 26,000
returned samples and 4,496 observations; and 6,400 dry-readout split checks. Source-derived, that is
about 2.4 × 10⁸ canonicalized records and 2.6 × 10⁷ split checks per plan in CPython. Hold
preparation in [raw_input_hold.py](tools/raw_input_hold.py) and
[streaming_midpoint_hold.py](tools/streaming_midpoint_hold.py) bisects every slot in Python, about
76 µs per hold (derived), although B2's single raw segment splits every slot the same way.

**Electrical stepping: the dominant cost.** The second measured slot took 2.24 s with the fastest
solver. If representative, a record costs about 1.4 × 10⁷ s, or 4.0 × 10³ CPU-h (derived); the
accounting of 19.2 M intervals at 0.5 s gives 2.7 × 10³. A 6,000-s attempt therefore needs a
1,600–2,400× speedup. The causes are visible in [conditioned_array_solver.py](tools/conditioned_array_solver.py),
[cpe_array.py](tools/cpe_array.py) and [radau_triangular_contact_schur.py](tools/radau_triangular_contact_schur.py):
1. Each interval rebuilds the full edge and conductance arrays and re-inspects them, including a
   Python loop over 4,096 contacts. The factor cache dies with the interval, so each late interval
   pays two band factorizations (about 0.11 s), although only 64 upstream configurations exist.
2. Partition doubling costs 3 two-stage Radau steps per 2-part interval and 15–63 after events,
   about 24 per slot. Each step evaluates 78,232-port forcing and full C/G residual and work.
3. Every mux change forces 8-part intervals. The trigger is the 20-ns hold transient
   (20 pF × 1 mS), which the large upstream system never sees.
4. A circuit that is linear apart from 64 rail clips runs through a general nonlinear Newton path,
   with CPython orchestration: a stage solve of about 2 × 10⁷ flops takes about 0.05 s.

**Readout and output: never reached, so estimated by count.** 104 M conversions and returned
samples, 18 M observations and 51 M support records pass through Python dataclasses into HDF5 rows
with 64-byte IDs: 10³–5 × 10³ s per record at 10–50 µs each (estimate), and ≥ 18 GB per record.

## 4. Structure a faster formulation can exploit (checked in source)

| ID | Structure | Source |
| --- | --- | --- |
| S1 | Linear between clip events: u = clip(8·v_s + e_s, ±50 mV). B-family fields are 10–50 µV RMS, far inside the ±6.25-mV source-referred rail, so events are rare | [analog_conditioning.py](tools/analog_conditioning.py) |
| S2 | Feed-forward: the 69,696 pad/CPE/source states have no C/G coupling to the 152 downstream states, and the drive reads only sources. The upstream can be solved first, without iteration | `prepare_triangular_contact_schur` checks the actual matrices; supported in attempt004 |
| S3 | 24 independent ADC groups of ≤ 7 states: ADC k holds sources k, k+24, k+48, plus G1 on ADC0 | [conditional_acquisition_recipe.py](tools/conditional_acquisition_recipe.py), [daq_mux_hold.py](tools/daq_mux_hold.py) |
| S4 | Periodic switching: upstream topology changes only at the 256,000 row switches (64 configurations); each ADC group has ≤ 4 slot configurations | schedule, recipe |
| S5 | Weak coupling between units, real modes: upstream C and passive G are exactly symmetric. Pad–pad 0.02 pF vs ≈ 2 pF self (≈ 1 %/edge); sources 0.2 vs 35 pF; TFT off 20 pS vs 2 µS on. Only the selected row couples strongly to the sources | recipe; symmetry check in source |
| S6 | Low-rank analytic inputs: field = Σ contact_modes × raw_drivers(t), with B2 = A·cos + B·sin at 11 Hz (rank 2). Noise = 9 fixed tones with fixed complex port amplitudes (18 real patterns). Holds and cubics approximate closed-form inputs | [surface_to_contact.py](tools/surface_to_contact.py), [spectral_noise.py](tools/spectral_noise.py) |
| S7 | Quantizer: one LSB is 76 nV at the hold, ≈ 9.5 nV source-referred, so the 10⁻⁸-V tolerance is ≈ 0.13 LSB | recipe |

## 5. Contract-preserving options (speedups are estimates against the current reference)

**O1 Template-level planning.** Validate the 4-ms template once (structure, seams, final publication
horizon), generate cycles as exact vectorized time shifts, compute holds in closed form, and digest
field and noise descriptors once. *Speedup:* preflight > 5,950 s → ≲ 60 s (≥ 100×). *Numerics:* none;
per-window replay digests become a structural guarantee plus spot checks. *Validation:* records
byte-equal to the legacy factory on windows 0–2 and 20 random windows; exact totals. *Effort:* 1–2 days.

**O2 Vectorized readout, compact complete output.** Per-window arrays with the same thresholds and
upper-tie rule (searchsorted); every output quantity kept as typed arrays with integer IDs.
*Speedup:* readout 10²–10³×; 0.1–0.5 GB per record. *Numerics:* none. *Validation:* bit-identical
codes, observations and timestamps from legacy hold means on ≥ 5 windows. *Effort:* 1–2 days.

**O3 Cached Radau, same algorithm.** Build each row's matrices and plan once; cache factors per (row,
step) with rank ≤ 128 Woodbury row updates on one base factor; one linear solve when no rail is
active; compiled kernels; partition doubling and tolerances unchanged. *Speedup:* 0.41 → 0.03–0.06 s
per late interval, ≈ 10× per record. *Numerics:* operation order only. *Validation:* attempt004's
eight intervals, all 69,848 states within 10⁻¹² V of the unmodified solver, identical partitions.
*Effort:* moderate, 1–2 weeks.

**O4 Feed-forward split and exact downstream.** Solve the upstream alone. Integrate each ADC group
exactly with precomputed 7 × 7 exponentials per slot configuration and interval length; aperture
means become exact integrals, clips are located as events on the analytic drive, and charges enter
as jumps. *Speedup:* upstream steps fall from about 24 to 12 per slot (about 2× on O3), and the
downstream costs 10¹⁰–10¹¹ flops per record. *Numerics:* the downstream becomes exact.
*Validation:* hold means, codes and work over mux-change slots and the 35-ms clipping fixture, within
the reference acceptance (10⁻⁸ V + 10⁻⁵ rel; 10⁻¹⁸ J + 10⁻⁵ rel). *Effort:* 2–3 days.

**O5 Large-step upstream.** An exact exponential via a 12–16-pole rational approximation with
precomputed shifted factors, or 3-stage Radau IIA on a fixed graded mesh after each row switch with
dense output (the passive `expm_multiply` path in [sparse_array_rc.py](tools/sparse_array_rc.py)
scales with ‖A‖h ≈ 10²–10³, so it is unsuitable). *Speedup:* about 300 → 20–30 upstream steps per
dwell (10–15×). *Numerics:* a new discretization. *Validation:* ≤ 10⁻⁹ V against the same solver at
rtol 10⁻⁸, atol 10⁻¹¹ V over one dwell and one frame. *Effort:* 1–2 weeks.

**O6 Exact low-rank inputs, with the same draw and ports.** Compute per-configuration particular
solutions for the 9 noise tones and sinusoidal drivers, at one complex solve per tone and
configuration per world. Event drivers get per-dwell polynomials with analytic remainders, so the
12.8 M holds and per-slot cubics disappear. *Speedup:* removes the per-step 78,232-port forcing and
the hold planner; O5 and O7 depend on it. *Numerics:* input error falls below the hold bound (10 nV)
and the cubic remainder (0.0187 of tolerance). *Validation:* first feed the reference's holds and
cubics, then report the exact-input difference. *Effort:* 2–3 days.

**O7 Multirate exact upstream, the recommended core.** Each 62.5-µs dwell is handled in three parts.
- *Active block.* The selected row's 64 pads with their CPE states, plus all 64 sources and their
  horizontal mutual capacitance, form 1,152 states. They are solved exactly in a precomputed
  C-orthonormal modal basis, so v_s(t) is available at any time. The basis takes one eigensolve per
  distinct row pattern: 2 for uniform G0, at most 64 with heterogeneity.
- *Units.* Every other pad is an independent 17-state unit, advanced exactly in large steps.
  Identical units in a column share their g_off response.
- *Coupling.* g_off column currents and mutual capacitance are exchanged by Gauss–Seidel waveform
  relaxation. Sweeps stop when the largest change is ≤ 0.1 nV, with a refusal cap. The standard
  frequency-domain WR bound suggests a contraction of about Σ C_mutual / C_self ≈ 0.04 per sweep
  (≤ 10⁻³ for g_off), so 4–6 sweeps (estimate, to be measured).
- *Outputs.* Work stays exact per window (exact integrals, heat from the energy identity, residual
  reported); per-interval work rows cost about 2× more if required. The same solver runs the probes.
- *Speedup.* About 10¹³ flops per record, mostly v_s(t) synthesis in BLAS-3 kernels. That is
  10²–10³ s if kernel-bound, or 10³–10⁴ s (0.3–3 CPU-h) with a 10× orchestration allowance:
  10³–10⁴× together with O1, O2, O4 and O6.
- *Numerics.* Exact block solutions plus a convergent iteration replace the monolithic implicit
  solve and its partition refinement.
- *Risks.* Modes span 13 decades: an eigenvalue error of ε‖A‖ ≈ 10⁻⁷ s⁻¹ gives ≲ 3 × 10⁻⁶ slow-mode
  drift over 16 s. Coupled identical units produce secular terms; near the row, use sub-interval
  polynomial coupling signals. Memory is about 1–4 GB.
- *Effort.* 1–3 weeks; this is the critical path.

**O9 GPU and O10 parallelism.** Porting today's sequential sparse/banded solves gains ≲ 2–5× or loses.
O7's small kernels suit GPUs batched over worlds sharing a design and device draw (estimate 10–100
GPU-s per record; CPU/GPU agreement ≤ 10⁻¹² V). Worlds run in parallel on ≤ 128 cores (throughput,
not latency); pads and ADC groups vectorize inside O4/O7; parallel-in-time is not worth it.

| Route | Options | Per record (estimate) | Speedup |
| --- | --- | --- | --- |
| A | O1 + O2 + O3 | 300–600 CPU-h | ≈ 10× |
| B (fallback, cross-check) | A + O4 + O5 + O6, batched right-hand sides | 4–40 CPU-h | 10²–10³× |
| C (recommended) | O1 + O2 + O4 + O6 + O7 | 0.3–3 CPU-h | 10³–10⁴× |
| C + O9 | batched GPU | 10–100 GPU-s | further 10–100× throughput |

**Study-level need (estimate).** About 25–40 hardware/device settings × 84 development worlds, plus
≥ 768 final worlds (3 × 4 × 64) × about 10 arms, each under every accepted model: about 10⁴ R3-class
records (up to 6 × 10⁴ with six models). 4,000 CPU-h then allows ≈ 0.4 CPU-h per record: ≈ 10⁴×, or GPU.

## 6. Options that change the design (each needs a separate scientist amendment)

| ID | Change | Saving | Scientific cost |
| --- | --- | --- | --- |
| A1 | Shorter records, e.g. 4 or 8 s | 2–4× | 1-Hz resolution, 0.5-Hz event conditions, event counts, λ/τ estimation |
| A2 | Fewer worlds, conditions, models or final realizations | proportional | interval widths; fixed 3 × 4 × 64 final support; ≥ 5 conditions per S/T cell |
| A3 | Fewer pads or a sub-array | large | breaks the fixed hardware and full-array crosstalk/settling; not recommended |
| A4 | Coarser timing: point samples, fewer simulated words, coarser mux | moderate | changes conversion and settling semantics |
| A5 | A different or reduced schedule | varies | changes the published frontier |
| A6 | Reduced-order CPE (former O8; H∞ bound 2Σσ) | ≤ 1.6× in states | changes a parameter and the high-frequency response behind settling and crosstalk |
| A7 | Calibrated linear surrogate, or tolerances loosened to e.g. 100 nV | large / small | changes the model of record or the stated tolerances |

## 7. Recommendation and staged plan

Adopt Route C, built as O1, O2, O4 and O6 first (low-risk, needed by every route, validated against
the unmodified upstream solver), then O7. Keep Route B (O3, O5), optionally GPU-batched, as a
time-boxed fallback: the 14-day clock makes O7's build time the main risk. If O7 has not passed
Stage 1 by 2026-10-18, move to Route B or seek an A1/A2 amendment. Route A cannot reach 1,600×.

**Open decisions for the scientist.** Adopt or reject this proposal; approve replacing per-window
replay digests with template validation; approve the new controls (WR sweep cap and tolerance, modal
residual check, event-location tolerance, per-window energy residual); decide whether per-interval
work rows are required outputs.

**Rules for every job.** One `sbatch` each, with explicit `--time`, `-c`, `--mem` and a partition from
`sh_part` (GPU only via `-G 1` on a GPU partition). Record `hostname` and `$SLURM_JOB_ID` first, pin
threads, set an internal deadline at 90 % of `--time`, and write a terminal JSON on every exit path.
No `srun --pty`, polling loops or automatic retries; chain with `--dependency=afterok`; log each
selection and job id before submission.

**Stage 1: equivalence** (development tests, not counted; 1 CPU, 8–16 GB, ≤ 2 h per job).
1. *Planner and readout:* byte-equal records and codes against the legacy code, with exact totals.
2. *Small arrays:* the reference's generic network classes against the new solver. Run passive
   R × S circuits through `SparsePassiveRCNetwork.advance` and conditioned R × 64 circuits (R = 1–4)
   through `ConditionedRCNetwork`, whose drive class fixes 64 sources. Add a dense-exponential
   oracle, requiring ≤ 10⁻¹² V against it and the reference tolerances against the reference.
3. *Full-size short windows,* new solver fed the reference's holds and cubics: the 5-µs case, one
   62.5-µs dwell and one 4-ms frame (reference ≈ 1 h, derived). All 69,848 final states and hold means
   within 10⁻⁸ V + 10⁻⁵|v|; work within 10⁻¹⁸ J + 10⁻⁵ rel; codes identical except ±1 where the
   reference lies within that tolerance of a threshold; pixels within 1/3 LSB. Partition agreement can
   falsely accept (log, cubic-repair entry): adjudicate excess with a tightened reference (rtol 10⁻⁸, 10⁻¹¹ V).
4. *Exact inputs and clipping:* report the exact-input difference on the frame (expected ≲ 10 nV).
   Run the 35-ms clipping fixture and one frame with a synthetic offset past the rail.

**Stage 2: speed at increasing sizes** (formal runs; a failure counts toward the 12). Run 1, 10, 100 and
1,000 frames of B2/2001 on one core, each `--time` ≥ 3× the projection from the previous size
(starting from Stage 1's frame timing), `--mem` 16–32 GB; a 100-frame GPU run is optional. Record
precomputation, per-frame cost, peak RSS, sweeps and energy residuals. **Gate:** per-frame cost within
± 20 % from 10 to 1,000 frames; projected record time ≤ 50 % of the Stage 3 limit; RSS ≤ 50 % of the
request; Stage 1 tolerances still met on frame 1.

**Stage 3: one full 16-s record**, only after the gate, with `--time` = max(1 h, 2 × projection), the
same checks and no retry. Report the per-record cost and the study-level projection before any
production panel. Stages 1–3 should total ≲ 10 CPU-h, plus ≲ 2 optional GPU-h.

## 8. Plain answer

- **Engineering alone can probably reach ≥ 1,600× per record, but only through Route C.** Tuning the
  current solver (A, ≈ 10×) or the conservative implicit route (B, 10²–10³×) falls short on one core.
  Route C's 10³–10⁴× already allows 10× for orchestration but is unmeasured; its low end (≈ 3 CPU-h)
  would still miss a 6,000-s attempt (≈ 1.7 CPU-h). Stage 2 decides.
- **No design amendment is needed to reach 1,600×.** One is likely for the full study, which needs
  about 10⁴ records, unless Route C lands high in its range or GPU batching works. The 14-day clock
  may also force one if O7 cannot be built in time.

## 9. Sources relied on

Contract and amendment: [GOAL.md](../GOAL.md), [DATASETS.md](../DATASETS.md), [SEARCH_POLICY.yaml](../SEARCH_POLICY.yaml)
(`scientist_amendment_20261012`). Records: [conditional_forward_recipe.md](conditional_forward_recipe.md),
[development_world_recipe.md](development_world_recipe.md), and [experiment_log.md](experiment_log.md) entries
10/12 21:46; 10/9 19:10, 17:02, 16:48; "selects hash-repaired full-preflight measurement"; "records full-preflight
timeout and selects bounded closeout"; "selects contact-block Radau repair"; "selects full-dimensional component
cost diagnosis"; "selects cubic repair attempt003"; 10:26, 10:24, 10:10. Run reports:
`runs/model_specification/conditional_component_cost_attempt004/` and `conditional_preflight_cost_attempt002/`.
Tools read, not run: `conditioned_array_solver.py`, `conditioned_radau_step.py`, `radau_triangular_contact_schur.py`,
`radau_contact_preconditioner.py`, `cpe_array.py`, `causal_cpe.py`, `sparse_array_rc.py`, `analog_conditioning.py`,
`daq_mux_hold.py`, `conditional_acquisition_recipe.py`, `repeating_acquisition_schedule.py`, `raw_input_hold.py`,
`streaming_midpoint_hold.py`, `streaming_array_acquisition.py`, `streaming_development_simulation.py`,
`run_conditional_preflight_cost.py`, `profile_conditional_component_cost.py`, `spectral_noise.py`,
`polynomial_noise.py`, `surface_to_contact.py`, `raw_world_time.py`.
