# Can a constrained NeuroCam redesign recover cortical voltage fields more faithfully?

A NeuroCam-class array cannot maximize spatial coverage and sampling speed at
the same time. Scanning all 4,096 pixels covers the array's full field of view,
but each pixel is revisited more slowly. Concentrating measurements on fewer
rows or source groups captures faster changes but leaves other locations less
well observed. Electrode size creates a related choice: a larger pad averages
over more tissue, whereas a smaller pad may preserve a focal event more
precisely.

EP20 asks whether those choices can be made more intelligently without making
the device larger, adding wires or conversion work, increasing latency, or
exceeding the same modeled power/thermal budget.
The candidate may combine a legal pattern of small and large pads with a legal
row/source acquisition schedule, and the reconstruction software may learn
the exact measurement pattern. The question is whether that pair recovers
broad activity, focal field transients, and propagating waves better than the
best eligible version of the published NeuroCam architecture under the same
resource limits.

The first decisive comparison is therefore not a new design against the
original reconstruction code. One locked co-designed candidate must be tested
on the same cortical fields and device draws as every resource-matched
NeuroCam operating mode, including NeuroCam with equally optimized software.
It must also beat its own hardware with conventional reconstruction and every
registered simple pad or scan heuristic given the same software and tuning
budget. These comparisons separate a genuine co-design benefit from a better
decoder, a hardware-only improvement, or an easy baseline.

An apparent win can still be misleading. A design may favor broad rhythms
while erasing focal events, exploit the exact simulator used during training,
or rely on idealized noise and crosstalk that do not survive fabrication
rounding. It may also look better only because one NeuroCam operating mode was
chosen as the reference after the result was known. The result therefore has
to hold across the complete frozen reference frontier, across registered
signal and device conditions, and under a sealed change in both the cortical
field generator and the device model. Before audit, one qualified empirical
ECoG or known-input physical replay must also be selected and frozen; none is
currently selected. That replay is a catastrophic-plausibility check, not a
validation of the unbuilt geometry.

If the candidate survives those tests, the deeper question is what the sensor
redesign actually preserves. Does multiscale electrode geometry recover both
broad and local spatial structure? Does nonuniform scan allocation retain fast
events without sacrificing slower waves? Or can optimized software on the
uniform NeuroCam array recover the same information, making new hardware
unnecessary?

The intended paper must do more than report one higher reconstruction score.
It should show which hardware and software changes contribute the gain, where
the gain appears across broad activity, focal events, and waves, where the
candidate still loses, and which physical experiment would most directly test
the proposed explanation. The final output should be a compiled pad map, scan
schedule, reconstruction recipe, resource ledger, and failure map that another
team could take into device validation.

This remains a virtual design study. A positive result would support the
narrow claim that one compiled design is worth fabrication or device testing
because it beat the complete optimized NeuroCam reference frontier within the
registered model ensemble and resource constraints. It would not show that a
fabricated device outperforms NeuroCam, is safe or stable in chronic use, or is
ready for foundry sign-off or in-vivo deployment.

## At a glance

| Question | EP20 design |
| --- | --- |
| What problem is being tested? | Whether electrode geometry, scan allocation, and reconstruction software can jointly preserve more of a cortical voltage field than an optimized NeuroCam reference. |
| What stays fixed? | The 64-by-64 lattice, 150-micrometre pitch, one-TFT row-column circuit, array area, and limit of 64 gate plus 64 source lines, together with matched conversion, latency, and power/thermal proxies. |
| What may change? | A legal uniform or two-scale pad pattern, a legal row/source acquisition schedule, and a capacity-capped reconstruction rule trained for that measurement pattern. |
| What is reconstructed? | The 1--100 Hz cortical-surface voltage field on the same dense reference grid for every design. |
| What is the reference? | Every eligible operating point of the paper-derived NeuroCam architecture, first with conventional reconstruction and then with equally optimized software. |
| What must the candidate beat? | The complete matched NeuroCam frontier, its own hardware-only version, and every registered simple hardware heuristic evaluated with the same software family and budget. |
| What must repeat? | The advantage must survive broad, focal, and propagating-wave signals and the registered device, noise, and resource conditions without an important stratum becoming meaningfully worse. |
| What checks simulator exploitation? | A sealed, independently implemented change in both the cortical-field generator and the device model. |
| What is the empirical plausibility gate? | Before audit, one qualified empirical ECoG or known-input phantom replay must be selected and frozen. It may expose catastrophic model mismatch but cannot validate the unbuilt geometry. |
| What can a positive virtual result conclude? | That a specific compiled design is a justified candidate for fabrication or device testing. |
| What can it not conclude? | That fabricated hardware outperforms NeuroCam, is chronically safe or stable, or is ready for manufacture or in-vivo use. |

## Scientific estimand

For a finite registered evaluation-stratum set `E`, let `D` be a compiled
hardware and software design and let `B(e)` contain every preregistered `D0`
and `D1` NeuroCam reference member eligible under stratum `e`'s resource
envelope. For each reference member `b`, the paired stratum estimand is

\[
\delta_{e,b}(D)
=
R^2_{\mathrm{SSE}}(D;e)
-
R^2_{\mathrm{SSE}}(b;e).
\]

For each reference member define its eligible support
`E_b = {e in E : b in B(e)}`. The confirmatory member-specific macro gain uses
the frozen equal stratum-family weights renormalized within that support, and
the frontier-wide estimand is the least gain against any eligible reference
member:

\[
\bar\delta_b(D)
=
\sum_{e\in\mathcal E_b}
\frac{w_e}{\sum_{j\in\mathcal E_b}w_j}\delta_{e,b}(D),
\qquad
\Delta_{\mathrm{frontier}}(D)
=
\min_{b\in\mathcal B}\bar\delta_b(D).
\]

A contrast is never imputed outside a member's registered resource-envelope
support. Every support, weight, and eligibility mapping is frozen before
candidate-hardware scores.

Promotion uses multiplicity-adjusted simultaneous lower bounds for every
member-specific macro gain, not a sample-wise oracle that changes reference
mode per scene. A separate robustness guard applies the simultaneous lower
bound to every registered `e` by `b` contrast. During development, search and
winner ranking use the minimum cell-mean paired gain across `e` and `b`; that
robust search objective is not substituted for the confirmatory macro
estimand.

This is paired regret relative to a complete reference frontier, not absolute
utility pooled across incomparable signal families. Each candidate and every
reference member receive identical cortical fields, device draws, nuisance
draws, and evaluation masks.

The primary target is the cortical-surface potential, not the unobserved
transmembrane-current source. Source recovery is an explicitly synthetic
diagnostic and cannot determine promotion.

## Competing explanations

1. **Uniform-reference robustness.** Once conversion, scan, and device shifts
   are matched, the uniform 64-by-64 reference remains the most robust design.
2. **Software sufficiency.** Optimized reconstruction on the reference
   hardware recovers essentially all attainable gain, so redesigning hardware
   is unwarranted.
3. **Multiscale-geometry benefit.** A legal two-scale pad pattern improves the
   broad-versus-local spatial trade-off after pad impedance, spatial averaging,
   and device noise are accounted for.
4. **Rate-allocation benefit.** A legal row/source-group schedule improves the
   spatial-versus-temporal trade-off without adding array lines, ADC work, or
   latency.
5. **Hardware-software complementarity.** A compiled hardware change and a
   measurement-aware reconstructor together outperform either component alone.
6. **Simulator exploitation.** Apparent gain disappears under an independently
   implemented forward model, device scaling law, nuisance family, or
   fabrication rounding.
7. **Underidentification.** Public characterization and planned replay data do
   not constrain the model ensemble enough to distinguish these explanations.

## Reference architecture and operating frontier

The named reference is derived from the published NeuroCam report and its
supplement. The paper describes a 64-by-64 Ln-IZO TFT array with 150
micrometre pixel pitch, approximately 9.6-by-9.6 mm active area, 70-by-135
micrometre gold sensing pads, 64 shared gate lines, and 64 shared source lines.

The publication reports several distinct operating points, including:

| Dynamic mode | Pixels acquired | Effective rate per pixel | Reported RMS noise |
| --- | ---: | ---: | ---: |
| `24S x 16G` | 384 | 1000 S/s | 8.8 +/- 4.2 microvolts |
| `24S x 64G` | 1536 | 250 S/s | 12.6 +/- 8.4 microvolts |
| `64S x 64G` | 4096 | 250 S/s | 63.6 +/- 24.1 microvolts |

It also reports a representative channel gain of 0.78 +/- 0.12, a channel
3-dB frequency above approximately 1 kHz, source-line-neighbour crosstalk near
-12.6 dB, gate-line-neighbour crosstalk near -37.2 dB, and more distant
crosstalk near -45 dB. These are characterization anchors, not a complete
generative electronics model.

The 1-kHz single-channel frequency response must not be confused with the
250-S/s effective full-array frame rate. A full 64-row scan with the reported
62.5-microsecond row dwell spans approximately 4 ms and has a nominal
per-pixel Nyquist frequency near 125 Hz. Full-array claims about higher
high-gamma frequencies or single-neuron action potentials are prohibited.
Focal `spike` in this episode means an epileptiform or other field transient,
never a sorted single-neuron spike.

The promoting field-reconstruction band is 1--100 Hz. Frequencies from
100--400 Hz may be examined only in the registered reduced-mode high-rate
diagnostic; that diagnostic cannot promote a design or rescue failure in the
primary band.

`NeuroCam` is a named reference architecture here, not a claim that it is the
best micro-ECoG system on every physical or scientific axis. Before audit, all
registered reference modes and software subclasses are evaluated on the same
environments and locked. Promotion is tested simultaneously against every
eligible member under the matched resource envelope; no audit-dependent member
is substituted after seeing the candidate result. A sample-wise pointwise
oracle envelope across registered modes is reported only as a labelled
diagnostic and never defines the primary contrast.

## Generative and measurement model

The simulator keeps neural source, cortical-surface potential, electrode
interface, electronics, acquisition action, and software distinct:

\[
(S,Z)\sim P_e,
\qquad
\phi=L_{\gamma,p}S,
\]

\[
v=E_{h,\zeta}\phi,
\qquad
a_t=\pi_h(y_{<t},q_{\le t}),
\]

\[
y_t
=
Q_{b(a_t)}
\left[
R_{h,\eta,a_t}(v_{0:t})+n_t
\right],
\qquad
(\widehat\phi,\widehat Z,\widehat\sigma)
=f_s(y_{\le t},a_{\le t}).
\]

Here:

- `S` is an optional latent neural-source field;
- `Z` is a task or mechanism label generated jointly with the field;
- `phi` is voltage on a common dense cortical-surface reference grid;
- `L` is a source-to-surface transfer model with tissue and placement state;
- `E` models pad aperture averaging, contact impedance, reference convention,
  curvature, and contact gaps;
- `R` models TFT gain, row-column timing, line RC, settling, switching
  transients, saturation, crosstalk, drift, and missing pixels;
- `eta` is a slowly varying device/process state drawn conditionally on the
  compiled hardware;
- `n_t` is stochastic measurement noise conditional on geometry, operating
  mode, and scan history;
- `a_t` is either a frozen schedule or a causal bounded acquisition action;
  and
- `q_t` contains only hardware state and information available at time `t`.

Some development mechanisms generate `phi` directly. Separate structural
audit mechanisms generate `S` and then apply an independently implemented
`L`. This separation prevents an inverse method from being evaluated only on
the exact forward operator used to train it.

All voltages are referenced. The reference electrode, common-mode convention,
and any guard cost are fixed and counted; absolute unreferenced potential is
not an observable.

## Fixed, searchable, and prohibited design components

| Fixed before development | Searchable in EP20 | Prohibited from promotion |
| --- | --- | --- |
| Ln-IZO process class and flexible substrate | One of three registered pad-pattern families | Material composition or transistor chemistry |
| 64-by-64 lattice and 150-micrometre pitch | Discrete legal small/large pad catalogue entries | Changing pixel count, pitch, or active area |
| One-TFT row-column addressing | Legal row order, dwell catalogue, and source-bank schedule | Hierarchical routing or extra addressing lines |
| At most 64 gate plus 64 source array lines | Uniform, fixed heterogeneous, or bounded causal revisit policy | Extra on-array buffer, memory, or free local compute |
| Frozen reference and guard convention | Capacity-capped reconstruction rule | Arbitrary analog mixing matrix |
| Matched AFE, conversion, bit-rate, latency, and power proxies | Registered precision/oversampling mode already supported by the resource model | New variable-resolution ADC or uncosted gain path |
| Common evaluation grid and masks | Candidate-specific weights trained under equal budget | Candidate-specific target, metric, or audit family |

The legal pad catalogue is finite. It is derived from a frozen design-rule
manifest and includes the reference pad plus at most two alternative pad
classes. A continuous relaxation may propose a design internally, but only
the compiled, rounded, discrete design is scored or promoted.

A row action addresses an entire legal gate group. A candidate cannot pretend
to revisit an arbitrary pixel independently when the row-column circuit
cannot do so. Likewise, source-bank precision is charged according to the
registered external ADC/AFE resource model.

Arbitrary learned analog projection

\[
y_k=\sum_j A_{kj}x_j
\]

is outside the promoting grammar. Sparse, local, or nonnegative coefficients
alone do not prove circuit realizability, and digital post-ADC mixing cannot
claim a conversion saving. Analog mixing may appear only as a labelled
non-promoting future-work diagnostic after all primary work is complete.

## Software grammar

Every hardware design is evaluated with the same software-family and training
rules. `Same software` means the same architecture, input contract, parameter
ceiling, optimizer, data volume, selection budget, seeds, and calibration
rule. Weights are retrained for each measurement operator; reusing weights
trained on an incompatible geometry is not a fair hardware-only control.

The promoting software grammar contains:

1. **Conventional reference (`s0`).** A fixed causal linear-Gaussian
   state-space or kriging rule operating on native timestamped observations,
   with a measurement-compatible adapter but no learned hardware-specific
   representation.
2. **Measurement-aware reconstructor (`s1`).** Three preregistered subclasses
   are evaluated on the complete reference frontier before any redesigned-
   hardware score is returned: a regularized physics-informed linear inverse,
   a regularized low-rank-plus-local structured state-space inverse, and a
   compact causal nonlinear inverse. The frozen selection rule chooses one
   subclass for all `D3`/`D4` comparisons; every evaluated `D1` point remains
   in the incumbent frontier. Each subclass receives value, timestamp,
   measurement identity, hardware state, gain state, scan state, and
   device-calibration state.
3. **Uncertainty head.** The registered reconstructor must emit calibrated
   field uncertainty under missing, irregular, and device-shifted observations.
4. **Transfer probes.** Frozen linear or low-rank probes assess event presence,
   source zone, and wave direction from the reconstructed representation.
   Transfer probes cannot update hardware or select the episode winner.

The exact parameter ceiling and training-step budget are frozen in the
admissible-space manifest before candidate-discriminating development. The
current planning ceiling is five million trainable parameters per
reconstructor; a lower profiled ceiling may be frozen before outcome access,
but it cannot change afterward.

Software-subclass selection is completed and hashed before any `D2`, `D3`, or
`D4` candidate-hardware outcome is exposed. For subclass `s`, its frozen score
is the minimum across resource envelopes of the equal-signal-family,
equal-device-stratum macro `R2_SSE` of `D1(s)`; the highest score wins, with
ties resolved by lower parameter count and then lexicographically lowest
recipe hash. Exact weights and normalization live in `SOFTWARE_BUDGET.yaml`.
The selection receives no extra calls and cannot be revisited because a
hardware candidate performs poorly.

`RESOURCE_MODEL.yaml` assigns every arm a frozen resource-envelope ID. For
each candidate envelope it names the eligible `D0` members, the same-mode `D0`
for every `D1`, the unique interaction-matched `D1` using the selected `s1`,
and every eligible registered `D4` heuristic.
`D2` always uses the identical compiled `D3` hardware and operating recipe
with `s0`. A missing or multiply defined matched `D1` invalidates the trial.
Every eligible `D4`, not merely a development-selected favourite, is carried
into the one-shot audit.

## Nested design ladder

| Design | Hardware | Software rule | Purpose |
| --- | --- | --- | --- |
| `D0` | Registered NeuroCam operating frontier | Conventional `s0` | Paper-derived reference analysis frontier |
| `D1` | Same NeuroCam frontier | All registered `s1` subclasses under equal budget | Maximum software-only gain and pre-hardware subclass selection |
| `D2` | Compiled redesigned geometry or legal readout | Conventional `s0` plus frozen compatible adapter | Hardware-only gain |
| `D3` | Compiled redesigned geometry/readout | Separately trained selected `s1` subclass | Joint co-design candidate |
| `D4` | Registered heuristic multiscale/readout controls | Separately trained selected `s1` subclass under the same budget | Whether learned search beats simple engineering rules |

The primary frontier gate is simultaneous against every eligible locked
`D0`/`D1` member within the matched resource envelope. `D3` must also exceed
its matched `D1` and `D2`, plus every resource-eligible `D4` control; beating
one weak conventional pipeline does not establish co-design value.

For a locked scalar primary utility, the four-cell contrast

\[
I_{\mathrm{co}}
=U(h_1,s_1)-U(h_1,s_0)-U(h_0,s_1)+U(h_0,s_0)
\]

is reported only as held-out descriptive complementarity. It depends on the
utility scale, is influenced by adaptive selection of `h1`, and is not a
causal decomposition or proof of a physical interaction mechanism.

## Reference-model qualification gate

No candidate search begins until a reference-model ensemble passes an
incumbent-reproduction gate. The ensemble is fitted only to the designated
calibration anchors and then tested on operating modes, trace segments, or
characterization summaries withheld from its fit.

At minimum the gate checks:

- array geometry, pad aperture, addressing, and scan timing;
- gain distribution and representative frequency response;
- all three registered dynamic mode rate/noise anchors;
- anisotropic first-neighbour and distant crosstalk anchors;
- settling and row-switch behaviour when raw traces become available;
- missing-pixel and gain-yield summaries only if a provenance-backed target is
  acquired; otherwise these remain adversarial stress dimensions, not
  qualification anchors; and
- monotonic resource accounting across source/gate configurations.

Pass tolerances, which anchors are fitted, and which anchors are held out must
be frozen before fitting. The planning tolerances are exact cardinalities; at
most 2% relative error for sample rate and throughput; at most 0.05 and 0.03
absolute error for the reported gain central value and dispersion; an
at-least-800-Hz representative cutoff check, with no unsupported upper bound;
at most 20% and 30% relative
error for the reported noise central value and dispersion; at most 3 dB error
for nearest-neighbour crosstalk;
distant crosstalk no greater than -42 dB; and exact recovery of the registered
directional-crosstalk and operating-mode trade-off orderings. These tolerances
remain provisional until outcome-blind fixture calibration and scientist
sign-off. A full frequency-response curve gate remains unset until the
1-mVpp/10-mVpp source conflict and curve-extraction tolerance are resolved.

The current aggregate publication does not by itself identify a
transistor-level model, arbitrary-layout RC law, power model, yield model, or
analog-mixing circuit. If the frozen reference ensemble fails qualification,
the episode returns `technical_reference_model_unqualified`; it does not widen
uncertainty until a candidate appears favourable.

Passing this gate licenses only the phrase `paper-calibrated NeuroCam-derived
model ensemble`. It does not license `validated digital twin`,
`fabrication-ready`, or physical equivalence.

## Signal and nuisance environments

The prior is an executable family, not a list of verbal labels. Every
environment manifest fixes amplitude, temporal spectrum, spatial spectrum,
correlation length, event rate, propagation speed, boundary conditions,
nonstationarity, mixture weight, reference convention, and nuisance range.

Development must cover at least:

- broad correlated oscillatory fields;
- focal field transients with no single-neuron interpretation;
- simple translating and expanding waves;
- broad-plus-local multiscale mixtures;
- multiple interacting sources;
- silent-region surprise events; and
- common-mode, line-noise, reference-drift, contact-gap, motion-like, and
  missing-pixel nuisances.

The audit uses sealed structural changes, not merely new random seeds. It must
include at least one independently implemented source-to-surface model, one
independently implemented electronics scaling or coupling law, one
nonstationary or moving-source family absent from development, one altered
spatial-spectrum family, and one unseen combination of crosstalk, noise,
drift, and process variation. Empirical replay cannot substitute for the
independent virtual electronics shift because it cannot evaluate an unbuilt
candidate geometry.

Real high-density ECoG replay can test ecological waveform and spectrum
coverage but is not dense cortical-surface ground truth for arbitrary new pad
geometries. It is therefore a required plausibility diagnostic, not the sole
promotion endpoint. If a known-input PBS/phantom replay becomes available, it
may provide a stronger electronics qualification endpoint under a separately
frozen role.

The development replay partition is evaluated only on `D0`/`D1` software and
simulator-envelope stress. Its outcomes may falsify software stability or the
simulated signal envelope, but they cannot score, rank, retire, or route a
`D2`/`D3`/`D4` hardware successor. The sealed plausibility partition opens with
the audit and likewise supplies no alternative-geometry ground truth.

## Objectives and reporting

### Primary field objective

For a common dense reference grid and frozen scoring mask,

\[
R^2_{\mathrm{SSE}}
=
1-
\frac{\sum_{r,t}(\phi(r,t)-\widehat\phi(r,t))^2}
{\sum_{r,t}(\phi(r,t)-\overline\phi_{\mathrm{train}})^2}.
\]

The training-fold mean is used in the denominator. Space-time samples are
repeated measurements; uncertainty is aggregated first within a generated
field, then within a structural-generator/device cell, and finally with equal
weight across registered mechanism families. The resampling unit is a whole
paired generated scene-by-device-draw world shared across `D0`--`D4`, never an
individual pixel or time point.

### Required secondary endpoints

- focal-event localization error in millimetres on the common surface grid;
- wave-direction angular error and propagation-speed relative error;
- transient detection proper log score;
- 50%, 80%, 90%, and 95% field-interval coverage and sharpness;
- worst-family and worst-device paired regret;
- performance after fabrication rounding and dead-pixel draws;
- latency, conversion load, output bit-rate, and power/thermal proxies; and
- fixed-probe transfer utility for the registered secondary labels.

Field reconstruction is primary. A candidate cannot trade an arbitrarily
large field loss for a simulated task gain. The task probes are reported on a
separate frontier and cannot rescue a failed field endpoint.

## Promotion rule

Exactly one compiled `D3` design may be locked.
`candidate_ready_virtual_codesign_specification` requires all of the following
in the one-shot audit:

1. the reference-model qualification gate was passed before search;
2. the one-sided multiplicity-adjusted simultaneous 95% paired
   cluster-bootstrap lower bound for the equal-weight macro `Delta R2_SSE` is
   at least `0.010` against every eligible preregistered `D0`/`D1` frontier
   member;
3. the corresponding overall lower bounds versus the matched `D1` and `D2`,
   and simultaneously versus every resource-eligible registered `D4` control,
   are each at least `0.005`;
4. the simultaneous 95% lower bound for every registered signal-mechanism by
   device-stratum by resource-envelope by eligible-reference contrast is at
   least `-0.005`;
5. localization, wave, transient, uncertainty, and downstream-transfer
   endpoints satisfy their frozen noninferiority margins;
6. the exact compiled, rounded design satisfies every area, line, AFE,
   conversion, bit-rate, latency, power/thermal-proxy, and reliability-proxy
   constraint;
7. the result survives the mandatory component ablations and influence
   checks; and
8. the sealed empirical replay or known-input physical-replay partition shows
   no prespecified catastrophic plausibility failure.

The provisional secondary noninferiority margins relative to the matched
control are 0.15 mm for localization error, 5 degrees for wave-direction
error, 0.05 for wave-speed relative error, 0.01 for normalized transient
proper-log-score utility, 0.01 for uncertainty ECE, and 0.01 for normalized
downstream-task utility. Transient AUROC is diagnostic only. Exact signs,
normalizations, estimators, and simultaneous-interval construction are frozen
before development outcome access and cannot change afterward.

If a Pareto archive contains several feasible candidates, choose the maximum
development worst-environment paired gain. Break a tie within `0.005` by lower
conversion-energy proxy, then lower scan latency, then lexicographically lowest
canonical design hash. Audit never chooses among multiple candidates.

## Pre-audit development lock

Configuration lock is evaluated only after a valid development stop. A `D3`
is development-lock eligible only if all of these pre-audit conditions hold:

1. reference qualification, 32 valid trials, exact 16-arm coverage, two
   adaptive successor cycles, two incumbent/challenger decisions, the 40%
   post-coverage falsifier share, ledger integrity, and deterministic replay
   are complete;
2. under the development interval family, its macro lower bound is at least
   `0.010` versus every resource-eligible `D0`/`D1` member, at least `0.005`
   versus the resource-map matched `D1` and `D2`, and at least `0.005`
   simultaneously versus every eligible `D4`;
3. every development stratum-by-reference simultaneous lower bound is at least
   `-0.005`, and every registered secondary noninferiority margin passes;
4. the rounded compiled design passes every hardware and resource constraint,
   mandatory development falsifier, ablation, and influence check; and
5. the development empirical partition passes its software-stability and
   simulator-envelope stress without contributing to hardware ranking.

If several designs qualify, the frozen tie rule above selects exactly one. If
none qualifies, no audit opens. Neither the sealed plausibility result nor any
virtual-audit outcome is a pre-lock prerequisite; those are knowable only after
the selected candidate and complete comparison bundle are immutable.

## Trial and fairness contract

One valid trial is one fully specified, compiled hardware/readout design plus
one software recipe evaluated on the complete common development environment
bank. It includes all frozen training seeds and paired reference reruns. An
internal continuous relaxation, architecture sweep, or hyperparameter search
is charged to that trial's compute budget and cannot hide an unbounded search.

A trial is invalid if it:

- fails the hardware constraint compiler;
- is scored before discrete fabrication rounding;
- changes the environment bank, metric, target, or reference frontier;
- receives more software capacity, training steps, calibration data, or
  feedback than its matched comparator;
- uses future signal in an acquisition action;
- reads an audit generator, parameter, trace, score, or diagnostic; or
- duplicates a prior compiled design and software recipe.

Every adaptive successor records `parent_trial_ids`, changed and unchanged
operators, directional prediction, falsifier, expected information gain,
expected cost, outcome evidence refs, successor-cycle ID, and a
`proposal_context_hash` over the complete visible ledger prefix, as required
by `../TRIAL_LEDGER.schema.json`.

## Search stages

1. **Source and reference qualification.** Freeze source manifests, design rules,
   resource model, development/audit roles, reference modes, generator bank,
   metrics, and qualification tolerances. Pass the reference-model gate.
2. **Reference frontier.** Evaluate every `D0` and `D1` operating mode with
   common paired environments and equal software budgets.
3. **Branch coverage.** Complete the exact 16-arm manifest in
   [`SEARCH_POLICY.yaml`](SEARCH_POLICY.yaml): one `D0` frontier arm, three
   `D1` software arms, four `D2` hardware arms, four matched `D3` co-design
   arms, and four registered `D4` heuristic arms. No adaptive-patience count
   begins before all 16 valid arms are complete.
4. **Adaptive search.** Propose compiled successors from development outcomes
   only. Maintain a feasible Pareto archive, but keep one nonterminal incumbent.
5. **Falsification and ablation.** Spend at least 40% of post-coverage trials
   on prior shift, structural-model alternatives, device misspecification,
   decoder capacity matching, fabrication rounding, component removal, and
   direct replication.
6. **Configuration lock.** Apply the explicit pre-audit development-lock rule,
   then freeze exactly one eligible `D3` winner, the full reference frontier,
   every resource-eligible `D4`, diagnostic runner-up if permitted, code,
   environment, resource compiler, design hash, software weights and training
   recipe, metrics, thresholds, report template, and audit command.
7. **One-shot audit.** A permission-separated evaluator opens the complete
   sealed audit once and emits only the predeclared aggregate and per-family
   outputs.

## Mandatory falsifiers and ablations

- reference-model structural ensemble, not just parameter resampling;
- independent source-to-surface operator and altered tissue/placement state;
- unseen spatial spectrum and propagation regime;
- surprise source outside the high-rate or large-pad region;
- mode-dependent noise, crosstalk, settling, drift, and contact-gap shifts;
- matched random and simple checkerboard/tiled geometry controls;
- pad-geometry-only, schedule-only, software-only, and uncertainty-head
  ablations;
- causal acquisition replay proving no future access;
- equal-capacity and equal-training-budget software twins;
- fabrication rounding, dead-pixel, gain-yield, and leave-one-device-model-out
  influence;
- reversal of the large-pad/small-pad spatial assignment;
- source/reference convention sensitivity;
- development empirical-partition spectral-envelope, amplitude-range, and
  missingness stress; and
- direct reruns of the incumbent and selected candidate with common seeds.

The 40% falsification requirement refers to development trials. It does not
permit viewing or adapting to the sealed audit.

## Budget and stopping

Run at least 32 and at most 64 valid trials after the reference-model
qualification gate. Qualified patience is 12 consecutive eligible valid
`D3` opportunities, each with its complete linked `D0`--`D4` ladder, after
minimum trials, exact branch coverage, two adaptive successor cycles, two
incumbent-challenger decisions, and the required falsification fraction,
without at least `0.010` improvement in development worst-environment paired
gain or a new feasible Pareto point. Standalone `D0`, `D1`, `D2`, `D4`, retry,
qualification, and falsifier executions do not increment patience. At most 12
engineering failures may be repaired outside scientific patience.
The thirteenth engineering failure stops the episode as `technical_failure`;
it is not converted into a scientific conclusion.

The planning resource ceiling is 4,000 aggregate CPU-core-hours, 960 GPU-hours,
336 wall-clock hours, 512 GB peak memory, 128 GB peak memory per trial, and
4,096 GB scratch, with at most 8 concurrent GPUs and 128 concurrent CPU cores.
Profiling must show
that reference qualification, all mandatory reference modes, minimum branch
coverage, and the audit can fit before candidate-discriminating outcomes are
opened. Hitting a resource ceiling at any stage yields `incomplete_search`, not
evidence for or against a hardware design. Exhausting the finite space before
minimum evidence is also incomplete unless the compiler has exhaustively
proved every `D3` branch infeasible. After minimum evidence, exhaustion uses
the terminal cascade: `closed_complete_incumbent_frontier_not_beaten` applies
only if every feasible `D3` fails the explicit development frontier
superiority inequalities; other non-locking cases remain `closed_unresolved`
unless a more specific rule applies.

The first positive score, appearance of an incumbent, or failure of one branch
is not a stop event. Search stops only on the first frozen maximum, resource,
qualified-patience, exhaustion, or all-branches-infeasible event.

## Configuration lock and one-shot audit

The lock bundle hashes:

- every source, exposure, role, and environment manifest;
- the fitted reference-model ensemble and qualification receipt;
- finite hardware and software grammar;
- design-rule and resource compiler versions;
- complete append-only trial ledger and proposal lineage;
- all reference, heuristic, incumbent, and diagnostic design hashes;
- compiled layouts, schedules, ADC/AFE allocations, and post-rounding costs;
- training code, environments, weights, seeds, and capacity receipts;
- metrics, aggregation, intervals, margins, tie and retry rules;
- disjoint development and sealed empirical-replay partition hashes,
  non-overlap proof, diagnostic, and failure threshold;
- exact audit runner and report template; and
- terminal mapping and claim text.

The evaluator accepts only that lock hash. It opens the audit at most once.
An audit result may change the conclusion class but cannot enqueue a new trial,
replace the winner, relax a constraint, or change a margin. A mechanical retry
is allowed only under the common protocol's no-output infrastructure rule.

## Terminal conclusion classes

The following cascade is evaluated from top to bottom; the first satisfied
rule wins, and every rule implicitly requires that no higher-precedence rule
has fired. All scientific comparison rules use the same multiplicity-adjusted
one-sided 95% paired interval family used for promotion.

| Conclusion class | Outer status | Meaning |
| --- | --- | --- |
| `policy_violation` | `technical_failure` | A forbidden action or post-outcome contract mutation invalidates the episode |
| `technical_reference_model_unqualified` | `technical_failure` | The reference ensemble fails its frozen pre-candidate qualification gate |
| `technical_audit_or_support_integrity_failure` | `technical_failure` | Leakage, lock mismatch, duplicate audit opening, or invalid support/evaluator state |
| `technical_failure` | `technical_failure` | Inputs, compiler, execution, or deterministic replay are invalid |
| `incomplete_search` | `closed_no_candidate` | A resource ceiling occurs at any stage, or another authorized stop/exhaustion occurs before minimum evidence without exhaustive compiler-infeasibility proof |
| `closed_resource_or_fabrication_constraint_failure` | `closed_no_candidate` | The frozen compiler exhaustively proves every legal `D3` grammar branch infeasible after rounding |
| `closed_software_only_sufficient` | `closed_no_candidate` | At least one eligible audited `D1` improves on its same-mode `D0` and is noninferior to `D3`, so hardware redesign is not justified |
| `closed_hardware_only_no_joint_gain` | `closed_no_candidate` | Audited `D2` improves on `D0` and is noninferior to `D3`, so the joint claim fails |
| `closed_heuristic_hardware_sufficient` | `closed_no_candidate` | A registered audited `D4` is noninferior to `D3` within the frozen margin |
| `closed_robustness_or_transfer_failure` | `closed_no_candidate` | Frontier and matched-control macro gates pass, but a stratum, secondary, falsifier, or plausibility gate fails |
| `closed_development_only_candidate` | `closed_no_candidate` | Every development lock rule passes, but an audit frontier or matched-control gate fails without a more specific equivalence label |
| `closed_complete_incumbent_frontier_not_beaten` | `closed_no_candidate` | After minimum evidence, the finite admissible `D3` space is exhausted and every feasible design fails the explicit development frontier-superiority inequalities |
| `closed_unresolved` | `closed_no_candidate` | Minimum evidence is complete, but no positive, specific negative, or exhaustive frontier-negative rule is satisfied |
| `candidate_ready_virtual_codesign_specification` | `candidate_ready` | One locked compiled `D3` passes every virtual promotion, falsifier, constraint, and plausibility gate |

For the three specific-control labels, `noninferior to D3` means the lower
bound for `U(control)-U(D3)` is at least `-0.005`. The software-only and
hardware-only labels additionally require the lower bound for `D1-D0` or
`D2-D0`, respectively, to be at least zero. A maximum- or patience-based stop
after minimum evidence is `closed_unresolved` unless a more specific frozen
rule applies; it is not silently promoted to proof that the complete frontier
cannot be beaten.

`candidate_ready_virtual_codesign_specification` is a local scientific
conclusion class. The canonical outer status remains `candidate_ready`; the
richer label does not invent a new server state.

## Deliverables and claim boundary

A positive virtual episode delivers:

- a compiled electrode-pad map and row/source acquisition schedule;
- an explicit external AFE/ADC allocation and resource ledger;
- the measurement-aware reconstruction recipe and calibration contract;
- the complete D0-D4 paired comparison and descriptive complementarity result;
- a reference-model qualification report and model-uncertainty envelope;
- an audit report over sealed mechanisms and device shifts;
- failure maps showing where the candidate loses; and
- a ranked physical-validation plan.

It may support:

> Within the registered paper-derived NeuroCam model ensemble, resource
> constraints, and sealed virtual audit, a compiled co-designed architecture
> improved cortical-surface field recovery over the complete optimized
> NeuroCam reference frontier and simple heuristic controls.

It may not support:

- superiority over a fabricated NeuroCam device;
- chronic safety, stability, biocompatibility, or reliability;
- fabrication readiness or foundry sign-off;
- superiority for unregistered species, cortical regions, or behaviours;
- recovery of unique biological current sources from surface voltage;
- recording of single-neuron spikes in full-array dynamic mode; or
- a universal claim that multiscale electrodes or adaptive scanning are best.

The next physical stage would require, in order, known-input analog replay,
PBS/phantom device testing, a withheld fabricated tile, a full fabricated
array, and appropriately governed in-vivo comparison. None is authorized by
this episode.

## Prior work and novelty boundary

Adaptive electrode selection on Neuropixels establishes that signal context
can guide use of a limited readout budget. General differentiable sensor
placement and computational electrophysiological-sensor layout optimization
also predate this episode. EP20 therefore does not claim novelty for `sensor
placement plus reconstruction` in the abstract.

Its intended contribution is narrower: a row-column-circuit-constrained,
device-shift-audited decomposition of software-only, hardware-only, heuristic,
and joint co-design gain for a NeuroCam-derived two-dimensional micro-ECoG
reference.

Primary references:

- Xie et al., *High-resolution spatial mapping of electrocorticographic
  activities with NeuroCam: a 4096-channel, multiplexed flexible thin-film
  transistor array*, Science Bulletin 70 (2025), 4133-4137,
  <https://doi.org/10.1016/j.scib.2025.11.030>.
- Choi et al., *Optimal Adaptive Electrode Selection to Maximize
  Simultaneously Recorded Neuron Yield*, NeurIPS 2020,
  <https://proceedings.neurips.cc/paper/2020/hash/445e1050156c6ae8c082a8422bb7dfc0-Abstract.html>.
- Liu et al., *Enhancing deep learning-based field reconstruction with a
  differentiable learning framework*, Nature Machine Intelligence 7 (2025),
  <https://doi.org/10.1038/s42256-025-01063-1>.
- Kim et al., *Computational design and optimization of electro-physiological
  sensors*, Nature Communications 12 (2021),
  <https://doi.org/10.1038/s41467-021-26442-1>.

## Requirements before candidate scoring and audit

Candidate scoring and audit access remain closed until all of the following
exist and are frozen:

- immutable published-paper, supplement, and characterization-source
  manifests with hashes and rights records;
- raw calibration traces or an explicitly accepted aggregate-only reference
  limitation;
- exact spatial subsets, channel identities, and marker semantics for every
  registered reduced NeuroCam operating mode;
- an exact fitted-versus-held-out reference-qualification split, a resolved
  frequency-response test-amplitude conflict, and frozen tolerances;
- a finite legal pad catalogue backed by a PDK or scientist-approved,
  provenance-bearing conservative surrogate design-rule deck;
- a complete resource compiler for array lines, AFE/ADC work, conversions,
  bit-rate, latency, defensibly bounded power/thermal proxy, and routing
  burden;
- executable development signal and device generators with registered
  parameter provenance and separate hashed manifests;
- an independently implemented, permission-separated audit generator and
  device-model ensemble, hidden-payload Merkle root, and dependency report;
- exact, legally usable empirical ECoG replay or known-input phantom sources
  with disjoint development-stress and sealed plausibility partitions;
- outcome-blind structural calibration showing that the proposed margins and
  sample counts are resolvable;
- the frozen admissible-space, environment, split, exposure, and audit
  manifests named in `DATASETS.md`;
- a trusted evaluator that returns only predeclared audit outputs;
- profiled compute demonstrating that minimum evidence fits the ceiling;
- human sign-off on the scientific margins and claim text.

Until then, this directory defines the research design; no candidate or audit
result exists.
