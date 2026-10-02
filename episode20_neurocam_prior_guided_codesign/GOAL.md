# When should a constrained NeuroCam redesign recover cortical voltage fields better?

NeuroCam uses a 64 × 64 multiplexed electrode array to observe voltage on the
cortical surface. Its architecture creates a basic trade-off. Reading more
pixels expands spatial coverage, while revisiting fewer rows or source groups
improves temporal resolution. Electrode size adds another trade-off between
spatial averaging and sensitivity to local events.

EP20 asks whether a legal change to **electrode geometry and scan allocation**,
paired with reconstruction software that knows the resulting measurement
pattern, can improve that trade-off. A proposed design must retain the
published 64 × 64 lattice, 150-micrometre pitch, approximately 9.6 × 9.6 mm
active area, one-TFT row-column topology, and no more than 64 gate plus 64
source lines. It cannot gain extra conversion work, latency, bitrate, modeled
power, or thermal capacity.

The comparison must distinguish true hardware-software complementarity from
four easier explanations. On the same cortical fields, device conditions,
masks, and resource budget, the co-designed candidate is compared with:

1. published NeuroCam hardware with conventional reconstruction;
2. published NeuroCam hardware with equally optimized reconstruction;
3. the candidate hardware with conventional reconstruction; and
4. simple geometry or scheduling rules with the same optimized software and
   tuning allowance.

Only after those comparisons are frozen do one representative from each D0–D3
factorial family and the required C4 controls face an independent final test
using separately implemented cortical-field and electronics models. A selected D3
that passes could justify fabrication or device testing of **one virtual
design**. For an in-depth paper, that is not enough: the same analysis must
produce a simple rule stating when smaller pads, faster revisiting, or their
combination should help, and that rule must predict winners under independent
conditions.
It would still not show that an unbuilt device outperforms fabricated NeuroCam,
works chronically in vivo, or is ready for manufacturing.

![EP20: field/device conditions, the correct design-family mapping and model uncertainty](outputs/ep20_question_imagegen-v2.png)

The D0–D3 table is the diagnostic factorial, not a replacement for the main
C0–C4 comparison. Columns vary spatial benefit; rows vary the bundled scan/
reconstruction benefit: D0 neither, D1 time-focused only, D2 geometry only,
D3 both. Array glyphs retain the same lattice, pad count and outline; the
small 4×4 drawings are schematic, not a proposed replacement for the actual
64×64 architecture. Pad sizes may change within that lattice.

Panel C compares the same proposed device and the same held-out field
condition under different accepted electrical models. Waveform glyphs denote
operators, not different test conditions or measurements. Disagreement calls
for a discriminating measurement; rule failure permits a descriptive comparison
only, not promotion. No virtual or physical result is shown. [Exact generation
and correction prompts](outputs/ep20_question_imagegen-v2-prompt.md) are saved.
Earlier concept images remain historical assets, not current design diagrams.

## The question at a glance

| Question | EP20 design |
| --- | --- |
| What may change? | A legal uniform or two-scale pad pattern, a legal row/source acquisition schedule, and a capacity-capped measurement-aware reconstruction method |
| What stays fixed? | 64 × 64 lattice, 150-micrometre pitch, approximately 9.6 × 9.6 mm active area, one-TFT row-column topology, and at most 128 array gate-plus-source lines |
| What is reconstructed? | Cortical-surface voltage on a common dense grid in the primary 1–100 Hz band |
| What is the main score? | Paired change in field-reconstruction R² against every eligible published-hardware reference setting |
| What prevents an easy win? | Equal software capacity, training effort, data, seeds, conversions, latency, bitrate, and modeled physical-resource accounting |
| What tests generality? | Registered challenge tests and one independent final comparison using separately implemented signal and electronics models |
| What makes the result transferable? | A frozen rule uses 90%-power spatial and temporal scales in micrometres and seconds, and independently characterized noise, impedance, settling, and crosstalk to predict the preferred D0–D3 family |
| What could make optimization meaningless? | Two reference models can both fit the published measurements yet rank candidate designs differently |
| What is the empirical check? | Separate development and final physical-signal or phantom checks for spectra, amplitudes, missingness, and catastrophic implausibility; these cannot validate an unbuilt geometry |
| What would success mean? | One virtual specification is worth fabricating, and a measurable design rule predicts where its advantage should and should not hold |

## The published reference

The named reference is the NeuroCam article and supplement described in
[DATASETS.md](DATASETS.md). The published array has 4,096 pixels, a 64 × 64
lattice, 150-micrometre pitch, approximately 9.6 × 9.6 mm active area, 70 ×
135 micrometre gold sensing pads, and 64 shared gate plus 64 shared source
lines.

Three reported dynamic settings anchor the virtual reference:

| Setting | Pixels acquired | Effective rate per pixel | Reported RMS noise |
| --- | ---: | ---: | ---: |
| 24 source × 16 gate | 384 | 1,000 samples/s | 8.8 ± 4.2 microvolts |
| 24 source × 64 gate | 1,536 | 250 samples/s | 12.6 ± 8.4 microvolts |
| 64 source × 64 gate | 4,096 | 250 samples/s | 63.6 ± 24.1 microvolts |

The paper also reports representative gain of 0.78 ± 0.12, channel bandwidth
above roughly 1 kHz, source-line neighbour crosstalk near −12.6 dB, gate-line
neighbour crosstalk near −37.2 dB, and more distant crosstalk near −45 dB.
These values constrain a **paper-derived NeuroCam model**; they do not define a
complete or transistor-accurate digital twin.

The [compact anchor assignment](DATASETS.md) now uses transfer, static/reduced-
mode noise and nearest-neighbour coupling for calibration, while holding full-
array dynamic noise and distant coupling aside from fitting. This is a public-
paper consistency check, not a blind or independent-device validation. Source
use and unresolved measurement conventions still need resolution before fitting.

The published anchors may admit more than one plausible scaling law. EP20 must
therefore retain every predeclared reference-model variant that passes the same
calibration and held-aside checks. Before optimizing a candidate, run the same
small, resource-matched sentinel panel through all accepted variants. Repeat
the check after one representative or deterministic selection rule has been
frozen for each D0–D3 family. If two accepted models produce a material
reversal—different unique winners whose simultaneous one-sided lower bounds are
each more than 0.005 above their own runner-up—the current measurements do not
identify an optimum. Optimization
stops. The next result is the additional noise, impedance, settling, or
crosstalk measurement needed to separate those models, not a chosen design.

The 1-kHz channel bandwidth does not imply 1-kHz full-array imaging. With the
reported 62.5-microsecond row dwell, a 64-row scan takes about 4 ms and has a
nominal per-pixel Nyquist frequency near 125 Hz. The primary endpoint is
therefore limited to 1–100 Hz. A 100–400 Hz reduced-mode result is descriptive
and cannot rescue a failed primary result. “Focal transient” refers to a field
event, not a sorted single-neuron spike.

## The five matched comparisons

The plain-language fairness comparisons receive **C** labels only for compact
tables. The later two-by-two prediction uses **D** labels, so a broad comparison
arm is never mistaken for a factorial family.

| Label | Hardware | Software | Question |
| --- | --- | --- | --- |
| **C0 — published/conventional** | Every eligible published NeuroCam setting | Conventional method | What does the paper-derived reference do with a standard analysis? |
| **C1 — published/optimized** | The same complete published-hardware frontier | Every registered optimized method with equal allowance | Is software alone sufficient? |
| **C2 — candidate/conventional** | One legal candidate design | Conventional method with a compatible adapter | Does the hardware change alone help? |
| **C3 — candidate/co-designed** | The same candidate design | One measurement-aware method chosen on the published hardware before candidate results | Does hardware-software co-design help? |
| **C4 — simple controls** | Every registered resource-matched geometry or schedule rule | The same optimized software family and allowance | Is a simple rule sufficient? |

For the transferable two-by-two rule, the distinct D0–D3 representatives form a
factorial panel. D0 uses the published geometry, published-like scan, and
conventional reconstruction. D1 holds geometry fixed and uses the frozen
time-focused legal scan/software recipe. D2 changes geometry while retaining
the D0 scan and conventional recipe. D3 combines the D2 geometry with the D1
time-focused acquisition and its separately trained measurement-aware inverse.
C0, C1, C2, and C3 respectively supply those frozen representatives, but their
remaining frontier members stay fairness comparators rather than prediction
families.
C4 remains a required simple-control set but is not one of the four predicted
families.

[Component details](outputs/component_details.md) define field-to-measurement
and inverse interfaces and the existing challenges to S/T separability. They
also distinguish geometry-only D2 from D3's matched C2, which retains D3's scan
as well as its geometry. The refinement keeps method choices open without
changing numerical criteria or the operative comparisons.

The [starting parameter/candidate catalogue](outputs/parameter_candidate_catalogue.md)
now selects evidence-contextualized field controls, 28 initial condition
classes, three proposed pad sizes and explicit scan sequences. These are
development starting choices, not a closed search or physically qualified
designs. The same endpoint, comparisons and trial budget remain in force.

The [electrical mechanisms and probes](outputs/electrical_model_and_probes.md)
select two contact-scaling alternatives and three persistent-state alternatives,
plus common finite probes. These are six initial structural templates, not
accepted reference models. Missing absolute circuit constants and physical ADC
configuration remain unknown; the existing held-aside checks decide acceptance.
The minimal reference fit now states which shared effective combinations the
published measurements constrain, without separately fitting each operating mode.

The [compact component choices](outputs/component_details.md) now add source
conductivity/depth/gap sensitivities and small inverse capacity, fitting and
causal-history allowances. These are starting choices, not a closed exploration
or demonstrated runtime/performance limits.

The full published-hardware C0/C1 frontier is established before any C2–C4
result is used. The three optimized software families are a physics-informed
linear inverse, a structured state-space inverse, and a compact past-only
nonlinear inverse. One family is chosen using only C1 performance; all C1
settings remain comparators.

C1, C3, and C4 use the same capacity, solver/optimizer, fitting allowance,
data, seeds and selection attempts within the selected optimized family.
Different families use appropriate fitting methods with shared data, seeds
and selection allowances. Every hardware measurement
operator is trained separately. Reusing favourable C3 weights for another arm
is prohibited.

## What a candidate may change

The physical design may use the published uniform pads, interleaved two-scale
pads, or tiled two-scale pads on the same lattice. At most three pad classes
are allowed. The starting classes are published 70 × 135 micrometres,
proposed 50 × 95 micrometres, and proposed 35 × 70 micrometres. Initial
proposals use balanced half-area interleaved and tiled maps; quarter-area
versions remain follow-ups. Exact placement and via/routing clearances still
need resolution before scoring; geometric subfootprints alone do not establish
physical legality. Continuous idealized shapes are not final candidates.

The readout may use the published uniform scan, fixed heterogeneous row dwell
and revisit, fixed source-bank precision allocation, or one bounded past-only
row-revisit rule. Legal dwell multipliers are 1, 2, and 4 times the 62.5-
microsecond base dwell; shorter dwell is not allowed. A row action addresses a
legal gate group, not an arbitrary pixel.

The design cannot add on-array memory or computation, arbitrary analog mixing,
local analog aggregation, a new ADC or front end, hierarchical routing, a new
material or transistor process, a changed active area or pitch, or more than
128 array lines.

Before a design is scored, a common design-legality and resource check applies
fabrication rounding and accounts for:

- active area and gate/source lines;
- reference, ground, guard, and supply cost;
- active front-end and ADC channels;
- raw conversions and effective pixel samples per second;
- output bitrate, latency, and scan skew;
- gate switching, modeled power, and thermal burden;
- routing area and electrical loading; and
- yield, dead-pixel, and dead-line burden.

Unknown cost is never treated as zero.

## The transferable result: a condition-to-design rule

The rule cannot use a candidate's observed reconstruction score or a vaguely
defined "penalty." Before any D0–D3 outcome is computed, condition *k* supplies
two field measurements from simulated truth or a separate known-input pilot.
Let `k90` be the smallest radial spatial frequency containing 90% of common-
mask 1–100 Hz field power and set `lambda_k = 1/(2*k90)` in micrometres, also
reported in common-grid pixels. Let `f90` be the smallest temporal frequency
containing 90% of the demeaned field power and set `tau_k = 1/(2*f90)` in
seconds. These definitions work for broad, focal, and travelling fields; they
cannot be estimated from the candidate's reconstruction.

These are spectral 90%-power scales, not autocorrelation decay lengths/times.
Identical radial spatial and temporal summaries can omit orientation and
space–time phase dependence. The already-required wave and coupled-device
challenges therefore test, rather than assume, the rule's S/T separability.

Each accepted reference model *m* also supplies an independently characterized
table for every device condition *k* and frozen family representative *d*.
On a common outcome-blind neural-site panel P, noise is equal-site pooled RMS
microvolts from a zero-input 1–100-Hz probe; impedance is the maximum site-level
90th percentile of contact magnitude over equally weighted integer 1–100-Hz
tones, in ohms; settling is worst-panel/history microseconds for the conditional
mean unit-step response to enter and remain within 1%; and crosstalk is the
largest nonnegative same-victim input-equivalent adjacent-line ratio
`abs(H_ij/H_ii)` over panel victims, physical neighbours and registered multisine
tones/phases. Decibels are supplementary. The
[probe protocol](outputs/electrical_model_and_probes.md) fixes P before C1 scores
from the verified intersection of published-setting readout masks, retains
off-gate aggressors, and defines waveforms, times, censoring and aggregation.
Panel characterizations are not full-array maxima; the dense primary scoring
mask is unchanged. These are absolute measurements or frozen model
outputs obtained without reconstruction results. A quantity such as "the
penalty of the proposed design" is not a legal input.

For arm *d* under model *m* and condition *k*, observed utility `U_mdk` is the
mean 1–100 Hz field R² over paired replicate worlds with the fixed truth mean
and mask. Development data fit a shallow monotone predictor
`Uhat_md(x_mdk)`, where `x_mdk` contains `lambda_k`, `tau_k`, and that arm's
absolute characterization values. Margins are applied later to paired
contrasts; they are not part of either utility. The predictors share the
registered form but accepted models are never averaged. They are frozen before
held-out outcomes. The predicted winner within each model is the arm with the
largest `Uhat_md`, not whichever candidate later reconstructs best.

The interpretable prediction is a two-by-two branch whose gates use those raw
measurements, not fitted outcomes. For each model and condition, the pad-noise
cost is the larger of the D2/D0 and D3/D1 noise ratios; the pad-impedance cost is
defined the same way. The schedule-settling cost is the larger of the D1−D0 and
D3−D2 settling increases; the schedule-crosstalk cost is the same maximum for
the nonnegative crosstalk ratio. Let **S** be on only when `lambda_k` is below
its frozen scale threshold and both pad costs are below their frozen limits.
Let **T** be on only when `tau_k` is below its frozen scale threshold and both
schedule costs are below their frozen limits. All six thresholds are learned on
development conditions and frozen before held-out outcomes.

| Spatial branch S | Temporal branch T | Predicted family |
| --- | --- | --- |
| no | no | D0: published-like acquisition and conventional reconstruction |
| no | yes | D1: time-focused published geometry with its frozen schedule/software rule |
| yes | no | D2: geometry-only candidate with the conventional rule |
| yes | yes | D3: joint geometry, schedule, and measurement-aware reconstruction |

A failed noise or impedance gate can switch **S** off but cannot erase a valid
**T** branch; a failed settling or crosstalk gate can switch **T** off but cannot
erase a valid **S** branch. The mapped family must also lie within 0.005 of the
largest frozen predicted utility. Otherwise the separable two-by-two rule has
failed before outcomes and must abstain. If a scale or cost interval crosses a
threshold, the prediction is the corresponding adjacent-family set rather than
a forced D0. For example, S tied with T off predicts {D0, D2}; S tied with T on
predicts {D1, D3}. Both tied predicts all four.

One representative or deterministic selection rule for each D0–D3 family is
chosen using development conditions under all accepted reference models and
then frozen. On every held-out and independent condition, all four are rerun on
the same worlds, separately for every model; performance is never averaged
across models. Each representative contains one global hardware specification.
A deterministic operating recipe may depend only on the registered resource
budget, and a permitted adaptive scheduler may use only its frozen past-only
inputs; neither may switch hardware or recipes after seeing the condition label,
`lambda_k`, or `tau_k`. The rule receives only the absolute inputs above and
predicts the winner before reconstruction. The observed winner set contains
every arm within 0.005 of the best R²; an uncertainty interval that cannot
separate the leaders is reported as unresolved. A unique prediction's regret is
best-arm R² minus predicted-arm R². For a threshold-tie prediction set, use its
worst member so an uninformative wide set cannot earn zero regret. The
simultaneous one-sided 95% upper bound on equal-weight mean regret must be at
most 0.005 in every accepted model and every S/T cell.

Also report the four-class confusion matrix, abstention rate, balanced winner
accuracy, and sign accuracy for gain over D0, with uncertainty resampled by
condition rather than by pixel or time point. Abstaining on an observed non-tie
counts as an error; ties are not silently removed to improve accuracy. These
classification summaries are descriptive because the final winner-class counts
cannot be guaranteed in advance. Promotion rests on the simultaneous regret
gate, with at least five pre-outcome conditions in every S/T cell under every
accepted model and no prespecified field family showing a confident reversal.
If one candidate wins on average but the rule fails, that result remains a
descriptive model-specific comparison. It cannot be nominated or advanced under
the current promotion rule, and no transferable acquisition principle is
established. This does not add a gate: the policy already requires the rule test.

## How the comparison is run

1. Divide published measurements into calibration and held-aside roles, fit the
   predeclared family of NeuroCam reference models, and retain every variant
   that passes the same check.
2. Before candidate optimization, run the small design panel across all accepted
   reference models. Stop design selection if a material candidate-ranking
   reversal remains.
3. Set the finite hardware choices, absolute condition measurements,
   family-specific utility form, resource accounting,
   software allowance, field and device models, endpoints, margins, and random
   draws before candidate results.
4. Evaluate every C0 and C1 setting and choose the optimized software family
   without candidate-hardware results.
5. Complete a 16-arm coverage panel: one C0, three C1, four C2, four C3, and
   four C4 arms.
6. Explore legal follow-up designs on development models while reserving at
   least 40% of post-coverage trials for challenge tests, ablations, and
   direct replications.
7. Freeze one representative or deterministic selection rule for each D0–D3
   family across all accepted reference models. Recheck that the models do not
   materially reverse their ranking.
8. Fit the bounded condition-to-design rule on development conditions, test it
   on held-out development conditions, and freeze it.
9. Choose exactly one joint C3/D3 candidate only if it passes every development
   comparison, secondary endpoint, resource check, required challenge test,
   rank-stability check, and design-rule check.
10. Rerun all four frozen D0–D3 representatives, plus required C4 controls, under
    every accepted reference model on the independent conditions. No new design,
    family representative, or rule may be chosen from that result.

The study uses one global hardware specification. A resource budget may select
a pre-set operating recipe, but the hardware cannot change after seeing a
signal family or device condition.

## Primary and secondary measurements

The primary endpoint is cortical-surface field reconstruction R² in the
1–100 Hz band on one common space-time mask. Every comparison is paired on the
same field realization, source-to-surface model, device and process draw,
nuisance draw, mask, and resource budget.

Results are first averaged within a signal-by-device condition. Device
conditions receive equal weight within each signal family, and signal families
receive equal weight. Pixels and time points are not treated as independent
replicates.

The candidate must beat every eligible C0 and C1 reference setting. The primary
summary is the least favourable paired gain across that complete frontier.
One-sided simultaneous 95% intervals determine whether each margin is met; the
study never chooses a different reference for each scene after seeing results.

Required secondary outcomes are:

- focal-event localization error;
- propagation direction and speed;
- transient-detection proper log-score utility;
- uncertainty coverage, sharpness, and calibration;
- worst signal-family and device-condition gain;
- performance after fabrication rounding;
- latency, conversions, bitrate, modeled power, thermal, routing, and yield;
  and
- transfer to one pre-set downstream probe.

Transient AUROC, latent-source recovery, 100–400 Hz performance, and a
descriptive interaction score cannot promote a candidate or rescue failure of
the field endpoint.

## What counts as a positive virtual result

| Comparison or safeguard | Required margin |
| --- | ---: |
| C3 versus every eligible C0/C1 reference setting | R² lower bound at least 0.010 |
| C3 versus its matched C1 software-only comparator | R² lower bound at least 0.005 |
| C3 versus its matched C2 hardware-only comparator | R² lower bound at least 0.005 |
| C3 versus every eligible C4 simple control | R² lower bound at least 0.005 |
| Every signal/device/reference condition | R² lower bound at least −0.005 |
| Localization-error noninferiority | 0.15 mm |
| Wave-direction-error noninferiority | 5 degrees |
| Wave-speed-relative-error noninferiority | 0.05 |
| Transient log-score-utility noninferiority | 0.01 |
| Uncertainty calibration-error noninferiority | 0.01 |
| Downstream normalized-utility noninferiority | 0.01 |

All margins require outcome-blind calibration on synthetic examples and
scientist approval before candidate scoring. Hardware and resource constraints
must pass again after discrete rounding.

If several designs pass every development requirement, choose the one with the
best worst-condition paired gain. Designs within 0.005 are tied; prefer lower
modeled conversion energy, then lower scan latency, then the earliest
preassigned design name. Exactly one design proceeds.

## Results the study can distinguish

| Evidence pattern | Scientific reading |
| --- | --- |
| Co-design beats the full frontier and all matched controls twice | One virtual design is justified for fabrication or device testing |
| Published hardware with optimized software matches C3 | Better reconstruction is sufficient; hardware redesign is not supported |
| Candidate hardware with conventional software matches C3 | The hardware change may help, but special co-designed reconstruction is not supported |
| A simple geometry or schedule matches C3 | The complex search did not beat the registered simple rule |
| Main comparisons pass but a condition, secondary endpoint, rounding, or plausibility check fails | The gain is too fragile to advance |
| Development passes but the independent model test fails | The result does not generalize beyond the development models |
| Accepted reference models reverse the candidate ranking | Published measurements do not identify a preferred design; obtain the measurement that separates the models before optimizing |
| One candidate wins but the condition-to-design rule fails | Retain a descriptive model-specific comparison; do not nominate or advance a design under the current promotion rule |
| The complete finite candidate space cannot beat the full reference frontier | No legal design in this search space improves the registered virtual comparison |
| Evidence remains too uncertain | Report what remains unresolved rather than selecting a design |

If the published measurements cannot support a credible reference model, or
if every legal branch fails physical or resource rules, the co-design question
was not tested. Neither case is evidence that the published architecture is
scientifically optimal.

## Required challenge tests

- sweep the complete C0/C1 published-hardware frontier;
- test candidate-rank stability across every reference model that passes the
  published calibration and held-aside checks;
- shift spatial spectrum, correlation length, propagation speed, direction,
  curvature, and broad/local mixtures;
- place focal events in previously silent or unexpected regions;
- change the source-to-surface operator;
- jointly vary noise, crosstalk, line RC, settling, drift, saturation, contact
  gaps, and dead pixels;
- match software capacity, training steps, seeds, and selection allowance;
- train C1, C3, and C4 separately;
- compare random and simple geometry or schedule controls;
- reverse large- and small-pad assignments;
- remove geometry, schedule, optimized software, and uncertainty modeling one
  at a time;
- verify one application of pad averaging, past-only scheduling, voltage
  reference, fabrication rounding, and post-rounding resource use;
- repeat the current best design and comparator on common random draws;
- test spectra, amplitude, missingness, and software stability on separate
  physical-signal or phantom data; and
- leave out each signal family and device condition in turn.
- freeze the absolute-input D0–D3 rule and test its regret, winner, and gain-sign
  predictions on held-out conditions.

These tests are required evidence, not optional follow-up figures.

## Development budget and stopping

The search requires **32–64 valid trials**. It begins with the exact 16-arm
coverage panel and must contain at least **two genuine follow-up cycles**, at
least **two recorded choices between the current best design and a new
proposal**, and at least **40% challenge tests, ablations, influence checks, or
direct replications** after coverage. Patience is 12 eligible co-design
opportunities after all minimum evidence is complete.

Up to 12 repairable engineering failures are allowed; the next one ends the
study as a technical failure. Resource limits are **4,000 CPU core-hours**,
**960 GPU-hours**, **336 wall-clock hours**, at most **8 GPUs** and **128 CPU
cores**, **512 GB total memory**, **128 GB per trial**, and **4,096 GB scratch**.
Reaching a resource limit is not a positive or negative scientific result.

Patience resets after either a 0.010 improvement in the worst-condition paired
gain or a new feasible trade-off point. One positive or negative trial never
ends a branch by itself.

## Independent final model test

The final comparison changes more than random seeds. It uses separately
implemented cortical-field mechanisms, a source-to-surface operator absent
from development, a separately implemented electronics scaling or coupling
law, altered spatial spectra, nonstationary sources, and unseen combinations
of noise, crosstalk, settling, drift, gaps, saturation, and dead pixels.

Planning requires at least three hidden signal mechanisms, four device-shift
conditions per mechanism, and 64 paired field realizations per condition. The
condition grid must also contain at least five conditions in every S/T cell
under every accepted model. Cell support may be filled using only the frozen
absolute inputs, never reconstruction scores. Exact counts may increase after
outcome-blind interval-width and resource calibration.

The complete C0 and C1 frontier, matched C2 arm, selected C3 candidate, and all
eligible C4 controls run together on identical paired worlds. No method,
margin, aggregation rule, hardware design, operating recipe, or runner-up may
replace the pre-set choice afterward.

For every independent condition and accepted reference model, the frozen rule
receives only `lambda_k`, `tau_k`, and the independently characterized
noise/impedance/settling/crosstalk table for the four frozen representatives.
All D0–D3 representatives are rerun together. The rule's predicted family and
gain sign are recorded before reconstruction; the observed winner and any tie
or unresolved result are then scored. Passing the selected C3 comparison
without passing this four-family prediction supports one virtual device, not
the general design rule.

A separate physical-signal or phantom check may reject catastrophic spectra,
amplitudes, missingness, or software behaviour. It cannot compare unbuilt pad
geometries or establish physical superiority.

## Claim boundary

The strongest permitted positive statement is:

> Under the paper-derived NeuroCam model, registered cortical-field and device
> conditions, matched software allowance, and modeled resource constraints,
> one virtual hardware-software design outperformed every eligible published-
> hardware reference setting and registered simple control, including under an
> independently implemented final model test; a frozen rule based on spatial
> scale, temporal timescale, noise/impedance, and settling/crosstalk predicted the
> conditions under which its design family was preferred.

That statement must immediately add that the design is unfabricated. EP20
cannot claim physical superiority, manufacturing readiness, chronic safety,
biocompatibility, reliability, lifetime, or in-vivo performance.

## Current status

EP20 is specified but not ready to run. The article and aggregate published
measurements and anchor roles are identified, but source-use approval, usable
measurement contexts, the accepted reference-model set and rank-stability check,
the paper-derived reference model, legal pad catalogue,
development field and device models, resource rules, independent final models,
and physical-signal or phantom data do not yet exist as episode inputs.

The starting catalogue is documented, but its physical eligibility is not
established. Published source-stream sample counts also do not identify
internal ADC conversion work; that distinction must be resolved for resource
parity rather than assuming equality from the nominal rate alone.

Raw NeuroCam traces, exact reduced-mode channel maps, marker meanings, a PDK,
compact transistor model, netlist, layout, DAQ code, and operating-mode power
traces remain unavailable. Their absence narrows EP20 to a paper-derived
virtual study; it does not permit guessed parameters or zero cost for unknowns.

Source details are in [DATASETS.md](DATASETS.md). Compact methods and budgets
are in [SEARCH_POLICY.yaml](SEARCH_POLICY.yaml). The proposed paper story is in
[outputs/paper_plan.md](outputs/paper_plan.md).
