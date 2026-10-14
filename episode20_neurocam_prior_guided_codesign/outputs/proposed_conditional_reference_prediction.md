# Adopted conditional reference prediction — October 13, 2026

The scientist explicitly adopted this proposal with “采用条件化预测方案（推荐）”.
The conditional route below is active; implementation remains necessary before
calibration. The existing live 4,000-frame engineering record remains unscored.

**Adopted decision:** permit a partial, explicitly conditional prediction of
the already assigned Fig. S5 second/far-neighbour response as EP20's next
scientific result. Fit only its assigned nearest source/gate observations,
then freeze the compatible parameter sets before evaluating the distant check.
The amendment permits a prospectively selected measurement operator where the
historical operator is unspecified; it does **not** accept a NeuroCam reference,
waive the remaining transfer/noise checks, or authorize candidate scoring.

## What the source resolves

The [article and supplement](https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf)
bind Fig. 2a to all 4,096 channels, a 12-Hz sinusoid, PBS and `Vout/Vin`.
The methods describe platinum-mesh stimulation and Gaussian histogram fitting.
They do not bind that gain's amplitude extractor, dispersion definition or
complete reference/acquisition operator. Fig. 1f supplies a fixed-gate
whole-path frequency response; its existing conditional time-constant bound
is already recorded and is not a new result. S5 uses a single metal-pad probe
in air, 5 mVpp at 12 Hz, but the ratio denominator and scan/DAQ configuration
remain unspecified in the inspected methods/caption. The generic processing
description does not resolve those omissions. In particular, a 1–199-Hz filter
cannot be applied after demultiplexing a 250-Hz pixel stream.

Article use is already authorized. The remaining issue is the observation
model, not permission to read the paper.

## Conditional operator and claim

Select one operator before fitting: drive zero-based Au pad `(31, 31)` with the
stated sine, referenced to the common electronics return; use the full
64-row, 62.5-microsecond scan and the existing explicitly conditional 24-ADC
mapping, apertures and third-to-fifth-sample aggregation. Fit sine and cosine
coefficients plus an intercept at the actual per-site timestamps. Define
`dB_i = 20 log10(A_i / A_driven_output)`, with amplitude from those coefficients.
For nearest-source calibration use the maximum amplitude ratio at `(30,31)`
and `(32,31)`; for nearest-gate calibration use the maximum at `(31,30)` and
`(31,32)`. The held-aside scalar is the maximum over every other physical pad
with Manhattan distance at least two from the driven pad. Retain the full
per-pad table and both directional distance curves. These mask/aggregation
choices are prospective interpretations, not established properties of S5.
Use the zero-noise conditional circuit response for this coherent-coupling
prediction; noisy observed amplitudes and detection floors are not fitted as
extra coupling. This is an assumed
S5 interpretation, not recovered wiring or the separate same-victim
`H_ij/H_ii` characterization probe.

The probe is a direct metal-pad voltage boundary. An air fixture cannot reuse
a PBS pre-contact voltage source without an explicit physical derivation.
For the conditional dry-air circuit, remove the PBS bath-source, spreading/
access and electrolyte contact-interface branches at every pad; retain the metal-pad reference/mutual capacitances,
declared leakage, TFT on/off paths, source loads and acquisition states. Those
air parasitics/load values require an explicit conditional family specification;
the PBS cost recipe is not an air measurement. Uncontacted metal pads float
through those retained paths rather than being grounded. The new fixture must
represent the direct probe boundary and its loading before fitting begins.
E-area and E-sublinear are consequently equivalent in this fixture: keep both
named templates and disclose that equality, not six independent evidence cases.
No steady-state or shortened-record surrogate
is authorized by this proposal; a time-domain prediction retains a complete
16-second record and declared initial/history treatment.

Retain E-area/E-sublinear crossed with R1/R2/R3. Parameter sharing, neighbour
topology, conditional nuisance constants or bounds, and initialization must be
written before the nearest-neighbour fit. Existing cost-case constants can be
used only as explicitly selected conditional assumptions, never as calibrated
values or source-derived bounds. The family specification must name every
direct or mediated distant path. If it selects the cost case's nearest-edge
graph, absent direct long-range edges are an explicit conditional assumption,
not physical absence inferred from the nearest data. Any additional allowed
unconstrained path remains in the prediction uncertainty. Profile all fitted shared coupling parameters
that remain compatible with the nearest observations; do not add a distant
floor, per-mode residual, or parameter chosen from the distant target.

The scientific artifact is a distance-versus-coupling figure containing every
compatible model prediction/envelope and the held-aside paper summary, plus
the calibration residuals and a three-line interpretation. A narrow prediction,
rejection under the chosen assumptions, or a demonstrably uninformative
prediction set can each be reported. None establishes a material R² rank
reversal, nominates hardware, or ends EP20's robust-ranking question.

Report a numerically explored compatible-parameter envelope unless the complete
declared domain has actually been bounded; a finite optimizer sample is not an
identified set. Neither range is a device-level confidence interval. The published spatial samples do not
provide independent-device replication. Keep the existing check tolerances;
do not announce formal reference acceptance before all existing prerequisites.

## Concrete work still required

This is not an executable fit today. The source exposes an actual gap:
`switched_line.py` is one R1 pole; the 64-line contact/neighbour/DAQ adapter is
unfinished. A complete six-template air-probe family, shared nuisance domain
and fitting runner also do not exist. Choosing a convenient neighbour law
after seeing the distant residual would invalidate the proposed prediction.

Implement new unused `tools/reference_s5_operator.py` (boundary,
clock, amplitude and ratio), `tools/reference_s5_family.py` (all six structures
and an explicit shared conditional parameterization), and
`tools/reference_s5_prediction.py` (nearest-only compatible-set fit followed by
frozen distant prediction and one figure). Finish the family specification
before a fitting job; preserve the active continuous-driver imports.
The first compute request is finite: one coordinated EP20 allocation of
1 CPU, 7 GiB and 2 hours for the calibration-only fixture/fit attempt, with
one numerical thread and no held-aside read. This is an initial work envelope,
not a promise that a 16-second fit finishes there; extend from measured cost
under standing authority and use clean frame checkpoints where applicable.
Do not release the distant comparison until the fitted family is frozen.

The primary 1–100-Hz full-16-second R² estimand, 64×64 geometry, resource parity,
paired independent-world uncertainty contract and closed independent-final
data remain unchanged. The original offset-equivalence result remains a
dependency for using the continuous production solver in scientific runs.

If this conditional interpretation is not adopted, the minimum original
experiment information is: S5 stimulus return/output reference, amplitude
denominator/extractor, gate scan and ADC configuration; for the full reference,
also the gain summary operator, noise filter placement/bandwidth/windows and
meaning of each spread. Original traces/configuration can supply these;
manufacturer capabilities and additional scalar optimizers cannot.
