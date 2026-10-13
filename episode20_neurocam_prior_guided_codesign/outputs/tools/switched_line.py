"""Causal R1 source-line pole from electrical_model_and_probes.md, section 4.

This component implements dx/dt = (gain * v_eff - x) / tau for a
piecewise-constant effective input and an explicit fixed-capacitance charge
jump. It has no NeuroCam parameter defaults, row/ADC clock, contact CPE,
off-gate coupling model or noise generator. The caller owns those components
and supplies their effective drive once; this is not a complete reference.
Optional explicitly reduced held current ports add I_eq/C to dx/dt. They do
not infer a noise PSD from C, tau or gain. There is no 64-line R1 acquisition
adapter here, nor a separate contact/DAQ model supplied by this scalar class.
"""

from dataclasses import dataclass
import math

from r1_noise import R1HeldCurrentNoise, R1NoiseWork


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class IntervalObservation:
    """Analog state after the switch, end state and exact time-average, in V.

    The interval average is available for a configured aperture integral; it
    does not assert that the actual NeuroCam ADC averages this interval.
    The endpoint is an ideal point sample. Neither includes quantization,
    clipping or an extra acquisition-time shift. Explicit held analog current
    noise, when supplied, is already propagated in these voltages.
    """

    post_switch_volts: float
    end_volts: float
    mean_volts: float
    duration_seconds: float
    noise_work: R1NoiseWork | None = None


class SourceLinePole:
    """One persistent voltage state in a fixed capacitance basis.

    Initial voltage and capacitance must be supplied explicitly. Keep the same
    instance across row changes; ``advance`` never resets state. Charge enters
    through q/C, not an arbitrary replacement voltage. A changing state basis
    or capacitance requires a different charge-conserving network model.
    """

    def __init__(self, *, initial_volts: float, capacitance_farad: float,
                 state_id: str | None = None,
                 voltage_reference_id: str | None = None) -> None:
        _finite("initial_volts", initial_volts)
        _finite("capacitance_farad", capacitance_farad)
        if capacitance_farad <= 0:
            raise ValueError("capacitance_farad must be positive")
        if (state_id is None) != (voltage_reference_id is None):
            raise ValueError("state and voltage reference must be supplied together")
        if state_id is not None and any(
            not isinstance(value, str) or not value.strip()
            for value in (state_id, voltage_reference_id)
        ):
            raise ValueError("state and voltage reference must be nonempty names")
        self._volts = initial_volts
        self._capacitance_farad = capacitance_farad
        self._state_id = state_id
        self._voltage_reference_id = voltage_reference_id

    @property
    def volts(self) -> float:
        return self._volts

    def advance(
        self,
        *,
        duration_seconds: float,
        effective_input_volts: float,
        gain: float,
        tau_seconds: float,
        switch_charge_coulomb: float,
        noise_sample: R1HeldCurrentNoise | None = None,
    ) -> IntervalObservation:
        """Apply the supplied boundary charge, then propagate one interval.

        All arguments are explicit, including a zero charge when the caller
        specifies no charge injection. ``effective_input_volts`` already
        contains the caller's registered physical-neighbour contribution.
        ``gain`` and ``tau_seconds`` are constant on this interval and must be
        supplied from the same shared physical model across maps/scans.

        Split at every drive, row or parameter change and at aperture edges.
        Advance through discarded settling intervals too: discarding a sample
        does not discard analog history. Combine adjacent interval means by
        duration when an actual aperture crosses a boundary. This class does
        not choose an acquisition schedule or claim any physical ADC work.

        A noise sample must name this state/reference and supply its physical
        conjugate reduction. Hold the SAME sample across numerical, aperture
        and storage splits; this method never draws or rescales noise. The
        caller splits at a genuine sample/reduction change. Original signed
        noise-port work is returned separately from algebraic voltage-offset
        work; it does not restore unspecified eliminated voltage sources.
        """
        for name, value in (
            ("duration_seconds", duration_seconds),
            ("effective_input_volts", effective_input_volts),
            ("gain", gain),
            ("tau_seconds", tau_seconds),
            ("switch_charge_coulomb", switch_charge_coulomb),
        ):
            _finite(name, value)
        if duration_seconds <= 0 or tau_seconds <= 0:
            raise ValueError("duration_seconds and tau_seconds must be positive")
        if noise_sample is not None:
            if not isinstance(noise_sample, R1HeldCurrentNoise):
                raise ValueError("noise_sample must be an explicit held R1 sample")
            if (self._state_id is None
                    or noise_sample.reduction.state_id != self._state_id
                    or noise_sample.reduction.voltage_reference_id != self._voltage_reference_id):
                raise ValueError("noise sample state/reference does not match this pole")

        post_switch = self._volts + switch_charge_coulomb / self._capacitance_farad
        target = gain * effective_input_volts
        _finite("post-switch voltage", post_switch)
        _finite("target voltage", target)
        if noise_sample is not None:
            current = noise_sample.equivalent_current_amp
            # Preserve a projected-zero sample's original port work without
            # introducing tau/C * 0 at extreme legacy time constants.
            if current != 0.0:
                shift = tau_seconds * current / self._capacitance_farad
                _finite("noise target shift", shift)
                target += shift
                _finite("noise-shifted target voltage", target)
        difference = target - post_switch
        _finite("target minus post-switch voltage", difference)
        ratio = duration_seconds / tau_seconds
        one_minus_decay = -math.expm1(-ratio)
        # Avoid cancellation in 1-(1-exp(-r))/r for short apertures. The
        # omitted next term at this numerical branch boundary is below 2e-28.
        if ratio < 1e-4:
            mean_progress = ratio * (
                0.5 + ratio * (-1 / 6 + ratio * (1 / 24 + ratio * (-1 / 120 + ratio / 720)))
            )
        else:
            mean_progress = 1.0 - one_minus_decay / ratio
        if ratio <= 1:
            end = post_switch + difference * one_minus_decay
            mean = post_switch + difference * mean_progress
        else:
            end = target - difference * math.exp(-ratio)
            mean = target - difference * (one_minus_decay / ratio)
        _finite("end voltage", end)
        _finite("mean voltage", mean)
        noise_work = None if noise_sample is None else noise_sample.work(
            mean_state_volts=mean, duration_seconds=duration_seconds)
        # ALL projection, state and original-port work checks precede commit.
        observation = IntervalObservation(post_switch, end, mean, duration_seconds, noise_work)
        self._volts = end
        return observation
