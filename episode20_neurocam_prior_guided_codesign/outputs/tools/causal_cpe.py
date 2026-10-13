"""Passive finite-band CPE realization; all physical/numerical choices explicit.

For 0 < alpha < 1, positive log-midpoint quadrature discretizes
  s**alpha = sin(pi*alpha)/pi * integral_0^inf s/(s+p) p**(alpha-1) dp.
Each term g*s/(s+p) is a resistor 1/g in series with a capacitor g/p.
This supplies the interface only: access resistance and noise belong to the
calling circuit exactly once. No NeuroCam parameter or reference is fitted.
"""

from dataclasses import dataclass
import math

import numpy as np

from passive_contact import ContactParameters, REFERENCE_PAD_AREA_UM2


def _real(name, value, *, positive=False):
    if np.iscomplexobj(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _positive_vector(name, values):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must be a nonempty positive finite vector")
    return result


@dataclass(frozen=True)
class ContactStamp:
    """Append these internal nodes and branches without changing terminal states.

    Supply initial voltages for every added node explicitly. Keep the same
    complete state basis and the contact branches across TFT/row switches.
    These entries neither ground a terminal nor supply a voltage source.
    """
    internal_state_ids: tuple
    internal_capacitance_to_reference_farad: tuple
    mutual_capacitance_edges: tuple
    resistor_edges: tuple


@dataclass(frozen=True)
class CPERealization:
    alpha: float
    scaled_q_siemens_seconds_alpha: float
    charge_transfer_siemens: float
    poles_rad_per_second: tuple
    branch_conductance_siemens: tuple
    branch_capacitance_farad: tuple
    ideal_capacitance_farad: float
    forward_band_hz: tuple
    checked_frequencies_hz: tuple
    maximum_grid_relative_error: float
    uniform_truncation_relative_bound: float

    @property
    def internal_state_count(self):
        return len(self.poles_rad_per_second)

    def admittance(self, *, frequencies_hz):
        """Interface admittance at supplied positive frequencies, in siemens.

        Evaluation outside the declared forward band is allowed for inspection;
        it does not extend the declared checks or approximation validity.
        """
        frequencies = _positive_vector("frequencies_hz", frequencies_hz)
        s = 2j * math.pi * frequencies
        result = self.charge_transfer_siemens + s * self.ideal_capacitance_farad
        for pole, conductance in zip(self.poles_rad_per_second, self.branch_conductance_siemens):
            result = result + conductance * s / (s + pole)
        if not np.all(np.isfinite(result)):
            raise ValueError("admittance is not finite at the supplied frequencies")
        return result

    def step_current(self, *, times_seconds, step_volts):
        """Zero-history voltage-step current at explicitly positive times.

        Fractional ideal current has a singular t=0 limit; a finite RC sum does
        not. In the alpha=1 case the capacitor's impulse is excluded because
        times must be strictly positive. No infinite-time-tail claim is made.
        """
        times = _positive_vector("times_seconds", times_seconds)
        voltage = _real("step_volts", step_volts)
        current = np.full(times.shape, self.charge_transfer_siemens)
        for pole, conductance in zip(self.poles_rad_per_second, self.branch_conductance_siemens):
            current += conductance * np.exp(-pole * times)
        current *= voltage
        if not np.all(np.isfinite(current)):
            raise ValueError("step current is not finite")
        return current

    def stamp(self, *, terminal_a, terminal_b, first_internal_index, state_prefix):
        """Stamp between two existing dynamic nodes, append internal RC states.

        For every branch: resistor a--internal, capacitor internal--b.
        The caller owns its base topology, initial states, reference/return,
        access resistance, current drive, noise and all switch events.
        """
        for name, value in (("terminal_a", terminal_a), ("terminal_b", terminal_b),
                            ("first_internal_index", first_internal_index)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if terminal_a == terminal_b or max(terminal_a, terminal_b) >= first_internal_index:
            raise ValueError("terminals must be distinct existing nodes before appended internal states")
        if not isinstance(state_prefix, str) or not state_prefix:
            raise ValueError("state_prefix must be a nonempty string")
        capacitances, conductances = [], []
        if self.charge_transfer_siemens:
            conductances.append((terminal_a, terminal_b, self.charge_transfer_siemens))
        if self.ideal_capacitance_farad:
            capacitances.append((terminal_a, terminal_b, self.ideal_capacitance_farad))
        for offset, (conductance, capacitance) in enumerate(zip(
                self.branch_conductance_siemens, self.branch_capacitance_farad)):
            internal = first_internal_index + offset
            conductances.append((terminal_a, internal, conductance))
            capacitances.append((internal, terminal_b, capacitance))
        return ContactStamp(
            tuple(f"{state_prefix}_{index}" for index in range(self.internal_state_count)),
            (0.0,) * self.internal_state_count, tuple(capacitances), tuple(conductances))


def realize_contact_cpe(*, parameters: ContactParameters, pad_area_um2,
                        pole_low_rad_per_second, pole_high_rad_per_second, branch_count,
                        forward_band_hz, check_frequencies_hz,
                        grid_relative_tolerance, truncation_relative_tolerance):
    """Construct positive log-midpoint branches with two distinct checks.

    The analytic truncation bound is uniform across the positive forward band
    but excludes quadrature error. The finite-grid error includes truncation
    and discretization, normalized by |q**gamma Q (j*w)**alpha|, never by Gct.
    Passing both checks is not a whole-band total-error or temporal-tail proof.
    No pole/order auto-tuning occurs. For alpha=1 supply both pole bounds None
    and branch_count=0: an exact mutual capacitor needs no internal state.
    """
    if not isinstance(parameters, ContactParameters):
        raise ValueError("parameters must be explicit ContactParameters")
    area = _real("pad_area_um2", pad_area_um2, positive=True)
    band = _positive_vector("forward_band_hz", forward_band_hz)
    if band.shape != (2,) or band[0] >= band[1]:
        raise ValueError("forward_band_hz must contain increasing positive endpoints")
    grid = _positive_vector("check_frequencies_hz", check_frequencies_hz)
    if grid.size < 2 or np.any(np.diff(grid) <= 0) or grid[0] != band[0] or grid[-1] != band[1]:
        raise ValueError("frequency grid must be increasing and include both forward-band endpoints")
    grid_tolerance = _real("grid_relative_tolerance", grid_relative_tolerance, positive=True)
    tail_tolerance = _real("truncation_relative_tolerance", truncation_relative_tolerance, positive=True)
    if isinstance(branch_count, bool) or not isinstance(branch_count, int) or branch_count < 0:
        raise ValueError("branch_count must be a nonnegative integer")
    scale = (area / REFERENCE_PAD_AREA_UM2) ** parameters.gamma
    q = _real("area-scaled Q", scale * parameters.q_l_siemens_seconds_alpha, positive=True)
    gct = _real("area-scaled Gct", scale * parameters.g_ct_l_siemens)
    alpha = parameters.alpha
    if alpha == 1:
        if branch_count != 0 or pole_low_rad_per_second is not None or pole_high_rad_per_second is not None:
            raise ValueError("alpha=1 requires no pole bounds and zero branches")
        return CPERealization(alpha, q, gct, (), (), (), q, tuple(band), tuple(grid), 0.0, 0.0)
    if branch_count < 1:
        raise ValueError("fractional alpha requires at least one branch")
    low = _real("pole_low_rad_per_second", pole_low_rad_per_second, positive=True)
    high = _real("pole_high_rad_per_second", pole_high_rad_per_second, positive=True)
    if low >= high:
        raise ValueError("pole bounds must be strictly increasing")
    coefficient = math.sin(math.pi * alpha) / math.pi
    omega_low, omega_high = 2 * math.pi * band
    # |s/(s+p)| <= 1 below low; <= |s|/p above high.
    tail_bound = coefficient * ((low / omega_low) ** alpha / alpha
                               + (omega_high / high) ** (1 - alpha) / (1 - alpha))
    if not math.isfinite(tail_bound) or tail_bound > tail_tolerance:
        raise ValueError("pole truncation exceeds the declared uniform relative bound")
    width = (math.log(high) - math.log(low)) / branch_count
    log_poles = math.log(low) + (np.arange(branch_count) + 0.5) * width
    poles = np.exp(log_poles)
    conductances = q * coefficient * width * np.exp(alpha * log_poles)
    capacitances = conductances / poles
    for name, values in (("poles", poles), ("conductances", conductances), ("capacitances", capacitances)):
        _positive_vector(name, values)  # Reject overflow/underflow instead of repairing parameters.
    s = 2j * math.pi * grid
    approximation = np.zeros(grid.shape, dtype=complex)
    for pole, conductance in zip(poles, conductances):
        approximation += conductance * s / (s + pole)
    exact = q * s ** alpha
    error = float(np.max(np.abs(approximation - exact) / np.abs(exact)))
    if not math.isfinite(error) or error > grid_tolerance:
        raise ValueError("CPE approximation exceeds the declared finite-grid tolerance")
    return CPERealization(alpha, q, gct, tuple(poles), tuple(conductances), tuple(capacitances),
                          0.0, tuple(band), tuple(grid), error, float(tail_bound))
