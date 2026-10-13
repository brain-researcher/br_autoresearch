"""Bind one frozen finite-band trajectory to explicit physical circuit ports.

This adapter supplies original port amplitudes, not projected nodal noise or
post-readout perturbations. Every hold uses the same existing midpoint key.
The bounds concern the stored analytic forcing trajectory, excluding spectral
quadrature error, floating-point evaluation error and circuit-output error.
"""
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType

from physical_noise_ports import PhysicalNoiseLayout
from polynomial_noise import NoisePolynomial, _real
from spectral_noise import SpectralNoise
from raw_world_time import _upper_mul, _upper_div, _upper_result


@dataclass(frozen=True, init=False)
class SpectralCircuitNoise:
    def __init__(self, *, layout, spectrum, port_absolute_tolerances, check):
        if (not isinstance(layout, PhysicalNoiseLayout)
                or not isinstance(spectrum, SpectralNoise) or not callable(check)):
            raise ValueError("explicit physical layout, spectral path and resource callback required")
        if layout.port_ids != spectrum.port_ids or layout.port_units != spectrum.port_units:
            raise ValueError("spectral ports and units must exactly match the physical layout")
        tolerances = []
        for value in port_absolute_tolerances:
            check()
            if (len(tolerances) >= len(layout.port_ids) or isinstance(value, bool)
                    or not isinstance(value, Real) or not math.isfinite(value) or value <= 0):
                raise ValueError("one positive finite absolute A/V tolerance per physical port is required")
            tolerances.append(float(value))
        if len(tolerances) != len(layout.port_ids):
            raise ValueError("physical-noise tolerance vector must cover every port")
        # The spectral bounds are global for this immutable realized path.
        # Cache the outward normalized maximum once, not once per physical
        # hold. No trajectory or Gaussian draw is evaluated here.
        _, derivative = spectrum.bounds(*spectrum.time_window_seconds)
        normalized_derivative = 0.
        for bound, tolerance in zip(derivative, tolerances):
            check()
            normalized_derivative = max(normalized_derivative, _upper_div(bound, tolerance))
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "spectrum", spectrum)
        object.__setattr__(self, "port_absolute_tolerances", tuple(tolerances))
        object.__setattr__(self, "_check", check)
        object.__setattr__(self, "_normalized_derivative_bound", normalized_derivative)

    @property
    def sample_numeric_bytes(self):
        return 8 * len(self.layout.port_ids)

    def __call__(self, midpoint):
        self._check()
        return self.layout.sample_from_values(self.spectrum.evaluate(midpoint))

    def hold_bound(self, start, end, midpoint):
        """Maximum normalized per-port midpoint error; no draw/evaluation occurs."""
        self._check()
        for value in (start, end, midpoint):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("finite real noise hold times required")
        if not start < midpoint < end:
            raise ValueError("noise midpoint must lie strictly inside its hold")
        # Preserve per-query support validation and the spectrum's callback.
        self.spectrum.bounds(start, end)
        radius = max(_upper_result(midpoint - start), _upper_result(end - midpoint))
        return _upper_mul(radius, self._normalized_derivative_bound)

    @property
    def metadata(self):
        return MappingProxyType(dict(
            model="explicit_finite_band_gaussian_spectral_atoms",
            physical_port_count=len(self.layout.port_ids),
            waveform_replayed_at_existing_midpoint_key=True,
            independent_redraw_at_numerical_or_storage_boundary=False,
            port_absolute_tolerance_numeric_bytes=8 * len(self.port_absolute_tolerances),
            normalized_derivative_cache_numeric_bytes=8,
            source_parameters_and_draws_require_retention_in_settings_reference=True,
            spectral_evaluation_accounting=self.spectrum.accounting(),
            analytic_hold_bound_excludes_spectral_and_float_and_output_errors=True,
            equilibrium_thermal_or_empirical_calibration_established=False,
            complete_device_noise_inventory_established=False))


@dataclass(frozen=True, init=False)
class SpectralCubicCircuitNoise(SpectralCircuitNoise):
    """One fixed cubic per original physical cell, with the same A/V errors.

    cell_for_key(key) explicitly returns (start,end,origin). It must describe
    disjoint, deterministic original physical cells, not newly recentered
    numerical/aperture/storage fragments. An inadmissible original cell refuses
    outright; subdividing its queries cannot repair its Taylor remainder.
    The executor binds support/origin/all four coefficient rows during replay.
    """
    def __init__(self, *, layout, spectrum, port_absolute_tolerances, cell_for_key, check):
        if not callable(cell_for_key):
            raise ValueError("an explicit original physical-cell mapping is required")
        super().__init__(layout=layout, spectrum=spectrum,
                         port_absolute_tolerances=port_absolute_tolerances, check=check)
        omega = max(spectrum.angular_frequencies_rad_per_second)
        fourth = self._normalized_derivative_bound
        for _ in range(3):
            fourth = _upper_mul(fourth, omega)
        object.__setattr__(self, "_normalized_fourth_derivative_bound", fourth)
        object.__setattr__(self, "_cell_for_key", cell_for_key)
        object.__setattr__(self, "_cached_bound_cell", None)
        object.__setattr__(self, "_cached_bound", None)
        object.__setattr__(self, "_cached_polynomial", None)

    @property
    def sample_numeric_bytes(self):
        return 32 * self.layout.numeric_value_count + 24

    def _cell(self, key):
        self._check()
        key = _real(key, "cubic noise query key")
        supplied = self._cell_for_key(key)
        if not isinstance(supplied, tuple) or len(supplied) != 3:
            raise ValueError("cell_for_key must return the exact (start,end,origin) tuple")
        start, end, origin = (_real(value, "original noise cell coordinate") for value in supplied)
        if not start < origin < end or not start < key < end or not math.isfinite(end - start):
            raise ValueError("original noise cell must strictly contain its key and fixed origin")
        self.spectrum.bounds(start, end)  # Domain and resource check, never a draw.
        cell = (start, end, origin)
        previous = self._cached_bound_cell
        if previous is not None and previous[0] < key < previous[1] and cell != previous:
            raise ValueError("original noise cell/origin changed within its declared support")
        polynomial = self._cached_polynomial
        if (polynomial is not None
                and polynomial.support_start_seconds < key < polynomial.support_end_seconds
                and cell != (polynomial.support_start_seconds, polynomial.support_end_seconds,
                             polynomial.origin_seconds)):
            raise ValueError("original noise polynomial cell changed during replay")
        return cell

    def _cell_bound(self, cell):
        if cell == self._cached_bound_cell:
            return self._cached_bound
        start, end, origin = cell
        radius = max(_upper_result(origin - start), _upper_result(end - origin))
        # M4/epsilon <= (M1/epsilon)*omega_max**3, evaluated by successive
        # outward basic operations. Four radius products avoid unbounded powers.
        bound = self._normalized_fourth_derivative_bound
        for _ in range(4):
            bound = _upper_mul(bound, radius)
        bound = _upper_div(bound, 24.)
        if bound > 1.:
            raise ValueError("original cubic noise cell exceeds its pointwise A/V tolerance; "
                             "subdivision or recentering is not permitted")
        object.__setattr__(self, "_cached_bound_cell", cell)
        object.__setattr__(self, "_cached_bound", bound)
        return bound

    def hold_bound(self, start, end, midpoint):
        start, end, midpoint = (_real(value, "noise interval coordinate") for value in (start, end, midpoint))
        if not start < midpoint < end:
            raise ValueError("noise midpoint must lie strictly inside its interval")
        cell = self._cell(midpoint)
        if not cell[0] <= start < end <= cell[1]:
            raise ValueError("a numerical noise interval cannot cross an original physical cell")
        return self._cell_bound(cell)

    def __call__(self, midpoint):
        cell = self._cell(midpoint)
        self._cell_bound(cell)
        polynomial = self._cached_polynomial
        if (polynomial is None or cell != (polynomial.support_start_seconds,
                                         polynomial.support_end_seconds, polynomial.origin_seconds)):
            # Release the prior cache before allocating the next original cell.
            object.__setattr__(self, "_cached_polynomial", None)
            coefficients = self.spectrum.cubic_coefficients(cell[2])
            self._check()
            polynomial = NoisePolynomial(self.layout, coefficients, cell[2], cell[0], cell[1])
            object.__setattr__(self, "_cached_polynomial", polynomial)
        self._check()
        return polynomial

    @property
    def metadata(self):
        result = dict(super().metadata)
        result.update(waveform_replayed_at_existing_midpoint_key=False,
            waveform_representation="degree3_Taylor_on_original_physical_cell",
            coefficients="ordinary_power_derivative_divided_by_factorial",
            original_cell_origin_unchanged_at_numerical_aperture_storage_splits=True,
            cache_maximum_live_polynomials=1, polynomial_numeric_bytes=self.sample_numeric_bytes,
            fourth_derivative_cache_numeric_bytes=8,
            analytic_remainder="outward_global_normalized_M1_times_omega_max_cubed_times_radius_fourth_over24",
            spectral_coefficient_and_Horner_float_errors_included=False,
            polynomial_sample_domain="original_cell_start_inclusive_end_exclusive")
        return MappingProxyType(result)
