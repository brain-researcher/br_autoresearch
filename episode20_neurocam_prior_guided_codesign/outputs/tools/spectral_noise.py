"""Finite-band Gaussian spectral-factor trajectories, with explicit port units.

For one-sided factors S_k=L_k L_k*, independent real a_k,b_k~N(0,I),
    x(t)=sum_k sqrt(w_k) Re[L_k(a_k+i*b_k) exp(i*omega_k*t)].
Then E[x(t+tau)x(t)^T]=sum_k w_k Re[S_k exp(i*omega_k*tau)].
L rows have port-unit/sqrt(Hz); w has Hz; there is NO extra sqrt(2).
This is a finite spectral-atom Gaussian expansion, not continuous white noise,
proof of ergodicity, a fitted PSD or a calibrated circuit-noise assignment.

Sparse factors and named frozen draws require no dense ports-by-ports matrix.
Paired designs must reuse the same frequency grid and named quadrature basis;
equal seeds on changed grids do not establish paired trajectories. Omitted COO
entries are zero in the caller's conditional model, not verified physical absence.
Bounds enclose the analytic trajectory defined by the stored rounded complex
amplitudes and angular frequencies. They exclude factor-contraction and later
trigonometric/evaluation arithmetic error, and are not circuit-output bounds.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class SpectralNoiseLimits:
    max_terms: int
    max_draws: int
    max_frequencies: int
    max_ports: int
    max_numeric_bytes: int
    max_evaluations: int
    max_term_evaluations: int

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("all spectral limits must be explicit positive integers")


@dataclass(frozen=True)
class LatentQuadrature:
    frequency_index: int
    latent_id: str
    a: float
    b: float

    def __post_init__(self):
        if type(self.frequency_index) is not int or self.frequency_index < 0:
            raise ValueError("frequency_index must be a nonnegative integer")
        if not isinstance(self.latent_id, str) or not self.latent_id:
            raise ValueError("latent_id must be a stable nonempty name")
        for name in ("a", "b"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("frozen real quadratures must be finite")
            object.__setattr__(self, name, float(value))


def _bounded(values, maximum, check):
    result = []
    for value in values:
        check()
        if len(result) >= maximum:
            raise ValueError("spectral input count exceeds declared bound")
        result.append(value)
    return tuple(result)


def draw_standard_normal_quadratures(*, frequency_latent_ids, seed, max_draws, check):
    """Draw a named common basis ONCE, then reuse the returned tuple across arms.

    Ordering is canonical by (frequency index, latent name). Adding/removing a
    basis changes this draw call: do not call it separately on design subsets.
    Retain/reuse the common returned draws, rather than assuming equal seeds on
    different bases imply shared trajectories. No RNG is used by evaluation.
    """
    if (type(seed) is not int or seed < 0 or type(max_draws) is not int or max_draws < 1
            or not callable(check)):
        raise ValueError("explicit seed, draw cap and callback required")
    pairs = _bounded(frequency_latent_ids, max_draws, check)
    canonical = []
    for pair in pairs:
        pair = _bounded(pair, 2, check)
        if (len(pair) != 2 or type(pair[0]) is not int or pair[0] < 0
                or not isinstance(pair[1], str) or not pair[1]):
            raise ValueError("draw basis needs (frequency index, stable latent name)")
        canonical.append(pair)
    if len(set(canonical)) != len(canonical) or not canonical:
        raise ValueError("draw basis must be nonempty and unique")
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    for frequency, latent in sorted(canonical):
        check()
        a, b = rng.standard_normal(2)
        result.append(LatentQuadrature(frequency, latent, float(a), float(b)))
    return tuple(result)


def _upper_add(a, b):
    value = a + b
    if not math.isfinite(value):
        raise ValueError("spectral analytic bound overflow")
    if a == 0 and b == 0:
        return 0.
    result = math.nextafter(value, math.inf)
    if not math.isfinite(result):
        raise ValueError("spectral analytic bound overflow")
    return result


def _upper_multiply(a, b):
    value = a * b
    if not math.isfinite(value):
        raise ValueError("spectral analytic bound overflow")
    if a == 0 or b == 0:
        return 0.
    result = math.nextafter(value, math.inf)
    if not math.isfinite(result):
        raise ValueError("spectral analytic bound overflow")
    return result


class SpectralNoise:
    """Owned replayable COO factors and immutable supplied latent quadratures.

    sparse_terms entries are (frequency_index, port_index, latent_index, L).
    Repeated COO triples refuse; different latent terms may sum at a port/tone.
    Every required named draw must exist; extra common-basis draws are retained
    and capped, permitting the SAME basis to serve different physical designs.
    Numeric-byte accounting covers retained numeric payload, including packed
    evaluation arrays, not Python object overhead or O(terms + ports + tones)
    evaluation temporaries; count caps also bound these objects. It is not
    peak RSS. Vector reduction may round differently from scalar math.fsum;
    the stored analytic amplitudes and their conservative bounds are unchanged.
    """
    def __init__(self, *, frequencies_hz, quadrature_widths_hz, port_ids, port_units,
                 latent_ids, sparse_terms, quadratures, time_window_seconds, limits, check):
        if not isinstance(limits, SpectralNoiseLimits) or not callable(check):
            raise ValueError("explicit spectral limits and callback required")
        self._limits, self._check = limits, check
        self._evaluations = self._term_evaluations = 0
        self._cubic_evaluations = 0
        frequencies = _bounded(frequencies_hz, min(limits.max_frequencies, limits.max_numeric_bytes // 24), check)
        widths = _bounded(quadrature_widths_hz, min(limits.max_frequencies, limits.max_numeric_bytes // 24), check)
        if (not frequencies or len(frequencies) != len(widths)
                or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
                       for v in frequencies + widths)):
            raise ValueError("explicit positive finite frequencies and Hz quadrature widths required")
        self._frequencies_hz = tuple(map(float, frequencies))
        self._quadrature_widths_hz = tuple(map(float, widths))
        if any(a >= b for a, b in zip(self.frequencies_hz, self.frequencies_hz[1:])):
            raise ValueError("positive frequencies must be unique and increasing")
        self._port_ids = _bounded(port_ids, min(limits.max_ports, limits.max_numeric_bytes // 16), check)
        self._port_units = _bounded(port_units, min(limits.max_ports, limits.max_numeric_bytes // 16), check)
        self._latent_ids = _bounded(latent_ids, limits.max_draws, check)
        for names in (self.port_ids, self.latent_ids):
            if (not names or any(not isinstance(name, str) or not name for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError("port/latent identifiers must be distinct nonempty names")
        # This set is bounded by max_draws. Membership must not scan a large
        # common physical-noise basis once for every supplied quadrature.
        latent_names = set(self.latent_ids)
        if len(self.port_units) != len(self.port_ids) or any(unit not in ("A", "V") for unit in self.port_units):
            raise ValueError("each physical port requires its explicit A or V unit")
        window = _bounded(time_window_seconds, 2, check)
        if (len(window) != 2 or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in window)
                or window[0] >= window[1]):
            raise ValueError("finite ordered trajectory time window required")
        self._time_window_seconds = tuple(map(float, window))
        self._omega = tuple(2 * math.pi * value for value in self.frequencies_hz)
        if any(not math.isfinite(w) or not math.isfinite(w * max(map(abs, window))) for w in self._omega):
            raise ValueError("angular frequencies and time phases must be representable")
        supplied_draws = _bounded(quadratures, min(limits.max_draws, limits.max_numeric_bytes // 32), check)
        draws = {}
        for draw in supplied_draws:
            check()
            if (not isinstance(draw, LatentQuadrature) or draw.frequency_index >= len(frequencies)
                    or draw.latent_id not in latent_names):
                raise ValueError("draw does not belong to the declared frequency/latent basis")
            key = (draw.frequency_index, draw.latent_id)
            if key in draws:
                raise ValueError("duplicate named quadrature")
            draws[key] = LatentQuadrature(draw.frequency_index, draw.latent_id, draw.a, draw.b)
        self._quadratures = tuple(draws[key] for key in sorted(draws))
        supplied_terms = _bounded(sparse_terms, min(limits.max_terms, limits.max_numeric_bytes // 40), check)
        terms, triples = [], set()
        for supplied in supplied_terms:
            check()
            values = _bounded(supplied, 4, check)
            if len(values) != 4:
                raise ValueError("COO terms require three indices and a complex factor")
            k, p, l, value = values
            if (any(type(v) is not int for v in (k, p, l)) or not 0 <= k < len(frequencies)
                    or not 0 <= p < len(self.port_ids) or not 0 <= l < len(self.latent_ids)):
                raise ValueError("COO indices outside the declared basis")
            if isinstance(value, bool) or not isinstance(value, (Real, complex, np.complexfloating)):
                raise ValueError("finite complex spectral factor required")
            value = complex(value)
            if not math.isfinite(value.real) or not math.isfinite(value.imag) or (k, p, l) in triples:
                raise ValueError("factor must be finite with unique COO triples")
            if (k, self.latent_ids[l]) not in draws:
                raise ValueError("required frozen latent quadrature is missing")
            triples.add((k, p, l))
            terms.append((k, p, l, value))
        if not terms:
            raise ValueError("at least one explicit factor term required")
        self._sparse_terms = tuple(sorted(terms, key=lambda v: (v[0], v[1], self.latent_ids[v[2]])))
        numeric_bytes = (40 * len(self.sparse_terms) + 32 * len(self.quadratures)
                         + 32 * len(frequencies) + 16 * len(self.port_ids) + 16)
        if numeric_bytes > limits.max_numeric_bytes:
            raise ValueError("retained spectral numeric payload exceeds declared bound")
        # No S=L L* is materialized. Contract only each sparse factor with its
        # supplied frozen Gaussian draw, retaining at most one amplitude/port/tone.
        grouped = {}
        for k, p, l, value in self.sparse_terms:
            check()
            draw = draws[k, self.latent_ids[l]]
            amplitude = math.sqrt(self.quadrature_widths_hz[k]) * value * complex(draw.a, draw.b)
            if not math.isfinite(amplitude.real) or not math.isfinite(amplitude.imag):
                raise ValueError("realized spectral amplitude overflow")
            if (k, p) not in grouped:
                # Retained tuple payload plus two int64 indices and two
                # float64 coefficient arrays. Refuse BEFORE their allocation.
                numeric_bytes += 64
                if numeric_bytes > limits.max_numeric_bytes:
                    raise ValueError("retained spectral numeric payload exceeds declared bound")
                grouped[k, p] = []
            grouped[k, p].append(amplitude)
        amplitude_records = []
        absolute, derivative = [0.] * len(self.port_ids), [0.] * len(self.port_ids)
        for (k, p), values in sorted(grouped.items()):
            check()
            amplitude = complex(math.fsum(v.real for v in values), math.fsum(v.imag for v in values))
            if not math.isfinite(amplitude.real) or not math.isfinite(amplitude.imag):
                raise ValueError("realized spectral sum overflow")
            amplitude_records.append((k, p, amplitude))
            # L1 of real/imag parts avoids un-enclosed sqrt/hypot rounding.
            envelope = _upper_add(abs(amplitude.real), abs(amplitude.imag))
            absolute[p] = _upper_add(absolute[p], envelope)
            derivative[p] = _upper_add(derivative[p], _upper_multiply(self._omega[k], envelope))
        self._amplitudes = tuple(amplitude_records)
        self._absolute_bounds, self._derivative_bounds = tuple(absolute), tuple(derivative)
        # Do not form port covariance or a ports-by-ports matrix. All arrays
        # below are owned and bounded by the admitted realized COO count.
        count = len(self._amplitudes)
        self._omega_array = np.array(self._omega, dtype=np.float64)
        check()
        self._frequency_indices = np.fromiter((v[0] for v in self._amplitudes), dtype=np.int64, count=count)
        check()
        self._port_indices = np.fromiter((v[1] for v in self._amplitudes), dtype=np.int64, count=count)
        check()
        self._amplitude_real = np.fromiter((v[2].real for v in self._amplitudes), dtype=np.float64, count=count)
        check()
        self._amplitude_imag = np.fromiter((v[2].imag for v in self._amplitudes), dtype=np.float64, count=count)
        for array in (self._omega_array, self._frequency_indices, self._port_indices,
                      self._amplitude_real, self._amplitude_imag):
            array.flags.writeable = False
        self._packed_evaluation_numeric_bytes = 8 * len(frequencies) + 32 * count
        self._numeric_storage_bytes = numeric_bytes
        check()

    frequencies_hz = property(lambda self: self._frequencies_hz)
    quadrature_widths_hz = property(lambda self: self._quadrature_widths_hz)
    port_ids = property(lambda self: self._port_ids)
    port_units = property(lambda self: self._port_units)
    latent_ids = property(lambda self: self._latent_ids)
    quadratures = property(lambda self: self._quadratures)
    sparse_terms = property(lambda self: self._sparse_terms)
    time_window_seconds = property(lambda self: self._time_window_seconds)
    numeric_storage_bytes = property(lambda self: self._numeric_storage_bytes)
    angular_frequencies_rad_per_second = property(lambda self: self._omega)

    def _time(self, value, *, endpoint=False):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("trajectory query time must be finite")
        value = float(value)
        start, end = self.time_window_seconds
        if not (start <= value <= end if endpoint else start <= value < end):
            raise ValueError("trajectory time is outside its explicit finite window")
        return value

    def evaluate(self, time_seconds):
        self._check()
        t = self._time(time_seconds)
        work = len(self._omega) + len(self._amplitudes)
        if (self._evaluations >= self._limits.max_evaluations
                or self._term_evaluations + work > self._limits.max_term_evaluations):
            raise ValueError("spectral evaluation/work limit exceeded")
        self._evaluations += 1
        self._term_evaluations += work
        # Work is still charged for every tone and coalesced term, including
        # a failed attempt. Resource callbacks bracket bounded vector kernels
        # instead of running one Python callback for every physical port/tone.
        try:
            with np.errstate(over="raise", invalid="raise"):
                phase = self._omega_array * t
                cosine, sine = np.cos(phase), np.sin(phase)
                self._check()
                contributions = (self._amplitude_real * cosine[self._frequency_indices]
                                 - self._amplitude_imag * sine[self._frequency_indices])
                self._check()
                values = np.bincount(self._port_indices, weights=contributions,
                                     minlength=len(self.port_ids))
        except FloatingPointError as exc:
            raise ValueError("spectral trajectory evaluation overflow") from exc
        self._check()
        if not np.all(np.isfinite(values)):
            raise ValueError("spectral trajectory evaluation overflow")
        return values

    def bounds(self, start_seconds, end_seconds):
        """Global analytic envelopes also enclose every requested subinterval.

        Returned units are each declared port unit and that unit per second.
        For a midpoint hold, L*(end-start)/2 bounds analytic input error only.
        Gaussian laws have no deterministic population amplitude bound; these
        finite bounds are conditional on this realized, unchanged draw.
        """
        self._check()
        start, end = self._time(start_seconds, endpoint=True), self._time(end_seconds, endpoint=True)
        if start > end:
            raise ValueError("bound interval endpoints must be ordered")
        return self._absolute_bounds, self._derivative_bounds

    def cubic_coefficients(self, origin_seconds):
        """Taylor rows x^(j)(origin)/j!, j=0..3, from the SAME frozen amplitudes.

        One call consumes one evaluation and frequencies+4*realized_terms work
        units, before numerical kernels. No draw, history or dense covariance
        is created. Returned 4*ports float64 payload and O(terms) temporaries
        are caller-owned workspace, not additional retained spectral storage.
        This computes coefficients only; the provider admits the cell/remainder.
        """
        self._check()
        origin = self._time(origin_seconds)
        work = len(self._omega) + 4 * len(self._amplitudes)
        if (self._evaluations >= self._limits.max_evaluations
                or self._term_evaluations + work > self._limits.max_term_evaluations):
            raise ValueError("spectral evaluation/work limit exceeded")
        self._evaluations += 1
        self._cubic_evaluations += 1
        self._term_evaluations += work
        coefficients = np.empty((4, len(self.port_ids)), dtype=np.float64)
        try:
            with np.errstate(over="raise", invalid="raise"):
                phase = self._omega_array * origin
                cosine, sine = np.cos(phase), np.sin(phase)
                omega = self._omega_array[self._frequency_indices]
                real = (self._amplitude_real * cosine[self._frequency_indices]
                        - self._amplitude_imag * sine[self._frequency_indices])
                imaginary = (self._amplitude_real * sine[self._frequency_indices]
                             + self._amplitude_imag * cosine[self._frequency_indices])
                self._check()
                # Fixed four rows; each contraction is bounded by the admitted
                # sparse count and uses the same port order as evaluate().
                for order in range(4):
                    self._check()
                    if order == 0:
                        contribution = real
                    elif order == 1:
                        contribution = -imaginary * omega
                    elif order == 2:
                        contribution = -(real * omega) * omega / 2.
                    else:
                        contribution = ((imaginary * omega) * omega) * omega / 6.
                    coefficients[order] = np.bincount(self._port_indices, weights=contribution,
                                                     minlength=len(self.port_ids))
        except FloatingPointError as error:
            raise ValueError("spectral cubic coefficients overflow") from error
        self._check()
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("spectral cubic coefficients must be finite")
        return coefficients

    def accounting(self):
        return {"evaluations": self._evaluations, "term_evaluations": self._term_evaluations,
                "cubic_coefficient_evaluations": self._cubic_evaluations,
                "numeric_storage_bytes": self.numeric_storage_bytes,
                "packed_evaluation_numeric_bytes": self._packed_evaluation_numeric_bytes,
                "evaluation_backend": "coalesced_sparse_vector_reduction",
                "factor_terms": len(self.sparse_terms), "realized_port_frequency_terms": len(self._amplitudes),
                "draws": len(self.quadratures), "ports": len(self.port_ids),
                "frequencies": len(self.frequencies_hz), "continuous_white_noise": False,
                "physical_injection_or_calibration_selected": False}
