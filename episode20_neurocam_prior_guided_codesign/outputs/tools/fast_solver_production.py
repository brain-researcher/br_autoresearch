"""Production adapters for the fast solver: inputs and the eight physical works.

Two input representations feed the same exact solver:
* ReferenceFedInputs: the reference's own midpoint field holds and the
  per-slot cubic noise polynomials (NoisePolynomial from the unmodified
  SpectralCubicCircuitNoise), projected with the reference layout.
* ExactInputs (O6): the analytic 11-Hz field quadratures and the nine frozen
  noise tones of the SAME saved draw and ports, expanded per piece as degree-10
  Taylor polynomials (remainder < 1e-19 relative on <= 2.5-us pieces).
  Projection to nodal currents is linear and done once per tone.

ProductionWorkModel reproduces the reference's physical work definitions
(cpe_array.advance_row and physical_noise_ports.PhysicalNoiseLayout.work) as
exact integrals of polynomial inputs against exact state moments.
"""
import math

import numpy as np

from fast_solver_engine import JMAX, MMAX, PieceInputs
from fast_solver_phi import product_integral, shift_monomials

_FACT = np.array([math.factorial(k) for k in range(JMAX + 1)], dtype=float)


class ProductionWorkModel:
    def __init__(self, array):
        layout = array.physical_noise_layout
        if layout is None:
            raise ValueError("bind the reference noise layout before building the work model")
        if layout.external_daq_current_ports:
            raise ValueError("external DAQ noise ports are not represented")
        self.layout = layout
        self.P = array.PADS
        n_up = array.BASE_STATES + array.resources.fractional_contact_states
        self.n_up = n_up
        nodal = layout._nodal_count
        positive, negative = np.asarray(layout._positive), np.asarray(layout._negative)
        self.nodal = nodal
        pad_mask = (positive < self.P) & (negative < 0)
        self.pad_ports = np.flatnonzero(pad_mask)
        self.pad_of_port = positive[pad_mask]
        src_mask = (positive >= self.P) & (positive < self.P + 64) & (negative < 0)
        self.source_ports = np.flatnonzero(src_mask)
        self.source_of_port = positive[src_mask] - self.P
        down_mask = positive >= n_up
        self.down_ports = np.flatnonzero(down_mask)
        self.down_pos = positive[down_mask] - n_up
        self.down_neg = np.where(negative[down_mask] >= 0, negative[down_mask] - n_up, -1)
        if len(self.pad_ports) + len(self.source_ports) + len(self.down_ports) != nodal:
            raise ValueError("unclassified nodal noise port")
        term = layout._term_count
        if term != 17 * self.P:
            raise ValueError("expected 17 star ports per pad")
        self.star_offset = nodal
        self.access_offset = nodal + term
        self.amp_offset = nodal + term + self.P
        self.inverse_d = np.asarray(layout._inverse_d)
        self.mix = np.asarray(layout._mix)
        self.r_div_d = np.asarray(layout._r_div_d)
        self.totals = np.asarray(layout._totals)
        self.fractions = np.asarray(layout._fractions).reshape(self.P, 17)
        group_nodes = np.asarray(layout._nodes).reshape(self.P, 17)
        if not np.array_equal(group_nodes[:, 0], np.arange(self.P)):
            raise ValueError("star port 0 must be the pad node")
        self.star_b = array._stars._b.reshape(self.P, 17)
        self.B = self.star_b.sum(axis=1)
        if np.any(array._direct_cap != 0):
            raise ValueError("direct-capacitance contacts not represented")

    def upstream_works(self, solver, h, inputs, unit_moments, source_mom):
        noise, field = inputs.noise_ports, inputs.field
        P = self.P
        width = noise.shape[1]
        eta = noise[self.star_offset:self.star_offset + 17 * P].reshape(P, 17, width)
        access = noise[self.access_offset:self.access_offset + P]
        total_eta = eta.sum(axis=1)
        U = np.zeros((P, width))
        U[:, :field.shape[1]] = field
        # star-node moments in original coordinates: node0 = pad, node i = pad + w_{i-1}
        X = unit_moments[:, :, :width]
        xs = X[:, :1, :] + X
        xs[:, 0, :] = X[:, 0, :]
        mean_v = np.einsum("pj,pjn->pn", self.fractions, xs)
        leak = np.zeros((P, width))
        leak[self.pad_of_port] = noise[self.pad_ports]
        ue = U + access
        current = (h * np.einsum("pn,pn->", leak, X[:, 0, :])
                   + h * np.einsum("pjn,pjn->", eta, xs)
                   - h * np.sum(self.inverse_d * product_integral(total_eta, ue))
                   - h * np.sum(self.mix * np.einsum("pn,pn->p", total_eta, mean_v))
                   + h * np.sum(self.r_div_d * product_integral(total_eta, total_eta)))
        src = noise[self.source_ports]
        current += h * np.einsum("qn,qn->", src, source_mom[self.source_of_port, :width])
        hd = self.totals * self.inverse_d
        access_work = h * np.sum(hd * (product_integral(access, ue) - np.einsum("pn,pn->p", access, mean_v))
                                 + self.inverse_d * product_integral(access, total_eta))
        correction = h * np.sum(hd * product_integral(U, access) + self.inverse_d * product_integral(U, total_eta))
        main = h * np.sum(self.B * product_integral(U, U) - np.einsum("pj,pn,pjn->p", self.star_b, U, xs))
        return dict(noise_current_source_interval_work_joule=float(current),
                    noise_access_voltage_source_interval_work_joule=float(access_work),
                    field_source_interval_work_joule=float(main + correction))

    def downstream_works(self, solver, sub, a, b, inputs, x_mom):
        noise = inputs.noise_ports[self.down_ports]
        if (a, b) != (0.0, 1.0):
            noise = shift_monomials(noise, a, b - a)
        width = noise.shape[1]
        drop = x_mom[self.down_pos, :width].copy()
        neg = self.down_neg >= 0
        drop[neg] -= x_mom[self.down_neg[neg], :width]
        return dict(noise_current_source_interval_work_joule=float(sub * np.einsum("qn,qn->", noise, drop)))


class ReferenceFedInputs:
    """The reference's held fields and per-slot cubic noise, per piece."""

    def __init__(self, array):
        self.array = array
        self.layout = array.physical_noise_layout
        self.stars = array._stars
        self._polynomial = None
        self._drive = None
        self._field = None

    def bind_polynomial(self, polynomial):
        layout = self.layout
        rows = polynomial.numeric_array
        self._rows = np.ascontiguousarray(rows.T)                      # (ports, 4)
        projected = np.stack([layout.project_current(layout.sample_from_values(row)) for row in rows], axis=1)
        amplifier = np.stack([np.asarray(layout.amplifier_input_volts(layout.sample_from_values(row)))
                              for row in rows], axis=1)
        self._projected, self._amplifier = projected, amplifier
        self._origin = polynomial.origin_seconds
        self._polynomial = polynomial

    def bind_field(self, field):
        field = np.asarray(field, dtype=float).ravel()
        self._field = field
        self._drive = self.stars.drive_current(port_volts=field)

    def piece(self, t_start, t_end):
        h = float(t_end) - float(t_start)
        offset = float(t_start) - self._origin
        current = shift_monomials(self._projected, offset, h)
        current[:, 0] += self._drive
        amplifier = shift_monomials(self._amplifier, offset, h)
        noise = shift_monomials(self._rows, offset, h)
        return PieceInputs(current=current, amplifier=amplifier, noise_ports=noise, field=self._field[:, None])


class ExactInputs:
    """O6: analytic field quadratures and frozen tones, Taylor-expanded per piece."""

    def __init__(self, array, noise_provider, raw_time, contacts, degree=JMAX):
        layout = array.physical_noise_layout
        spectrum = noise_provider.spectrum
        self.degree = degree
        self.layout = layout
        omega = np.asarray(spectrum._omega_array)
        k_index = np.asarray(spectrum._frequency_indices)
        p_index = np.asarray(spectrum._port_indices)
        count = len(spectrum.port_ids)
        tones = []
        for k in range(len(omega)):
            sel = k_index == k
            real = np.bincount(p_index[sel], weights=spectrum._amplitude_real[sel], minlength=count)
            imag = np.bincount(p_index[sel], weights=spectrum._amplitude_imag[sel], minlength=count)
            proj_r = layout.project_current(layout.sample_from_values(real))
            proj_i = layout.project_current(layout.sample_from_values(imag))
            amp_r = np.asarray(layout.amplifier_input_volts(layout.sample_from_values(real)))
            amp_i = np.asarray(layout.amplifier_input_volts(layout.sample_from_values(imag)))
            tones.append(dict(omega=float(omega[k]), port=real + 1j * imag, current=proj_r + 1j * proj_i,
                              amplifier=amp_r + 1j * amp_i))
        self.tones = tones
        bound = raw_time.bind_contact_factors(contacts)
        modes = np.asarray(bound._modes)
        if modes.shape != (4096, 2) or raw_time._condition[0] not in "BNW":
            raise ValueError("exact field adapter implements the two-quadrature carrier worlds only")
        self.field_omega = 2 * np.pi * raw_time._frequency
        self.field_complex = 1e-6 * raw_time._scale * (modes[:, 0] - 1j * modes[:, 1])
        self.stars = array._stars
        b = array._stars._b
        self._star_nodes = np.asarray(array._stars._nodes)
        self._star_b = np.asarray(b)
        self._star_group = np.asarray(array._stars._group)
        self.N = len(array.state_ids)

    def _taylor(self, amplitude, omega, t_start, h):
        n = np.arange(self.degree + 1)
        factor = (1j * omega * h) ** n / _FACT[:self.degree + 1] * np.exp(1j * omega * t_start)
        return np.real(amplitude[:, None] * factor[None, :])

    def piece(self, t_start, t_end):
        h = float(t_end) - float(t_start)
        t0 = float(t_start)
        width = self.degree + 1
        current = np.zeros((self.N, width))
        amplifier = np.zeros((64, width))
        noise = np.zeros((len(self.tones[0]["port"]), width))
        for tone in self.tones:
            current += self._taylor(tone["current"], tone["omega"], t0, h)
            amplifier += self._taylor(tone["amplifier"], tone["omega"], t0, h)
            noise += self._taylor(tone["port"], tone["omega"], t0, h)
        field = self._taylor(self.field_complex, self.field_omega, t0, h)
        # star drive b_i * U_pad(t) at every star node
        np.add.at(current, self._star_nodes, self._star_b[:, None] * field[self._star_group])
        return PieceInputs(current=current, amplifier=amplifier, noise_ports=noise, field=field)
