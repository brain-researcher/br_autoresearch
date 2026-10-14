"""Prospective dry-air S5 family. No PBS branches or historical-device claims.

R1 is a fixed passive Galerkin projection, not a rowwise Schur elimination.
R2 retains every unforced metal pad, source and physical ADC hold. R3 adds
the selected unity-gain buffer/filter path. E_area/E_sublinear are duplicate
named alternatives in dry air, where their electrolyte branches are absent.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from conditional_acquisition_recipe import (ADC_IDS, HOLD_IDS, HOLD_CAPACITANCE_FARAD,
    TRACK_CONDUCTANCE_SIEMENS, OFF_CONDUCTANCE_SIEMENS, HOLD_LEAK_CONDUCTANCE_SIEMENS)
from fast_solver_continuous_kernel import BoundedWeightCache


FAMILY_ID = "conditional_S5_dry_air_v1"
STRUCTURES = ("R1", "R2", "R3")
CONTACT_NAMES = ("E_area", "E_sublinear")
FREQUENCY_HZ = 12.0
AMPLITUDE_VOLTS = 2.5e-3
OMEGA = 2 * math.pi * FREQUENCY_HZ
PROBE_SITE = (31, 31)
PAD_C = 2e-12
PAD_LEAK = 1e-12
PAD_NEIGHBOUR_C = 0.02e-12
SOURCE_C = 30e-12
TFT_ON, TFT_OFF = 2e-6, 2e-11
INPUT_C, INPUT_G = 5e-12, 1e-9
BUFFER_C, BUFFER_G = 100e-12, 20e-6
FILTER_C, OUTPUT_G, FILTER_G = 2.2e-9, 200e-6, 1e-9
CONTROLLED_GAIN, CONTROLLED_RAIL = 1.0, 0.050


@dataclass(frozen=True)
class Parameters:
    source_load_siemens: float = 1e-9
    adjacent_source_capacitance_farad: float = 0.2e-12

    def __post_init__(self):
        if not (math.isfinite(self.source_load_siemens)
                and 1e-9 <= self.source_load_siemens <= 20e-6):
            raise ValueError("source loading outside the selected conditional domain")
        if not (math.isfinite(self.adjacent_source_capacitance_farad)
                and 0 <= self.adjacent_source_capacitance_farad <= 3e-9):
            raise ValueError("source capacitance outside the selected conditional domain")


def specification():
    """All fixed assumptions/domain are declared before calibration outcomes."""
    return dict(family_id=FAMILY_ID, structures=list(STRUCTURES),
        contact_names=list(CONTACT_NAMES), electrolyte_contact_names_equivalent=True,
        geometry=dict(rows=64, columns=64, pitch_um=150, pad_um=[70, 135]),
        probe=dict(site=list(PROBE_SITE), volts_peak=AMPLITUDE_VOLTS, frequency_hz=FREQUENCY_HZ,
                   source_impedance_ohm=0, reference="conditional_R3_zero_reference"),
        source_load_domain_siemens=[1e-9, 20e-6],
        adjacent_source_capacitance_domain_farad=[0, 3e-9],
        domain_interpretation="selected conditional rectangle, not an evidence bound or posterior",
        pad_reference_capacitance_farad=PAD_C, pad_leak_siemens=PAD_LEAK,
        nearest_horizontal_vertical_pad_capacitance_farad=PAD_NEIGHBOUR_C,
        source_reference_capacitance_farad=SOURCE_C,
        tft_on_siemens=TFT_ON, tft_off_siemens=TFT_OFF,
        r3=dict(input_capacitance_farad=INPUT_C, input_load_siemens=INPUT_G,
                buffer_capacitance_farad=BUFFER_C, buffer_drive_siemens=BUFFER_G,
                filter_capacitance_farad=FILTER_C, output_path_siemens=OUTPUT_G,
                filter_leak_siemens=FILTER_G, gain=CONTROLLED_GAIN,
                controlled_target_rails_volts=[-CONTROLLED_RAIL, CONTROLLED_RAIL]),
        r1_basis="fixed all-off DC pad lift onto all64 source coordinates, plus unchanged24 holds",
        r1_charge="Cr delta(z)=T.T q, fixed T; never per-row voltage reset",
        initial_state="all unforced physical/reduced states zero; sine starts at zero",
        noise="zero stochastic noise; deterministic switch charges, G1 and quantization retained",
        removed=["PBS bath source", "PBS spreading/access resistance", "CPE", "charge transfer"],
        direct_edges="nearest horizontal/vertical metal pads and adjacent source lines only",
        omitted_direct_long_range_edges="explicit conditional graph assumption, not verified absence",
        distant_paths="all mediated capacitive/TFT/shared-source/mux/conditioning paths retained",
        calibration_parameters=list(asdict(Parameters())),
        parameter_sharing="one pair across every pad/row/scan within each structural fit")


def initial_grid():
    """Small prefit exploration, not exhaustive coverage of the rectangle."""
    return tuple(Parameters(g, c) for g in (1e-9, math.sqrt(1e-9 * 20e-6), 20e-6)
                 for c in (0., 0.2e-12, 30e-12, 3e-9))


def branch_matrix(shunts, edges):
    n = len(shunts)
    rows, cols, values = list(range(n)), list(range(n)), list(shunts)
    for a, b, value in edges:
        if value == 0:
            continue
        rows.extend((a, b, a, b)); cols.extend((a, b, b, a))
        values.extend((value, value, -value, -value))
    return sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()


class AirFamily:
    """Full dry-air circuit and one fixed coordinate map x=e_d*u+T*z.

    Small dimensions are available only to independent arithmetic tests;
    the production acquisition constructor requires the full 64x64 family.
    """
    def __init__(self, structure, parameters, *, check=lambda: None,
                 rows=64, cols=64, probe=None, adc_count=24, cache_bytes=256 * 1024**2):
        if structure not in STRUCTURES or not isinstance(parameters, Parameters):
            raise ValueError("an explicit structural alternative and parameter pair are required")
        if min(rows, cols, adc_count) < 1:
            raise ValueError("positive circuit dimensions required")
        self.structure, self.parameters, self.check = structure, parameters, check
        self.rows, self.cols, self.adc_count = rows, cols, adc_count
        self.pads = rows * cols
        self.probe_site = PROBE_SITE if probe is None else tuple(probe)
        r, c = self.probe_site
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError("probe outside array")
        self.driven = r * cols + c
        self.sources = np.arange(self.pads, self.pads + cols)
        self.buffers = np.arange(self.pads + cols, self.pads + 2 * cols) if structure == "R3" else np.zeros(0, int)
        self.filters = self.buffers + cols if structure == "R3" else np.zeros(0, int)
        self.hold_first = self.pads + cols + (2 * cols if structure == "R3" else 0)
        self.holds = np.arange(self.hold_first, self.hold_first + adc_count)
        self.full_size = self.hold_first + adc_count
        self.daq_sources = self.filters if structure == "R3" else self.sources
        shunt_c = np.zeros(self.full_size)
        shunt_c[:self.pads] = PAD_C; shunt_c[self.sources] = SOURCE_C
        shunt_c[self.holds] = HOLD_CAPACITANCE_FARAD
        if structure == "R3":
            shunt_c[self.sources] += INPUT_C
            shunt_c[self.buffers] = BUFFER_C; shunt_c[self.filters] = FILTER_C
        edges = []
        for p in range(self.pads):
            if p % cols + 1 < cols:
                edges.append((p, p + 1, PAD_NEIGHBOUR_C))
            if p // cols + 1 < rows:
                edges.append((p, p + cols, PAD_NEIGHBOUR_C))
        edges.extend((int(self.sources[s]), int(self.sources[s + 1]),
                      parameters.adjacent_source_capacitance_farad) for s in range(cols - 1))
        self.full_capacitance = branch_matrix(shunt_c, edges)
        self.reference_capacitance = shunt_c
        self.free = np.delete(np.arange(self.full_size), self.driven)
        if structure == "R1":
            tr, tc, tv = [], [], []
            attenuation = TFT_OFF / (TFT_OFF + PAD_LEAK)
            for p in range(self.pads):
                if p != self.driven:
                    tr.append(p); tc.append(p % cols); tv.append(attenuation)
            for s, node in enumerate(self.sources):
                tr.append(int(node)); tc.append(s); tv.append(1.)
            for k, node in enumerate(self.holds):
                tr.append(int(node)); tc.append(cols + k); tv.append(1.)
            self.projection = sp.csr_matrix((tv, (tr, tc)), shape=(self.full_size, cols + adc_count))
        else:
            self.projection = sp.csr_matrix((np.ones(len(self.free)), (self.free, np.arange(len(self.free)))),
                                            shape=(self.full_size, len(self.free)))
        self.size = self.projection.shape[1]
        self.capacitance = (self.projection.T @ self.full_capacitance @ self.projection).tocsc()
        self.cap_factor = sla.splu(self.capacitance)
        self.cap_probe = np.asarray((self.projection.T @ self.full_capacitance[:, self.driven]).toarray()).ravel()
        self.hold_observer = self.projection[self.holds].tocsr()
        self.source_observer = self.projection[self.sources].tocsr()
        self.cache = BoundedWeightCache(cache_bytes)
        self.matrix_cache = self.cache.scope("circuit")
        self.step_cache = self.cache.scope("dense_step")
        self.stats = dict(operator_products=0, dense_exponentials=0, sparse_exponentials=0,
                          largest_certified_source_bound_volts=0.)
        self._rail_particular = {}
        self.check()

    @property
    def state_ids(self):
        names = [f"pad_{p:04d}" for p in range(self.pads)]
        names += [f"source_{s:02d}" for s in range(self.cols)]
        if self.structure == "R3":
            names += [f"s5_buffer_{s:02d}" for s in range(self.cols)]
            names += [f"s5_filter_{s:02d}" for s in range(self.cols)]
        names += list(HOLD_IDS) if self.adc_count == 24 else [f"hold_{k}" for k in range(self.adc_count)]
        return tuple(names)

    def row_key(self, row):
        # Fixed uniform lift: all ordinary rows have the same projected G.
        return int(row == self.probe_site[0]) if self.structure == "R1" else row

    def full_conductance(self, row, slot):
        if not (0 <= row < self.rows and 0 <= slot < 4):
            raise ValueError("row or physical ADC slot outside fixture")
        shunts = np.zeros(self.full_size)
        shunts[:self.pads] = PAD_LEAK
        shunts[self.sources] = self.parameters.source_load_siemens
        shunts[self.holds] = HOLD_LEAK_CONDUCTANCE_SIEMENS
        if self.structure == "R3":
            shunts[self.sources] += INPUT_G
            shunts[self.buffers] = BUFFER_G
            shunts[self.filters] = FILTER_G
        edges = [(p, int(self.sources[p % self.cols]), TFT_ON if p // self.cols == row else TFT_OFF)
                 for p in range(self.pads)]
        if self.structure == "R3":
            edges.extend((int(b), int(f), OUTPUT_G) for b, f in zip(self.buffers, self.filters))
        marker = np.zeros(self.full_size)
        for adc, hold in enumerate(self.holds):
            selected = adc + self.adc_count * slot
            for source in range(adc, self.cols, self.adc_count):
                g = TRACK_CONDUCTANCE_SIEMENS if selected == source else OFF_CONDUCTANCE_SIEMENS
                edges.append((int(self.daq_sources[source]), int(hold), g))
            if adc == 0:
                g = TRACK_CONDUCTANCE_SIEMENS if slot == 3 else OFF_CONDUCTANCE_SIEMENS
                shunts[hold] += g
                marker[hold] = g
        result = branch_matrix(shunts, edges)
        if self.structure == "R3":
            result += sp.csr_matrix((-np.full(self.cols, BUFFER_G * CONTROLLED_GAIN),
                                     (self.buffers, self.sources)), shape=result.shape)
        return result.tocsr(), marker

    def matrices(self, row, slot):
        key = (self.row_key(row), slot)
        saved = self.matrix_cache.get(key)
        if saved is not None:
            return saved
        self.check()
        full_g, marker = self.full_conductance(row, slot)
        g = (self.projection.T @ full_g @ self.projection).tocsr()
        g_probe = np.asarray((self.projection.T @ full_g[:, self.driven]).toarray()).ravel()
        # f_s*sin(omega*t) + f_c*cos(omega*t), before multiplication by C^-1.
        forcing = np.column_stack((-AMPLITUDE_VOLTS * g_probe,
            -AMPLITUDE_VOLTS * OMEGA * self.cap_probe,
            np.asarray(self.projection.T @ marker).ravel()))
        saved = (g, self.cap_factor.solve(forcing))
        self.matrix_cache[key] = saved
        return saved

    def reconstruct(self, state, time):
        result = np.asarray(self.projection @ state).ravel()
        result[self.driven] = AMPLITUDE_VOLTS * math.sin(OMEGA * time)
        return result

    def physical_energy(self, state, time):
        # Full physical storage for x=T*z+e_d*u, including drive cross terms.
        # This also avoids reconstructing4096 pads at every R1 switch boundary.
        u = AMPLITUDE_VOLTS * math.sin(OMEGA * time)
        return float(0.5 * state @ (self.capacitance @ state)
                     + u * (state @ self.cap_probe)
                     + 0.5 * u * u * self.full_capacitance[self.driven, self.driven])

    def charge_delta(self, charge):
        charge = np.asarray(charge)
        if charge.shape != (self.full_size,):
            raise ValueError("charge must name every physical coordinate")
        return self.cap_factor.solve(np.asarray(self.projection.T @ charge).ravel())

    def dense_step(self, row, slot, marker_volts, duration):
        """Physical-state/phase/constant map plus exact hold integral map."""
        if self.structure != "R1" or duration <= 0:
            raise ValueError("positive R1 interval required")
        key = (self.row_key(row), slot, float(marker_volts).hex(), float(duration).hex())
        saved = self.step_cache.get(key)
        if saved is not None:
            return saved
        self.check()
        g, forcing = self.matrices(row, slot)
        n, h = self.size, self.adc_count
        generator = np.zeros((n + 3 + h, n + 3 + h))
        generator[:n, :n] = -self.cap_factor.solve(g.toarray())
        generator[:n, n:n + 2] = forcing[:, :2]
        generator[:n, n + 2] = forcing[:, 2] * marker_volts
        generator[n, n + 1], generator[n + 1, n] = OMEGA, -OMEGA
        generator[n + 3:, :n] = self.hold_observer.toarray()
        value = la.expm(duration * generator)
        saved = (value[:n + 3, :n + 3].copy(), value[n + 3:, :n + 3].copy())
        self.stats["dense_exponentials"] += 1
        self.step_cache[key] = saved
        return saved

    def certify_controlled_rails(self, state, row, time):
        """Bound the complete R3 interval, not just sampled rail values.

        Its passive upstream transient contracts in the C norm. A sinusoidal
        particular solution plus that transient bounds every source for all
        later times with this fixed row. C>=diag(reference C) supplies a
        conservative diagonal inverse bound. Refusal leaves state unchanged.
        """
        if self.structure != "R3":
            return
        count = self.pads + self.cols
        keep = np.delete(np.arange(count), self.driven)
        c = self.full_capacitance[keep][:, keep].tocsc()
        particular = self._rail_particular.get(row)
        if particular is None:
            full_g, _ = self.full_conductance(row, 0)
            g = full_g[keep][:, keep].tocsc()
            gp = full_g[keep, self.driven].toarray().ravel()
            cp = self.full_capacitance[keep, self.driven].toarray().ravel()
            amplitude = -1j * AMPLITUDE_VOLTS
            particular = sla.spsolve(g.astype(complex) + 1j * OMEGA * c,
                                     -(gp + 1j * OMEGA * cp) * amplitude)
            self._rail_particular[row] = particular
        # R2/R3 free coordinates preserve the full ordering except the drive.
        transient = state[:len(keep)] - np.real(particular * np.exp(1j * OMEGA * time))
        norm2 = float(transient @ (c @ transient))
        if norm2 < 0 or not math.isfinite(norm2):
            raise ArithmeticError("invalid passive-upstream transient energy")
        source_local = np.searchsorted(keep, self.sources)
        bounds = np.abs(particular[source_local]) + np.sqrt(norm2 / self.reference_capacitance[self.sources])
        largest = float(np.max(bounds * CONTROLLED_GAIN))
        self.stats["largest_certified_source_bound_volts"] = max(
            self.stats["largest_certified_source_bound_volts"], largest)
        if not largest < CONTROLLED_RAIL:
            raise ArithmeticError("linear R3 step cannot certify controlled target remains inside rails")

    def sparse_step(self, state, row, slot, marker_volts, start, end, *, max_products=100000):
        """Transient sine-augmented expm; reference baseline for R2/R3."""
        self.check()
        duration = float(end) - float(start)
        if duration <= 0:
            raise ValueError("positive physical interval required")
        self.certify_controlled_rails(state, row, start)
        g, forcing = self.matrices(row, slot)
        operator = _SineIntegralOperator(self, g, forcing, marker_volts, duration, max_products)
        n = self.size
        initial = np.zeros(operator.shape[0])
        initial[:n] = state
        initial[n:n + 3] = (math.sin(OMEGA * start), math.cos(OMEGA * start), 1.)
        # Approximate trace is a shift/preconditioner only, as in the existing
        # sparse kernel; it does not replace C^-1 or alter the propagated A.
        trace = -duration * float(np.sum(g.diagonal() / self.capacitance.diagonal()))
        value = sla.expm_multiply(operator, initial, traceA=trace)
        if not np.all(np.isfinite(value)):
            raise ArithmeticError("nonfinite dry-air propagated state")
        self.stats["sparse_exponentials"] += 1
        return value[:n], value[n + 3:]  # integral in V*s, not a point sample


class _SineIntegralOperator(sla.LinearOperator):
    def __init__(self, family, g, forcing, marker_volts, duration, maximum):
        self.family, self.g, self.duration = family, g, duration
        self.forcing = forcing.copy(); self.forcing[:, 2] *= marker_volts
        self.maximum, self.products = maximum, 0
        n = family.size + 3 + family.adc_count
        super().__init__(dtype=np.dtype(float), shape=(n, n))

    def _count(self, count):
        self.family.check()
        self.products += count
        self.family.stats["operator_products"] += count
        if self.products > self.maximum:
            raise RuntimeError("sine interval exceeded its explicit operator-product budget")

    def _matmat(self, x):
        self._count(x.shape[1]); n = self.family.size
        result = np.zeros_like(x)
        result[:n] = -self.family.cap_factor.solve(self.g @ x[:n]) + self.forcing @ x[n:n + 3]
        result[n], result[n + 1] = OMEGA * x[n + 1], -OMEGA * x[n]
        result[n + 3:] = self.family.hold_observer @ x[:n]
        return result * self.duration

    def _rmatmat(self, x):
        self._count(x.shape[1]); n = self.family.size
        result = np.zeros_like(x)
        result[:n] = -self.g.T @ self.family.cap_factor.solve(x[:n], trans="T")
        result[:n] += self.family.hold_observer.T @ x[n + 3:]
        result[n:n + 3] = self.forcing.T @ x[:n]
        result[n] -= OMEGA * x[n + 1]; result[n + 1] += OMEGA * x[n]
        return result * self.duration

    def _matvec(self, x):
        return self._matmat(np.asarray(x).reshape(-1, 1)).ravel()

    def _rmatvec(self, x):
        return self._rmatmat(np.asarray(x).reshape(-1, 1)).ravel()
