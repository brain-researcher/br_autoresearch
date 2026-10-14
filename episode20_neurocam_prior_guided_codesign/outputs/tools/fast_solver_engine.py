"""Exact multirate solver for the EP20 conditional R3 circuit (O4 + O7 core).

Same physical model as the reference (`NeurocamCPEArray` + `ConditionedRCNetwork`):
circuit matrices are read from the reference objects, never rebuilt from
nominal recipe values. Only the numerical representation changes.

Upstream (pads, sources, contact branches) is feed-forward to downstream
(buffers, filters, holds); this is verified on the actual sparse matrices.
Upstream uses relative contact coordinates w = branch - pad, in which C is
block diagonal and every block pencil (G, C) is symmetric-definite.

* Active block (O7): the gated row's pads with their contact states plus all
  sources, solved exactly in a C-orthonormal modal basis (one eigensolve per
  distinct row matrix pair).
* Units: every other pad with its contact states, solved exactly per mode.
  Identical units share one eigensolve and one response table.
* Coupling (TFT-off conductances and mutual capacitances between blocks) is
  exchanged by Gauss-Seidel waveform relaxation on each piece; coupling
  signals are degree-d polynomials interpolated at Chebyshev-Lobatto nodes from
  the exact block solutions. Sweeps stop when the largest nodal change is
  <= wr_tolerance (0.1 nV by default) and refuse beyond wr_max_sweeps.
* Downstream (O4): each ADC group is a small exact modal system driven by the
  clipped controlled voltage; clip crossings are located on the polynomial
  drive (event-location tolerance) and the piece is split there.

Every block is integrated exactly for polynomial inputs (fast_solver_phi).
Remaining numerical error: WR truncation, coupling-polynomial representation
(measured a posteriori at check nodes), eigensolve roundoff (modal residual
check) and floating-point arithmetic.
"""
from dataclasses import dataclass
import math

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from fast_solver_phi import (PieceWeights, chebyshev_lobatto, vandermonde_inverse, shift_monomials,
                             polynomial_values)

JMAX = 10          # highest input polynomial degree on a piece
MMAX = 10          # highest moment needed by work integrals
GAUSS_NODES = 16   # Gauss-Legendre nodes for quadratic downstream work


class FastSolverError(RuntimeError):
    """A declared numerical control (WR cap, modal residual, event) refused."""


def _gauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


# --------------------------------------------------------------------------
# Modal blocks
# --------------------------------------------------------------------------

class ModalBlock:
    """C-orthonormal modes of one symmetric-definite block: G phi = mu C phi."""

    def __init__(self, capacitance, conductance, label, residual_limit=1e-8):
        c = np.array(capacitance, dtype=float)
        g = np.array(conductance, dtype=float)
        if not (np.array_equal(c, c.T) and np.array_equal(g, g.T)):
            raise FastSolverError(f"{label}: block matrices must be exactly symmetric")
        mu, phi = scipy.linalg.eigh(g, c)
        if np.any(mu < 0):
            raise FastSolverError(f"{label}: negative computed rate {mu.min():.3e}; refusing to project")
        cphi = c @ phi
        # Backward-error style modal residual: |G phi - mu C phi| / (|G| |phi| + |mu| |C| |phi|).
        gnorm = np.abs(g).sum(axis=1).max()
        cnorm = np.abs(c).sum(axis=1).max()
        column = np.abs(phi).max(axis=0)
        residual = np.abs(g @ phi - cphi * mu[None, :]).max(axis=0) / ((gnorm + mu * cnorm) * column)
        orth = np.abs(phi.T @ cphi - np.eye(len(mu))).max()
        self.label = label
        self.size = len(mu)
        self.rate = mu
        self.lam = -mu
        self.phi = phi
        self.capacitance = c
        self.conductance = g
        self.modal_residual = float(residual.max())
        self.orthonormality_error = float(orth)
        if self.modal_residual > residual_limit or self.orthonormality_error > 1e-8:
            raise FastSolverError(f"{label}: modal residual {self.modal_residual:.3e} / orthonormality "
                                  f"{self.orthonormality_error:.3e} exceed the declared check")
        self._weights = {}

    def to_modal(self, x):
        """x (..., n) state coordinates -> modal y = phi^T C x."""
        return (np.asarray(x) @ self.capacitance) @ self.phi

    def from_modal(self, y):
        return np.asarray(y) @ self.phi.T

    def weights(self, key, h, nodes, nodes_key):
        full = (key, nodes_key)
        cached = self._weights.get(full)
        if cached is None:
            cached = PieceWeights(self.lam, h, JMAX, nodes, MMAX)
            if len(self._weights) > 4096:
                self._weights.clear()
            self._weights[full] = cached
        return cached


# --------------------------------------------------------------------------
# Circuit description extracted from the reference objects
# --------------------------------------------------------------------------

@dataclass
class ArraySpec:
    rows: int
    cols: int
    branches: int
    capacitance: object            # csr, original coordinates (N x N)
    shunts: np.ndarray             # base shunts (N,)
    static_edges: np.ndarray       # (E, 3) non-TFT, non-mux resistor edges
    tft_on: np.ndarray             # (P,)
    tft_off: np.ndarray            # (P,)
    star_h: np.ndarray             # (P, K+1) weights for nodes (pad, branch0..)
    star_r: np.ndarray             # (P,) series resistance
    buffers: np.ndarray            # (64,)
    filters: np.ndarray            # (64,)
    holds: np.ndarray              # (n_adc,)
    gain: np.ndarray
    offset: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    gb: np.ndarray
    frontend: object = None        # PassiveDAQFrontend or None
    daq_first_state: int = 0
    daq_source_offset: int = 0
    stars_object: object = None    # PassiveStarConductance (exact b, mix, diagonal)
    partition: str = "rows"        # "rows" or "single"
    tft_pad_source: object = None  # optional explicit (pad -> source) map; default pad % cols
    n_pads_override: object = None  # for non-grid fixtures (single partition)
    n_sources_override: object = None

    @property
    def pads(self):
        return self.rows * self.cols if self.n_pads_override is None else int(self.n_pads_override)

    @property
    def sources(self):
        return self.cols if self.n_sources_override is None else int(self.n_sources_override)

    @property
    def size(self):
        return self.capacitance.shape[0]

    @property
    def upstream_size(self):
        return self.pads + self.sources + self.pads * self.branches

    def contact_index(self, pad, branch):
        return self.pads + self.sources + self.branches * pad + branch


def spec_from_cpe_array(array):
    """Read every matrix/parameter from an unmodified NeurocamCPEArray instance."""
    drives = array._controlled_drives
    if drives is None:
        raise ValueError("the conditional R3 solver requires assembled analog conditioning")
    stars = array._stars
    pads = array.PADS
    k = len(stars._nodes) // pads - 1
    nodes = stars._nodes.reshape(pads, k + 1)
    if (not np.array_equal(nodes[:, 0], np.arange(pads))
            or not np.array_equal(nodes[:, 1:], array.BASE_STATES + np.arange(pads * k).reshape(pads, k))):
        raise ValueError("unexpected contact-star layout")
    if np.any(array._direct_cap != 0):
        raise ValueError("direct (alpha=1, R=0) contacts are not represented by this solver")
    h = stars._h.reshape(pads, k + 1)
    resistance = np.array([float(r) for _, _, r in array._contact_noise_stars])
    return ArraySpec(rows=array.ROWS, cols=array.SOURCES, branches=k,
                     capacitance=array._network._capacitance.tocsr(), shunts=np.array(array._shunts),
                     static_edges=np.array(array._resistor_edges).reshape(-1, 3), tft_on=np.array(array._on),
                     tft_off=np.array(array._off), star_h=np.array(h), star_r=resistance,
                     buffers=np.array(drives._buffer_index),
                     filters=np.arange(array._daq_source_offset, array._daq_source_offset + 64),
                     holds=np.arange(array._daq_first_state, array._size),
                     gain=np.array(drives._gain), offset=np.array(drives._offset),
                     lower=np.array(drives._lower), upper=np.array(drives._upper),
                     gb=np.array(drives._conductance), frontend=array._daq_frontend,
                     daq_first_state=array._daq_first_state, daq_source_offset=array._daq_source_offset,
                     stars_object=stars)


# --------------------------------------------------------------------------
# Relative-coordinate model
# --------------------------------------------------------------------------

class UpstreamModel:
    """Port-level matrices and pad-unit blocks in relative contact coordinates."""

    def __init__(self, spec):
        self.spec = spec
        P, S, K = spec.pads, spec.sources, spec.branches
        self.P, self.S, self.K = P, S, K
        n_up = spec.upstream_size
        c = spec.capacitance.tocsr()
        coo = c.tocoo()
        if np.any(((coo.row < n_up) != (coo.col < n_up)) & (coo.data != 0)):
            raise ValueError("capacitance couples upstream and downstream")
        cdiag = c.diagonal()
        if K:
            contact = spec.contact_index(np.arange(P)[:, None], np.arange(K)[None, :])
            branch_c = cdiag[contact]
            rows_contact = c[spec.contact_index(0, 0):n_up, :].tocoo()
            owner = rows_contact.row // K
            self_index = rows_contact.row + spec.contact_index(0, 0)
            on_pad = rows_contact.col == owner
            ok = on_pad | (rows_contact.col == self_index)
            if np.any(~ok & (rows_contact.data != 0)):
                raise ValueError("contact capacitance outside its own pad")
            pad_entries = np.zeros((P, K))
            pad_entries[owner[on_pad], rows_contact.row[on_pad] % K] = rows_contact.data[on_pad]
            if not np.array_equal(pad_entries, -branch_c):
                raise ValueError("contact capacitance is not an exact relative branch stamp")
            pad_c = np.array([math.fsum([float(cdiag[p]), *(-float(v) for v in branch_c[p])])
                              for p in range(P)])
        else:
            branch_c = np.zeros((P, 0))
            pad_c = cdiag[:P].copy()
        self.branch_c = branch_c
        ports = np.arange(P + S)
        cpp = c[ports, :][:, ports].tolil()
        for p in range(P):
            cpp[p, p] = pad_c[p]
        self.cpp = cpp.tocsr()
        if (abs(self.cpp - self.cpp.T) > 0).nnz != 0:
            raise ValueError("port capacitance must be exactly symmetric")
        h = np.array(spec.star_h, dtype=float)
        r = np.array(spec.star_r, dtype=float)
        total = h.sum(axis=1) if h.size else np.zeros(P)
        self.star_h, self.star_total = h, total
        if spec.stars_object is not None:
            stars = spec.stars_object
            b = stars._b.reshape(P, K + 1)
            mix = np.array(stars._mix)
            diag = np.array(stars._diagonal)
        elif h.size:
            b = h / (1.0 + r * total)[:, None]
            mix = r * total / (1.0 + r * total)
            diag = None
        else:
            b, mix, diag = np.zeros((P, 0)), np.zeros(P), None
        self.star_b, self.star_mix = b, mix
        self.star_fraction = (np.divide(h, total[:, None], out=np.zeros_like(h), where=total[:, None] > 0)
                              if h.size else np.zeros((P, 0)))
        self.star_r = r
        self.B = (np.array([math.fsum(map(float, row)) for row in b]) if h.size else np.zeros(P))
        if K:
            gbr = np.empty((P, K, K))
            for p in range(P):
                hb = h[p, 1:]
                # -(h_i h_j) * (mix/H): exactly symmetric in floating point; equals the reference's
                # -h_i*mix*(h_j/H) up to rounding (its operator form is symmetric mathematically).
                scale = mix[p] / total[p] if total[p] > 0 else 0.0
                block = -np.outer(hb, hb) * scale
                if diag is not None:
                    block[np.diag_indices(K)] = diag[spec.contact_index(p, np.arange(K))]
                else:
                    block[np.diag_indices(K)] = hb - r[p] * hb * hb / (1.0 + r[p] * total[p])
                gbr[p] = block
            self.branch_g = gbr
        else:
            self.branch_g = np.zeros((P, 0, 0))
        self.unit_size = K + 1
        self.pad_source = (np.arange(P) % S if spec.tft_pad_source is None
                           else np.asarray(spec.tft_pad_source, dtype=np.int64))

    def port_conductance(self, row):
        """G at pad/source ports for one gated row: shunts, TFT edges, star B on pads."""
        spec = self.spec
        P, S = self.P, self.S
        tft = np.array(spec.tft_off, dtype=float)
        if row is not None and spec.partition == "rows":
            tft[row * S:(row + 1) * S] = spec.tft_on[row * S:(row + 1) * S]
        elif row is not None:
            tft = np.array(spec.tft_on, dtype=float)
        pads = np.arange(P)
        sources = P + self.pad_source
        live = tft > 0
        rows_ = np.concatenate((pads[live], sources[live], pads[live], sources[live]))
        cols_ = np.concatenate((pads[live], sources[live], sources[live], pads[live]))
        vals = np.concatenate((tft[live], tft[live], -tft[live], -tft[live]))
        g = sp.coo_matrix((vals, (rows_, cols_)), shape=(P + S, P + S)).tocsr()
        g = g + sp.diags(np.asarray(spec.shunts[:P + S], dtype=float))
        g = g + sp.diags(np.concatenate((self.B, np.zeros(S))))
        g.sum_duplicates()
        return g.tocsr(), tft

    def unit_matrices(self, pad, g_pad):
        K = self.K
        cmat = np.zeros((K + 1, K + 1))
        cmat[0, 0] = self.cpp[pad, pad]
        if K:
            cmat[np.arange(1, K + 1), np.arange(1, K + 1)] = self.branch_c[pad]
        gmat = np.zeros((K + 1, K + 1))
        gmat[0, 0] = g_pad
        if K:
            gmat[0, 1:] = self.star_b[pad, 1:]
            gmat[1:, 0] = self.star_b[pad, 1:]
            gmat[1:, 1:] = self.branch_g[pad]
        return cmat, gmat

    def to_relative(self, x):
        P, S, K = self.P, self.S, self.K
        pads = np.empty((P, K + 1))
        pads[:, 0] = x[:P]
        if K:
            contacts = np.asarray(x[P + S:P + S + P * K]).reshape(P, K)
            pads[:, 1:] = contacts - pads[:, :1]
        return pads, np.array(x[P:P + S], dtype=float)

    def from_relative(self, pads, sources):
        P, S, K = self.P, self.S, self.K
        x = np.empty(P + S + P * K)
        x[:P] = pads[:, 0]
        x[P:P + S] = sources
        if K:
            x[P + S:] = (pads[:, 1:] + pads[:, :1]).ravel()
        return x

    def currents_to_relative(self, current):
        """Nodal currents (n_up, ...) -> relative forcing (P, K+1, ...), (S, ...)."""
        P, S, K = self.P, self.S, self.K
        trailing = current.shape[1:]
        pads = np.empty((P, K + 1) + trailing,
                        dtype=np.result_type(current.dtype, np.float64)
                        if np.iscomplexobj(current) else np.float64)
        if K:
            contacts = current[P + S:P + S + P * K].reshape((P, K) + trailing)
            pads[:, 1:] = contacts
            pads[:, 0] = current[:P] + contacts.sum(axis=1)
        else:
            pads[:, 0] = current[:P]
        return pads, current[P:P + S]


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

@dataclass
class PieceInputs:
    """Polynomial inputs on one piece in s=(t-t_a)/h, original coordinates.

    current excludes the external-DAQ drive (added from the selection) and the
    controlled buffer drive (computed from the clipped target)."""
    current: np.ndarray            # (N, deg+1) nodal currents (field, noise, constant drives)
    amplifier: np.ndarray          # (64, deg+1) amplifier input voltage (before gain)
    noise_ports: object = None     # optional (n_ports, deg+1) original noise-port coefficients
    field: object = None           # optional (P, deg+1) field polynomial (V)


@dataclass
class PieceResult:
    t_start: float
    t_end: float
    hold_mean: np.ndarray
    hold_end: np.ndarray
    source_mean: np.ndarray
    sweeps: int
    wr_change: float
    representation_error: dict
    works: dict
    clip_events: int
    pieces: int


class FastArraySolver:
    def __init__(self, spec, *, degree=8, wr_tolerance=1e-10, wr_max_sweeps=12, grade_levels=8,
                 event_tolerance=1e-15, residual_limit=1e-8, work_model=None, energy_audit=True):
        self.energy_audit = energy_audit
        self.spec = spec
        self.model = UpstreamModel(spec)
        self.degree = degree
        self.nodes = chebyshev_lobatto(degree)
        self.vinv = vandermonde_inverse(self.nodes)
        self.check_nodes = 0.5 * (self.nodes[1:] + self.nodes[:-1])
        self.check_vdm = self.check_nodes[:, None] ** np.arange(degree + 1)[None, :]
        self.gauss_nodes, self.gauss_weights = _gauss(GAUSS_NODES)
        self.wr_tolerance, self.wr_max_sweeps = wr_tolerance, wr_max_sweeps
        self.grade_levels = grade_levels
        self.event_tolerance = event_tolerance
        self.residual_limit = residual_limit
        self.work_model = work_model
        m = self.model
        P, S, K = m.P, m.S, m.K
        self.N = spec.size
        self.n_up = spec.upstream_size
        self._check_feed_forward()
        self._build_unit_types()
        self._a_blocks = {}
        self._down_cache = {}
        self._groups = self._downstream_groups()
        self.unit_y = np.zeros((P, K + 1))
        self.source_x = np.zeros(S)
        self.down_x = np.zeros(self.N - self.n_up)
        self.row = None
        self.a_y = None
        self.stats = dict(pieces=0, subpieces=0, sweeps=0, max_sweeps=0, max_wr_change=0.0,
                          max_rep_error_pad=0.0, max_rep_error_source=0.0, clip_events=0,
                          modal_residual=max(b.modal_residual for b in self.unit_blocks),
                          active_block_eigensolves=0, downstream_eigensolves=0)

    # ---------------- structure -------------------------------------------
    def _check_feed_forward(self):
        spec, n_up = self.spec, self.n_up
        for a, b, g in spec.static_edges:
            if (a < n_up) != (b < n_up) and g != 0:
                raise ValueError("static resistor edge crosses the feed-forward split")
        for nodes in (spec.buffers, spec.filters, spec.holds):
            if np.any(np.asarray(nodes) < n_up):
                raise ValueError("downstream nodes must follow the upstream block")
        if np.any(np.asarray(spec.shunts[n_up:]) < 0):
            raise ValueError("negative downstream shunt")

    def _build_unit_types(self):
        m = self.model
        P = m.P
        g_ports, _ = m.port_conductance(None)
        gdiag = g_ports.diagonal()
        keys, self.unit_type, self.unit_blocks = {}, np.empty(P, dtype=np.int64), []
        for pad in range(P):
            cmat, gmat = m.unit_matrices(pad, gdiag[pad])
            key = (cmat.tobytes(), gmat.tobytes())
            if key not in keys:
                keys[key] = len(self.unit_blocks)
                self.unit_blocks.append(ModalBlock(cmat, gmat, f"unit_type_{len(self.unit_blocks)}",
                                                   self.residual_limit))
            self.unit_type[pad] = keys[key]
        self.unit_members = [np.flatnonzero(self.unit_type == t) for t in range(len(self.unit_blocks))]

    def _a_block(self, row):
        if row in self._a_blocks:
            return self._a_blocks[row]
        m, spec = self.model, self.spec
        P, S, K = m.P, m.S, m.K
        u = K + 1
        g_ports, _ = m.port_conductance(row)
        pads = np.arange(P) if spec.partition == "single" else np.arange(row * S, (row + 1) * S)
        npads = len(pads)
        n = npads * u + S
        cmat = np.zeros((n, n))
        gmat = np.zeros((n, n))
        gdiag = g_ports.diagonal()
        port_pos = np.empty(npads + S, dtype=np.int64)
        for local, pad in enumerate(pads):
            cu, gu = m.unit_matrices(pad, gdiag[pad])
            sl = slice(local * u, (local + 1) * u)
            cmat[sl, sl] = cu
            gmat[sl, sl] = gu
            port_pos[local] = local * u
        port_pos[npads:] = npads * u + np.arange(S)
        port_global = np.concatenate((pads, P + np.arange(S)))
        for matrix, target in ((m.cpp, cmat), (g_ports, gmat)):
            sub = matrix[port_global, :][:, port_global].tocoo()
            for i, j, v in zip(sub.row, sub.col, sub.data):
                if i == j and i < npads:
                    continue  # pad diagonals already inside the unit blocks
                target[port_pos[i], port_pos[j]] = v
        key = (cmat.tobytes(), gmat.tobytes())
        block = None
        for other in self._a_blocks.values():
            if other["key"] == key:
                block = other["block"]
                break
        if block is None:
            block = ModalBlock(cmat, gmat, f"active_row_{row}", self.residual_limit)
            self.stats["active_block_eigensolves"] += 1
            self.stats["modal_residual"] = max(self.stats["modal_residual"], block.modal_residual)
        in_block = np.zeros(P + S, dtype=bool)
        in_block[port_global] = True
        cpp = m.cpp.tocoo()
        keep = (cpp.row != cpp.col) & ~(in_block[cpp.row] & in_block[cpp.col]) & (cpp.data != 0)
        c_off = sp.csr_matrix((cpp.data[keep], (cpp.row[keep], cpp.col[keep])), shape=(P + S, P + S))
        gpp = g_ports.tocoo()
        keep = (gpp.row != gpp.col) & ~(in_block[gpp.row] & in_block[gpp.col]) & (gpp.data != 0)
        g_off = sp.csr_matrix((gpp.data[keep], (gpp.row[keep], gpp.col[keep])), shape=(P + S, P + S))
        source_pos = npads * u + np.arange(S)
        g_coo = g_off.tocoo()
        entry = dict(key=key, block=block, pads=pads, port_pos=port_pos, port_global=port_global,
                     c_off=c_off, g_off=g_off, in_block=in_block, unit_size=u, source_pos=source_pos,
                     c_off_a=c_off[port_global].tocsr(), g_off_a=g_off[port_global].tocsr(),
                     c_off_u=c_off[:P].tocsr(), g_off_u=g_off[:P].tocsr(),
                     g_off_edges=(g_coo.row, g_coo.col, g_coo.data))
        self._a_blocks[row] = entry
        return entry

    # ---------------- downstream ------------------------------------------
    def _downstream_groups(self):
        spec = self.spec
        n_up = self.n_up
        nd = self.N - n_up
        adjacency = sp.lil_matrix((nd, nd))
        for a, b, g in spec.static_edges:
            a, b = int(a), int(b)
            if a >= n_up and b >= n_up:
                adjacency[a - n_up, b - n_up] = adjacency[b - n_up, a - n_up] = 1
        if spec.frontend is not None:
            for index, group in enumerate(spec.frontend.groups):
                hold = spec.daq_first_state + index
                for source in group.source_indices:
                    f = spec.daq_source_offset + source
                    adjacency[f - n_up, hold - n_up] = adjacency[hold - n_up, f - n_up] = 1
        if nd == 0:
            return []
        count, labels = csgraph.connected_components(adjacency.tocsr(), directed=False)
        return [np.flatnonzero(labels == g) for g in range(count)]

    def _down_matrices(self, selection):
        key = None if selection is None else tuple(sorted((k, _selection_key(v)) for k, v in selection.items()))
        cached = self._down_cache.get(key)
        if cached is not None:
            return cached
        spec = self.spec
        n_up, nd = self.n_up, self.N - self.n_up
        cmat = spec.capacitance[n_up:, n_up:].toarray()
        gmat = np.diag(np.asarray(spec.shunts[n_up:], dtype=float))
        external = []
        edges = [tuple(e) for e in spec.static_edges if e[0] >= n_up and e[1] >= n_up]
        if spec.frontend is not None:
            edges += list(spec.frontend.resistor_edges(selected_sources=selection,
                                                       source_node_offset=spec.daq_source_offset,
                                                       first_hold_state=spec.daq_first_state))
            for node, conductance, volts in spec.frontend.external_drives(
                    selected_sources=selection, first_hold_state=spec.daq_first_state,
                    voltage_reference_id=_reference_id(selection)):
                gmat[node - n_up, node - n_up] += conductance
                external.append((int(node), float(conductance), float(volts)))
        for a, b, g in edges:
            a, b = int(a) - n_up, int(b) - n_up
            gmat[a, a] += g
            gmat[b, b] += g
            gmat[a, b] -= g
            gmat[b, a] -= g
        blocks = []
        for gi, members in enumerate(self._groups):
            blocks.append(ModalBlock(cmat[np.ix_(members, members)], gmat[np.ix_(members, members)],
                                     f"down_group_{gi}", self.residual_limit))
            self.stats["downstream_eigensolves"] += 1
            self.stats["modal_residual"] = max(self.stats["modal_residual"], blocks[-1].modal_residual)
        entry = dict(blocks=blocks, external=tuple(external), cmat_diag=np.diag(cmat).copy(), gmat=gmat,
                     cmat=cmat, weights={})
        if len(self._down_cache) > 64:
            self._down_cache.clear()
        self._down_cache[key] = entry
        return entry

    # ---------------- state management -------------------------------------
    def set_state(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.N,):
            raise ValueError("state must cover the complete basis")
        pads, sources = self.model.to_relative(x[:self.n_up])
        for t, block in enumerate(self.unit_blocks):
            members = self.unit_members[t]
            self.unit_y[members] = block.to_modal(pads[members])
        self.source_x = sources.copy()
        self.down_x = np.array(x[self.n_up:], dtype=float)
        self.row = None
        self.a_y = None

    def _relative_pads(self):
        P = self.model.P
        pads = np.empty((P, self.model.unit_size))
        for t, block in enumerate(self.unit_blocks):
            members = self.unit_members[t]
            pads[members] = block.from_modal(self.unit_y[members])
        sources = self.source_x.copy()
        if self.row is not None:
            entry = self._a_blocks[self.row]
            xa = entry["block"].from_modal(self.a_y)
            u, npads = entry["unit_size"], len(entry["pads"])
            pads[entry["pads"]] = xa[:npads * u].reshape(npads, u)
            sources = xa[npads * u:]
        return pads, sources

    def state(self):
        pads, sources = self._relative_pads()
        return np.concatenate((self.model.from_relative(pads, sources), self.down_x))

    def _activate(self, row):
        if row == self.row:
            return self._a_blocks[row]
        pads, sources = self._relative_pads()
        if self.row is not None:
            old = self._a_blocks[self.row]["pads"]
            for t, block in enumerate(self.unit_blocks):
                members = np.intersect1d(self.unit_members[t], old)
                if len(members):
                    self.unit_y[members] = block.to_modal(pads[members])
        entry = self._a_block(row)
        xa = np.concatenate((pads[entry["pads"]].ravel(), sources))
        self.a_y = entry["block"].to_modal(xa)
        self.source_x = sources
        self.row = row
        return entry

    def _node_values(self, entry, nodes):
        """Original-coordinate values at selected nodes (sources/downstream fast path)."""
        nodes = np.asarray(nodes, dtype=np.int64)
        values = np.empty(len(nodes))
        P, S = self.model.P, self.model.S
        full = None
        for i, node in enumerate(nodes):
            if P <= node < P + S:
                values[i] = entry["block"].phi[entry["source_pos"][node - P]] @ self.a_y
            elif node >= self.n_up:
                values[i] = self.down_x[node - self.n_up]
            else:
                if full is None:
                    full = self.state()
                values[i] = full[node]
        return values

    # ---------------- piece advance ----------------------------------------
    def advance(self, *, gate_row, t_start, t_end, inputs, charge=None, selection=None, grade=False):
        h = float(t_end) - float(t_start)
        if not h > 0:
            raise ValueError("positive piece length required")
        entry = self._activate(gate_row)
        works = {}
        if charge is not None and np.any(charge):
            charged = np.flatnonzero(charge)
            before = self._node_values(entry, charged)
            self._apply_charge(entry, charge)
            after = self._node_values(entry, charged)
            works["supplied_charge_boundary_work_joule"] = float(0.5 * ((before + after) @ charge[charged]))
        key = round(h * 1e15)
        if abs(key * 1e-15 - h) > 1e-20:
            key = ("exact", h)
        levels = self.grade_levels if grade else 0
        fractions = [0.0] + [2.0 ** (l - levels) for l in range(levels + 1)] if levels else [0.0, 1.0]
        down = self._down_matrices(selection) if len(self.spec.buffers) else None
        hold_means = np.zeros(len(self.spec.holds))
        source_means = np.zeros(self.model.S)
        sweeps_total, change_max = 0, 0.0
        rep = dict(pad=0.0, source=0.0)
        events = 0
        for a, b in zip(fractions[:-1], fractions[1:]):
            sub_h = h * (b - a)
            sub_inputs = _restrict_inputs(inputs, a, b)
            result = self._advance_sub(entry, down, sub_h, (key, a, b), sub_inputs)
            hold_means += (b - a) * result["hold_mean"]
            source_means += (b - a) * result["source_mean"]
            for name, value in result["works"].items():
                works[name] = works.get(name, 0.0) + value
            sweeps_total += result["sweeps"]
            change_max = max(change_max, result["wr_change"])
            rep["pad"] = max(rep["pad"], result["rep_pad"])
            rep["source"] = max(rep["source"], result["rep_source"])
            events += result["events"]
        count = len(fractions) - 1
        stats = self.stats
        stats["pieces"] += 1
        stats["subpieces"] += count
        stats["sweeps"] += sweeps_total
        stats["max_sweeps"] = max(stats["max_sweeps"], result["sweeps"])
        stats["max_wr_change"] = max(stats["max_wr_change"], change_max)
        stats["max_rep_error_pad"] = max(stats["max_rep_error_pad"], rep["pad"])
        stats["max_rep_error_source"] = max(stats["max_rep_error_source"], rep["source"])
        stats["clip_events"] += events
        hold_end = self.down_x[np.asarray(self.spec.holds) - self.n_up].copy()
        return PieceResult(float(t_start), float(t_end), hold_means, hold_end, source_means, sweeps_total,
                           change_max, rep, works, events, count)

    def _apply_charge(self, entry, charge):
        m = self.model
        P = m.P
        up = np.asarray(charge[:self.n_up], dtype=float)
        if np.any(up):
            pads_q, sources_q = m.currents_to_relative(up[:, None])
            pads_q, sources_q = pads_q[..., 0], sources_q[..., 0]
            active = entry["pads"]
            qa = np.concatenate((pads_q[active].ravel(), sources_q))
            self.a_y = self.a_y + qa @ entry["block"].phi
            inactive = np.ones(P, dtype=bool)
            inactive[active] = False
            if np.any(pads_q[inactive]):
                for t, block in enumerate(self.unit_blocks):
                    members = self.unit_members[t][inactive[self.unit_members[t]]]
                    self.unit_y[members] += pads_q[members] @ block.phi
        down = np.asarray(charge[self.n_up:], dtype=float)
        if np.any(down):
            cdown = self.spec.capacitance[self.n_up:, self.n_up:].toarray()
            self.down_x = self.down_x + np.linalg.solve(cdown, down)

    def _advance_sub(self, entry, down, h, key, inputs):
        m = self.model
        P, S, K = m.P, m.S, m.K
        u = K + 1
        d = self.degree
        nodes = self.nodes
        block = entry["block"]
        active = entry["pads"]
        n_active = len(active)
        inactive_mask = np.ones(P, dtype=bool)
        inactive_mask[active] = False
        current = np.asarray(inputs.current, dtype=float)
        if current.shape[1] > JMAX + 1:
            raise ValueError("input polynomial degree exceeds JMAX")
        cur = np.zeros((current.shape[0], JMAX + 1))
        cur[:, :current.shape[1]] = current
        pads_f, sources_f = m.currents_to_relative(cur[:self.n_up])
        # ---- active block own response
        wa = block.weights(key, h, nodes, "lobatto")
        fa_phys = np.concatenate((pads_f[active].reshape(n_active * u, JMAX + 1), sources_f))
        fa = block.phi.T @ fa_phys
        ya0 = self.a_y
        phi_ports = block.phi[entry["port_pos"]]
        ya_nodes_own = wa.node_exp * ya0[:, None] + np.einsum("kj,kji->ki", fa, wa.node_force)
        # ---- units own response
        unit_data = []
        unit_port_own = np.zeros((P, d + 1))
        for t, ublock in enumerate(self.unit_blocks):
            members = self.unit_members[t][inactive_mask[self.unit_members[t]]]
            if not len(members):
                continue
            wu = ublock.weights(key, h, nodes, "lobatto")
            fu = np.einsum("qk,mqj->mkj", ublock.phi, pads_f[members])
            y0 = self.unit_y[members]
            phi0 = ublock.phi[0]
            own_nodes = (y0 * phi0[None, :]) @ wu.node_exp + np.einsum("mkj,kji->mi", fu * phi0[None, :, None],
                                                                     wu.node_force)
            response = np.einsum("k,kji->ji", phi0 * phi0, wu.node_force)
            unit_port_own[members] = own_nodes
            unit_data.append((t, members, wu, fu, y0, phi0, response))
        # ---- Gauss-Seidel waveform relaxation on port waveforms
        port_values = np.zeros((P + S, d + 1))
        port_values[:P] = unit_port_own
        a_ports = entry["port_global"]
        c_off, g_off = entry["c_off"], entry["g_off"]
        powers = np.arange(1, d + 1)[None, :] / h
        a_coupling = np.zeros((len(a_ports), JMAX + 1))
        unit_coupling = np.zeros((P, JMAX + 1))
        change, sweeps = math.inf, 0
        while change > self.wr_tolerance:
            if sweeps >= self.wr_max_sweeps:
                raise FastSolverError(f"waveform relaxation exceeded {self.wr_max_sweeps} sweeps "
                                      f"(last change {change:.3e} V)")
            sweeps += 1
            previous = port_values.copy()
            coeff = port_values @ self.vinv.T
            deriv = np.zeros_like(coeff)
            deriv[:, :-1] = coeff[:, 1:] * powers
            a_coupling[:, :d + 1] = -(entry["c_off_a"] @ deriv + entry["g_off_a"] @ coeff)
            fac = phi_ports.T @ a_coupling
            ya_nodes = ya_nodes_own + np.einsum("kj,kji->ki", fac, wa.node_force)
            port_values[a_ports] = phi_ports @ ya_nodes
            coeff = port_values @ self.vinv.T
            deriv[:, :-1] = coeff[:, 1:] * powers
            unit_coupling[:, :d + 1] = -(entry["c_off_u"] @ deriv + entry["g_off_u"] @ coeff)
            for t, members, wu, fu, y0, phi0, response in unit_data:
                port_values[members] = unit_port_own[members] + unit_coupling[members] @ response
            change = float(np.abs(port_values - previous).max())
        # ---- final states, moments and representation check
        fa_total = fa + phi_ports.T @ a_coupling
        ya_end = wa.end_exp * ya0 + np.einsum("kj,kj->k", fa_total, wa.end_force)
        ya_mom = wa.mom_exp * ya0[:, None] + np.einsum("kj,kmj->km", fa_total, wa.mom_force)
        wa_check = block.weights(key, h, self.check_nodes, "check")
        exact_check = phi_ports @ (wa_check.node_exp * ya0[:, None]
                                   + np.einsum("kj,kji->ki", fa_total, wa_check.node_force))
        poly_check = (port_values[a_ports] @ self.vinv.T) @ self.check_vdm.T
        rep = np.abs(exact_check - poly_check)
        rep_source = float(rep[n_active:].max()) if S else 0.0
        rep_pad = float(rep[:n_active].max()) if n_active else 0.0
        unit_moments = np.zeros((P, u, MMAX + 1))
        audit_input = audit_heat = 0.0
        if self.energy_audit:
            # Norton identity terms: int x^T f_external and int x^T G x (G incl. off-block TFT).
            wag = block.weights(key, h, self.gauss_nodes, "gauss")
            ya_g = wag.node_exp * ya0[:, None] + np.einsum("kj,kji->ki", fa_total, wag.node_force)
            audit_heat += h * float(np.sum(self.gauss_weights[None, :] * block.rate[:, None] * ya_g ** 2))
            audit_input += h * float(np.einsum("kj,kj->", fa, ya_mom[:, :JMAX + 1]))
            port_gauss = np.zeros((P + S, GAUSS_NODES))
            port_gauss[a_ports] = phi_ports @ ya_g
        for t, members, wu, fu, y0, phi0, response in unit_data:
            ftot = fu + phi0[None, :, None] * unit_coupling[members][:, None, :]
            y_end = wu.end_exp[None, :] * y0 + np.einsum("mkj,kj->mk", ftot, wu.end_force)
            y_mom = wu.mom_exp[None, :, :] * y0[:, :, None] + np.einsum("mkj,kij->mki", ftot, wu.mom_force)
            if self.energy_audit:
                wug = self.unit_blocks[t].weights(key, h, self.gauss_nodes, "gauss")
                y_g = wug.node_exp[None, :, :] * y0[:, :, None] + np.einsum("mkj,kji->mki", ftot, wug.node_force)
                audit_heat += h * float(np.einsum("g,k,mkg->", self.gauss_weights, self.unit_blocks[t].rate,
                                                  y_g ** 2))
                audit_input += h * float(np.einsum("mkj,mkj->", fu, y_mom[:, :, :JMAX + 1]))
                port_gauss[members] = np.einsum("k,mkg->mg", phi0, y_g)
            self.unit_y[members] = y_end
            unit_moments[members] = np.einsum("qk,mki->mqi", self.unit_blocks[t].phi, y_mom)
            wcheck = self.unit_blocks[t].weights(key, h, self.check_nodes, "check")
            exact_u = (y0 * phi0[None, :]) @ wcheck.node_exp + np.einsum("mkj,kji->mi", ftot * phi0[None, :, None],
                                                                       wcheck.node_force)
            poly_u = (port_values[members] @ self.vinv.T) @ self.check_vdm.T
            rep_pad = max(rep_pad, float(np.abs(exact_u - poly_u).max()))
        if self.energy_audit:
            rows_e, cols_e, vals_e = entry["g_off_edges"]
            if len(vals_e):
                audit_heat += h * float(np.sum(self.gauss_weights[None, :] * vals_e[:, None]
                                               * port_gauss[rows_e] * port_gauss[cols_e]))
        self.a_y = ya_end
        xa_mom = block.phi @ ya_mom
        unit_moments[active] = xa_mom[:n_active * u].reshape(n_active, u, MMAX + 1)
        source_mom = xa_mom[n_active * u:]
        # ---- downstream (O4)
        source_poly = port_values[P:P + S] @ self.vinv.T
        down_result = (self._downstream(down, h, key, inputs, cur, source_poly) if down is not None
                       else dict(hold_mean=np.zeros(0), works={}, events=0))
        works = dict(down_result["works"])
        if self.energy_audit:
            works["audit_input_joule"] = audit_input + works.pop("audit_down_input_joule", 0.0)
            works["audit_heat_joule"] = audit_heat + works.pop("audit_down_heat_joule", 0.0)
        if self.work_model is not None:
            for name, value in self.work_model.upstream_works(self, h, inputs, unit_moments, source_mom).items():
                works[name] = works.get(name, 0.0) + value
        return dict(hold_mean=down_result["hold_mean"], source_mean=source_mom[:, 0], works=works,
                    sweeps=sweeps, wr_change=change, rep_pad=rep_pad, rep_source=rep_source,
                    events=down_result["events"])

    def _downstream(self, down, h, key, inputs, cur, source_poly):
        spec = self.spec
        n_up = self.n_up
        nd = self.N - n_up
        d = self.degree
        width = JMAX + 1
        requested = np.zeros((64, width))
        requested[:, :d + 1] = spec.gain[:, None] * source_poly
        amp = np.asarray(inputs.amplifier, dtype=float)
        requested[:, :amp.shape[1]] += spec.gain[:, None] * amp
        requested[:, 0] += spec.offset
        cuts = {0.0, 1.0}
        for channel in range(64):
            for rail in (spec.lower[channel], spec.upper[channel]):
                cuts.update(_poly_roots_in_unit(requested[channel], rail, self.event_tolerance))
        cuts = sorted(cuts)
        events = len(cuts) - 2
        hold_local = np.asarray(spec.holds) - n_up
        buffer_local = np.asarray(spec.buffers) - n_up
        hold_mean = np.zeros(len(hold_local))
        works = dict(controlled_voltage_source_interval_work_joule=0.0,
                     conditioning_drive_branch_dissipation_joule=0.0,
                     daq_external_source_interval_work_joule=0.0)
        noise_down = cur[n_up:]
        for a, b in zip(cuts[:-1], cuts[1:]):
            frac = b - a
            sub = h * frac
            if not sub > 0:
                continue
            middle = 0.5 * (a + b)
            req_mid = polynomial_values(requested, np.array([middle]))[:, 0]
            clipped_hi = req_mid >= spec.upper
            clipped_lo = req_mid <= spec.lower
            target = shift_monomials(requested, a, frac) if (a, b) != (0.0, 1.0) else requested.copy()
            target[clipped_hi] = 0.0
            target[clipped_hi, 0] = spec.upper[clipped_hi]
            target[clipped_lo] = 0.0
            target[clipped_lo, 0] = spec.lower[clipped_lo]
            forcing = shift_monomials(noise_down, a, frac) if (a, b) != (0.0, 1.0) else noise_down.copy()
            forcing[buffer_local] += spec.gb[:, None] * target
            for node, conductance, volts in down["external"]:
                forcing[node - n_up, 0] += conductance * volts
            full_piece = (a, b) == (0.0, 1.0)
            x0 = self.down_x
            x_end = np.empty(nd)
            x_mom = np.empty((nd, MMAX + 1))
            x_gauss = np.empty((nd, GAUSS_NODES))
            for gi, members in enumerate(self._groups):
                blk = down["blocks"][gi]
                if full_piece:
                    w = down["weights"].get((gi, key))
                    if w is None:
                        w = PieceWeights(blk.lam, sub, JMAX, self.gauss_nodes, MMAX)
                        down["weights"][(gi, key)] = w
                else:
                    w = PieceWeights(blk.lam, sub, JMAX, self.gauss_nodes, MMAX)
                y0 = blk.to_modal(x0[members])
                f = blk.phi.T @ forcing[members]
                x_end[members] = blk.from_modal(w.end_exp * y0 + np.einsum("kj,kj->k", f, w.end_force))
                x_mom[members] = blk.phi @ (w.mom_exp * y0[:, None] + np.einsum("kj,kmj->km", f, w.mom_force))
                x_gauss[members] = blk.phi @ (w.node_exp * y0[:, None] + np.einsum("kj,kji->ki", f, w.node_force))
            self.down_x = x_end
            hold_mean += frac * x_mom[hold_local, 0]
            tg = polynomial_values(target, self.gauss_nodes)
            bg = x_gauss[buffer_local]
            gw = self.gauss_weights * sub
            works["controlled_voltage_source_interval_work_joule"] += float(
                np.sum(spec.gb[:, None] * tg * (tg - bg) * gw[None, :]))
            works["conditioning_drive_branch_dissipation_joule"] += float(
                np.sum(spec.gb[:, None] * (tg - bg) ** 2 * gw[None, :]))
            for node, conductance, volts in down["external"]:
                works["daq_external_source_interval_work_joule"] += conductance * volts * sub * (
                    volts - x_mom[node - n_up, 0])
            if self.energy_audit and nd:
                works["audit_down_input_joule"] = works.get("audit_down_input_joule", 0.0) + sub * float(
                    np.einsum("nj,nj->", forcing, x_mom[:, :forcing.shape[1]]))
                works["audit_down_heat_joule"] = works.get("audit_down_heat_joule", 0.0) + sub * float(
                    np.einsum("g,ng,nm,mg->", self.gauss_weights, x_gauss, down["gmat"], x_gauss))
            if self.work_model is not None:
                for name, value in self.work_model.downstream_works(self, sub, a, b, inputs, x_mom).items():
                    works[name] = works.get(name, 0.0) + value
        return dict(hold_mean=hold_mean, works=works, events=events)


def _selection_key(value):
    from daq_mux_hold import DAQGroupDrive
    if isinstance(value, DAQGroupDrive):
        return ("drive", value.selected_input, value.external_volts, value.voltage_reference_id)
    return ("member", value)


def _reference_id(selection):
    from daq_mux_hold import DAQGroupDrive
    if selection:
        for value in selection.values():
            if isinstance(value, DAQGroupDrive):
                return value.voltage_reference_id
    return "unused_reference"


def _restrict_inputs(inputs, a, b):
    """Re-express piece inputs on the sub-piece [a, b] of the unit interval."""
    if a == 0.0 and b == 1.0:
        return inputs
    def shift(values):
        return None if values is None else shift_monomials(values, a, b - a)
    return PieceInputs(current=shift(inputs.current), amplifier=shift(inputs.amplifier),
                       noise_ports=shift(inputs.noise_ports), field=shift(inputs.field))


def _poly_roots_in_unit(coefficients, rail, tolerance):
    """Real crossings of p(s) = rail for s in (0, 1); tangencies are immaterial."""
    c = np.array(coefficients, dtype=float)
    c[0] -= rail
    if np.sum(np.abs(c[1:])) < abs(c[0]):
        return []  # |p(s) - c0| <= sum|c_j| on [0,1]: no crossing possible
    while len(c) > 1 and c[-1] == 0.0:
        c = c[:-1]
    if len(c) < 2:
        return []
    candidates = np.polynomial.polynomial.polyroots(c)
    roots = []
    for z in np.atleast_1d(candidates):
        x = float(z.real)
        if abs(z.imag) > 1e-7 or not -1e-9 < x < 1.0 + 1e-9:
            continue
        lo, hi = max(0.0, x - 1e-6), min(1.0, x + 1e-6)
        flo = polynomial_values(c, np.array([lo]))[0]
        fhi = polynomial_values(c, np.array([hi]))[0]
        if flo == 0.0 or fhi == 0.0 or (flo < 0) == (fhi < 0):
            continue  # tangency or root at an endpoint: no change of clip state inside
        while hi - lo > tolerance:
            mid = 0.5 * (lo + hi)
            fmid = polynomial_values(c, np.array([mid]))[0]
            if fmid == 0.0:
                lo = hi = mid
                break
            if (fmid < 0) == (flo < 0):
                lo, flo = mid, fmid
            else:
                hi = mid
        root = 0.5 * (lo + hi)
        if 0.0 < root < 1.0:
            roots.append(root)
    return roots
