"""O6 particular solutions and O7 multirate homogeneous relaxation (fast path).

Inputs are the exact O6 tones of the saved draw (nine noise tones) plus the
11-Hz field quadrature tone. For each gated-row configuration r and tone k the
upstream particular solution X_{r,k} solves (i w_k C + G_r) X = F_k exactly
(contact states eliminated per pad, one sparse complex port solve). Within a
dwell the upstream state is x_part(t; r) + x_hom(t), where x_hom obeys the
AUTONOMOUS circuit C x' = -G_r x; at a row switch x_hom absorbs
x_part(t; r_old) - x_part(t; r_new) and the injected source charge.

x_hom is solved by the same exact blocks as the per-piece solver: the active
row block and the pad units, coupled by Gauss-Seidel waveform relaxation, but
on a coarse geometrically graded segment grid per dwell (units in large exact
steps). A fine pass re-advances only the active block on the reference splits
(cut at segment boundaries) to supply the downstream drive. The downstream is
the unchanged exact O4 path. Works: exact closed forms for particular parts,
exact Fourier moments for homogeneous parts, exact polynomial moments for the
downstream. No physics, parameter, schedule or output definition changes.
"""
import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from fast_solver_engine import JMAX, MMAX, FastSolverError, PieceInputs
from fast_solver_fourier import FourierWeights, phi_complex, tone_integral
from fast_solver_phi import PieceWeights, shift_monomials, product_integral

_FACT = np.array([math.factorial(k) for k in range(JMAX + 1)], dtype=float)


def _taylor(amplitude, omega, t0, h, degree=JMAX):
    """Coefficients (in s = (t - t0)/h) of Re[a exp(i w t)]: amplitude (..., ) complex."""
    n = np.arange(degree + 1)
    factor = (1j * omega * h) ** n / _FACT[:degree + 1] * np.exp(1j * omega * t0)
    return np.real(np.asarray(amplitude)[..., None] * factor)


class ToneSet:
    """All exact tones: nine noise tones and the field tone (last)."""

    def __init__(self, exact_inputs, array):
        self.exact = exact_inputs
        layout = array.physical_noise_layout
        tones = exact_inputs.tones
        N = len(array.state_ids)
        field_current = np.zeros(N, dtype=complex)
        np.add.at(field_current, exact_inputs._star_nodes,
                  exact_inputs._star_b * exact_inputs.field_complex[exact_inputs._star_group])
        self.omega = np.array([t["omega"] for t in tones] + [exact_inputs.field_omega])
        self.current = np.vstack([t["current"] for t in tones] + [field_current])        # (M, N)
        self.amplifier = np.vstack([t["amplifier"] for t in tones] + [np.zeros(64)])      # (M, 64)
        self.port = np.vstack([t["port"] for t in tones])                                 # (Mn, ports)
        self.field = exact_inputs.field_complex
        self.field_omega = exact_inputs.field_omega
        self.noise_count = len(tones)
        self.layout = layout


class ParticularSolutions:
    def __init__(self, solver, toneset):
        self.solver, self.tones = solver, toneset
        m = solver.model
        P, S, K = m.P, m.S, m.K
        self.P, self.S, self.K = P, S, K
        n_up = solver.n_up
        M = len(toneset.omega)
        self.M = M
        # contact types (exact bytes of the contact block data)
        keys, self.contact_type, reps = {}, np.empty(P, dtype=np.int64), []
        for p in range(P):
            key = (m.branch_c[p].tobytes(), m.branch_g[p].tobytes(), m.star_b[p, 1:].tobytes())
            if key not in keys:
                keys[key] = len(reps)
                reps.append(p)
            self.contact_type[p] = keys[key]
        self.contact_reps = reps
        # relative forcing per tone
        self.f_pads = np.empty((M, P, K + 1), dtype=complex)
        self.f_src = np.empty((M, S), dtype=complex)
        for k in range(M):
            pads, src = m.currents_to_relative(toneset.current[k][:n_up, None])
            self.f_pads[k], self.f_src[k] = pads[..., 0], src[:, 0]
        # contact elimination per tone and type
        self.dinv_b = np.empty((M, P, K), dtype=complex)
        self.sigma = np.empty((M, P), dtype=complex)
        self.w0 = np.empty((M, P, K), dtype=complex)
        for k in range(M):
            omega = toneset.omega[k]
            for t, rep in enumerate(reps):
                members = np.flatnonzero(self.contact_type == t)
                if K:
                    d = 1j * omega * np.diag(m.branch_c[rep]) + m.branch_g[rep]
                    b = m.star_b[rep, 1:]
                    sol = np.linalg.solve(d, np.column_stack((b, self.f_pads[k, members, 1:].T)))
                    self.dinv_b[k, members] = sol[:, 0]
                    self.sigma[k, members] = b @ sol[:, 0]
                    self.w0[k, members] = sol[:, 1:].T
                else:
                    self.dinv_b[k, members] = 0
                    self.sigma[k, members] = 0
                    self.w0[k, members] = 0
        self.f_eff = self.f_pads[:, :, 0] - np.einsum("pk,mpk->mp", m.star_b[:, 1:], self.w0) if K else \
            self.f_pads[:, :, 0].copy()
        self._ports = {}
        self.max_residual = 0.0

    def ports(self, row):
        """X_v (M, P) and X_s (M, S) for one gated row (cached)."""
        cached = self._ports.get(row)
        if cached is not None:
            return cached
        m = self.solver.model
        P, S = self.P, self.S
        g_ports, _ = m.port_conductance(row)
        xv = np.empty((self.M, P), dtype=complex)
        xs = np.empty((self.M, S), dtype=complex)
        for k in range(self.M):
            omega = self.tones.omega[k]
            matrix = (1j * omega * m.cpp + g_ports).tocsc().astype(complex)
            matrix = matrix - sp.diags(np.concatenate((self.sigma[k], np.zeros(S))))
            rhs = np.concatenate((self.f_eff[k], self.f_src[k]))
            solution = splu(matrix.tocsc()).solve(rhs)
            residual = np.abs(matrix @ solution - rhs).max() / max(np.abs(rhs).max(), 1e-300)
            self.max_residual = max(self.max_residual, float(residual))
            xv[k], xs[k] = solution[:P], solution[P:]
        if len(self._ports) > 70:
            self._ports.clear()
        self._ports[row] = (xv, xs)
        return xv, xs

    def contacts(self, xv):
        """w phasors (M, P, K) for pad phasors xv (M, P)."""
        return self.w0 - self.dinv_b * xv[:, :, None]

    def relative_state(self, row, t):
        """Particular upstream state at time t in relative coordinates (pads (P,K+1), sources (S,))."""
        xv, xs = self.ports(row)
        phase = np.exp(1j * self.tones.omega * t)[:, None]
        pads = np.empty((self.P, self.K + 1))
        pads[:, 0] = np.real(np.sum(xv * phase, axis=0))
        if self.K:
            pads[:, 1:] = np.real(np.sum(self.contacts(xv) * phase[:, :, None], axis=0))
        sources = np.real(np.sum(xs * phase, axis=0))
        return pads, sources

    def source_taylor(self, row, t0, h, degree=JMAX):
        _, xs = self.ports(row)
        total = np.zeros((self.S, degree + 1))
        for k in range(self.M):
            total += _taylor(xs[k], self.tones.omega[k], t0, h, degree)
        return total


class MultirateSolver:
    """Fast path: particular solutions + multirate homogeneous WR + exact downstream."""

    _audit_optimize_tones = False

    def __init__(self, base, toneset, work_model, *, segment_levels=12, dwell_seconds=62.5e-6):
        self.base, self.tones, self.work_model = base, toneset, work_model
        self.part = ParticularSolutions(base, toneset)
        self.segment_levels = segment_levels
        self.dwell = float(dwell_seconds)
        self.fractions = [0.0] + [2.0 ** (l - segment_levels) for l in range(segment_levels + 1)]
        self.row = None
        self.t_hom = None
        self._a_weights = {}
        self._noise_full = None
        self.stats = dict(dwells=0, segments=0, sweeps=0, max_sweeps=0, max_wr_change=0.0,
                          max_rep_error_pad=0.0, max_rep_error_source=0.0, fine_pieces=0,
                          max_fine_coarse_mismatch=0.0, clip_events=0)
        m = base.model
        P, S, K = m.P, m.S, m.K
        u = K + 1
        # work functionals (relative coordinates) per pad: noise current, access, field
        wm = work_model
        Mn = toneset.noise_count
        star_port = wm.star_offset + 17 * np.arange(P)[:, None] + np.arange(17)[None, :]
        A = toneset.port                                                   # (Mn, ports)
        mstar = np.zeros((17, 17))
        mstar[:, 0] = 1.0
        mstar[np.arange(1, 17), np.arange(1, 17)] = 1.0
        f = wm.fractions                                                   # (P, 17)
        fm = f @ mstar                                                     # (P, 17) f . M
        a_cur = np.einsum("mpj,jq->mpq", A[:, star_port], mstar) - (wm.mix[:, None] * fm)[None] * \
            A[:, star_port].sum(axis=2)[:, :, None]
        leak = np.zeros((Mn, P), dtype=complex)
        leak[:, wm.pad_of_port] = A[:, wm.pad_ports]
        a_cur[:, :, 0] += leak
        acc = A[:, wm.access_offset + np.arange(P)]
        a_acc = -(acc * (wm.totals * wm.inverse_d)[None, :])[:, :, None] * fm[None]
        f_vec = wm.star_b.copy()
        f_vec_rel = np.empty((P, 17))
        f_vec_rel[:, 0] = wm.B
        f_vec_rel[:, 1:] = wm.star_b[:, 1:]
        a_fld = -(toneset.field[:, None] * f_vec_rel)[None]                  # (1, P, 17)
        self.functionals = dict(noise_current_source_interval_work_joule=(np.arange(Mn), a_cur),
                                noise_access_voltage_source_interval_work_joule=(np.arange(Mn), a_acc),
                                field_source_interval_work_joule=(np.array([len(toneset.omega) - 1]), a_fld))
        src = np.zeros((Mn, S), dtype=complex)
        np.add.at(src, (slice(None), wm.source_of_port), A[:, wm.source_ports])
        self.source_functional = src                                         # noise current at sources
        # modal functionals for units: g[name] (Mt, P, u) = Phi_T^T a  (x = Phi y)
        self.g_units = {}
        for name, (tone_idx, a) in self.functionals.items():
            g = np.zeros(a.shape, dtype=complex)
            for t, block in enumerate(base.unit_blocks):
                members = base.unit_members[t]
                g[:, members] = a[:, members] @ block.phi
            self.g_units[name] = (tone_idx, g)
        self._g_types = [{name: (tone_idx, g[:, base.unit_members[t]].copy())
                          for name, (tone_idx, g) in self.g_units.items()} for t in range(len(base.unit_blocks))]
        # particular-part coefficient maps (row independent): a.X = alpha X_v + beta
        self.part_maps = {}
        for name, (tone_idx, a) in self.functionals.items():
            av, aw = a[:, :, 0], a[:, :, 1:]
            alpha = av[:, None, :] - np.einsum("mpk,npk->mnp", aw, self.part.dinv_b)
            beta = np.einsum("mpk,npk->mnp", aw, self.part.w0)
            alpha_c = av[:, None, :] - np.einsum("mpk,npk->mnp", aw, np.conj(self.part.dinv_b))
            beta_c = np.einsum("mpk,npk->mnp", aw, np.conj(self.part.w0))
            self.part_maps[name] = (tone_idx, alpha, beta, alpha_c, beta_c)
        self._fourier_unit = {}
        self._a_func = {}
        self._fourier_a = {}
        self._audit_row = None
        self._audit_modal = None
        self._audit_rules = {}

    # ---------------- helpers --------------------------------------------
    def _key(self, h):
        key = round(h * 1e15)
        return key if abs(key * 1e-15 - h) <= 1e-20 else ("exact", h)

    def _hom_relative(self):
        return self.base._relative_pads()

    def _set_hom_relative(self, pads, sources):
        base = self.base
        for t, block in enumerate(base.unit_blocks):
            members = base.unit_members[t]
            base.unit_y[members] = block.to_modal(pads[members])
        base.source_x = np.array(sources, dtype=float)
        base.row = None
        base.a_y = None

    def full_state(self, t):
        """Complete original-coordinate state at the current time t (part + hom)."""
        pads, sources = self._hom_relative()
        if self.row is not None:
            pp, ps = self.part.relative_state(self.row, t)
            pads, sources = pads + pp, sources + ps
        x = self.base.model.from_relative(pads, sources)
        return np.concatenate((x, self.base.down_x))

    def set_full_state(self, x, t, row):
        """Initialize from a full state at time t with gated row (hom = x - part)."""
        base = self.base
        pads, sources = base.model.to_relative(np.asarray(x)[:base.n_up])
        pp, ps = self.part.relative_state(row, t)
        self._set_hom_relative(pads - pp, sources - ps)
        base.down_x = np.array(x[base.n_up:], dtype=float)
        base._activate(row)
        self.row = row

    # ---------------- dwell ------------------------------------------------
    def begin_dwell(self, row, t_r, source_charge):
        """Row switch at t_r: x_hom += part(old) - part(new) + C^-1 q (sources)."""
        base = self.base
        pads, sources = self._hom_relative()
        if self.row is not None and self.row != row:
            p_old, s_old = self.part.relative_state(self.row, t_r)
            p_new, s_new = self.part.relative_state(row, t_r)
            pads = pads + p_old - p_new
            sources = sources + s_old - s_new
        self._set_hom_relative(pads, sources)
        entry = base._activate(row)
        self.row = row
        work = 0.0
        if source_charge is not None and np.any(source_charge):
            charged = np.flatnonzero(source_charge)
            part_now = self.part.relative_state(row, t_r)[1]
            before = base._node_values(entry, charged) + part_now[charged - base.model.P]
            base._apply_charge(entry, source_charge)
            after = base._node_values(entry, charged) + part_now[charged - base.model.P]
            work = float(0.5 * ((before + after) @ source_charge[charged]))
        return entry, work

    def coarse_dwell(self, entry, t_r):
        """Homogeneous WR on the graded segment grid; returns per-segment records."""
        base = self.base
        m = base.model
        P, S, K = m.P, m.S, m.K
        u = K + 1
        d = base.degree
        nodes = base.nodes
        block = entry["block"]
        active = entry["pads"]
        n_active = len(active)
        inactive_mask = np.ones(P, dtype=bool)
        inactive_mask[active] = False
        phi_ports = block.phi[entry["port_pos"]]
        a_ports = entry["port_global"]
        records = []
        for a, b in zip(self.fractions[:-1], self.fractions[1:]):
            h = self.dwell * (b - a)
            key = ("seg", self._key(self.dwell), a, b)
            wa = block.weights(key, h, nodes, "lobatto")
            ya0 = base.a_y.copy()
            ya_nodes_own = wa.node_exp * ya0[:, None]
            unit_data = []
            unit_port_own = np.zeros((P, d + 1))
            for t, ublock in enumerate(base.unit_blocks):
                members = base.unit_members[t][inactive_mask[base.unit_members[t]]]
                if not len(members):
                    continue
                wu = ublock.weights(key, h, nodes, "lobatto")
                y0 = base.unit_y[members].copy()
                phi0 = ublock.phi[0]
                unit_port_own[members] = (y0 * phi0[None, :]) @ wu.node_exp
                response = np.einsum("k,kji->ji", phi0 * phi0, wu.node_force)
                unit_data.append((t, members, wu, y0, phi0, response))
            port_values = np.zeros((P + S, d + 1))
            port_values[:P] = unit_port_own
            powers = np.arange(1, d + 1)[None, :] / h
            a_coupling = np.zeros((len(a_ports), JMAX + 1))
            unit_coupling = np.zeros((P, JMAX + 1))
            change, sweeps = math.inf, 0
            while change > base.wr_tolerance:
                if sweeps >= base.wr_max_sweeps:
                    raise FastSolverError(f"coarse waveform relaxation exceeded {base.wr_max_sweeps} sweeps "
                                          f"(last change {change:.3e} V)")
                sweeps += 1
                previous = port_values.copy()
                coeff = port_values @ base.vinv.T
                deriv = np.zeros_like(coeff)
                deriv[:, :-1] = coeff[:, 1:] * powers
                a_coupling[:, :d + 1] = -(entry["c_off_a"] @ deriv + entry["g_off_a"] @ coeff)
                fac = phi_ports.T @ a_coupling
                ya_nodes = ya_nodes_own + np.einsum("kj,kji->ki", fac, wa.node_force)
                port_values[a_ports] = phi_ports @ ya_nodes
                coeff = port_values @ base.vinv.T
                deriv[:, :-1] = coeff[:, 1:] * powers
                unit_coupling[:, :d + 1] = -(entry["c_off_u"] @ deriv + entry["g_off_u"] @ coeff)
                for t, members, wu, y0, phi0, response in unit_data:
                    port_values[members] = unit_port_own[members] + unit_coupling[members] @ response
                change = float(np.abs(port_values - previous).max())
            fa_total = phi_ports.T @ a_coupling
            ya_end = wa.end_exp * ya0 + np.einsum("kj,kj->k", fa_total, wa.end_force)
            wcheck = block.weights(key, h, base.check_nodes, "check")
            exact_check = phi_ports @ (wcheck.node_exp * ya0[:, None]
                                       + np.einsum("kj,kji->ki", fa_total, wcheck.node_force))
            poly_check = (port_values[a_ports] @ base.vinv.T) @ base.check_vdm.T
            rep = np.abs(exact_check - poly_check)
            rep_pad = float(rep[:n_active].max()) if n_active else 0.0
            rep_src = float(rep[n_active:].max())
            for t, members, wu, y0, phi0, response in unit_data:
                c = unit_coupling[members]
                base.unit_y[members] = wu.end_exp[None, :] * y0 + phi0[None, :] * (c @ wu.end_force.T)
                wck = base.unit_blocks[t].weights(key, h, base.check_nodes, "check")
                exact_u = (y0 * phi0[None, :]) @ wck.node_exp + c @ np.einsum("k,kji->ji", phi0 * phi0, wck.node_force)
                poly_u = (port_values[members] @ base.vinv.T) @ base.check_vdm.T
                rep_pad = max(rep_pad, float(np.abs(exact_u - poly_u).max()))
            records.append(dict(a=a, b=b, h=h, key=key, ya0=ya0, a_coupling=a_coupling.copy(),
                                unit_y0=[(t, members, y0) for t, members, wu, y0, phi0, response in unit_data],
                                unit_coupling=unit_coupling.copy(), sweeps=sweeps, change=change,
                                rep_pad=rep_pad, rep_source=rep_src))
            base.a_y = ya_end
            st = self.stats
            st["segments"] += 1
            st["sweeps"] += sweeps
            st["max_sweeps"] = max(st["max_sweeps"], sweeps)
            st["max_wr_change"] = max(st["max_wr_change"], change)
            st["max_rep_error_pad"] = max(st["max_rep_error_pad"], rep_pad)
            st["max_rep_error_source"] = max(st["max_rep_error_source"], rep_src)
        self.stats["dwells"] += 1
        return records

    def a_fine(self, entry, record, a, b):
        """Advance the active block over [a, b] (fractions of the segment) from base.a_y."""
        base = self.base
        block = entry["block"]
        h = record["h"] * (b - a)
        # These weights contain the active block's eigenvalues. Different
        # gated-row blocks can share a time grid without sharing those modes.
        key = ("fine", id(block), record["key"], a, b)
        w = self._a_weights.get(key)
        if w is None:
            w = PieceWeights(block.lam, h, JMAX, base.nodes, MMAX)
            self._a_weights[key] = w
        coupling = record["a_coupling"] if (a, b) == (0.0, 1.0) else shift_monomials(record["a_coupling"], a, b - a)
        phi_ports = block.phi[entry["port_pos"]]
        fa = phi_ports.T @ coupling
        y0 = base.a_y
        nodes_values = w.node_exp * y0[:, None] + np.einsum("kj,kji->ki", fa, w.node_force)
        source_values = block.phi[entry["source_pos"]] @ nodes_values
        base.a_y = w.end_exp * y0 + np.einsum("kj,kj->k", fa, w.end_force)
        self.stats["fine_pieces"] += 1
        return source_values @ base.vinv.T, h

    def _audit_modal_data(self, entry):
        """Modal particular states and EXTERNAL Norton forces for the current row.

        These are derived from the original tone forcing, not from a stored
        energy difference. Only one row is retained to bound audit memory.
        """
        if self._audit_row == self.row:
            return self._audit_modal
        base, part = self.base, self.part
        xv, xs = part.ports(self.row)
        pads = np.concatenate((xv[:, :, None], part.contacts(xv)), axis=2)
        active, block = entry["pads"], entry["block"]
        xa = np.concatenate((pads[:, active].reshape(part.M, -1), xs), axis=1)
        fa = np.concatenate((part.f_pads[:, active].reshape(part.M, -1), part.f_src), axis=1)
        data = {"active": (block.to_modal(xa), fa @ block.phi), "units": {}}
        for t, unit in enumerate(base.unit_blocks):
            members = base.unit_members[t]
            data["units"][t] = (unit.to_modal(pads[:, members]), part.f_pads[:, members] @ unit.phi)
        self._audit_row, self._audit_modal = self.row, data
        return data

    def _audit_weights(self, lam, h, nodes):
        """Default audit kernel; subclasses may omit unused moment arrays."""
        return PieceWeights(lam, h, JMAX, nodes, 0)

    def _audit_unit_response(self, force, node_force):
        """Polynomial response per member, mode and quadrature node."""
        return np.einsum("pkj,kjg->pkg", force, node_force)

    def upstream_energy_audit(self, entry, record, t_r, order):
        """Independently integrate x.F_external and x.G.x over one coarse segment.

        The total state includes particular, homogeneous and their cross
        terms. G contains the modal blocks plus the original off-block
        conductance edges. This never reads endpoint energy and never sets
        heat from an energy balance. Order doubling is performed by the caller.
        The coarse upstream path and actual downstream path are audited
        separately; their endpoint discrepancy remains visible in the residual.
        """
        base = self.base
        if order not in self._audit_rules:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            self._audit_rules[order] = (0.5 * (nodes + 1.0), 0.5 * weights)
        nodes, weights = self._audit_rules[order]
        data = self._audit_modal_data(entry)
        h = record["h"]
        t0 = t_r + self.dwell * record["a"]
        phase = np.exp(1j * self.tones.omega[:, None] * (t0 + h * nodes)[None, :])
        block = entry["block"]
        # Keep quadrature weights local. Caching 32-node tables for every
        # active-row block would multiply the propagation cache footprint.
        w = self._audit_weights(block.lam, h, nodes)
        force = block.phi[entry["port_pos"]].T @ record["a_coupling"]
        hom = w.node_exp * record["ya0"][:, None] + np.einsum("kj,kjg->kg", force, w.node_force)
        yp, fp = data["active"]
        state = hom + np.real(yp.T @ phase)
        external = np.real(fp.T @ phase)
        input_work = h * float(np.einsum("g,kg,kg->", weights, state, external))
        heat = h * float(np.einsum("g,k,kg->", weights, block.rate, state ** 2))
        ports = np.zeros((base.model.P + base.model.S, order))
        ports[entry["port_global"]] = block.phi[entry["port_pos"]] @ state
        for t, members, y0 in record["unit_y0"]:
            unit = base.unit_blocks[t]
            rows = np.searchsorted(base.unit_members[t], members)
            w = self._audit_weights(unit.lam, h, nodes)
            force = unit.phi[0][None, :, None] * record["unit_coupling"][members, None, :]
            hom = (w.node_exp[None, :, :] * y0[:, :, None]
                   + self._audit_unit_response(force, w.node_force))
            yp, fp = data["units"][t]
            state = hom + np.real(np.einsum("mpk,mg->pkg", yp[:, rows], phase,
                                          optimize=self._audit_optimize_tones))
            external = np.real(np.einsum("mpk,mg->pkg", fp[:, rows], phase,
                                       optimize=self._audit_optimize_tones))
            input_work += h * float(np.einsum("g,pkg,pkg->", weights, state, external))
            heat += h * float(np.einsum("g,k,pkg->", weights, unit.rate, state ** 2))
            ports[members] = np.einsum("k,pkg->pg", unit.phi[0], state)
        rows, cols, values = entry["g_off_edges"]
        heat += h * float(np.einsum("g,e,eg,eg->", weights, values, ports[rows], ports[cols]))
        if not np.all(np.isfinite((input_work, heat))):
            raise FastSolverError("nonfinite independently integrated upstream energy audit")
        return dict(input_work=input_work, heat=heat)

    # ---------------- works -------------------------------------------------
    def particular_works(self, row, t0, t1):
        """Exact state-dependent work integrals of x_part over [t0, t1] (one row config)."""
        xv, xs = self.part.ports(row)
        omega = self.tones.omega
        result = {}
        for name, (tone_idx, alpha, beta, alpha_c, beta_c) in self.part_maps.items():
            total = 0.0
            for mi, m_tone in enumerate(tone_idx):
                plus = tone_integral(omega[m_tone] + omega, t0, t1)        # (K,)
                minus = tone_integral(omega[m_tone] - omega, t0, t1)
                ax = alpha[mi] * xv + beta[mi]                              # (K, P)
                axc = alpha_c[mi] * np.conj(xv) + beta_c[mi]
                total += 0.5 * float(np.real(np.sum(ax * plus[:, None]) + np.sum(axc * minus[:, None])))
            result[name] = total
        # noise current ports at sources: eta . v_s
        total = 0.0
        for mi in range(self.tones.noise_count):
            plus = tone_integral(omega[mi] + omega, t0, t1)
            minus = tone_integral(omega[mi] - omega, t0, t1)
            a = self.source_functional[mi]
            total += 0.5 * float(np.real(np.sum((a[None, :] * xs) * plus[:, None])
                                         + np.sum((a[None, :] * np.conj(xs)) * minus[:, None])))
        result["noise_current_source_interval_work_joule"] += total
        return result

    def _a_functionals(self, row, entry):
        """Active-block work functionals in its modal basis (cached per gated row)."""
        cached = self._a_func.get(row)
        if cached is not None:
            return cached
        block = entry["block"]
        active = entry["pads"]
        u = self.base.model.unit_size
        a_rows = {}
        for name, (tone_idx, a) in self.functionals.items():
            av = np.zeros((len(tone_idx), block.size), dtype=complex)
            av[:, :len(active) * u] = a[:, active].reshape(len(tone_idx), -1)
            if name == "noise_current_source_interval_work_joule":
                av[:, len(active) * u:] = self.source_functional
            a_rows[name] = (tone_idx, av @ block.phi)
        self._a_func[row] = a_rows
        return a_rows

    def _unit_tables(self, key, h, t):
        """Fourier weights of unit type t and its coupling-response contraction per pad."""
        full = (key, t)
        cached = self._fourier_unit.get(full)
        if cached is not None:
            return cached
        base = self.base
        ublock = base.unit_blocks[t]
        members = base.unit_members[t]
        fw = FourierWeights(ublock.lam, self.tones.omega, h, JMAX)
        phi0 = ublock.phi[0]
        tables = {}
        for name, (tone_idx, g) in self._g_types[t].items():
            coupled = np.einsum("mpk,kmj->mpj", g * phi0[None, None, :], fw.F[:, tone_idx, :])  # (Mt, Mm, J+1)
            tables[name] = (tone_idx, g, fw.E[:, tone_idx].T, coupled)
        cached = (members, tables)
        if len(self._fourier_unit) > 256:
            self._fourier_unit.clear()
        self._fourier_unit[full] = cached
        return cached

    def homogeneous_works(self, entry, records, t_r):
        """Exact Fourier-moment work integrals of x_hom over the dwell's coarse segments."""
        base = self.base
        omega = self.tones.omega
        block = entry["block"]
        result = dict.fromkeys(self.functionals, 0.0)
        a_rows = self._a_functionals(self.row, entry)
        phi_ports = block.phi[entry["port_pos"]]
        for record in records:
            t_a = t_r + self.dwell * record["a"]
            fw_key = (record["key"], id(block))
            fw = self._fourier_a.get(fw_key)
            if fw is None:
                fw = FourierWeights(block.lam, omega, record["h"], JMAX)
                self._fourier_a[fw_key] = fw
            ua = phi_ports.T @ record["a_coupling"]                           # (n, J+1)
            phase = np.exp(1j * omega * t_a)
            for name, (tone_idx, g) in a_rows.items():
                inner = (
                    np.einsum("mk,km,k->m", g, fw.E[:, tone_idx], record["ya0"])
                    + np.einsum("mk,kmj,kj->m", g, fw.F[:, tone_idx, :], ua))
                result[name] += float(np.real(np.sum(phase[tone_idx] * inner)))
            for t, members, y0 in record["unit_y0"]:
                all_members, tables = self._unit_tables(record["key"], record["h"], t)
                rows = np.searchsorted(all_members, members)
                y0_full = np.zeros((len(all_members), y0.shape[1]))
                y0_full[rows] = y0
                c_full = np.zeros((len(all_members), JMAX + 1))
                c_full[rows] = record["unit_coupling"][members]
                for name, (tone_idx, g, e_weights, coupled) in tables.items():
                    free = np.sum(e_weights * np.einsum("mpk,pk->mk", g, y0_full), axis=1)
                    forced = np.einsum("mpj,pj->m", coupled, c_full)
                    result[name] += float(np.real(np.sum(phase[tone_idx] * (free + forced))))
        return result

    def input_works(self, t0, t1):
        """Work terms that involve only inputs (closed forms over [t0, t1])."""
        wm, tones = self.work_model, self.tones
        P = wm.P
        Mn = tones.noise_count
        omega = tones.omega
        A = tones.port
        star = wm.star_offset + 17 * np.arange(P)[:, None] + np.arange(17)[None, :]
        tot = A[:, star].sum(axis=2)                                        # (Mn, P) total eta phasors
        acc = A[:, wm.access_offset + np.arange(P)]                         # (Mn, P)
        U = tones.field                                                     # (P,) field phasor, tone f
        wf = tones.field_omega

        def cross(x, wx, y, wy):
            # int Re[x e^{i wx t}] Re[y e^{i wy t}] dt summed over pads; x,y (P,)
            plus = tone_integral(np.array([wx + wy]), t0, t1)[0]
            minus = tone_integral(np.array([wx - wy]), t0, t1)[0]
            return 0.5 * float(np.real(np.sum(x * y) * plus + np.sum(x * np.conj(y)) * minus))

        def weighted_cross(weight, x, wx, y, wy):
            return cross(weight * x, wx, y, wy)

        hd = wm.totals * wm.inverse_d
        current = access = field = 0.0
        for a in range(Mn):
            for b in range(Mn):
                # -inverse_d totEta (U + e) + r_div_d totEta^2  (noise current, input part)
                current += weighted_cross(-wm.inverse_d, tot[a], omega[a], acc[b], omega[b])
                current += weighted_cross(wm.r_div_d, tot[a], omega[a], tot[b], omega[b])
                # access: hd e (e) + inverse_d e totEta
                access += weighted_cross(hd, acc[a], omega[a], acc[b], omega[b])
                access += weighted_cross(wm.inverse_d, acc[a], omega[a], tot[b], omega[b])
            current += weighted_cross(-wm.inverse_d, tot[a], omega[a], U, wf)
            access += weighted_cross(hd, acc[a], omega[a], U, wf)
            field += weighted_cross(hd, U, wf, acc[a], omega[a])
            field += weighted_cross(wm.inverse_d, U, wf, tot[a], omega[a])
        field += weighted_cross(wm.B, U, wf, U, wf)
        return dict(noise_current_source_interval_work_joule=current,
                    noise_access_voltage_source_interval_work_joule=access,
                    field_source_interval_work_joule=field)

    # ---------------- downstream piece -----------------------------------
    def downstream_piece(self, down, h, key, t0, source_hom_poly, row):
        """Exact downstream over [t0, t0+h] with v_s = part + hom and exact tone inputs."""
        base = self.base
        tones = self.tones
        nd = base.N - base.n_up
        S = base.model.S
        amp = np.zeros((64, JMAX + 1))
        for k in range(len(tones.omega)):
            amp += _taylor(tones.amplifier[k], tones.omega[k], t0, h)
        amp[:S] += self.part.source_taylor(row, t0, h)             # v_s,part folded into the gain term
        cur = np.zeros((base.N, JMAX + 1))
        for k in range(len(tones.omega)):
            cur[base.n_up:] += _taylor(tones.current[k][base.n_up:], tones.omega[k], t0, h)
        if self._noise_full is None:
            self._noise_full = np.zeros((tones.port.shape[1], JMAX + 1))
        noise = self._noise_full
        ports = self.work_model.down_ports
        noise[ports] = 0.0
        for k in range(tones.noise_count):
            noise[ports] += _taylor(tones.port[k, ports], tones.omega[k], t0, h)
        inputs = PieceInputs(current=cur, amplifier=amp, noise_ports=noise, field=None)
        hom = np.zeros((S, base.degree + 1))
        hom[:, :source_hom_poly.shape[1]] = source_hom_poly[:, :base.degree + 1]
        result = base._downstream(down, h, key, inputs, cur, hom)
        self.stats["clip_events"] += result["events"]
        return result
