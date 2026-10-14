"""Stage 1 item 2 (and the 35-ms clipping fixture): small circuits.

Each circuit is advanced by (a) the unmodified reference class
(SparsePassiveRCNetwork or ConditionedRCNetwork), (b) a dense matrix-exponential
oracle of the complete linear system (scipy.linalg.expm of the augmented
generator; clip events located by bisection on that exact solution) and (c) the
new fast solver through its normal O7 block/WR and O4 downstream path.
Requirements: |fast - oracle| <= 1e-12 V; |fast - reference| within the
reference's own tolerances (atol + rtol |v|). Results are also written as JSON
to $FAST_SOLVER_TEST_JSON when set. Development tests; run inside Slurm only.
"""
import json
import math
import os
import unittest

import numpy as np
from scipy.linalg import expm
import scipy.sparse as sp

from fast_solver_engine import ArraySpec, FastArraySolver, PieceInputs, JMAX
from fast_solver_phi import PieceWeights, phi_table, shift_monomials

RESULTS = {}
ORACLE_TOL = 1e-12


def _record(name, **values):
    RESULTS[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in values.items()}
    path = os.environ.get("FAST_SOLVER_TEST_JSON")
    if path:
        with open(path, "w") as stream:
            json.dump(RESULTS, stream, indent=2, sort_keys=True, default=float)


# ---------------------------------------------------------------------------
# Dense oracle
# ---------------------------------------------------------------------------

def oracle_interval(c, a_extra, g, x0, forcing, h, mean_rows=None):
    """Exact x(h) and selected means for C x' = (-G + A_extra) x + sum_j F_j (t/h)^j."""
    n = len(x0)
    jdim = forcing.shape[1]
    cinv = np.linalg.inv(c)
    rows = np.arange(n) if mean_rows is None else np.asarray(mean_rows)
    size = n + jdim + len(rows)
    m = np.zeros((size, size))
    m[:n, :n] = cinv @ (-g + a_extra)
    m[:n, n:n + jdim] = cinv @ forcing
    for j in range(1, jdim):
        m[n + j, n + j - 1] = j / h       # d/dt (t/h)^j = (j/h)(t/h)^(j-1)
    m[n + jdim:, :n] = np.eye(n)[rows]
    z0 = np.zeros(size)
    z0[:n] = x0
    z0[n] = 1.0
    z = expm(m * h) @ z0
    return z[:n], z[n + jdim:] / h


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------

class KernelTests(unittest.TestCase):
    def test_phi_matches_augmented_exponential(self):
        worst = 0.0
        for z in (0.0, -1e-6, -0.3, -0.5, -1.0, -7.0, -39.0, -41.0, -300.0, -2.5e4):
            k = 12
            m = np.zeros((k + 1, k + 1))
            m[0, 0] = z
            for i in range(k):
                m[i, i + 1] = 1.0
            if abs(z) <= 300:
                expected = expm(m)[0]
                values = phi_table(np.array([z]), k)[:, 0]
                rel = np.max(np.abs(values - expected) / np.abs(expected))
                worst = max(worst, rel)
                self.assertLess(rel, 1e-13, msg=f"z={z}")
            else:
                values = phi_table(np.array([z]), k)[:, 0]
                # asymptotics: phi_k(z) ~ -1/((k-1)! z) for large negative z (k >= 1)
                for kk in range(1, 5):
                    approx = -1.0 / (math.factorial(kk - 1) * z)
                    self.assertLess(abs(values[kk] - approx) / approx, 1e-3)
        _record("phi_vs_expm", max_relative_error=worst)

    def test_piece_weights_match_oracle(self):
        nodes = np.array([0.1, 0.37, 0.8, 1.0])
        worst = 0.0
        for lam in (-1e-3, -2.0, -39.5, -40.5, -300.0, -5e5):
            h = 1.0
            w = PieceWeights(np.array([lam]), h, JMAX, nodes, 6)
            # The expm oracle's error is relative to |exp(M)|; use it where that does not dominate.
            for j in (range(4) if abs(lam * h) <= 1000 else ()):
                forcing = np.zeros((1, JMAX + 1))
                forcing[0, j] = 1.0
                end, mean = oracle_interval(np.eye(1), np.zeros((1, 1)), np.array([[-lam]]), np.array([0.7]),
                                            forcing[:, :j + 1], h)
                got_end = w.end_exp[0] * 0.7 + w.end_force[0, j]
                got_mean = w.mom_exp[0, 0] * 0.7 + w.mom_force[0, 0, j]
                for value, ref in ((got_end, end[0]), (got_mean, mean[0])):
                    rel = abs(value - ref) / max(abs(ref), 1e-300)
                    worst = max(worst, rel)
                    self.assertLess(rel, 1e-12, msg=f"lam={lam} j={j}")
                # node values: on [0, s h] the forcing (tau/h)^j equals s^j (tau/(s h))^j
                for i, s in enumerate(nodes):
                    local = np.zeros((1, j + 1))
                    local[0, j] = s ** j
                    ref_s, _ = oracle_interval(np.eye(1), np.zeros((1, 1)), np.array([[-lam]]), np.array([0.7]),
                                               local, h * s)
                    got_s = w.node_exp[0, i] * 0.7 + w.node_force[0, j, i]
                    rel = abs(got_s - ref_s[0]) / max(abs(ref_s[0]), 1e-300)
                    worst = max(worst, rel)
                    self.assertLess(rel, 1e-12, msg=f"node lam={lam} j={j} s={s}")
            # moments by a fine composite Gauss rule on the exact solution
            from fast_solver_phi import phi_table as phi
            xs, ws = np.polynomial.legendre.leggauss(64)
            edges = np.unique(np.concatenate(([0.0], np.geomspace(1e-9, 1.0, 40))))
            for j in range(3):
                for mom in range(4):
                    total = 0.0
                    for a, b in zip(edges[:-1], edges[1:]):
                        s = 0.5 * (b - a) * xs + 0.5 * (a + b)
                        p = phi(lam * h * s, j + 1)
                        y = h * s ** (j + 1) * math.factorial(j) * p[j + 1]
                        total += 0.5 * (b - a) * np.sum(ws * s ** mom * y)
                    got = w.mom_force[0, mom, j]
                    rel = abs(got - total) / abs(total)
                    worst = max(worst, rel)
                    self.assertLess(rel, 1e-11, msg=f"moment lam={lam} j={j} m={mom}")
        _record("piece_weights_vs_oracle", max_relative_error=worst)


# ---------------------------------------------------------------------------
# Passive R x S arrays through SparsePassiveRCNetwork
# ---------------------------------------------------------------------------

def _grid_edges(rows, cols, value):
    edges = []
    for r in range(rows):
        for c in range(cols):
            p = r * cols + c
            if c + 1 < cols:
                edges.append((p, p + 1, value))
            if r + 1 < rows:
                edges.append((p, p + cols, value))
    return edges


def build_passive(rows, cols, seed=0):
    rng = np.random.default_rng(seed)
    P, S = rows * cols, cols
    n = P + S
    ground = np.concatenate((np.full(P, 2e-12), np.full(S, 35e-12)))
    cap_edges = _grid_edges(rows, cols, 0.02e-12) + [(P + s, P + s + 1, 0.2e-12) for s in range(S - 1)]
    from sparse_array_rc import passive_matrix
    cmat = passive_matrix(shunts=ground, edges=cap_edges, positive_shunts=False)
    drive_g = np.full(P, 1e-7) * (1 + 0.1 * rng.random(P))
    leak = np.concatenate((np.full(P, 1e-12), np.full(S, 2e-9)))
    on = np.full(P, 2e-6)
    off = np.full(P, 2e-11)
    return dict(P=P, S=S, n=n, ground=ground, cap_edges=cap_edges, cmat=cmat, drive_g=drive_g, leak=leak,
                on=on, off=off, rows=rows, cols=cols)


def passive_g(circuit, row):
    P, S = circuit["P"], circuit["S"]
    tft = circuit["off"].copy()
    tft[row * S:(row + 1) * S] = circuit["on"][row * S:(row + 1) * S]
    edges = [(p, P + p % S, tft[p]) for p in range(P)]
    from sparse_array_rc import passive_matrix
    shunts = circuit["leak"].copy()
    shunts[:P] += circuit["drive_g"]
    return shunts, edges, passive_matrix(shunts=shunts, edges=edges, positive_shunts=False).toarray()


class PassiveArrayTests(unittest.TestCase):
    def test_passive_rows_against_reference_and_oracle(self):
        from sparse_array_rc import SolverControls, SparsePassiveRCNetwork
        summary = {}
        for rows, cols in ((1, 4), (3, 5), (4, 8)):
            circuit = build_passive(rows, cols, seed=rows)
            P, S, n = circuit["P"], circuit["S"], circuit["n"]
            ids = tuple(f"s{i}" for i in range(n))
            reference = SparsePassiveRCNetwork(state_ids=ids, initial_volts=np.zeros(n),
                capacitance_to_reference_farad=circuit["ground"], mutual_capacitance_edges=circuit["cap_edges"])
            controls = SolverControls(relative_tolerance=1e-5, voltage_atol=1e-8, charge_atol_coulomb=1e-20,
                                      max_refinements=8, max_operator_products=200000)
            spec = ArraySpec(rows=rows, cols=cols, branches=0, capacitance=circuit["cmat"].tocsr(),
                             shunts=circuit["leak"], static_edges=np.zeros((0, 3)), tft_on=circuit["on"],
                             tft_off=circuit["off"], star_h=circuit["drive_g"][:, None], star_r=np.zeros(P),
                             buffers=np.zeros(0, np.int64), filters=np.zeros(0, np.int64),
                             holds=np.zeros(0, np.int64), gain=np.zeros(0), offset=np.zeros(0),
                             lower=np.zeros(0), upper=np.zeros(0), gb=np.zeros(0))
            fast = FastArraySolver(spec, degree=8, wr_tolerance=1e-17, wr_max_sweeps=40, grade_levels=10,
                                   energy_audit=False)
            fast.set_state(np.zeros(n))
            x_oracle = np.zeros(n)
            cmat = circuit["cmat"].toarray()
            rng = np.random.default_rng(7)
            t = 0.0
            worst_oracle = worst_ref_scaled = 0.0
            schedule = [(r % rows, d) for r, d in zip(range(6), (5e-7, 7.5e-7, 5e-7, 2e-6, 1e-6, 3e-6))]
            for index, (row, duration) in enumerate(schedule):
                field = 1e-4 * np.sin(np.arange(P) + 0.3 * index) + 2e-5 * rng.standard_normal(P)
                current = np.zeros(n)
                current[:P] = circuit["drive_g"] * field
                charge = np.zeros(n)
                if index == 0 or schedule[index - 1][0] != row:
                    charge[P:] = 3.5e-17 * np.where(np.arange(S) % 2 == 0, 1.0, -1.0)
                shunts, edges, gmat = passive_g(circuit, row)
                observation = sp.csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
                result = reference.advance(duration_seconds=duration, conductance_to_reference_siemens=shunts,
                    resistor_edges=edges, drive_current_amp=current, observation_matrix=observation,
                    feedthrough_volts=np.zeros(n), injected_charge_coulomb=charge, controls=controls)
                x_start = x_oracle + np.linalg.solve(cmat, charge)
                x_oracle, mean_oracle = oracle_interval(cmat, np.zeros((n, n)), gmat, x_start, current[:, None],
                                                        duration)
                piece = fast.advance(gate_row=row, t_start=t, t_end=t + duration,
                                     inputs=PieceInputs(current=current[:, None], amplifier=np.zeros((64, 1))),
                                     charge=charge, grade=True)
                t += duration
                x_fast = fast.state()
                ref_end = np.asarray(result.observation.end_volts)
                worst_oracle = max(worst_oracle, float(np.abs(x_fast - x_oracle).max()))
                scale = controls.voltage_atol + controls.relative_tolerance * np.abs(ref_end)
                worst_ref_scaled = max(worst_ref_scaled, float(np.max(np.abs(x_fast - ref_end) / scale)))
                self.assertLess(np.abs(x_fast - x_oracle).max(), ORACLE_TOL, msg=f"{rows}x{cols} interval {index}")
                self.assertLessEqual(np.max(np.abs(x_fast - ref_end) / scale), 1.0)
            summary[f"{rows}x{cols}"] = dict(max_abs_vs_oracle=worst_oracle, max_scaled_vs_reference=worst_ref_scaled,
                                             max_state=float(np.abs(x_oracle).max()), stats=dict(fast.stats))
        _record("passive_rows", **summary)


# ---------------------------------------------------------------------------
# Conditioned R x 64 arrays through ConditionedRCNetwork
# ---------------------------------------------------------------------------

def build_conditioned(rows, branches):
    from analog_conditioning import AnalogConditioning, SourceConditioning
    from conditional_acquisition_recipe import build_conditional_frontend
    from conditional_forward_recipe import CONDITIONING, REFERENCE_ID
    from sparse_array_rc import PassiveStarConductance, passive_matrix
    cols, K = 64, branches
    P, S = rows * cols, cols
    up = P + S + P * K
    frontend = build_conditional_frontend()
    n = up + 128 + len(frontend.groups)
    ground = np.zeros(n)
    ground[:P] = 2e-12
    ground[P:P + S] = 30e-12 + 5e-12
    shunts = np.zeros(n)
    shunts[:P] = 1e-12
    shunts[P:P + S] = 1e-9 + 1e-9
    cap_edges = _grid_edges(rows, cols, 0.02e-12) + [(P + s, P + s + 1, 0.2e-12) for s in range(S - 1)]
    poles = np.geomspace(2 * np.pi * 1e-2, 2 * np.pi * 1e7, K) if K else np.zeros(0)
    g_branch = 2e-8 / np.pi * 1.7 * np.sqrt(poles)
    c_branch = g_branch / poles if K else np.zeros(0)
    groups = []
    for p in range(P):
        nodes = [p]
        weights = [5e-9]
        for i in range(K):
            node = P + S + K * p + i
            nodes.append(node)
            weights.append(float(g_branch[i]))
            cap_edges.append((node, p, float(c_branch[i])))
        groups.append((tuple(nodes), tuple(weights), 2000.0))
    stars = PassiveStarConductance(state_count=n, groups=groups)
    conditioning = AnalogConditioning(channels=(SourceConditioning(**CONDITIONING),) * 64,
        source_load_role="additional_fixed_shunt_excludes_conditioning",
        source_capacitance_role="additional_fixed_capacitance_excludes_conditioning",
        voltage_reference_id=REFERENCE_ID)
    stamp = conditioning.stamp(state_count=n, source_node_offset=P, first_conditioning_state=up,
                               voltage_reference_id=REFERENCE_ID)
    ground[up:up + 128] = stamp.added_capacitance_farad
    shunts[up:up + 128] = stamp.added_shunts_siemens
    holds = np.arange(up + 128, n)
    for index, group in enumerate(frontend.groups):
        ground[holds[index]] = group.hold_capacitance_farad
        shunts[holds[index]] = group.hold_leak_conductance_siemens
    cmat = passive_matrix(shunts=ground, edges=cap_edges, positive_shunts=False)
    static = np.array(stamp.resistor_edges, dtype=float)
    return dict(rows=rows, cols=cols, K=K, P=P, S=S, up=up, n=n, ground=ground, shunts=shunts,
                cap_edges=cap_edges, cmat=cmat, stars=stars, drives=stamp.controlled_drives, static=static,
                frontend=frontend, holds=holds, filters=np.array(stamp.output_nodes), buffers=np.arange(up, up + 64),
                on=np.full(P, 2e-6), off=np.full(P, 2e-11), reference_id=REFERENCE_ID)


def conditioned_matrices(circuit, row, selection):
    from sparse_array_rc import passive_matrix
    P, S, n = circuit["P"], circuit["S"], circuit["n"]
    tft = circuit["off"].copy()
    tft[row * S:(row + 1) * S] = circuit["on"][row * S:(row + 1) * S]
    edges = [(p, P + p % S, tft[p]) for p in range(P)] + [tuple(e) for e in circuit["static"]]
    frontend = circuit["frontend"]
    edges += list(frontend.resistor_edges(selected_sources=selection, source_node_offset=P + S + P * circuit["K"] + 64,
                                         first_hold_state=int(circuit["holds"][0])))
    shunts = circuit["shunts"].copy()
    drive = np.zeros(n)
    for node, conductance, volts in frontend.external_drives(selected_sources=selection,
            first_hold_state=int(circuit["holds"][0]), voltage_reference_id=circuit["reference_id"]):
        shunts[node] += conductance
        drive[node] += conductance * volts
    g = passive_matrix(shunts=shunts, edges=edges, positive_shunts=False).toarray() + circuit["stars"] @ np.eye(n)
    return shunts, edges, drive, g


def conditioned_spec(circuit):
    d = circuit["drives"]
    return ArraySpec(rows=circuit["rows"], cols=64, branches=circuit["K"], capacitance=circuit["cmat"].tocsr(),
                     shunts=circuit["shunts"], static_edges=circuit["static"], tft_on=circuit["on"],
                     tft_off=circuit["off"], star_h=circuit["stars"]._h.reshape(circuit["P"], circuit["K"] + 1),
                     star_r=np.full(circuit["P"], 2000.0), buffers=np.array(d._buffer_index),
                     filters=circuit["filters"], holds=circuit["holds"], gain=np.array(d._gain),
                     offset=np.array(d._offset), lower=np.array(d._lower), upper=np.array(d._upper),
                     gb=np.array(d._conductance), frontend=circuit["frontend"],
                     daq_first_state=int(circuit["holds"][0]), daq_source_offset=int(circuit["filters"][0]),
                     stars_object=circuit["stars"])


class _PolyForcing:
    """Reference-side polynomial forcing (currents and amplifier input), zero port work."""

    def __init__(self, drives, currents, amplifier, duration):
        self.drives, self.currents, self.amplifier, self.duration = drives, currents, amplifier, duration

    def _at(self, t):
        s = t / self.duration
        basis = s ** np.arange(self.currents.shape[1])
        return self.currents @ basis, self.drives.with_input_noise_volts(self.amplifier @ basis[:self.amplifier.shape[1]])

    def evaluate(self, t):
        return self._at(t)

    def evaluate_left(self, t):
        return self._at(t)

    def work_rates(self, t, state):
        return np.zeros(4)

    def work_rates_left(self, t, state):
        return np.zeros(4)


class ConditionedArrayTests(unittest.TestCase):
    def test_conditioned_rows_against_reference_and_oracle(self):
        from conditioned_array_solver import ConditionedRCNetwork, ConditionedSolverControls
        from conditional_acquisition_recipe import build_conditional_schedule
        summary = {}
        for rows, branches in ((1, 16), (2, 2), (3, 2), (4, 2)):
            circuit = build_conditioned(rows, branches)
            P, S, n, up = circuit["P"], circuit["S"], circuit["n"], circuit["up"]
            ids = tuple(f"x{i}" for i in range(n))
            reference = ConditionedRCNetwork(state_ids=ids, initial_volts=np.zeros(n),
                capacitance_to_reference_farad=circuit["ground"], mutual_capacitance_edges=circuit["cap_edges"])
            controls = ConditionedSolverControls(relative_tolerance=1e-5, voltage_atol=1e-8,
                charge_atol_coulomb=1e-20, max_refinements=8, max_operator_products=200000,
                linear_relative_tolerance=1e-9, linear_charge_atol_coulomb=1e-22, energy_atol_joule=1e-18,
                max_newton_iterations=12, max_linear_iterations=10000, gmres_restart=40,
                max_workspace_bytes=512 * 1024**2, integration_method="radau_iia_2",
                radau_preconditioner="stage_jacobi" if branches != 16 else "contact_block",
                radau_linear_solver="triangular_contact_schur" if (branches == 16) else "gmres")
            fast = FastArraySolver(conditioned_spec(circuit), degree=8, wr_tolerance=1e-17, wr_max_sweeps=40,
                                   grade_levels=10, energy_audit=False)
            fast.set_state(np.zeros(n))
            # production-like mux slots for the first actions of the production schedule
            actions = []
            for tick in range(6):
                slot = tick % 4
                selection = {}
                for index, group in enumerate(circuit["frontend"].groups):
                    source = index + 24 * slot
                    member = source if source < 64 else None
                    if index == 0:
                        from daq_mux_hold import DAQGroupDrive
                        member = "G1" if slot == 3 else member
                        selection[group.adc_id] = DAQGroupDrive(member, (("G1", 0.005),), circuit["reference_id"])
                    else:
                        selection[group.adc_id] = member
                actions.append((tick, selection))
            cmat = circuit["cmat"].toarray()
            rng = np.random.default_rng(rows)
            x_oracle = np.zeros(n)
            t, worst_oracle, worst_ref, worst_hold = 0.0, 0.0, 0.0, 0.0
            previous = None
            for index, (tick, selection) in enumerate(actions):
                row = (index // 3) % rows
                duration = (5e-7, 7.5e-7, 1.25e-6)[index % 3]
                shunts, edges, ext_drive, gmat = conditioned_matrices(circuit, row, selection)
                field = 1e-4 * np.sin(0.1 * np.arange(P) + index)
                base = circuit["stars"].drive_current(port_volts=field) + ext_drive
                noise = np.zeros((n, 4))
                noise[:up] = 1e-13 * rng.standard_normal((up, 4))
                noise[up:] = 1e-14 * rng.standard_normal((n - up, 4))
                amplifier = 1e-6 * rng.standard_normal((64, 4))
                charge = np.zeros(n)
                if previous is None or previous[0] != row:
                    charge[P:P + S] = 3.5e-17 * np.where(np.arange(S) % 2 == 0, 1.0, -1.0)
                charge[circuit["holds"]] += 2e-17 * (1 if tick % 2 == 0 else -1)
                observation = sp.csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
                forcing = _PolyForcing(circuit["drives"], noise, amplifier, duration)
                result = reference.advance(duration_seconds=duration, conductance_to_reference_siemens=shunts,
                    resistor_edges=edges, drive_current_amp=base, observation_matrix=observation,
                    feedthrough_volts=np.zeros(n), injected_charge_coulomb=charge, controls=controls,
                    controlled_drives=circuit["drives"], check=lambda: None, passive_stars=circuit["stars"],
                    time_varying_forcing=forcing)
                # oracle: linear controlled drive (no clipping in this fixture)
                d = circuit["drives"]
                active = np.zeros((n, n))
                gb, gain = np.array(d._conductance), np.array(d._gain)
                for s in range(64):
                    active[d._buffer_index[s], d._source_index[s]] = gb[s] * gain[s]
                forcing_matrix = noise.copy()
                forcing_matrix[:, 0] += base
                forcing_matrix[d._buffer_index, :] += (gb * gain)[:, None] * amplifier
                forcing_matrix[d._buffer_index, 0] += gb * np.array(d._offset)
                x_start = x_oracle + np.linalg.solve(cmat, charge)
                x_oracle, mean_oracle = oracle_interval(cmat, active, gmat, x_start, forcing_matrix, duration,
                                                        mean_rows=circuit["holds"])
                current = noise.copy()
                current[:, 0] += circuit["stars"].drive_current(port_volts=field)
                piece = fast.advance(gate_row=row, t_start=t, t_end=t + duration,
                                     inputs=PieceInputs(current=current, amplifier=amplifier), charge=charge,
                                     selection=selection, grade=True)
                t += duration
                previous = (row, tick)
                x_fast = fast.state()
                ref_end = np.asarray(result.observation.end_volts)
                worst_oracle = max(worst_oracle, float(np.abs(x_fast - x_oracle).max()))
                worst_hold = max(worst_hold, float(np.abs(piece.hold_mean - mean_oracle).max()))
                scale = 1e-8 + 1e-5 * np.abs(ref_end)
                worst_ref = max(worst_ref, float(np.max(np.abs(x_fast - ref_end) / scale)))
                ref_hold_mean = np.asarray(result.observation.mean_volts)[circuit["holds"]]
                worst_ref = max(worst_ref, float(np.max(np.abs(piece.hold_mean - ref_hold_mean)
                                                       / (1e-8 + 1e-5 * np.abs(ref_hold_mean)))))
            summary[f"{rows}x64_K{branches}"] = dict(max_abs_vs_oracle=worst_oracle, max_hold_mean_vs_oracle=worst_hold,
                                                     max_scaled_vs_reference=worst_ref,
                                                     max_state=float(np.abs(x_oracle).max()),
                                                     stats=dict(fast.stats))
            _record("conditioned_rows_partial", **summary)
            self.assertLess(worst_oracle, ORACLE_TOL, msg=f"rows={rows}")
            self.assertLess(worst_hold, ORACLE_TOL, msg=f"rows={rows}")
            self.assertLessEqual(worst_ref, 1.0, msg=f"rows={rows}")
        _record("conditioned_rows", **summary)


# ---------------------------------------------------------------------------
# 35-ms clipping fixture (test_radau_triangular_contact_schur.finite_problem)
# ---------------------------------------------------------------------------

class ClippingFixtureTests(unittest.TestCase):
    def test_35ms_clipping_fixture(self):
        from conditioned_array_solver import ConditionedRCNetwork
        from test_conditioned_array_solver import controls
        from test_radau_triangular_contact_schur import (BUFFERS, CONTACTS, FILTERS, HOLDS, SIZE, SOURCES,
                                                         finite_problem)
        problem, duration = finite_problem(clipped=True), .035
        base_current = problem["currents"][0]

        def forcing_poly():
            # current(t) = base + (.004 + .003 s) sin(i); amplifier(t) = .003 sin(k), channel0 = .5 - s
            current = np.zeros((SIZE, 2))
            current[:, 0] = base_current + .004 * np.sin(np.arange(SIZE))
            current[:, 1] = .003 * np.sin(np.arange(SIZE))
            amplifier = np.zeros((64, 2))
            amplifier[:, 0] = .003 * np.sin(np.arange(64))
            amplifier[0] = (.5, -1.0)
            return current, amplifier

        current, amplifier = forcing_poly()

        class Forcing:
            def _at(self, time):
                noise = .003 * np.sin(np.arange(64))
                noise[0] = .5 - time / duration
                value = (.004 + .003 * time / duration) * np.sin(np.arange(SIZE))
                return value, problem["drive"].with_input_noise_volts(noise)

            def evaluate(self, time):
                return self._at(time)

            def evaluate_left(self, time):
                return self._at(time)

            def work_rates(self, time, state):
                return np.zeros(4)

            def work_rates_left(self, time, state):
                return np.zeros(4)

        outputs = {}
        for strategy in ("gmres", "triangular_contact_schur"):
            network = ConditionedRCNetwork(state_ids=tuple(f"finite:{i}" for i in range(SIZE)),
                initial_volts=problem["initial"], capacitance_to_reference_farad=problem["ground"],
                mutual_capacitance_edges=problem["cap_edges"])
            result = network.advance(duration_seconds=duration, conductance_to_reference_siemens=problem["shunts"],
                resistor_edges=problem["edges"], drive_current_amp=base_current,
                observation_matrix=sp.csr_matrix(np.eye(SIZE)), feedthrough_volts=np.zeros(SIZE),
                injected_charge_coulomb=problem["injected"], controlled_drives=problem["drive"],
                passive_stars=problem["stars"], time_varying_forcing=Forcing(), check=lambda: None,
                controls=controls(integration_method="radau_iia_2", radau_preconditioner="contact_block",
                                  radau_linear_solver=strategy, max_refinements=12,
                                  max_workspace_bytes=64 * 1024**2))
            outputs[strategy] = result
        # new solver: one upstream block (2 pads x 17 + 64 sources), exact downstream with events
        channels = problem["channels"]
        pad_edges = [e for e in problem["edges"] if e[0] < 2]
        static = np.array([e for e in problem["edges"] if e[0] >= 2], dtype=float)
        spec = ArraySpec(rows=1, cols=64, branches=16, capacitance=problem["sparse_c"], shunts=problem["shunts"],
                         static_edges=static, tft_on=np.array([e[2] for e in pad_edges]),
                         tft_off=np.array([e[2] for e in pad_edges]),
                         star_h=problem["stars"]._h.reshape(2, 17),
                         star_r=np.array([g[2] for g in problem["groups"]]), buffers=BUFFERS, filters=FILTERS,
                         holds=HOLDS, gain=np.array([c.gain for c in channels]),
                         offset=np.array([c.offset_volts for c in channels]),
                         lower=np.array([c.drive_lower_volts for c in channels]),
                         upper=np.array([c.drive_upper_volts for c in channels]),
                         gb=np.array([c.buffer_conductance_siemens for c in channels]), frontend=None,
                         stars_object=problem["stars"], partition="single",
                         tft_pad_source=np.array([e[1] - 2 for e in pad_edges]), n_pads_override=2,
                         n_sources_override=64)
        fast = FastArraySolver(spec, degree=8, wr_tolerance=1e-15, grade_levels=6, energy_audit=False)
        fast.set_state(problem["initial"])
        piece = fast.advance(gate_row=0, t_start=0.0, t_end=duration,
                             inputs=PieceInputs(current=current, amplifier=amplifier),
                             charge=problem["injected"], selection=None, grade=False)
        x_fast = fast.state()
        # exact oracle: full dense system, clip state of channel 0 piecewise, events by bisection
        c, g = problem["c"], problem["g"]
        gb = np.array([ch.buffer_conductance_siemens for ch in channels])
        gain = np.array([ch.gain for ch in channels])

        def generator(state0):
            active = np.zeros((SIZE, SIZE))
            forcing = current.copy()
            for k in range(64):
                if k == 0 and state0 != "linear":
                    rail = channels[0].drive_upper_volts if state0 == "upper" else channels[0].drive_lower_volts
                    forcing[BUFFERS[0], 0] += gb[0] * rail
                else:
                    active[BUFFERS[k], SOURCES[k]] = gb[k] * gain[k]
                    forcing[BUFFERS[k]] += gb[k] * gain[k] * amplifier[k]
            return active, forcing

        def propagate(x0, t0, t1, state0):
            active, forcing = generator(state0)
            # forcing is polynomial in s = t/duration: re-express in local (t - t0)/(t1 - t0)
            local = shift_monomials(forcing, t0 / duration, (t1 - t0) / duration)
            return oracle_interval(c, active, g, x0, local, t1 - t0, mean_rows=HOLDS)

        def requested(x, t):
            return gain[0] * (x[SOURCES[0]] + (.5 - t / duration))

        x0 = problem["initial"] + np.linalg.solve(c, problem["injected"])
        state0 = "upper" if requested(x0, 0.0) >= channels[0].drive_upper_volts else "linear"
        t0, events, x_cur, hold_integral = 0.0, [], x0, np.zeros(2)
        grid = np.linspace(0, duration, 351)
        while t0 < duration:
            # advance on the grid until the clip state changes, then bisect
            nxt = duration
            for t1 in grid[grid > t0]:
                xt, _ = propagate(x_cur, t0, t1, state0)
                r = requested(xt, t1)
                new_state = "upper" if r >= channels[0].drive_upper_volts else (
                    "lower" if r <= channels[0].drive_lower_volts else "linear")
                if new_state != state0:
                    lo, hi = max(t0, t1 - duration / 350), t1
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        xm, _ = propagate(x_cur, t0, mid, state0)
                        rm = requested(xm, mid)
                        sm = "upper" if rm >= channels[0].drive_upper_volts else (
                            "lower" if rm <= channels[0].drive_lower_volts else "linear")
                        if sm == state0:
                            lo = mid
                        else:
                            hi = mid
                    nxt = 0.5 * (lo + hi)
                    break
            xt, mean = propagate(x_cur, t0, nxt, state0)
            hold_integral += mean * (nxt - t0)
            if nxt < duration:
                events.append(nxt)
                rm = requested(xt, nxt + 1e-12)
                state0 = "upper" if rm >= channels[0].drive_upper_volts else (
                    "lower" if rm <= channels[0].drive_lower_volts else "linear")
            x_cur, t0 = xt, nxt
        hold_mean_oracle = hold_integral / duration
        oracle_err = float(np.abs(x_fast - x_cur).max())
        hold_err = float(np.abs(piece.hold_mean - hold_mean_oracle).max())
        ref_scaled = {}
        for strategy, result in outputs.items():
            ref_end = np.asarray(result.observation.end_volts)
            ref_scaled[strategy] = float(np.max(np.abs(x_fast - ref_end) / (4e-8 + 2e-5 * np.abs(ref_end))))
        _record("clipping_fixture_35ms", max_abs_vs_oracle=oracle_err, max_hold_mean_vs_oracle=hold_err,
                oracle_events=events, fast_events=int(piece.clip_events), max_state=float(np.abs(x_cur).max()),
                max_scaled_vs_reference=ref_scaled, reference_subintervals={k: v.subintervals for k, v in outputs.items()})
        self.assertEqual(len(events), piece.clip_events)
        self.assertLess(oracle_err, 1e-10)
        for value in ref_scaled.values():
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
