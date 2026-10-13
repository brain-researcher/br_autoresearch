"""Optional exact two-stage solve for the declared feed-forward contact circuit.

This helper owns no physical state or acceptance decision. It returns all
original-coordinate stages; the caller MUST apply its original full C/G,
charge/voltage residual, work, refinement and commit checks. Unsupported
structure returns None during preparation. Numerical/resource failures raise.

The selected 4096-pad circuit reduces to 8320 real retained stage variables.
Column ordering bounds their band by 131; pivoted general-band LU has bounded
fill. Relative contact coordinates avoid subtracting branch capacitances in
the Schur complement. Their capacitance is transformed from the actual matrix,
never reconstructed from nominal recipe values. No scientific result is implied.
"""
from dataclasses import dataclass
import math
from time import perf_counter
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, get_lapack_funcs, lu_factor, lu_solve
from scipy.sparse import isspmatrix_csc, isspmatrix_csr

from analog_conditioning import ControlledVoltageDrives
from sparse_array_rc import PassiveStarConductance, SolverConvergenceError, _ConductanceWithStars, _array


BRANCHES = 16
MAX_CONTACTS = 4096
MAX_GROUPS = 32
MAX_BAND = 131
_A = ((5. / 12., -1. / 12.), (.75, .25))


def estimate_triangular_workspace_bytes(*, state_count, contact_count, sparse_nonzeros):
    """Added peak numeric bound; the caller also retains its original workspace.

    Inventory (all integer/pivot elements conservatively charged as 8 bytes):
    * 16*(N+Z): sparse copies/COO views, structural masks/indices, owner maps,
      transformed retained C/G, full diagonals and preparation scratch.
    * 3*(3*b+1)*R + 2*(2*b+1)*R: three expanded general-band arrays and two
      compact-band equivalents, covering assembly, Fortran/LAPACK copies and
      factor output. Old exact-h factors are dropped before replacement.
    * 48*N: full stage/current/state/RHS/recovery vectors and simultaneous
      gather, scaling, finite-check and transform temporaries.
    * 16*m*q: grouped contact RHS, solutions, two pad coupling directions,
      recovery and grouping indices; q=32 includes both stages.
    * 6*d*d: downstream matrices, equilibration, factor/library copies.
    * K*(8*q*q+16*q): at most 32 distinct contact matrices, local factors,
      coupling solves, exact grouping keys and scaling/pivot vectors.
    * 4*R+2*d: band/downstream scaling, pivots and small solve work.

    Preparation visits one 17-node contact at a time: it never allocates an
    m-by-q-by-q tensor. Library/Python object overhead remains RSS-bound. The
    existing full capacitance factor stays in its original assembly budget.
    """
    if (any(type(v) is not int or v < 0 for v in (state_count, contact_count, sparse_nonzeros))
            or not 1 <= contact_count <= MAX_CONTACTS):
        raise ValueError("explicit bounded triangular dimensions required")
    n, m, z = state_count, contact_count, sparse_nonzeros
    r, q = 2 * (m + 64), 2 * BRANCHES
    d = 2 * (n - (m + 64 + m * BRANCHES))
    if not 256 <= d <= 304:
        raise ValueError("triangular downstream basis must contain 64 buffers/filters and at most 24 holds")
    b = min(MAX_BAND, r - 1)
    return 8 * (16 * (n + z) + 3 * (3 * b + 1) * r + 2 * (2 * b + 1) * r
        + 48 * n + 16 * m * q + 6 * d * d
        + MAX_GROUPS * (8 * q * q + 16 * q) + 4 * r + 2 * d)


def _finite(value, message):
    if not np.all(np.isfinite(value)):
        raise SolverConvergenceError(message)
    return value


def _dense_factor(matrix, check):
    check()
    matrix = np.array(matrix, dtype=float, order="F", copy=True)
    _finite(matrix, "nonfinite triangular dense matrix")
    rows = np.max(np.abs(matrix), axis=1)
    if np.any(rows <= 0):
        raise SolverConvergenceError("singular triangular dense row")
    matrix /= rows[:, None]
    columns = np.max(np.abs(matrix), axis=0)
    if np.any(columns <= 0):
        raise SolverConvergenceError("singular triangular dense column")
    matrix /= columns[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            lu, pivots = lu_factor(matrix, overwrite_a=True, check_finite=False)
        except (LinAlgWarning, ValueError, np.linalg.LinAlgError) as error:
            raise SolverConvergenceError("triangular dense factorization failed") from error
    _finite(lu, "nonfinite triangular dense factor")
    if np.any(np.diag(lu) == 0):
        raise SolverConvergenceError("singular triangular dense factor")
    check()
    return lu, pivots, rows, columns


def _dense_solve(factor, rhs, check):
    check()
    lu, pivots, rows, columns = factor
    right = np.array(rhs, dtype=float, order="F", copy=True)
    right /= rows if right.ndim == 1 else rows[:, None]
    _finite(right, "nonfinite scaled triangular right-hand side")
    try:
        value = lu_solve((lu, pivots), right, overwrite_b=True, check_finite=False)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise SolverConvergenceError("triangular dense solve failed") from error
    value /= columns if value.ndim == 1 else columns[:, None]
    _finite(value, "nonfinite triangular dense solution")
    check()
    return value


@dataclass(frozen=True)
class _ContactGroup:
    pads: np.ndarray
    nodes: np.ndarray
    capacitance: np.ndarray
    conductance: np.ndarray
    coupling: np.ndarray


class TriangularContactSchurPlan:
    """One physical-interval plan with one exact-h factor set, not a global cache."""

    def __init__(self, *, state_count, contacts, c_reduced, g_reduced, c_downstream,
                 g_downstream, position, groups, source_nodes, buffer_nodes, buffer_conductances, workspace):
        self.state_count, self.contact_count = state_count, contacts
        self.retained_count = contacts + 64
        self.upstream_count = self.retained_count + contacts * BRANCHES
        self.estimated_workspace_bytes = workspace
        self._c, self._g = c_reduced, g_reduced
        self._cd, self._gd = c_downstream, g_downstream
        self._position, self._groups = position, tuple(groups)
        self._sources, self._buffers = source_nodes, buffer_nodes
        self._buffer_conductances = buffer_conductances
        self._bandwidth = min(MAX_BAND, 2 * self.retained_count - 1)
        self._cached_step, self._factors = None, None
        self.factor_builds = self.contact_factorizations = 0
        self.band_factorizations = self.downstream_factorizations = 0
        self.stage_solves = self.band_solves = self.contact_solves = self.downstream_solves = 0
        self.factor_build_seconds = self.solve_seconds = 0.0

    @property
    def metadata(self):
        return dict(state_count=self.state_count, contact_count=self.contact_count,
            retained_stage_dimension=2 * self.retained_count,
            downstream_stage_dimension=2 * (self.state_count - self.upstream_count),
            lower_bandwidth=self._bandwidth, upper_bandwidth=self._bandwidth,
            distinct_contact_blocks=len(self._groups),
            estimated_workspace_bytes=self.estimated_workspace_bytes,
            factor_builds=self.factor_builds, contact_factorizations=self.contact_factorizations,
            band_factorizations=self.band_factorizations,
            downstream_factorizations=self.downstream_factorizations,
            stage_solves=self.stage_solves, band_solves=self.band_solves,
            contact_solves=self.contact_solves, downstream_solves=self.downstream_solves,
            factor_build_seconds=self.factor_build_seconds, solve_seconds=self.solve_seconds,
            timing_scope="nested_direct_solver_timings_not_additive",
            cached_step_seconds=self._cached_step,
            scope="original_states_recovered_original_residual_work_refinement_required")

    def _prepare_step(self, step, check):
        if self._cached_step == step:
            return
        self._cached_step, self._factors = None, None
        started = perf_counter()
        try:
            check()
            count, width = 2 * self.retained_count, self._bandwidth
            band = np.zeros((3 * width + 1, count), order="F")
            center = 2 * width
            for matrix, is_capacitance in ((self._c, True), (self._g, False)):
                coo = matrix.tocoo(copy=False)
                rows, columns = 2 * self._position[coo.row], 2 * self._position[coo.col]
                for i in range(2):
                    for j in range(2):
                        if is_capacitance and i != j:
                            continue
                        check()
                        coefficient = 1.0 if is_capacitance else step * _A[i][j]
                        band[center + rows + i - columns - j, columns + j] += coefficient * coo.data
            contact_factors = []
            for group in self._groups:
                check()
                c, g, coupling = np.diag(group.capacitance), group.conductance, group.coupling
                matrix = np.block([[c + step * _A[0][0] * g, step * _A[0][1] * g],
                                   [step * _A[1][0] * g, c + step * _A[1][1] * g]])
                # B and D have distinct stage orientation: Radau A is not symmetric.
                b = np.block([[step * _A[0][0] * coupling[None, :], step * _A[0][1] * coupling[None, :]],
                              [step * _A[1][0] * coupling[None, :], step * _A[1][1] * coupling[None, :]]])
                d = np.block([[step * _A[0][0] * coupling[:, None], step * _A[0][1] * coupling[:, None]],
                              [step * _A[1][0] * coupling[:, None], step * _A[1][1] * coupling[:, None]]])
                factor = _dense_factor(matrix, check)
                self.contact_factorizations += 1
                inverse_d = _dense_solve(factor, d, check)
                self.contact_solves += 1
                correction = b @ inverse_d
                _finite(correction, "nonfinite contact Schur correction")
                indices = 2 * self._position[group.pads]
                for i in range(2):
                    for j in range(2):
                        band[center + i - j, indices + j] -= correction[i, j]
                contact_factors.append((factor, b, inverse_d))
            _finite(band, "nonfinite reduced triangular band")
            row_scale = np.zeros(count)
            for offset in range(-width, width + 1):
                start, end = max(0, -offset), min(count, count - offset)
                row_scale[start + offset:end + offset] = np.maximum(
                    row_scale[start + offset:end + offset], np.abs(band[center + offset, start:end]))
            if np.any(row_scale <= 0):
                raise SolverConvergenceError("singular triangular band row")
            column_scale = np.zeros(count)
            for offset in range(-width, width + 1):
                start, end = max(0, -offset), min(count, count - offset)
                band[center + offset, start:end] /= row_scale[start + offset:end + offset]
                column_scale[start:end] = np.maximum(column_scale[start:end],
                    np.abs(band[center + offset, start:end]))
            if np.any(column_scale <= 0):
                raise SolverConvergenceError("singular triangular band column")
            band /= column_scale[None, :]
            _finite(band, "nonfinite equilibrated triangular band")
            check()
            gbtrf = get_lapack_funcs("gbtrf", (band,))
            try:
                band_lu, pivots, info = gbtrf(band, width, width, overwrite_ab=True)
            except (ValueError, np.linalg.LinAlgError) as error:
                raise SolverConvergenceError("triangular band factorization failed") from error
            if info != 0:
                raise SolverConvergenceError("triangular band factorization refused")
            _finite(band_lu, "nonfinite triangular band factor")
            self.band_factorizations += 1
            check()
            downstream = np.block([
                [self._cd + step * _A[0][0] * self._gd, step * _A[0][1] * self._gd],
                [step * _A[1][0] * self._gd, self._cd + step * _A[1][1] * self._gd]])
            downstream_factor = _dense_factor(downstream, check)
            self.downstream_factorizations += 1
            self._factors = (tuple(contact_factors), (band_lu, pivots, row_scale, column_scale), downstream_factor)
            self._cached_step = step
            self.factor_builds += 1
        finally:
            self.factor_build_seconds += perf_counter() - started

    def solve_stages(self, *, step_seconds, old, stage_currents, stage_drives, passive_product, check):
        """Return (2,N) full stages; caller must verify original physical residuals.

        No forcing callback is re-evaluated here. The two already-resolved
        snapshots preserve the caller's evaluate/evaluate_left ordering.
        """
        if (isinstance(step_seconds, (bool, np.bool_)) or not math.isfinite(step_seconds)
                or step_seconds <= 0 or not callable(check) or not callable(passive_product)):
            raise ValueError("finite positive step and checked original passive operator required")
        if len(stage_currents) != 2 or len(stage_drives) != 2:
            raise ValueError("exactly two resolved stage snapshots required")
        for drives in stage_drives:
            if (type(drives) is not ControlledVoltageDrives or drives.state_count != self.state_count
                    or drives.source_nodes != self._sources or drives.buffer_nodes != self._buffers
                    or drives.buffer_conductance_siemens != self._buffer_conductances):
                raise ValueError("triangular stage drive must preserve its verified dependency basis")
        started = perf_counter()
        try:
            check()
            step = float(step_seconds)
            old = _array("triangular post-charge state", old, (self.state_count,))
            currents = tuple(_array("triangular stage current", value, (self.state_count,)) for value in stage_currents)
            self._prepare_step(step, check)
            factors, band_factor, downstream_factor = self._factors
            passive_old = _array("triangular original passive product", passive_product(old), (self.state_count,))
            force = np.vstack((currents[0] - passive_old, currents[1] - passive_old))
            rhs = step * np.vstack((_A[0][0] * force[0] + _A[0][1] * force[1],
                                    _A[1][0] * force[0] + _A[1][1] * force[1]))
            _finite(rhs, "nonfinite triangular increment right-hand side")
            retained = rhs[:, :self.retained_count].copy()
            contact_rhs = rhs[:, self.retained_count:self.upstream_count].reshape(2, self.contact_count, BRANCHES)
            retained[:, :self.contact_count] += np.sum(contact_rhs, axis=2)
            solved_contacts = []
            for group, (factor, b, inverse_d) in zip(self._groups, factors):
                check()
                local_rhs = np.concatenate((rhs[0, group.nodes].T, rhs[1, group.nodes].T), axis=0)
                solved = _dense_solve(factor, local_rhs, check)
                self.contact_solves += 1
                retained[:, group.pads] -= b @ solved
                solved_contacts.append(solved)
            count, width = 2 * self.retained_count, self._bandwidth
            band_lu, pivots, row_scale, column_scale = band_factor
            right = np.empty((count, 1), order="F")
            right[2 * self._position, 0] = retained[0]
            right[2 * self._position + 1, 0] = retained[1]
            right[:, 0] /= row_scale
            _finite(right, "nonfinite reduced triangular right-hand side")
            check()
            gbtrs = get_lapack_funcs("gbtrs", (band_lu,))
            try:
                solution, info = gbtrs(band_lu, width, width, right, pivots, trans=0, overwrite_b=True)
            except (ValueError, np.linalg.LinAlgError) as error:
                raise SolverConvergenceError("triangular band solve failed") from error
            if info != 0:
                raise SolverConvergenceError("triangular band solve refused")
            solution[:, 0] /= column_scale
            _finite(solution, "nonfinite triangular band solution")
            self.band_solves += 1
            check()
            increments = np.zeros((2, self.state_count))
            increments[0, :self.retained_count] = solution[2 * self._position, 0]
            increments[1, :self.retained_count] = solution[2 * self._position + 1, 0]
            for group, (_, _, inverse_d), solved in zip(self._groups, factors, solved_contacts):
                check()
                relative = solved - inverse_d @ increments[:, group.pads]
                increments[0, group.nodes] = relative[:BRANCHES].T + increments[0, group.pads, None]
                increments[1, group.nodes] = relative[BRANCHES:].T + increments[1, group.pads, None]
            stages = old[None, :] + increments
            # The amplifier depends on recovered upstream sources, including
            # stage-specific clipping/noise; downstream state does not feed it.
            active = []
            for index, drives in enumerate(stage_drives):
                check()
                evaluation = drives.evaluate(stages[index])
                if tuple(evaluation.node_indices) != self._buffers:
                    raise ValueError("triangular controlled node order changed")
                active.append(evaluation.current_amp)
            downstream_rhs = rhs[:, self.upstream_count:].copy()
            downstream_buffer = np.asarray(self._buffers, dtype=np.int64) - self.upstream_count
            for index in range(2):
                downstream_rhs[index, downstream_buffer] += step * (
                    _A[index][0] * active[0] + _A[index][1] * active[1])
            solved_downstream = _dense_solve(downstream_factor, downstream_rhs.ravel(), check)
            self.downstream_solves += 1
            stages[:, self.upstream_count:] = old[None, self.upstream_count:] + solved_downstream.reshape(2, -1)
            _finite(stages, "nonfinite recovered original triangular stages")
            check()
            self.stage_solves += 1
            return stages
        finally:
            self.solve_seconds += perf_counter() - started


def prepare_triangular_contact_schur(capacitance, passive_conductance, controlled_drives,
                                    *, max_workspace_bytes, check, fallback_info=None):
    """Admit actual structure before boundary mutation, or return None.

    Layout is the production pad/source/contact/buffer/filter/hold order. A
    finite prefix of its pads is accepted for independent small algebra tests;
    production still uses all 4096. Every actual nonzero is inspected. No edge,
    transformed remainder, state or small coefficient is tolerance-zeroed.
    """
    if type(max_workspace_bytes) is not int or max_workspace_bytes < 0 or not callable(check):
        raise ValueError("explicit remaining workspace and callback required")
    if fallback_info is not None and not isinstance(fallback_info, dict):
        raise ValueError("fallback_info must be a caller-owned dictionary")

    def unsupported(reason):
        if fallback_info is not None:
            fallback_info.update(supported=False, reason=reason)
        return None

    check()
    if (type(passive_conductance) is not _ConductanceWithStars
            or type(passive_conductance.stars) is not PassiveStarConductance
            or type(controlled_drives) is not ControlledVoltageDrives):
        return unsupported("requires_owned_passive_stars_and_controlled_drive_dependency")
    base, stars = passive_conductance.base, passive_conductance.stars
    if any(not (isspmatrix_csr(item) or isspmatrix_csc(item)) or not item.has_canonical_format
           for item in (capacitance, base)):
        return unsupported("requires_canonical_CSR_or_CSC_actual_matrices")
    n, m = controlled_drives.state_count, stars.port_count
    if (not 1 <= m <= MAX_CONTACTS or stars.term_count != 17 * m
            or any(item.shape != (n, n) for item in (capacitance, base, stars))):
        return unsupported("requires_disjoint_sixteen_branch_contacts_in_one_basis")
    retained, upstream = m + 64, m + 64 + BRANCHES * m
    sources, buffers = tuple(range(m, retained)), tuple(range(upstream, upstream + 64))
    if (not upstream + 128 <= n <= upstream + 152
            or controlled_drives.source_nodes != sources or controlled_drives.buffer_nodes != buffers):
        return unsupported("requires_declared_pad_source_contact_buffer_filter_hold_order")
    workspace = estimate_triangular_workspace_bytes(state_count=n, contact_count=m,
        sparse_nonzeros=int(capacitance.nnz + base.nnz))
    # This precedes every allocation proportional to the basis or sparse data.
    if workspace > max_workspace_bytes:
        raise SolverConvergenceError("triangular contact-Schur numeric-workspace limit exceeded")
    for matrix in (capacitance, base):
        if np.iscomplexobj(matrix.data) or not np.all(np.isfinite(matrix.data)):
            raise SolverConvergenceError("nonfinite or complex triangular matrix")
        if (matrix != matrix.T).nnz:
            return unsupported("requires_exact_symmetric_actual_passive_matrices")
    nodes = stars._nodes.reshape(m, 17)
    expected_contact = np.arange(retained, upstream, dtype=np.int64).reshape(m, BRANCHES)
    if (not np.array_equal(nodes[:, 0], np.arange(m))
            or not np.array_equal(nodes[:, 1:], expected_contact)
            or not np.array_equal(stars._reference, np.arange(m))
            or not np.array_equal(stars._group.reshape(m, 17), np.broadcast_to(np.arange(m)[:, None], (m, 17)))):
        return unsupported("requires_complete_disjoint_ordered_contact_groups")
    cdiag = capacitance.diagonal()
    if np.any(cdiag <= 0) or not np.all(np.isfinite(cdiag)):
        raise SolverConvergenceError("positive finite actual capacitance diagonal required")
    # Inspect both directions, including stored explicit zeros without inventing
    # any numerical sparsity threshold. Contact rows touch only their own pad.
    for matrix, is_capacitance in ((capacitance, True), (base, False)):
        check()
        coo = matrix.tocoo(copy=False)
        nonzero = coo.data != 0
        row_up, col_up = coo.row < upstream, coo.col < upstream
        if np.any(nonzero & (row_up != col_up)):
            return unsupported("passive_upstream_downstream_coupling")
        row_q = (coo.row >= retained) & row_up
        col_q = (coo.col >= retained) & col_up
        if not is_capacitance:
            if np.any(nonzero & (row_q | col_q)):
                return unsupported("extra_base_conductance_on_contact_states")
            continue
        row_owner = np.maximum(coo.row - retained, 0) // BRANCHES
        col_owner = np.maximum(coo.col - retained, 0) // BRANCHES
        diagonal = coo.row == coo.col
        if (np.any(nonzero & row_q & ~diagonal & (coo.col != row_owner))
                or np.any(nonzero & col_q & ~diagonal & (coo.row != col_owner))):
            return unsupported("contact_capacitance_outside_own_pad_branch")
        coupling = row_q & (coo.col == row_owner) & nonzero
        if (np.count_nonzero(coupling) != BRANCHES * m
                or not np.array_equal(coo.data[coupling], -cdiag[coo.row[coupling]])):
            return unsupported("contact_capacitance_is_not_exact_relative_branch_stamp")
    c_reduced = capacitance[:retained, :retained].tocsr(copy=True)
    g_reduced = base[:retained, :retained].tocsr(copy=True)
    c_reduced_diagonal, g_reduced_diagonal = c_reduced.diagonal(), g_reduced.diagonal()
    memberships, representatives, mapping = [], [], {}
    weights = stars._h.reshape(m, 17)
    port_drive = stars._b.reshape(m, 17)
    for pad in range(m):
        if pad % 64 == 0:
            check()
        ci = cdiag[expected_contact[pad]].copy()
        # Congruence of the ACTUAL assembled C: this retains its rounding and
        # every nonzero remainder; no nominal ground capacitance is substituted.
        c_reduced_diagonal[pad] = math.fsum([float(cdiag[pad]), *(-float(v) for v in ci)])
        g_reduced_diagonal[pad] += math.fsum(float(v) for v in port_drive[pad])
        h, total, mix = weights[pad, 1:], stars._totals[pad], stars._mix[pad]
        fraction = h / total if total > 0 else np.zeros(BRANCHES)
        gi = -h[:, None] * mix * fraction[None, :]
        gi[np.diag_indices(BRANCHES)] = stars._diagonal[expected_contact[pad]]
        coupling = port_drive[pad, 1:].copy()
        _finite(gi, "nonfinite transformed contact conductance")
        key = (ci.tobytes(), gi.tobytes(), coupling.tobytes())
        group = mapping.get(key)
        if group is None:
            if len(mapping) == MAX_GROUPS:
                return unsupported("more_than_32_distinct_exact_contact_blocks")
            group = len(mapping)
            mapping[key] = group
            representatives.append((ci, gi, coupling))
            memberships.append([])
        memberships[group].append(pad)
    _finite(c_reduced_diagonal, "nonfinite transformed retained capacitance")
    if np.any(c_reduced_diagonal <= 0):
        raise SolverConvergenceError("nonpositive actual relative-coordinate retained capacitance")
    _finite(g_reduced_diagonal, "nonfinite transformed retained conductance")
    c_reduced.setdiag(c_reduced_diagonal)
    g_reduced.setdiag(g_reduced_diagonal)
    c_reduced.eliminate_zeros()
    g_reduced.eliminate_zeros()
    order = np.array([node for column in range(64)
        for node in (*range(column, m, 64), m + column)], dtype=np.int64)
    position = np.empty(retained, dtype=np.int64)
    position[order] = np.arange(retained)
    width = min(MAX_BAND, 2 * retained - 1)
    for matrix, extra in ((c_reduced, 0), (g_reduced, 1)):
        coo = matrix.tocoo(copy=False)
        if np.any(2 * np.abs(position[coo.row] - position[coo.col]) + extra > width):
            return unsupported("retained_coupling_outside_admitted_stage_band")
    groups = []
    for pads, (ci, gi, coupling) in zip(memberships, representatives):
        pads = np.array(pads, dtype=np.int64)
        contact_nodes = expected_contact[pads].copy()
        for value in (pads, contact_nodes, ci, gi, coupling):
            value.flags.writeable = False
        groups.append(_ContactGroup(pads, contact_nodes, ci, gi, coupling))
    cd = capacitance[upstream:, upstream:].toarray()
    gd = base[upstream:, upstream:].toarray()
    for matrix in (c_reduced, g_reduced):
        for value in (matrix.data, matrix.indices, matrix.indptr):
            value.flags.writeable = False
    for value in (position, cd, gd):
        value.flags.writeable = False
    plan = TriangularContactSchurPlan(state_count=n, contacts=m, c_reduced=c_reduced,
        g_reduced=g_reduced, c_downstream=cd, g_downstream=gd, position=position,
        groups=groups, source_nodes=sources, buffer_nodes=buffers,
        buffer_conductances=controlled_drives.buffer_conductance_siemens, workspace=workspace)
    if fallback_info is not None:
        fallback_info.update(supported=True, reason="actual_triangular_relative_contact_band_structure")
    check()
    return plan
