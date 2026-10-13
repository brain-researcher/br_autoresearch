"""Bounded contact-block inverse for an unchanged full-state Radau Jacobian.

Only disjoint 17-node PassiveStarConductance groups are supported. Each block
retains the exact principal capacitance and passive conductance, including
external-edge diagonal contributions. Cross-block and controlled-drive terms
remain in the caller's original Jacobian and residual. No state is eliminated
from the physical solve. Unsupported layouts use the caller's stage Jacobi.
"""
from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.sparse import isspmatrix_csc, isspmatrix_csr
from scipy.sparse.linalg import LinearOperator

from sparse_array_rc import PassiveStarConductance, SolverConvergenceError, _ConductanceWithStars, _array


CONTACT_SIZE = 17
MAX_DISTINCT_BLOCKS = 32


def estimate_contact_workspace_bytes(*, state_count, contact_count, sparse_nonzeros):
    """Conservative added peak, including preparation and one exact-h cache.

    Covers dense local C/G and star construction temporaries, sparse extraction
    indices/masks, grouping keys/copies, grouped LU/scales/pivots, and full-state
    fallback/gather/solve/scatter vectors. Inputs already owned by the solver
    are excluded. Python/library overhead remains under the process RSS cap.
    This is added to, never substituted for, the existing Radau workspace.
    """
    if any(type(value) is not int or value < 0 for value in (state_count, contact_count, sparse_nonzeros)):
        raise ValueError("contact workspace dimensions must be nonnegative integers")
    k, stages = CONTACT_SIZE, 2 * CONTACT_SIZE
    return 8 * (12 * contact_count * k * k + 24 * (state_count + sparse_nonzeros)
                + MAX_DISTINCT_BLOCKS * (4 * stages * stages + 10 * stages)
                + 24 * contact_count * k + 48 * state_count)


@dataclass(frozen=True)
class _BlockGroup:
    nodes: np.ndarray
    capacitance: np.ndarray
    conductance: np.ndarray


class ContactBlockPlan:
    """Owned local matrices; one step-size factor set, never a topology cache.

    Construct through prepare_contact_blocks once for the current physical
    interval. A returned operator becomes invalid if for_step replaces its
    exact-h factor set; overlapping use is unsupported and explicitly refused.
    """
    def __init__(self, *, state_count, groups, workspace):
        self.state_count = state_count
        self.estimated_workspace_bytes = workspace
        self._groups = tuple(groups)
        self._cached_step = None
        self._factors = ()
        self.factorizations = 0
        self.applications = 0

    @property
    def metadata(self):
        return dict(contact_count=sum(len(group.nodes) for group in self._groups),
            contact_size=CONTACT_SIZE, distinct_block_count=len(self._groups),
            estimated_workspace_bytes=self.estimated_workspace_bytes,
            factorizations=self.factorizations, applications=self.applications,
            cached_step_seconds=self._cached_step,
            scope="preconditioner_only_original_full_residual_and_operator_counters_unchanged")

    @property
    def numeric_storage_bytes(self):
        return (sum(value.nbytes for group in self._groups
                    for value in (group.nodes, group.capacitance, group.conductance))
                + sum(value.nbytes for factors in self._factors for value in factors))

    def for_step(self, step_seconds, *, fallback, check):
        if (isinstance(step_seconds, (bool, np.bool_)) or not math.isfinite(step_seconds)
                or step_seconds <= 0 or not callable(check)):
            raise ValueError("contact inverse requires a finite positive exact step and check callback")
        if isinstance(fallback, LinearOperator):
            if fallback.shape != (2 * self.state_count, 2 * self.state_count):
                raise ValueError("contact fallback must retain both complete stages")
            fallback_product = fallback.matvec
        elif callable(fallback):
            fallback_product = fallback
        else:
            raise ValueError("contact inverse requires the original stage-Jacobi fallback")
        step = float(step_seconds)
        check()
        if step != self._cached_step:
            # Drop the old factor set before creating its replacement. Operators
            # access this owned set, so closures cannot retain stale LU arrays.
            self._cached_step, self._factors = None, ()
            factors = []
            for group in self._groups:
                check()
                c, q = group.capacitance, step * group.conductance
                matrix = np.block([[c + (5. / 12.) * q, -q / 12.],
                                   [.75 * q, c + .25 * q]])
                if not np.all(np.isfinite(matrix)):
                    raise SolverConvergenceError("nonfinite contact Radau block")
                # Equilibrate only this preconditioner, without changing a
                # residual scale, physical coefficient or accepted tolerance.
                row_scale = np.max(np.abs(matrix), axis=1)
                if np.any(row_scale <= 0):
                    raise SolverConvergenceError("singular contact Radau block")
                matrix /= row_scale[:, None]
                column_scale = np.max(np.abs(matrix), axis=0)
                if np.any(column_scale <= 0):
                    raise SolverConvergenceError("singular contact Radau block")
                matrix /= column_scale[None, :]
                with warnings.catch_warnings():
                    warnings.simplefilter("error", LinAlgWarning)
                    try:
                        lu, pivots = lu_factor(matrix, overwrite_a=True, check_finite=False)
                    except (LinAlgWarning, ValueError, np.linalg.LinAlgError) as error:
                        raise SolverConvergenceError("contact Radau factorization failed") from error
                if not np.all(np.isfinite(lu)) or np.any(np.diag(lu) == 0):
                    raise SolverConvergenceError("invalid contact Radau factors")
                factors.append((lu, pivots, row_scale, column_scale))
                self.factorizations += 1
            self._factors, self._cached_step = tuple(factors), step
            check()

        def apply(value):
            check()
            if self._cached_step != step:
                raise ValueError("stale contact Radau step operator")
            right = _array("contact Radau right-hand side", value, (2 * self.state_count,))
            result = _array("contact stage-Jacobi fallback", fallback_product(right), (2 * self.state_count,))
            rhs_stages, result_stages = right.reshape(2, self.state_count), result.reshape(2, self.state_count)
            for group, (lu, pivots, row_scale, column_scale) in zip(self._groups, self._factors):
                check()
                # One LAPACK call per distinct block, with all equal contacts
                # as columns. There is no Python solve loop over 4096 pads.
                rhs = np.concatenate((rhs_stages[0, group.nodes].T,
                                      rhs_stages[1, group.nodes].T), axis=0)
                rhs /= row_scale[:, None]
                if not np.all(np.isfinite(rhs)):
                    raise SolverConvergenceError("unrepresentable scaled contact right-hand side")
                solution = lu_solve((lu, pivots), rhs, overwrite_b=True, check_finite=False)
                solution /= column_scale[:, None]
                if not np.all(np.isfinite(solution)):
                    raise SolverConvergenceError("nonfinite contact Radau preconditioner result")
                result_stages[0, group.nodes] = solution[:CONTACT_SIZE].T
                result_stages[1, group.nodes] = solution[CONTACT_SIZE:].T
            check()
            self.applications += 1
            return result

        return LinearOperator((2 * self.state_count, 2 * self.state_count), matvec=apply, dtype=np.dtype(float))


def prepare_contact_blocks(capacitance, passive_conductance, *, max_workspace_bytes, check,
                           fallback_info=None):
    """Prepare exact local principal blocks or return None with a reason.

    The caller passes the unused part of its ONE original solver workspace cap.
    Insufficient workspace and nonfinite data raise; they are not permission to
    allocate more or weaken checks. Optional fallback_info is a caller-owned
    dictionary containing only this selection's supported flag/reason.
    """
    if type(max_workspace_bytes) is not int or max_workspace_bytes < 0 or not callable(check):
        raise ValueError("contact blocks require an explicit remaining workspace budget and callback")
    if fallback_info is not None and not isinstance(fallback_info, dict):
        raise ValueError("contact fallback_info must be a caller-owned dictionary")

    def unsupported(reason):
        if fallback_info is not None:
            fallback_info.update(supported=False, reason=reason)
        return None

    check()
    if (type(passive_conductance) is not _ConductanceWithStars
            or type(passive_conductance.stars) is not PassiveStarConductance):
        return unsupported("requires_owned_passive_stars_and_sparse_base")
    base, stars = passive_conductance.base, passive_conductance.stars
    if any(not (isspmatrix_csr(item) or isspmatrix_csc(item))
           or not item.has_canonical_format for item in (capacitance, base)):
        return unsupported("requires_canonical_CSR_or_CSC_capacitance_and_base")
    size = capacitance.shape[0]
    if capacitance.shape != (size, size) or base.shape != (size, size) or stars.shape != (size, size):
        return unsupported("incompatible_full_state_bases")
    count = stars.port_count
    if stars.term_count != count * CONTACT_SIZE:
        return unsupported("requires_seventeen_nodes_per_contact")
    workspace = estimate_contact_workspace_bytes(state_count=int(size), contact_count=int(count),
        sparse_nonzeros=int(capacitance.nnz + base.nnz))
    if workspace > max_workspace_bytes:
        raise SolverConvergenceError("contact preconditioner numeric-workspace limit exceeded")
    for item in (capacitance, base):
        if np.iscomplexobj(item.data) or not np.all(np.isfinite(item.data)):
            raise ValueError("contact principal matrices must be finite and real")
    nodes = stars._nodes.reshape(count, CONTACT_SIZE)
    if (not np.array_equal(stars._group.reshape(count, CONTACT_SIZE),
                           np.broadcast_to(np.arange(count)[:, None], nodes.shape))
            or not np.array_equal(stars._reference, nodes[:, 0])):
        return unsupported("requires_contiguous_seventeen_node_contact_groups")
    if len(np.unique(nodes)) != nodes.size:
        return unsupported("overlapping_contact_groups")
    check()
    owner, local = np.full(size, -1, dtype=np.int64), np.full(size, -1, dtype=np.int64)
    owner[nodes] = np.arange(count)[:, None]
    local[nodes] = np.arange(CONTACT_SIZE)[None, :]

    def principal_blocks(matrix):
        check()
        coo = matrix.tocoo(copy=False)
        row_owner, col_owner = owner[coo.row], owner[coo.col]
        inside = (row_owner >= 0) & (row_owner == col_owner)
        blocks = np.zeros((count, CONTACT_SIZE, CONTACT_SIZE))
        blocks[row_owner[inside], local[coo.row[inside]], local[coo.col[inside]]] = coo.data[inside]
        check()
        return blocks

    cblocks, gblocks = principal_blocks(capacitance), principal_blocks(base)
    weights = stars._h.reshape(count, CONTACT_SIZE)
    fractions = np.divide(weights, stars._totals[:, None], out=np.zeros_like(weights),
                          where=stars._totals[:, None] > 0)
    star_blocks = -weights[:, :, None] * stars._mix[:, None, None] * fractions[:, None, :]
    # The existing diagonal uses the stable centered star formulation; do not
    # reconstruct it by subtracting near-equal conductances.
    diagonal = np.arange(CONTACT_SIZE)
    star_blocks[:, diagonal, diagonal] = stars._diagonal[nodes]
    gblocks += star_blocks
    if not np.all(np.isfinite(cblocks)) or not np.all(np.isfinite(gblocks)):
        raise ValueError("contact principal blocks must be finite")
    check()
    distinct, membership, representatives = {}, np.empty(count, dtype=np.int64), []
    for index in range(count):
        if index % 64 == 0:
            check()
        # Exact bytes only: no numerical rounding, tolerance grouping or
        # assumption that nominally identical contacts actually match.
        key = (cblocks[index].tobytes(), gblocks[index].tobytes())
        group_index = distinct.get(key)
        if group_index is None:
            if len(distinct) == MAX_DISTINCT_BLOCKS:
                return unsupported("more_than_32_distinct_contact_blocks")
            group_index = len(distinct)
            distinct[key] = group_index
            representatives.append(index)
        membership[index] = group_index
    groups = []
    for group_index, representative in enumerate(representatives):
        check()
        selected_nodes = nodes[membership == group_index].copy()
        c, g = cblocks[representative].copy(), gblocks[representative].copy()
        for value in (selected_nodes, c, g):
            value.flags.writeable = False
        groups.append(_BlockGroup(selected_nodes, c, g))
    plan = ContactBlockPlan(state_count=size, groups=groups, workspace=workspace)
    if fallback_info is not None:
        fallback_info.update(supported=True, reason="grouped_seventeen_node_principal_blocks")
    check()
    return plan
