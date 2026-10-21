"""Exact graph step for EP09's proposed matched-donor qualification.

This module consumes adjacency lists only. It neither reads scientific payloads
nor selects donor calipers, populations, roles, or a donor-plan distribution.
Using it does not adopt the pending companion contract.
"""

from dataclasses import dataclass
from typing import Mapping, Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching


@dataclass(frozen=True)
class GraphQualification:
    edges: dict[int, frozenset[int]]
    disposition: str
    iterations: int
    infeasible_edges_removed: int


def qualify_support_graph(
    adjacency: Mapping[int, Iterable[int]], *, minimum_donors: int = 4
) -> GraphQualification:
    """Prune degrees and infeasible matching edges to a fixed point.

Recipients and donors are the same cell set. Each retained vertex must have
``minimum_donors`` distinct non-self choices that participate in a complete
bijective plan on that same retained set. A failed Hall check makes the entire
remaining pool unsupported; this function does not search for a better subset.

Given one perfect matching, contract each matched recipient/donor pair. An
unmatched edge is in another perfect matching exactly when its contracted edge
lies on a directed cycle, identified by strong components. This avoids one
matching computation per edge while preserving the exact feasible edge set.
"""
    if isinstance(minimum_donors, bool) or not isinstance(minimum_donors, int):
        raise ValueError("minimum_donors must be a positive integer")
    if minimum_donors < 1:
        raise ValueError("minimum_donors must be a positive integer")
    edges = {cell: set(donors) for cell, donors in adjacency.items()}
    vertices = set(edges)
    if any(cell in donors for cell, donors in edges.items()):
        raise ValueError("self-donor edges are forbidden")
    if any(not donors <= vertices for donors in edges.values()):
        raise ValueError("every donor must also be a recipient vertex")

    removed = 0
    iteration = 0
    while edges:
        iteration += 1
        # Simultaneous pruning: remove these cells from both sides of the graph.
        drop = {cell for cell, donors in edges.items() if len(donors) < minimum_donors}
        if drop:
            edges = {cell: donors - drop for cell, donors in edges.items() if cell not in drop}
            continue

        order = sorted(edges)
        index = {cell: i for i, cell in enumerate(order)}
        rows, columns = zip(*(
            (index[cell], index[donor])
            for cell in order for donor in sorted(edges[cell])
        ))
        matrix = csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, columns)),
            shape=(len(order), len(order)),
        )
        # perm_type="column" returns one matched donor-column per recipient-row.
        matching = maximum_bipartite_matching(matrix, perm_type="column")
        if np.any(matching < 0):
            return GraphQualification({}, "no_perfect_matching", iteration, removed)

        owner = np.empty(len(order), dtype=np.intp)
        owner[matching] = np.arange(len(order))
        contracted = csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, owner[np.asarray(columns)])),
            shape=matrix.shape,
        )
        _, component = connected_components(contracted, directed=True, connection="strong")
        feasible = {
            cell: {
                donor for donor in edges[cell]
                if component[index[cell]] == component[owner[index[donor]]]
            }
            for cell in order
        }
        newly_removed = sum(len(edges[cell]) - len(feasible[cell]) for cell in order)
        if not newly_removed:
            return GraphQualification(
                {cell: frozenset(feasible[cell]) for cell in order},
                "qualified", iteration, removed,
            )
        removed += newly_removed
        edges = feasible

    return GraphQualification({}, "empty_after_degree_pruning", iteration, removed)
