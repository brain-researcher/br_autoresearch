"""Independent finite-graph oracle; no episode data or scientific inference."""

import itertools
import random
import unittest

from ep09_donor_graph import qualify_support_graph
from ep09_donor_support_census import supported_vertices


def exhaustive_qualification(adjacency, minimum_donors):
    """Enumerate every permutation instead of using graph/matching algorithms."""
    edges = {cell: set(donors) for cell, donors in adjacency.items()}
    while edges:
        drop = {cell for cell, donors in edges.items() if len(donors) < minimum_donors}
        if drop:
            edges = {cell: donors - drop for cell, donors in edges.items() if cell not in drop}
            continue
        order = sorted(edges)
        plans = [
            dict(zip(order, permutation))
            for permutation in itertools.permutations(order)
            if all(donor in edges[cell] for cell, donor in zip(order, permutation))
        ]
        if not plans:
            return {}
        feasible = {cell: {plan[cell] for plan in plans} for cell in order}
        if feasible == edges:
            return feasible
        edges = feasible
    return {}


class DonorGraphTests(unittest.TestCase):
    def assert_matches_oracle(self, adjacency, minimum_donors):
        actual = qualify_support_graph(adjacency, minimum_donors=minimum_donors)
        self.assertEqual(actual.edges, exhaustive_qualification(adjacency, minimum_donors))
        for cell, donors in actual.edges.items():
            self.assertNotIn(cell, donors)
            self.assertGreaterEqual(len(donors), minimum_donors)
        return actual

    def test_all_three_vertex_graphs(self):
        candidates = [(i, j) for i in range(3) for j in range(3) if i != j]
        for mask in range(1 << len(candidates)):
            edges = {i: set() for i in range(3)}
            for bit, (i, j) in enumerate(candidates):
                if mask & (1 << bit):
                    edges[i].add(j)
            for minimum in (1, 2, 4):
                with self.subTest(mask=mask, minimum=minimum):
                    self.assert_matches_oracle(edges, minimum)

    def test_asymmetric_and_symmetric_larger_graphs(self):
        rng = random.Random(20261020)
        for n in range(4, 8):
            for symmetric in (False, True):
                for repeat in range(15):
                    edges = {i: set() for i in range(n)}
                    for i in range(n):
                        for j in range(n):
                            if i != j and rng.random() < 0.65:
                                edges[i].add(j)
                                if symmetric:
                                    edges[j].add(i)
                    for minimum in (1, 2, 4):
                        with self.subTest(n=n, symmetric=symmetric, repeat=repeat, minimum=minimum):
                            self.assert_matches_oracle(edges, minimum)

    def test_edge_with_no_feasible_completion_is_removed(self):
        # Only (0->1, 1->0, 2->3, 3->2) is a complete donor plan.
        graph = {0: {1, 2}, 1: {0}, 2: {3}, 3: {2}}
        result = self.assert_matches_oracle(graph, 1)
        self.assertEqual(result.edges[0], {1})
        self.assertEqual(result.infeasible_edges_removed, 1)

    def test_hall_failure_does_not_search_for_a_supported_subset(self):
        graph = {0: {2}, 1: {2}, 2: {0, 1}}
        result = self.assert_matches_oracle(graph, 1)
        self.assertEqual(result.disposition, "no_perfect_matching")
        self.assertEqual(result.edges, {})

    def test_cascade_removes_recipients_and_donors_together(self):
        graph = {0: {1, 2}, 1: {0, 2}, 2: {0, 3}, 3: {2}}
        result = self.assert_matches_oracle(graph, 2)
        self.assertEqual(result.edges, {})
        self.assertGreater(result.iterations, 1)

    def test_invalid_graphs_are_rejected(self):
        for graph in ({0: {0}}, {0: {1}}):
            with self.assertRaises(ValueError):
                qualify_support_graph(graph)
        for minimum in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                qualify_support_graph({}, minimum_donors=minimum)

    def test_complete_nonself_pool_and_id_order(self):
        ids = [91, 2, 17, 4, 63]
        graph = {i: set(ids) - {i} for i in ids}
        result = self.assert_matches_oracle(graph, 4)
        self.assertEqual(result.disposition, "qualified")
        self.assertEqual(result.edges, graph)
        reversed_graph = dict(reversed(list(graph.items())))
        self.assertEqual(qualify_support_graph(reversed_graph), result)

    def test_census_opt_in_rechecks_degrees_but_preserves_historical_default(self):
        # Two internally complete pools, with one-way edges from the four-cell
        # block to the five-cell block. Those extra edges cannot occur in any
        # complete plan, so the first block has only three feasible donors.
        first, second = set(range(4)), set(range(4, 9))
        graph = {i: (first - {i}) | {4} for i in first}
        graph.update({i: second - {i} for i in second})
        original = {i: set(donors) for i, donors in graph.items()}
        self.assertEqual(supported_vertices(graph), first | second)
        self.assertEqual(supported_vertices(graph, qualify_feasible_edges=True), second)
        self.assertEqual(graph, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
