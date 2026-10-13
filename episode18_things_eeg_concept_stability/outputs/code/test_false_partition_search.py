"""Two functional regressions; no scientific recovery or cost pilot.

Execution belongs in the coordinator's verified Slurm allocation. Fixtures
shrink only the number of groups; retain twelve blocks, six folds and families.
"""
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import false_partition_search as search


def fixture():
    generator = np.random.default_rng(718)
    lookup = np.arange(search.IMAGES, dtype=np.int32).reshape(search.GROUPS, 12)
    events = {"concept_by_block_image_row": lookup, "concept": np.repeat(np.arange(search.GROUPS), 12),
              "block": np.tile(np.arange(12), search.GROUPS)}
    pairs = generator.uniform(0., 2., (36, search.IMAGES*(search.IMAGES-1)//2))
    contexts = [generator.uniform(0., 1., (search.IMAGES, 3)) for _ in range(6)]
    for values in contexts:
        values[:, 2] = 1.  # Invariant zero-SD mean and variance descriptors.
    rate = np.array([False, True, False])
    return events, pairs, contexts, rate


def blocks_for(events, pairs, contexts, rate, assignment):
    blocks = search.prepare_metric_blocks(events["concept_by_block_image_row"], pairs, contexts, rate, lambda: None)
    for block, values in zip(blocks, search.group_metrics(assignment, pairs, contexts)):
        block.bind(values)
    return blocks


class FalsePartitionSearchTests(unittest.TestCase):
    def test_two_group_updates_match_complete_recomputation_and_exact_refusal(self):
        with patch.multiple(search, GROUPS=31, IMAGES=31*12):
            events, pairs, contexts, rate = fixture()
            assignment, rng = search.initial_assignment(events, 0, 0)
            blocks = blocks_for(events, pairs, contexts, rate, assignment)
            for _ in range(7):
                rows, proposed, _ = search.legal_proposal(assignment, events["concept"], rng)
                local = search.group_metrics(proposed, pairs, contexts)
                changes = [block.change(rows, value) for block, value in zip(blocks, local)]
                assignment[rows] = proposed
                search.valid_assignment(assignment, events)
                full = search.group_metrics(assignment, pairs, contexts)
                for block, expected, value, (total, squares) in zip(blocks, full, local, changes):
                    np.testing.assert_array_equal(value, expected[:, rows])
                    block.values[:, rows] = value
                    np.testing.assert_array_equal(block.values, expected)
                    block.reanchor()
                    np.testing.assert_allclose(total, block.total, atol=1e-10, rtol=1e-12)
                    np.testing.assert_allclose(squares, block.squares, atol=1e-10, rtol=1e-12)
                self.assertAlmostEqual(search.adjusted_rand(assignment, assignment[::-1]), 1.)
            # A passing aggregate mean cannot certify matching distributions.
            reference = np.arange(31., dtype=float)[None, :, None]
            block = search.MetricBlock(reference, np.array([False]))
            block.bind(np.full_like(reference, reference.mean()))
            self.assertFalse(block.fails_necessary_mean())
            self.assertFalse(block.exact_acceptance()[0])
            zero = search.MetricBlock(np.ones_like(reference), np.array([False]))
            zero.bind(np.ones_like(reference))
            self.assertTrue(zero.exact_acceptance()[0])
            zero.values[0, 0, 0] += 2e-12
            self.assertFalse(zero.exact_acceptance()[0])

    def test_checkpoint_continuation_preserves_rng_moments_and_refuses_bad_history(self):
        with patch.multiple(search, GROUPS=31, IMAGES=31*12), tempfile.TemporaryDirectory() as temporary:
            events, pairs, contexts, rate = fixture()
            assignment, rng = search.initial_assignment(events, 0, 0)
            blocks = blocks_for(events, pairs, contexts, rate, assignment)
            accepted_count = 0
            for step in range(17):
                accepted_count += search.advance_assignment(assignment, events["concept"], pairs, contexts, blocks, rng, step)
            root = Path(temporary)
            prior, cache = root/"prior", root/"cache"
            prior.mkdir(); cache.mkdir()
            search.atomic_json(cache/"manifest.json", {"context_names": ["a", "b", "c"]})
            config = {"cache_dir": str(cache), "segment_byte_cap": 64*1024**2, "recipe_id": search.RECIPE}
            state = {"config_id": search.digest(config), "recipe_id": search.RECIPE, "member": 0, "start": 0,
                "swaps": 17, "phase": "search", "last_checked": 0, "total_legal_proposals": 17,
                "total_accepted_proposals": int(accepted_count), "exact_checks": 0, "acceptance_positions": []}
            search.atomic_npz(prior/"checkpoint.npz",
                search.checkpoint_payload(state, assignment, [], [], blocks, rng), config["segment_byte_cap"]//4)
            search.atomic_json(prior/"report.json", {"status": "resource_incomplete", "checkpoint_available": True})
            original_bytes = (prior/"checkpoint.npz").read_bytes()
            restored, continued, accepted, diagnostics, saved = search.restore_checkpoint(prior/"checkpoint.npz", config, events)
            self.assertEqual(accepted, []); self.assertEqual(diagnostics, [])
            continuation = blocks_for(events, pairs, contexts, rate, continued)
            for index, block in enumerate(continuation):
                block.total, block.squares = saved[f"total_{index}"], saved[f"squares_{index}"]
            second_rng = np.random.Generator(np.random.PCG64())
            second_rng.bit_generator.state = restored["rng_state"]
            for step in range(17, 34):
                first_accept = search.advance_assignment(assignment, events["concept"], pairs, contexts, blocks, rng, step)
                next_accept = search.advance_assignment(continued, events["concept"], pairs, contexts, continuation, second_rng, step)
                self.assertEqual(first_accept, next_accept)
            np.testing.assert_array_equal(assignment, continued)
            self.assertEqual(rng.bit_generator.state, second_rng.bit_generator.state)
            for first, second in zip(blocks, continuation):
                np.testing.assert_array_equal(first.values, second.values)
                np.testing.assert_array_equal(first.total, second.total)
                np.testing.assert_array_equal(first.squares, second.squares)
            self.assertEqual(original_bytes, (prior/"checkpoint.npz").read_bytes())
            search.atomic_json(prior/"report.json", {"status": "stopped_error", "checkpoint_available": True})
            with self.assertRaises(Exception):
                search.restore_checkpoint(prior/"checkpoint.npz", config, events)
            self.assertEqual(original_bytes, (prior/"checkpoint.npz").read_bytes())
            # Deliberate RSS admission stops are resumable budget events;
            # genuine allocation failures still follow the error disposition.
            with patch.object(search.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=2)), \
                    search.budget(time.time()+10., 1024) as check:
                with self.assertRaisesRegex(search.BudgetStop, "rss_cap"):
                    check()


if __name__ == "__main__":
    unittest.main()
