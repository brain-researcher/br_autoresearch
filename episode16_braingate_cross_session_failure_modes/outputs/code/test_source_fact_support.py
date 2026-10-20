"""Focused checks for the 2026-10-16 source-fact classification rules (synthetic layouts only)."""
import math
import unittest

import numpy as np

from apply_source_fact_support import BASE, session_nodes, load_session, two_sides

RING8 = [(0.5 * math.cos(math.pi * k / 4), 0.5 * math.sin(math.pi * k / 4)) for k in range(8)]


def record(blocks, day=100, participant="T9"):
    """blocks: list of target sequences; each trial lasts 10 samples inside its block."""
    targets, trial_block, starts, ends, block_ids, block_starts, block_ends = [], [], [], [], [], [], []
    t = 0
    for b, sequence in enumerate(blocks, start=1):
        block_ids.append(b)
        block_starts.append(t)
        for target in sequence:
            targets.append(list(target))
            trial_block.append(b)
            starts.append(t)
            ends.append(t + 10)
            t += 10
        block_ends.append(t)
    whole = {str(b): len(s) for b, s in enumerate(blocks, start=1)}
    return {"fields": {"trial_targets": targets, "electrodes": {"array": ["lateral", "medial"]},
                       "trials": {"block_id": trial_block, "trial_start_index": starts, "trial_end_index": ends},
                       "blocks": {"block_id": block_ids, "sample_start_index": block_starts, "sample_end_index": block_ends}},
            "summary": {"participant": participant, "post_implant_day": day, "whole_trial_counts_by_block": whole}}


def center_out(center, ring, cycles):
    sequence = []
    for _ in range(cycles):
        for target in ring:
            sequence += [center, target]
    return sequence


def run(rec, **override):
    cfg = {**BASE, **override}
    return session_nodes(load_session(rec, cfg), cfg)


class SourceFactRules(unittest.TestCase):
    def test_center_variants_placeholder_and_strict_alternative(self):
        a = [(-1.0, -1.0)] + center_out((0.0012, 0.0), RING8, 5)
        b = center_out((0.0025, 0.0), RING8, 5)
        result = run(record([a, b]))
        self.assertEqual(result["status"], "single_schedule")
        node = result["nodes"][0]
        self.assertEqual(node["family"], "radial_8")
        self.assertTrue(node["capable"])
        self.assertEqual(node["whole_trials"], 160)
        self.assertLess(np.abs(node["canonical"][0]).max(), 1e-12)
        strict = run(record([a, b]), slot_tolerance=0.001, canonical_center="stored")
        self.assertEqual(strict["nodes"], [])
        self.assertIn("radial_multiple_instances", strict["notes"])

    def test_oval_radial8_and_radial4_family(self):
        oval = [(484, 0), (316, 316), (0, 409), (-316, 316), (-484, 0), (-316, -316), (0, -409), (316, -316)]
        self.assertEqual(run(record([center_out((0, 0), oval, 5)] * 2))["nodes"][0]["family"], "radial_8")
        four = [(0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5)]
        rec = record([center_out((0, 0), four, 9)] * 2)
        self.assertEqual(run(rec)["nodes"][0]["family"], "radial_4")
        self.assertEqual(run(rec, families=("radial_8", "grid"))["nodes"], [])

    def test_grid_with_roundoff_duplicate(self):
        cells = [(-0.4 + 0.2 * i, -0.4 + 0.2 * j) for i in range(5) for j in range(5)]
        rng = np.random.default_rng(0)
        sequence = [cells[k] for k in rng.integers(0, 25, 80)] + [(3.05e-17, -1.67e-17), (0.0, 0.0)] + cells
        result = run(record([sequence[:60], sequence[60:]]))
        node = result["nodes"][0]
        self.assertEqual((node["family"], node["grid_n"]), ("grid", 5))

    def test_mixed_session_excluded_or_split(self):
        cells = [(-0.4 + 0.16 * i, -0.4 + 0.16 * j) for i in range(6) for j in range(6)]
        grid_block = cells + cells
        rec = record([center_out((0, 0), RING8, 5), center_out((0, 0), RING8, 5), grid_block, grid_block])
        self.assertEqual(run(rec)["nodes"], [])
        families = sorted(n["family"] for n in run(rec, split_mixed_sessions=True)["nodes"])
        self.assertEqual(families, ["grid", "radial_8"])

    def test_conflicting_transitions_and_multiring(self):
        shuffled = [RING8[k % 8] for k in range(0, 80, 3)] + [(0, 0)] * 2
        rec = record([center_out((0, 0), RING8, 5), shuffled])
        self.assertEqual(run(rec)["nodes"], [])
        rings = center_out((0, 0), RING8 + [(x / 2, y / 2) for x, y in RING8], 4)
        self.assertEqual(run(record([rings, rings]))["nodes"], [])

    def test_finite_sampling_check(self):
        four = [RING8[k] for k in (0, 2, 4, 6)]
        rec = record([center_out((0, 0), RING8, 5), center_out((0, 0), four, 5), center_out((0, 0), RING8, 5)])
        self.assertEqual(run(rec)["nodes"], [])
        self.assertEqual(run(rec, finite_sampling_check=False)["nodes"][0]["family"], "radial_8")
        self.assertEqual(run(rec, split_mixed_sessions=True)["nodes"][0]["blocks"], [1, 3])

    def test_two_sides_coverage(self):
        full = {"C", 0, 1}
        self.assertTrue(two_sides([70, 70], [full, full], full))
        self.assertFalse(two_sides([100, 40, 40], [full, {"C", 0}, {"C", 0}], full))
        self.assertTrue(two_sides([100, 40, 40]))


if __name__ == "__main__":
    unittest.main()
