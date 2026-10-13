"""Source-free identity, guide-only access and frozen subset qualification."""
import copy
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np

import qualify_primary_intersection as qualify
from source_adapter import EVENTS


def records_for(session):
    return [{"trial_id": f"{session}_trial{i+1:06d}", "source_trial_index_zero_based": i,
             "direction": i//2, "idx_startTime": 0, "idx_tgtOnTime": 1,
             "idx_goCueTime": 2, "idx_movement_on": 4, "idx_endTime": 19}
            for i in range(16)]


def guides():
    return (np.array([[e, *band] for e in (2, 4, 7) for band in ((0., 0.), (100., 200.))]),
            np.array([[2., 0.], [2., 7.], [7., -1.], [9., 1.]]))


class ShapeOnly:
    def __init__(self, shape, name, accesses):
        self.shape, self.name, self.accesses = shape, name, accesses

    def __array__(self, *args, **kwargs):
        raise AssertionError(f"Forbidden numerical read of {self.name}")

    def __getitem__(self, key):
        raise AssertionError(f"Forbidden numerical slice of {self.name}")


class FakeSession:
    guides = qualify.M1Session.guides

    def __init__(self, source, animal, implant, day):
        self.session_id = f"{animal}_{day}"
        self.path = Path(source)/f"{animal}_CO_{day}.mat"
        self.rows, self.count, self.accesses = records_for(self.session_id), 16, []
        lfp, units = guides()
        self.lfp_guides = [lfp.copy() for _ in self.rows]
        self.unit_guides = [units.copy() for _ in self.rows]
        self.shapes = {"M1_lfp": (len(lfp), 20), "M1_spikes": (len(units), 20), "vel": (2, 20)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def metadata(self, trial):
        return {**self.rows[trial], "result": "R", "direction_circular_error": 0., "native_bin_seconds": .03}

    def field(self, trial, name):
        self.accesses.append((trial, name))
        if name == "M1_lfp_guide":
            return self.lfp_guides[trial].T
        if name == "M1_unit_guide":
            return self.unit_guides[trial].T
        if name not in self.shapes:
            raise AssertionError(f"Undeclared source field: {name}")
        return ShapeOnly(self.shapes[name], name, self.accesses)


class IntersectionMetadataTest(unittest.TestCase):
    def test_unique_physical_intersection_retains_all_finite_unit_labels(self):
        lfp, units = guides()
        result = qualify.guide_intersection(lfp, units, [2, 4, 7])
        self.assertEqual(result["intersection_electrode_ids"], [2, 7])
        self.assertEqual(result["unit_electrode_ids"], [2, 7, 9])
        self.assertEqual(result["unit_guide_pairs"], units.tolist())
        self.assertEqual(result["listed_unit_count"], 4)
        self.assertEqual(result["original_electrodes_without_listed_units"], [4])
        self.assertEqual(result["unit_electrodes_outside_original_LMP"], [9])
        self.assertFalse(result["unit_quality_filter_applied"])
        for other_lfp, other_units, original in (
                (lfp, np.vstack((units, units[0])), [2, 4, 7]),
                (np.vstack((lfp, lfp[0])), units, [2, 4, 7]),
                (lfp, np.array([[8., 0.]]), [2, 4, 7]),
                (lfp, np.array([[2.5, 0.]]), [2, 4, 7]),
                (lfp, np.array([[2., np.nan]]), [2, 4, 7]),
                (lfp, units, [2, 7]), (lfp, units, [2, 2, 7])):
            with self.assertRaises(ValueError):
                qualify.guide_intersection(other_lfp, other_units, original)

    def test_exact_ordinary_subset_literals_fixed_points_and_full_support_reuse(self):
        electrodes, intersection = [2, 4, 7, 9, 11, 19], [4, 11]
        supports = qualify.support_sets(electrodes, intersection, "example")
        self.assertEqual(len(supports), 9)
        self.assertEqual(supports[0]["electrode_ids"], intersection)
        self.assertEqual(supports[0]["electrode_positions"], [1, 4])
        for replicate, support in enumerate(supports[1:]):
            literal = f"ep05-HW-unit-intersection-random-support-v1|20261004|example|{replicate}"
            generator = random.Random()
            generator.seed(literal, version=2)
            queue = electrodes.copy()
            generator.shuffle(queue)
            self.assertEqual(support["seed_literal"], literal)
            self.assertEqual(support["electrode_queue"], queue)
            self.assertEqual(support["electrode_ids"], sorted(queue[:2]))
            self.assertEqual(support["electrode_ids"], [electrodes[p] for p in support["electrode_positions"]])
        class Identity:
            def seed(self, literal, version):
                if version != 2:
                    raise AssertionError("Wrong seed version")
            def shuffle(self, values):
                pass
        with patch.object(qualify.random, "Random", Identity):
            repeated = qualify.support_sets(electrodes, electrodes[:2], "example")
        self.assertTrue(all(support["electrode_ids"] == electrodes[:2] for support in repeated))
        full = qualify.support_sets(electrodes, electrodes, "example")
        self.assertTrue(all(support["full_support"] for support in full))

    def test_source_adapter_reads_only_guides_and_shapes_for_every_original_trial(self):
        session = FakeSession(Path("unopened"), "Chewie", "Chewie-L", "20150313")
        expected = {"session_id": session.session_id, "complete_trials": 16,
                    "original_electrode_ids": [2, 4, 7], "direction_counts": [2]*8}
        result = qualify.qualify_session(session, session.rows, expected)
        self.assertEqual(result["trial_ids"], [row["trial_id"] for row in session.rows])
        self.assertEqual(result["intersection_electrode_ids"], [2, 7])
        self.assertTrue(result["constant_guides_on_all_original_trials"])
        self.assertFalse(result["spike_values_read"])
        self.assertFalse(result["LFP_XY_values_read"])
        self.assertFalse(result["scores_computed"])
        self.assertEqual(sum(name == "M1_unit_guide" for _, name in session.accesses), 16)
        self.assertEqual(sum(name == "M1_lfp_guide" for _, name in session.accesses), 16)
        self.assertEqual(result["guide_logical_bytes"], 16*(guides()[0].nbytes+guides()[1].nbytes))
        unique = {tuple(s["electrode_ids"]) for s in result["support_sets"] if not s["full_support"]}
        self.assertEqual(result["unique_new_fit_supports"], len(unique))
        self.assertEqual(result["workload_proxy"], len(unique)*(105*16*8**2+8*8**3))

    def test_changed_guides_events_or_dimensions_fail_without_neural_value_reads(self):
        for damage in ("unit_order", "lfp_order", "unit_identity", "event", "spike_time", "empty_intersection"):
            session = FakeSession(Path("unopened"), "Chewie", "Chewie-L", "20150313")
            expected = {"session_id": session.session_id, "complete_trials": 16,
                        "original_electrode_ids": [2, 4, 7], "direction_counts": [2]*8}
            rows = copy.deepcopy(session.rows)
            if damage == "unit_order":
                session.unit_guides[-1] = session.unit_guides[-1][::-1]
            elif damage == "lfp_order":
                session.lfp_guides[-1] = session.lfp_guides[-1][::-1]
            elif damage == "unit_identity":
                session.unit_guides[-1][0, 1] = 22.
            elif damage == "event":
                rows[-1]["idx_goCueTime"] += 1
            elif damage == "spike_time":
                session.shapes["M1_spikes"] = (4, 19)
            else:
                session.unit_guides = [np.array([[8., 0.], [8., 1.], [9., 1.], [9., 2.]]) for _ in rows]
            with self.assertRaises(ValueError):
                qualify.qualify_session(session, rows, expected)

    def test_all_sixteen_qualification_and_metadata_pilot_choice_are_required(self):
        records = {s: records_for(s) for s in qualify.SESSIONS}
        original = {s: {"session_id": s, "complete_trials": 16, "direction_counts": [2]*8,
                       "original_electrode_ids": [2, 4, 7]} for s in qualify.SESSIONS}
        opened = []
        def factory(*args):
            result = FakeSession(*args)
            opened.append(result.session_id)
            return result
        result = qualify.qualify_all(Path("unopened"), records, original, session_factory=factory)
        self.assertEqual(opened, list(qualify.SESSIONS))
        self.assertTrue(result["all_sixteen_qualified_before_values"])
        self.assertEqual(result["pilot_session"], max(result["sessions"], key=lambda row: row["workload_proxy"])["session_id"])
        equal = [{"session_id": s, "workload_proxy": 7} for s in reversed(qualify.SESSIONS)]
        self.assertEqual(qualify.choose_pilot(equal), qualify.SESSIONS[0])
        self.assertIsNone(qualify.choose_pilot([{**row, "workload_proxy": 0} for row in equal]))
        opened.clear()
        broken = dict(records)
        del broken[qualify.SESSIONS[-1]]
        with self.assertRaises(ValueError):
            qualify.qualify_all(Path("unopened"), broken, original, session_factory=factory)
        self.assertEqual(opened, [])

    def test_original_scalar_HW_metadata_reconcile_before_source(self):
        session = qualify.SESSIONS[0]
        rows = records_for(session)
        source = {"session_id": session, "complete_trials": 16, "common_LMP_electrodes": [2, 4, 7]}
        scalar = {"status": "complete_session_score_bank", "recipe": qualify.SCALAR_RECIPE,
                  "session_id": session, "source": source, "direction_counts": [2]*8}
        history = {"status": "complete_two_history_repeatability_weight_session_banks", "session_id": session,
            "exact_original_LMP_baseline_target_trial_replay": True,
            "exact_original_current_weight_state_and_maps_replay": True,
            "source": {**source, "native_LMP_offsets": [-3, 14], "native_target_offsets": [0, 14],
                       "native_bin_seconds": .03, "lag_offsets": [0, -1, -2, -3]},
            "banks": {qualify.REFERENCE: {"direction_counts": [2]*8}}}
        result = qualify.original_metadata(scalar, history, session, rows)
        self.assertEqual(result["original_electrode_ids"], [2, 4, 7])
        for damage in ("channels", "count", "window", "status", "replay"):
            changed = copy.deepcopy(history)
            if damage == "channels":
                changed["source"]["common_LMP_electrodes"] = [2, 7]
            elif damage == "count":
                changed["banks"][qualify.REFERENCE]["direction_counts"][0] = 3
            elif damage == "window":
                changed["source"]["native_LMP_offsets"] = [-4, 14]
            elif damage == "status":
                changed["status"] = "started"
            else:
                changed["exact_original_LMP_baseline_target_trial_replay"] = False
            with self.assertRaises(ValueError):
                qualify.original_metadata(scalar, changed, session, rows)


if __name__ == "__main__":
    unittest.main()
