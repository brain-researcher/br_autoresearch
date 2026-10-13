"""Changed availability-axis integrity and reader joins; run in Slurm."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import development_handoff_reader as reader
from finite_development_support import select_support
from test_development_handoff_reader import member_fixture, release_fixture


class FiniteSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.summaries = member_fixture()
        cls.bindings, cls.original, cls.archives = release_fixture(cls.rows)
        cls.header = cls.bindings[0].anatomy_headers["space-SUIT_xfm.nii"]
        _, _, ids = select_support(cls.original, cls.header)
        keep = np.ones(208, dtype=bool)
        keep[[0, 3, 8, 90, 207]] = False
        cls.selection = dict(keep=keep, original_voxel_ids=np.array(ids),
            selected_voxel_ids=np.array(ids)[keep], completed_maps=np.array(9024))

    def test_exact_common_order_original_unchanged_and_mismatch_refused(self):
        original = {key: value.copy() for key, value in self.original.items()}
        selected, indices, ids = select_support(original, self.header, self.selection)
        self.assertEqual(len(ids), 203)
        np.testing.assert_array_equal(indices, self.original["source_transform_voxel_ijk"][self.selection["keep"]])
        np.testing.assert_array_equal(selected["world_xyz_mm"], self.original["world_xyz_mm"][self.selection["keep"]])
        for key in original:
            np.testing.assert_array_equal(original[key], self.original[key])
        for key in ("original_voxel_ids", "selected_voxel_ids"):
            changed = deepcopy(self.selection)
            changed[key] = changed[key][::-1]
            with self.assertRaises(ValueError):
                select_support(original, self.header, changed)
        changed = deepcopy(self.selection)
        changed["completed_maps"] = np.array(9023)
        with self.assertRaises(ValueError):
            select_support(original, self.header, changed)

    def test_reader_maps_same_axis_for_all_twelve_people_and_all_views(self):
        seen = []
        def native(_archive, _member, _header, indices):
            seen.append(indices.copy())
            return np.zeros((len(indices), 3))
        def sample(_archive, _member, _header, _mask, world):
            return np.arange(len(world), dtype=np.float64), 0
        with patch.object(reader.zipfile, "ZipFile", side_effect=lambda path: self.archives[Path(path).stem]), \
                patch.object(reader, "_native_coordinates", side_effect=native), \
                patch.object(reader, "sample_beta", side_effect=sample):
            result = reader.read_development_handoffs(release_bindings=self.bindings,
                member_rows=self.rows, condition_multiplicity_by_session=self.summaries,
                common_support=self.original, derived_support=self.selection, check=lambda: None)
        self.assertEqual(len(result.participants), 12)
        self.assertEqual(result.report["support_locations"], 203)
        self.assertEqual(len(seen), 12)
        for record, packet, observation in zip(result.participants, result.task_a_packets, result.observations):
            self.assertEqual(record.voxel_ids, result.voxel_ids)
            self.assertEqual(packet.voxel_ids, result.voxel_ids)
            self.assertEqual(observation.voxel_ids, result.voxel_ids)
            self.assertEqual(record.task_a_maps.shape, (29, 203))
            self.assertEqual(observation.task_b_partitions.shape, (4, 18, 203))
        # Default is still the distinct original208 route.
        self.assertEqual(len(select_support(self.original, self.header)[2]), 208)


if __name__ == "__main__":
    unittest.main()
