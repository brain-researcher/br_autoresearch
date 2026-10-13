"""Focused anatomy/basis fixtures; not empirical fits or class qualification."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import local_anatomical_basis as anatomy


# Fixture tolerances only; callers must freeze their actual recipe explicitly.
TOL = anatomy.BasisTolerances(eigen_tie_atol=1e-10, eigen_tie_rtol=1e-10,
                             rank_rtol=1e-10, numerical_atol=1e-10)


class LocalAnatomicalBasisTests(unittest.TestCase):
    def setUp(self):
        self.xyz = np.indices((13, 4, 4)).reshape(3, -1).T.astype(float)
        self.ids = np.arange(500, 708, dtype=np.int64)

    def test_reference_preserves_support_centering_and_hard_locality(self):
        result = anatomy.build_local_anatomical_basis(self.xyz, self.ids, tolerances=TOL)
        self.assertEqual(result.B.shape, (208, 16))
        self.assertEqual(result.blocks, tuple((2*i, 2*i+2) for i in range(8)))
        np.testing.assert_array_equal(result.voxel_ids, self.ids)
        np.testing.assert_array_equal(result.coordinates, self.xyz)
        np.testing.assert_array_equal(np.bincount(result.cell_labels), np.full(8, 26))
        self.assertEqual(sorted(sum(result.cell_voxel_ids, ())), self.ids.tolist())
        np.testing.assert_allclose(result.B.T@result.B, np.eye(16), atol=1e-11, rtol=0)
        np.testing.assert_allclose(result.B.sum(axis=0), 0, atol=1e-11, rtol=0)
        max_distance = 0.0
        for cell, (start, stop) in enumerate(result.blocks):
            rows = result.cell_labels == cell
            np.testing.assert_array_equal(result.B[~rows, start:stop], 0)
            points = self.xyz[rows]
            max_distance = max(max_distance, float(np.max(np.linalg.norm(
                points[:, None]-points[None, :], axis=2))))
        self.assertEqual(result.max_cell_diameter_mm, max_distance)
        # A block-orthogonal operator cannot connect different spatial cells.
        O = np.zeros((16, 16))
        for start, stop in result.blocks:
            O[start:stop, start:stop] = [[0, -1], [1, 0]]
        Q = np.eye(208)+result.B@(O-np.eye(16))@result.B.T
        cross_cell = result.cell_labels[:, None] != result.cell_labels[None, :]
        np.testing.assert_array_equal(Q[cross_cell], 0)
        np.testing.assert_allclose(Q.T@Q, np.eye(208), atol=1e-11, rtol=0)
        self.assertFalse(result.B.flags.writeable)

    def test_exact_spatial_ties_use_axis_then_native_id_and_ignore_row_order(self):
        # Root extents tie on all axes; next split again ties within each half.
        cubes = np.array([[x, y, z] for x in (0., 1.)
                          for y in (0., 1.) for z in (0., 1.)])
        repeated = np.repeat(cubes, 26, axis=0)
        cells, splits = anatomy._partition(repeated, self.ids, 2)
        self.assertEqual([item["axis"] for item in splits], [0, 1, 1])
        # A flat coordinate tie is resolved by numeric native ID, not row index.
        flat = np.zeros_like(self.xyz)
        reversed_ids = self.ids[::-1]
        tied_cells, _ = anatomy._partition(flat, reversed_ids, 3)
        self.assertEqual([reversed_ids[rows].tolist() for rows in tied_cells],
                         [self.ids[26*i:26*(i+1)].tolist() for i in range(8)])
        baseline = anatomy.build_local_anatomical_basis(self.xyz, self.ids, tolerances=TOL)
        order = np.roll(np.arange(208)[::-1], 37)
        reordered = anatomy.build_local_anatomical_basis(
            self.xyz[order], self.ids[order], tolerances=TOL)
        np.testing.assert_allclose(reordered.B, baseline.B[order], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(reordered.cell_labels, baseline.cell_labels[order])
        self.assertEqual(reordered.cell_voxel_ids, baseline.cell_voxel_ids)

    def test_boundary_tie_uses_complete_eigenspace_and_canonical_projection(self):
        # K4 has a three-dimensional tied nonconstant eigenspace; retain only2.
        laplacian = 4*np.eye(4)-np.ones((4, 4))
        expected, diagnostics = anatomy._low_modes(laplacian, 2, TOL)
        eigh = np.linalg.eigh
        rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., -1.]])

        def rotated(matrix):
            values, vectors = eigh(matrix)
            return values, vectors@rotation

        with patch.object(anatomy.np.linalg, "eigh", side_effect=rotated):
            actual, changed_diagnostics = anatomy._low_modes(laplacian, 2, TOL)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        first = np.array([.75, -.25, -.25, -.25])
        first /= np.linalg.norm(first)
        np.testing.assert_allclose(expected[:, 0], first, atol=1e-12, rtol=0)
        group = diagnostics["retained_groups"][0]
        self.assertEqual((group["start"], group["stop"], group["retained"]), (0, 3, 2))
        self.assertEqual(changed_diagnostics["nonconstant_rank"], 3)

    def test_explicit_locality_capacity_variants_keep_all_208_rows(self):
        for depth, modes in ((2, 2), (4, 2), (3, 1), (3, 3)):
            with self.subTest(depth=depth, modes=modes):
                result = anatomy.build_local_anatomical_basis(
                    self.xyz, self.ids, tolerances=TOL,
                    split_depth=depth, modes_per_cell=modes)
                self.assertEqual(result.B.shape, (208, 2**depth*modes))
                self.assertEqual(sum(len(ids) for ids in result.cell_voxel_ids), 208)
                np.testing.assert_array_equal(result.voxel_ids, self.ids)
                for cell, (start, stop) in enumerate(result.blocks):
                    self.assertEqual(stop-start, modes)
                    np.testing.assert_array_equal(result.B[result.cell_labels != cell, start:stop], 0)

    def test_no_support_replacement_or_undefined_graph_repair(self):
        with self.assertRaises(ValueError):
            anatomy.build_local_anatomical_basis(self.xyz[:-1], self.ids[:-1], tolerances=TOL)
        with self.assertRaises(ValueError):
            anatomy.build_local_anatomical_basis(self.xyz, np.repeat(1, 208), tolerances=TOL)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            anatomy.build_local_anatomical_basis(np.zeros_like(self.xyz), self.ids, tolerances=TOL)
        with self.assertRaisesRegex(ValueError, "more voxels"):
            anatomy.build_local_anatomical_basis(self.xyz, self.ids, tolerances=TOL,
                                                modes_per_cell=26)


if __name__ == "__main__":
    unittest.main()
