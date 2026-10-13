"""Focused geometric, scaling and finite-stencil checks; no study data."""
import io
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_anatomical_support import (SUFFIXES, lattice_from_header, participant_support,
                                      read_image, selected_sform, strict_trilinear, world_to_voxel)
from inspect_registration_headers import nifti_header


def image(values, slope=1., intercept=0., affine=None):
    return {"values": np.asarray(values), "slope": slope, "intercept": intercept,
            "affine": np.eye(4) if affine is None else affine}


def nifti_fixture(units=2):
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, 2, 2, 2, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 2)
    struct.pack_into("<h", header, 72, 8)
    struct.pack_into("<8f", header, 76, 1, 1, 1, 1, 0, 0, 0, 0)
    struct.pack_into("<f", header, 108, 352)
    struct.pack_into("<2f", header, 112, 1 / 255, 0)
    header[123] = units
    struct.pack_into("<h", header, 254, 2)
    for offset, row in zip((280, 296, 312), ((1, 0, 0, 10), (0, 1, 0, 20), (0, 0, 1, 30))):
        struct.pack_into("<4f", header, offset, *row)
    header[344:348] = b"n+1\0"
    return bytes(header) + bytes([0, 32, 64, 96, 128, 160, 192, 255])


class MemoryArchive:
    def __init__(self, payload):
        self.payload = payload
        self.read_sizes = []

    def getinfo(self, member):
        return SimpleNamespace(file_size=len(self.payload))

    def open(self, member):
        archive = self

        class Stream(io.BytesIO):
            def read(self, count=-1):
                archive.read_sizes.append(count)
                return super().read(count)

        return Stream(self.payload)


class AnatomicalSupportTests(unittest.TestCase):
    def test_nifti_column_major_data_and_scaled_probability(self):
        payload = nifti_fixture()
        loaded = read_image(MemoryArchive(payload), "fixture.nii", nifti_header(io.BytesIO(payload)))
        self.assertEqual(int(loaded["values"][1, 0, 0]), 32)
        self.assertEqual(int(loaded["values"][0, 0, 1]), 128)
        values, valid = strict_trilinear(loaded, [[0, 0, 1], [1, 1, 1]])
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_allclose(values, [128 / 255, 1], atol=1e-7, rtol=0)

    def test_linear_field_and_exact_boundary_without_clamping(self):
        x, y, z = np.indices((2, 2, 2))
        values, valid = strict_trilinear(image(x + 2*y + 3*z),
                                         [[.25, .5, .75], [1, 1, 1], [-1e-12, 0, 0], [1 + 1e-12, 1, 1]])
        np.testing.assert_array_equal(valid, [True, True, False, False])
        np.testing.assert_allclose(values[:2], [3.5, 6])
        self.assertTrue(np.all(np.isnan(values[2:])))

    def test_zero_weight_nan_ignored_but_positive_weight_nan_invalid(self):
        volume = np.ones((2, 2, 2))
        volume[1, 1, 1] = np.nan
        values, valid = strict_trilinear(image(volume), [[0, 0, 0], [.5, .5, .5], [np.nan, 0, 0]])
        np.testing.assert_array_equal(valid, [True, False, False])
        self.assertEqual(values[0], 1)
        self.assertTrue(np.all(np.isnan(values[1:])))

    def test_incomplete_mask_stencil_is_not_renormalized(self):
        mask = np.ones((2, 2, 2))
        mask[1, 1, 1] = 0
        values, valid = strict_trilinear(image(mask), [[.5, .5, .5], [0, 0, 0]], require_positive=True)
        np.testing.assert_array_equal(valid, [False, True])
        self.assertTrue(np.isnan(values[0]))

    def test_affine_orientation_and_stride3_origin(self):
        affine = np.array([[-2, 0, 0, 10], [0, 3, 0, 20], [0, 0, 4, -5], [0, 0, 0, 1.]])
        np.testing.assert_allclose(world_to_voxel(affine, [[9.5, 21.5, -2]]), [[.25, .5, .75]])
        header = {"shape": [141, 95, 87, 1, 3], "sform_code": 2, "xyzt_units": 2,
                  "sform_rows": [[-1, 0, 0, 70], [0, 1, 0, -100], [0, 0, 1, -75]]}
        indices, world, shape, lattice_affine = lattice_from_header(header)
        self.assertEqual(shape, (47, 32, 29))
        self.assertEqual(len(indices), 43616)
        np.testing.assert_array_equal(indices[-1], [138, 93, 84])
        np.testing.assert_allclose(world[[0, -1]], [[70, -100, -75], [-68, -7, 9]])
        np.testing.assert_allclose(lattice_affine[:3, :3], np.diag([-3, 3, 3]))

    def test_absolute_forward_xyz_probability_threshold_and_no_byte_unit_guess(self):
        forward = np.ones((2, 2, 2, 1, 3)) * .5
        images = {SUFFIXES[0]: image(forward), SUFFIXES[1]: image(np.ones((2, 2, 2))),
                  SUFFIXES[2]: image(np.full((2, 2, 2), 128, dtype=np.uint8), slope=1/255)}
        support, _ = participant_support(images, np.array([[0, 0, 0]]))
        self.assertTrue(support[0])
        images[SUFFIXES[2]]["values"][:] = 127
        self.assertFalse(participant_support(images, np.array([[0, 0, 0]]))[0][0])
        images[SUFFIXES[2]]["slope"] = 1
        with self.assertRaisesRegex(ValueError, "probability units"):
            participant_support(images, np.array([[0, 0, 0]]))

    def test_unknown_units_stop_before_anatomical_values_are_read(self):
        payload = nifti_fixture(units=0)
        archive = MemoryArchive(payload)
        with self.assertRaisesRegex(ValueError, "millimetres"):
            read_image(archive, "fixture.nii", nifti_header(io.BytesIO(payload)))
        self.assertEqual(archive.read_sizes, [348])

    def test_unsampled_gmc_nan_does_not_change_probability_units(self):
        forward = np.zeros((2, 2, 2, 1, 3))
        gray = np.full((2, 2, 2), .6)
        gray[1, 1, 1] = np.nan
        images = {SUFFIXES[0]: image(forward), SUFFIXES[1]: image(np.ones((2, 2, 2))),
                  SUFFIXES[2]: image(gray)}
        indices = np.array([[0, 0, 0]])
        self.assertTrue(participant_support(images, indices)[0][0])
        forward[:] = .5
        self.assertFalse(participant_support(images, indices)[0][0])
        gray[:] = np.nan
        with self.assertRaisesRegex(ValueError, "no finite values"):
            participant_support(images, indices)


if __name__ == "__main__":
    unittest.main()
