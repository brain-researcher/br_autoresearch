"""Focused leakage, polarity, exact-source and anatomical-support checks on compute."""

import copy
from decimal import Decimal
import gzip
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from prepare_mask_and_signed_text import (bind_target_paths, check_target_headers,
                                          exact_definition, prepare_text, read_reference_mask)


def text_fixture():
    names = ["trial_type.private_a", "trial_type.private_b"]
    weights = [Decimal("0.1234567890123456789"), Decimal("-0.1234567890123456789")]
    row = {"route": "ds000001/task-private", "session": "", "contrast": "private_contrast", "s": "-1",
           "definition": " ".join(f"{float(w):+g}*{c}" for c, w in zip(names, weights))}
    identifier = row["route"] + "||" + row["contrast"]
    prep = {"maps": [{"map_id": identifier, "polarity": -1, "metadata_exclusions": ["timing_hold"],
                      "role": "historically_exposed_development_only"}]}
    contrast = {"name": row["contrast"], "condition_list": names, "weights": weights}
    model = {row["route"]: {"name": "SECRET_DATASET_TITLE", "nodes": [{"contrasts": [contrast]}]}}
    roles = {"routes": {row["route"]: {
        "factors": {"outcome": {"type": "valence", "levels": {"gain": 1, "loss": -1},
                                "note": "SECRET_NEURAL_RESULT", "evidence": "SECRET_EVIDENCE"}},
        "conditions": {names[0]: {"outcome": "loss"}, names[1]: {"outcome": "gain"}},
        "note": "SECRET_METHODS_PROSE"}}}
    return prep, row, model, roles


def mask_fixture(path, values, qoffset=0, uint8_scaled=False):
    """Tiny asymmetrical mask on a 2-mm grid; no empirical inputs."""
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *values.shape, 1, 1, 1, 1)
    struct.pack_into("<2h", header, 70, *( (2, 8) if uint8_scaled else (16, 32) ))
    struct.pack_into("<8f", header, 76, 1, 2, 2, 2, 1, 1, 1, 1)
    struct.pack_into("<3f", header, 108, 352, 1/255 if uint8_scaled else 1, 0)
    header[123] = 2
    struct.pack_into("<2h", header, 252, 2, 2)
    struct.pack_into("<6f", header, 256, 0, 0, 0, qoffset, 0, 0)
    affine = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]
    struct.pack_into("<12f", header, 280, *np.asarray(affine)[:3].ravel())
    header[344:348] = b"n+1\x00"
    with gzip.open(path, "wb") as handle:
        handle.write(header)
        handle.write(values.astype("u1" if uint8_scaled else "<f4").tobytes(order="F"))
    return {"shape": list(values.shape), "affine": affine}, header


class MaskAndSignedTextTests(unittest.TestCase):
    def test_exact_coefficients_sign_and_redaction(self):
        prep, row, models, roles = text_fixture()
        record = prepare_text(prep, [row], models, roles)[0]
        self.assertEqual(record["terms"][0]["source_weight"], "0.1234567890123456789")
        self.assertEqual(record["terms"][0]["canonical_weight"], "-0.1234567890123456789")
        self.assertEqual(record["metadata_exclusions"], ["timing_hold"])
        self.assertEqual(record["role"], prep["maps"][0]["role"])
        for forbidden in ["SECRET", "ds000001", "private", "evidence", "note"]:
            self.assertNotIn(forbidden, record["model_text"])
        self.assertIn("outcome: loss", record["model_text"])
        self.assertIn("outcome: gain", record["model_text"])
        # Reversing both source definition and adopted s preserves canonical text.
        row["s"], prep["maps"][0]["polarity"] = "1", 1
        definition = models[row["route"]]["nodes"][0]["contrasts"][0]
        definition["weights"] = [-w for w in definition["weights"]]
        row["definition"] = " ".join(f"{float(w):+g}*{c}" for c, w in
                                     zip(definition["condition_list"], definition["weights"]))
        self.assertEqual(prepare_text(prep, [row], models, roles)[0]["model_text"], record["model_text"])

    def test_unknown_meaning_keeps_identity_and_exclusions_without_invention(self):
        prep, row, models, roles = text_fixture()
        roles["routes"][row["route"]]["conditions"].clear()
        result = prepare_text(prep, [row], models, roles)[0]
        self.assertIsNone(result["model_text"])
        self.assertIn("condition_meaning_unresolved", result["text_issues"])
        self.assertEqual(result["map_id"], prep["maps"][0]["map_id"])
        self.assertEqual(result["metadata_exclusions"], prep["maps"][0]["metadata_exclusions"])

    def test_executed_definition_beats_route_model_and_ambiguity_is_not_rounded_away(self):
        prep, row, models, roles = text_fixture()
        definition = copy.deepcopy(models[row["route"]]["nodes"][0]["contrasts"][0])
        census = [{"route": row["route"], "session": None, "contrast": row["contrast"],
                   "definitions": [definition]}]
        models[row["route"]]["nodes"][0]["contrasts"] = []
        result = prepare_text(prep, [row], models, roles, census)[0]
        self.assertEqual(result["weight_binding"], "map_specific_executed_definition")
        self.assertIsNotNone(result["model_text"])
        # Producer's exact-map census maps cue_LvNeut to filename cueLvNeut.
        alias_definition = dict(definition, name="cue_LvNeut")
        alias_row = dict(row, contrast="cueLvNeut")
        self.assertEqual(exact_definition(alias_row, None, [alias_definition])[1], definition["weights"])
        with self.assertRaises(ValueError):
            exact_definition(alias_row, {"nodes": [{"contrasts": [alias_definition]}]})
        other = copy.deepcopy(definition)
        other["weights"][0] += Decimal("0.0000000000000000001")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            exact_definition(row, None, [definition, other])

    def test_unreviewed_parametric_prose_never_becomes_a_predictor(self):
        prep, row, models, roles = text_fixture()
        condition = "quantity_private"
        row["definition"] = "+1*" + condition
        models[row["route"]]["nodes"][0]["contrasts"][0].update(condition_list=[condition], weights=[1])
        roles["routes"][row["route"]]["conditions"] = {condition: {
            "_kind": "pm", "_dir": 1, "_constr": "O", "_quantity": "SECRET_NEURAL_RESULT"}}
        result = prepare_text(prep, [row], models, roles)[0]
        self.assertIsNone(result["model_text"])
        self.assertIn("parametric_quantity_not_in_reviewed_allowlist", result["text_issues"])
        roles["routes"][row["route"]]["conditions"][condition].update(
            _constr="U", _quantity="stimulus coherence (original column)")
        self.assertIn("parametric_direction_unresolved", prepare_text(prep, [row], models, roles)[0]["text_issues"])

    def test_mask_order_binary_support_and_conflicting_geometry(self):
        values = np.zeros((2, 3, 4))
        values[1, 0, 2] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.nii.gz"
            binding, _ = mask_fixture(path, values)
            mask, report = read_reference_mask(path, binding)
            np.testing.assert_array_equal(mask, values.astype(bool))
            self.assertEqual(report["foreground_voxels"], 1)
            self.assertEqual(report["target_arrays_read"], 0)
            # Observed reference encoding: uint8 0/255 with a float32 1/255 slope.
            mask_fixture(path, values * 255, uint8_scaled=True)
            scaled_mask, scaled_report = read_reference_mask(path, binding)
            np.testing.assert_array_equal(scaled_mask, values.astype(bool))
            self.assertEqual(scaled_report["raw_levels"], [0, 255])
            self.assertEqual(scaled_report["scaled_levels"], [0, 1.0000000591389835])
            self.assertEqual(scaled_report["foreground_voxels"], 1)
            mask_fixture(path, values, qoffset=2)
            with self.assertRaisesRegex(ValueError, "qform"):
                read_reference_mask(path, binding)
            values[0, 0, 0] = .5
            mask_fixture(path, values)
            with self.assertRaisesRegex(ValueError, "binary"):
                read_reference_mask(path, binding)
            values[1, 0, 2] = .5  # Exactly two levels 0/.5 are not a binary mask.
            mask_fixture(path, values)
            with self.assertRaisesRegex(ValueError, "binary"):
                read_reference_mask(path, binding)

    def test_exact_path_join_and_header_phase_does_not_need_voxels(self):
        prep, row, _, _ = text_fixture()
        census = [{"route": row["route"], "session": None, "contrast": row["contrast"],
                   "dataset_id": "ds000001", "path": "task-private/a_stat-z_statmap.nii.gz", "map_commit": "source"}]
        bindings = bind_target_paths(prep, census, Path("/authorized"))
        self.assertEqual(bindings[0]["map_id"], prep["maps"][0]["map_id"])
        self.assertEqual(bindings[0]["path"], "/authorized/ds000001/task-private/a_stat-z_statmap.nii.gz")
        census[0]["path"] = "../outside_stat-z_statmap.nii.gz"
        with self.assertRaises(ValueError):
            bind_target_paths(prep, census, Path("/authorized"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "header_only.nii.gz"
            binding, header = mask_fixture(path, np.zeros((2, 3, 4)))
            with gzip.open(path, "wb") as handle:
                handle.write(header)  # No voxel payload exists to read.
            report = check_target_headers([{"map_id": "a", "path": str(path)}], binding)
            self.assertEqual(report["status_counts"], {"compatible_header_only": 1})
            self.assertEqual(report["target_arrays_read"], 0)
            self.assertFalse(report["baseline_target_access"])


if __name__ == "__main__":
    unittest.main()
