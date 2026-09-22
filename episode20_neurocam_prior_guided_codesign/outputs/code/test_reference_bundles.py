#!/usr/bin/env python3
"""Focused standard-library checks for the EP20 reference examples."""

import copy
import unittest

import validate_reference_bundles as validator


def valid_boundary_bundle(bundle_id):
    spec = validator.BUNDLE_SPECS[bundle_id]
    bundle = {
        "schema_version": validator.SCHEMA_VERSION,
        "episode_id": "ep20",
        "bundle_id": bundle_id,
        "bundle_role": spec["role"],
        "reference_only": True,
        "example_only": True,
        "estimation_only": spec["estimation_only"],
        "launch_gate_effect": "none",
        "calibration_status": "uncalibrated_reference",
        "allowed_uses": sorted(validator.ALLOWED_USES),
        "prohibited_uses": sorted(validator.PROHIBITED_USES),
        "canonical_role_contracts": {
            "anchor_manifest": "../NEUROCAM_ANCHORS.yaml",
            "anchor_split_manifest": "../REFERENCE_ANCHOR_SPLIT.yaml",
            "this_bundle_is_canonical": False,
        },
        "source": {
            "doi": "10.1016/j.scib.2025.11.030",
            "pmid": "41309324",
            "source_artifact_included": False,
        },
    }
    for field in validator.FALSE_BOUNDARY_FIELDS:
        bundle[field] = False
    return bundle


class ReferenceBundleTests(unittest.TestCase):
    def test_provisioned_bundles_validate(self):
        missing = [
            spec["filename"]
            for spec in validator.BUNDLE_SPECS.values()
            if not (validator.bundle_directory() / spec["filename"]).is_file()
        ]
        if missing:
            self.skipTest(
                "requires separately provisioned ignored inputs: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        bundles = validator.validate_all()
        self.assertEqual(set(bundles), set(validator.BUNDLE_SPECS))

    def test_gate_lifting_is_rejected(self):
        bundle = copy.deepcopy(valid_boundary_bundle("neurocam_paper_direct_v1"))
        bundle["launch_gate_effect"] = "lift"
        with self.assertRaises(ValueError):
            validator.validate_boundary(
                bundle,
                "neurocam_paper_direct_v1",
                validator.BUNDLE_SPECS["neurocam_paper_direct_v1"],
            )

    def test_estimation_label_is_required(self):
        bundle = copy.deepcopy(valid_boundary_bundle("neurocam_figure_derived_v1"))
        bundle["estimation_only"] = False
        with self.assertRaises(ValueError):
            validator.validate_boundary(
                bundle,
                "neurocam_figure_derived_v1",
                validator.BUNDLE_SPECS["neurocam_figure_derived_v1"],
            )


if __name__ == "__main__":
    unittest.main()
