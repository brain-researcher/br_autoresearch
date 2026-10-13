#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("audit_unmanifested_assets.py")
SPEC = importlib.util.spec_from_file_location("ep21_unmanifested_assets", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UnmanifestedAssetAuditTest(unittest.TestCase):
    def test_hard_coded_checkpoints_and_unpinned_bdpy_are_detected(self) -> None:
        script = '''
if encoder_name == "AlexNet":
    encoder = AlexNet()
    encoder_param_file = '/home/kiss/x/bvlc_alexnet.pt'
elif encoder_name == "VGG19":
    encoder = VGG19()
    encoder_param_file = '/home/kiss/x/VGG_ILSVRC_19_layers.pt'
true_image_dir = '../../data/test_image/source/'
'''
        report = MODULE.build_report(
            attempt=6,
            process_id=1,
            started_at="2026-09-29T00:00:00+00:00",
            script_text=script,
            environment_text="dependencies:\n  - bdpy\n",
            declared_names={"VGG_ILSVRC_19_layers.zip", "bvlc_reference_caffenet.zip"},
        )
        gaps = report["gaps"]
        self.assertEqual(
            gaps["alexnet_identification_checkpoint"]["state"],
            "required_unmanifested",
        )
        self.assertFalse(
            gaps["alexnet_identification_checkpoint"][
                "any_alexnet_named_manifest_asset"
            ]
        )
        self.assertEqual(
            gaps["vgg19_identification_checkpoint_binding"]["state"],
            "candidate_archive_metadata_verified_payload_binding_unproven",
        )
        self.assertEqual(gaps["main_bdpy_dependency"]["state"], "declared_but_unpinned")
        self.assertEqual(report["payload_bytes_downloaded"], 0)


if __name__ == "__main__":
    unittest.main()
