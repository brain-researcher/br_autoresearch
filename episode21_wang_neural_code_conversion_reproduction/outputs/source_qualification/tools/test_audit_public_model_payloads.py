#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import io
import os
import stat
import tempfile
import unittest
import zipfile
from email.message import Message
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse


RUNNER = Path(__file__).with_name("audit_public_model_payloads.py")
SPEC = importlib.util.spec_from_file_location("ep21_public_model_payloads", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


EVALUATOR_SOURCE = '''
class DNNHandler:
    def __init__(self, encoder_name="AlexNet", device="cpu"):
        self.encoder_name = encoder_name
        self.device = device
        if encoder_name == "AlexNet":
            self.encoder = AlexNet()
            encoder_param_file = "/home/kiss/data/models_shared/pytorch/bvlc_alexnet/bvlc_alexnet.pt"
            self.encoder.to(device)
            self.encoder.load_state_dict(torch.load(encoder_param_file))
            self.encoder.eval()
            self.image_size = [227, 227]
            self.mean_image = np.float32([104., 117., 123.])
            self.layer_mapping = layer_map("alexnet")
        elif encoder_name == "VGG19":
            self.encoder = VGG19()
            encoder_param_file = "/home/kiss/data/models_shared/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt"
            self.encoder.to(device)
            self.encoder.load_state_dict(torch.load(encoder_param_file))
            self.encoder.eval()
            self.image_size = [224, 224]
            self.mean_image = np.float32([104., 117., 123.])
            self.layer_mapping = layer_map("vgg19")
        else:
            raise RuntimeError("unsupported")
        self.layers = list(self.layer_mapping.keys())
        self.feature_extractor = FeatureExtractor(self.encoder, self.layers, self.layer_mapping, device=self.device, detach=True)

    def get_activation(self, img_obj_list):
        for a_img in img_obj_list:
            a_img = a_img.resize(self.image_size, Image.LANCZOS)
            x = np.asarray(a_img)
            if self.encoder_name in ["AlexNet", "VGG19"]:
                x = np.transpose(x, (2, 0, 1))[::-1]
                x = np.float32(x) - np.reshape(self.mean_image, (3, 1, 1))
            features = self.feature_extractor.run(x)
'''


MANIFEST_SOURCE = '''
{
  "recon_demo": {
    "save_in": "./",
    "files": [
      {"name": "VGG_ILSVRC_19_layers.zip", "url": "https://figshare.com/ndownloader/files/38225868", "md5sum": "49f4d6dbf1024f33149352dfd309dc4d", "postproc": [{"name": "unzip", "destination": "./models"}]},
      {"name": "bvlc_reference_caffenet.zip", "url": "https://figshare.com/ndownloader/files/52398902", "md5sum": "2e4fe2126c7ed13554e33d716a72c65f", "postproc": [{"name": "unzip", "destination": "./models"}]}
    ]
  }
}
'''


VGG_CONFIG_SOURCE = '''
name: vgg19
parameters_file: ./data/models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt
input_image_shape: [224, 224, 3]
image_mean_file: ./data/models/VGG_ILSVRC_19_layers/ilsvrc_2012_mean.npy
feature_std_file: ./data/models/VGG_ILSVRC_19_layers/estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat
layers:
  - conv1_1
  - conv1_2
  - conv2_1
  - conv2_2
  - conv3_1
  - conv3_2
  - conv3_3
  - conv3_4
  - conv4_1
  - conv4_2
  - conv4_3
  - conv4_4
  - conv5_1
  - conv5_2
  - conv5_3
  - conv5_4
  - fc6
  - fc7
  - fc8
'''


def write_regular_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o444) << 16
    archive.writestr(info, payload)


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        length: str | None = None,
        final_url: str = "https://s3-eu-west-1.amazonaws.com/bucket/file.zip?secret=yes",
    ) -> None:
        super().__init__(body)
        self._status = status
        self._final_url = final_url
        self.headers = Message()
        if length is not None:
            self.headers.add_header("Content-Length", length)

    def getcode(self) -> int:
        return self._status

    def geturl(self) -> str:
        return self._final_url


class PublicModelPayloadAuditTests(unittest.TestCase):
    def test_pinned_asset_identities(self) -> None:
        self.assertEqual(
            MODULE.ASSETS["vgg19"]["md5"],
            "49f4d6dbf1024f33149352dfd309dc4d",
        )
        self.assertEqual(MODULE.ASSETS["vgg19"]["size_bytes"], 533_710_127)
        self.assertEqual(
            MODULE.ASSETS["caffenet"]["md5"],
            "2e4fe2126c7ed13554e33d716a72c65f",
        )
        self.assertEqual(MODULE.ASSETS["caffenet"]["size_bytes"], 226_699_018)

    def test_download_endpoints_bypass_human_facing_waf(self) -> None:
        self.assertEqual(
            {urlparse(asset["url"]).hostname for asset in MODULE.ASSETS.values()},
            {"ndownloader.figshare.com"},
        )

    def test_archive_member_safety(self) -> None:
        self.assertTrue(MODULE.safe_member_name("models/checkpoint.pt"))
        self.assertFalse(MODULE.safe_member_name("../checkpoint.pt"))
        self.assertFalse(MODULE.safe_member_name("/absolute/checkpoint.pt"))
        self.assertFalse(MODULE.safe_member_name(r"..\checkpoint.pt"))

    def test_zip_inventory_hashes_without_extracting(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            archive_path = Path(name) / "asset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                write_regular_member(
                    archive, "models/VGG_ILSVRC_19_layers.pt", b"weights"
                )
                write_regular_member(archive, "README.md", b"metadata")
            inventory = MODULE.inspect_zip(archive_path)
            self.assertEqual(inventory["file_count"], 2)
            self.assertEqual(inventory["unsafe_member_names"], [])
            self.assertEqual(inventory["non_regular_member_names"], [])
            matches = MODULE.matching_basename(
                inventory, "VGG_ILSVRC_19_layers.pt"
            )
            self.assertEqual(len(matches), 1)
            self.assertEqual(
                matches[0]["sha256"],
                "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
            )

    def test_zip_inventory_rejects_symlink_member(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            archive_path = Path(name) / "asset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                info = zipfile.ZipInfo("models/checkpoint.pt")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, b"target")
            inventory = MODULE.inspect_zip(archive_path)
            self.assertEqual(inventory["file_count"], 0)
            self.assertEqual(
                inventory["non_regular_member_names"], ["models/checkpoint.pt"]
            )

    def test_missing_exact_checkpoint_is_explicit(self) -> None:
        inventory = {
            "members": [
                {
                    "path": "models/bvlc_reference.caffemodel",
                    "basename": "bvlc_reference.caffemodel",
                }
            ]
        }
        self.assertEqual(
            MODULE.matching_basename(inventory, "bvlc_alexnet.pt"), []
        )

    def test_source_requirement_parser(self) -> None:
        parsed = MODULE.parse_source_requirements(
            EVALUATOR_SOURCE, "script.py", "source-md5"
        )
        self.assertEqual(
            parsed["evaluator_branches"]["alexnet"]["checkpoint_path"],
            "/home/kiss/data/models_shared/pytorch/bvlc_alexnet/bvlc_alexnet.pt",
        )
        self.assertEqual(
            parsed["evaluator_branches"]["vgg19"]["image_size"], [224, 224]
        )
        self.assertTrue(all(parsed["preprocessing_assertions"].values()))

    def test_source_archive_hash_and_parse_share_one_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            archive_path = directory / "source.zip"
            replacement_path = directory / "replacement.zip"

            def write_source(path: Path, evaluator: str) -> None:
                with zipfile.ZipFile(path, "w") as archive:
                    write_regular_member(
                        archive,
                        MODULE.SOURCE_ROOT + MODULE.IDENTIFICATION_SCRIPT_SUFFIX,
                        evaluator.encode(),
                    )
                    write_regular_member(
                        archive,
                        MODULE.SOURCE_ROOT + MODULE.RECONSTRUCTION_MANIFEST_SUFFIX,
                        MANIFEST_SOURCE.encode(),
                    )
                    write_regular_member(
                        archive,
                        MODULE.SOURCE_ROOT + MODULE.VGG_CONFIG_SUFFIX,
                        VGG_CONFIG_SOURCE.encode(),
                    )

            write_source(archive_path, EVALUATOR_SOURCE)
            write_source(
                replacement_path,
                EVALUATOR_SOURCE.replace(
                    "self.image_size = [227, 227]",
                    "self.image_size = [224, 224]",
                    1,
                ),
            )
            pinned_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
            original_stream_md5 = MODULE.source_archive_verifier._stream_md5
            replaced = False

            def digest_then_replace(handle):
                nonlocal replaced
                digest = original_stream_md5(handle)
                if not replaced:
                    os.replace(replacement_path, archive_path)
                    replaced = True
                return digest

            try:
                with mock.patch.object(
                    MODULE, "SOURCE_ARCHIVE_MD5", pinned_md5
                ), mock.patch.object(
                    MODULE, "SOURCE_ARCHIVE_SIZE_BYTES", archive_path.stat().st_size
                ), mock.patch.object(
                    MODULE.source_archive_verifier,
                    "_stream_md5",
                    side_effect=digest_then_replace,
                ):
                    requirements = MODULE.source_requirements(archive_path)
            except ValueError:
                requirements = None

            self.assertTrue(replaced)
            if requirements is not None:
                self.assertEqual(
                    requirements["evaluator_branches"]["alexnet"]["image_size"],
                    [227, 227],
                )

    def test_source_parser_rejects_branch_local_mutations_and_decoys(self) -> None:
        mutations = [
            EVALUATOR_SOURCE.replace("self.image_size = [227, 227]", "self.image_size = [224, 224]", 1),
            EVALUATOR_SOURCE.replace('self.layer_mapping = layer_map("vgg19")', 'self.layer_mapping = layer_map("alexnet")'),
            EVALUATOR_SOURCE.replace("self.encoder.load_state_dict(torch.load(encoder_param_file))", "self.encoder.eval()", 1),
            EVALUATOR_SOURCE.replace('if self.encoder_name in ["AlexNet", "VGG19"]:', 'if self.encoder_name in ["AlexNet"]:'),
            EVALUATOR_SOURCE.replace("Image.LANCZOS", "Image.BILINEAR"),
        ]
        for source in mutations:
            with self.subTest(source=source[:80]), self.assertRaises(ValueError):
                MODULE.parse_source_requirements(source, "script.py", "source-md5")

    def test_manifest_and_vgg_config_are_exact(self) -> None:
        manifest = MODULE.parse_reconstruction_manifest(MANIFEST_SOURCE)
        self.assertEqual(manifest["vgg19"]["figshare_file_id"], 38225868)
        config = MODULE.parse_vgg_config(VGG_CONFIG_SOURCE)
        self.assertEqual(
            config["archive_member_paths"]["checkpoint"],
            MODULE.VGG_CHECKPOINT_MEMBER,
        )
        for bad in [
            MANIFEST_SOURCE.replace("38225868", "52398902", 1),
            MANIFEST_SOURCE.replace('"./models"', '"models"', 1),
            MANIFEST_SOURCE.replace(
                '"files": [',
                '"files": [{"name":"VGG_ILSVRC_19_layers.zip","url":"https://figshare.com/ndownloader/files/38225868","md5sum":"49f4d6dbf1024f33149352dfd309dc4d","postproc":[{"name":"unzip","destination":"./models"}]},',
                1,
            ),
        ]:
            with self.subTest(kind="manifest"), self.assertRaises(ValueError):
                MODULE.parse_reconstruction_manifest(bad)
        for bad in [
            VGG_CONFIG_SOURCE.replace("./data/models/", "./models/", 1),
            VGG_CONFIG_SOURCE.replace("  - conv1_1\n", "", 1),
            VGG_CONFIG_SOURCE.replace("  - conv1_1\n  - conv1_2", "  - conv1_2\n  - conv1_1"),
        ]:
            with self.subTest(kind="config"), self.assertRaises(ValueError):
                MODULE.parse_vgg_config(bad)

    def test_valid_download_is_bounded_atomic_and_reusable(self) -> None:
        body = b"public-model"
        asset = {
            "filename": "asset.zip",
            "url": "https://ndownloader.figshare.com/files/1",
            "size_bytes": len(body),
            "md5": hashlib.md5(body).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as name:
            cache = Path(name)
            with mock.patch.object(
                MODULE,
                "_open_download",
                return_value=FakeResponse(body, length=str(len(body))),
            ) as opened:
                report, identity = MODULE.download_or_reuse(asset, cache)
            self.assertTrue(report["md5_matches"])
            self.assertNotIn("?secret", report["sanitized_final_url"])
            self.assertEqual(stat.S_IMODE((cache / "asset.zip").stat().st_mode), 0o440)
            self.assertEqual(identity[2], len(body))
            with mock.patch.object(
                MODULE, "_open_download", side_effect=AssertionError("network used")
            ):
                reused, _ = MODULE.download_or_reuse(asset, cache)
            self.assertTrue(reused["cache_reused_after_identity_check"])
            self.assertEqual(opened.call_count, 1)

    def test_download_rejects_symlink_status_length_and_oversize(self) -> None:
        body = b"abcd"
        asset = {
            "filename": "asset.zip",
            "url": "https://ndownloader.figshare.com/files/1",
            "size_bytes": len(body),
            "md5": hashlib.md5(body).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as name:
            cache = Path(name)
            sentinel = cache / "sentinel"
            sentinel.write_bytes(body)
            (cache / "asset.zip").symlink_to(sentinel)
            with mock.patch.object(
                MODULE, "_open_download", side_effect=AssertionError("network used")
            ), self.assertRaises(ValueError):
                MODULE.download_or_reuse(asset, cache)
            self.assertEqual(sentinel.read_bytes(), body)
        cases = [
            FakeResponse(body, status=206, length="4"),
            FakeResponse(body, length=None),
            FakeResponse(body, length="5"),
            FakeResponse(body + b"x", length="4"),
            FakeResponse(body, length="4", final_url="http://example.com/file"),
        ]
        for response in cases:
            with tempfile.TemporaryDirectory() as name, mock.patch.object(
                MODULE, "_open_download", return_value=response
            ), self.subTest(response=response), self.assertRaises(ValueError):
                MODULE.download_or_reuse(asset, Path(name))
            self.assertFalse(any(Path(name).glob(".asset.zip.part-*")))


if __name__ == "__main__":
    unittest.main()
