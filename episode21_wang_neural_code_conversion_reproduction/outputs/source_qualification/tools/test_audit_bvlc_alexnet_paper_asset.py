from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_bvlc_alexnet_paper_asset.py")
SPEC = importlib.util.spec_from_file_location("audit_bvlc_alexnet_paper_asset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeHeaders:
    def __init__(self, lengths):
        self.lengths = lengths

    def get_all(self, key: str):
        if key == "Content-Length":
            return self.lengths
        return None


class FakeResponse:
    def __init__(self, payload: bytes, url: str, *, status: int = 200, lengths=None):
        self.payload = io.BytesIO(payload)
        self.url = url
        self.status = status
        self.headers = FakeHeaders(
            [str(len(payload))] if lengths is None else lengths
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url

    def read(self, size: int = -1):
        return self.payload.read(size)


def qualification_fixture():
    caffenet = {
        "basename": "bvlc_reference_caffenet.pt",
        "path": "bvlc_reference_caffenet/bvlc_reference_caffenet.pt",
        "sha256": "24016c5822dfc778796f9410fc5d2d474bb451d86094747c8c5b1274f8bd9a00",
        "uncompressed_bytes": 243866815,
    }
    lab_path = "/home/kiss/data/models_shared/pytorch/bvlc_alexnet/bvlc_alexnet.pt"
    return {
        "schema_version": MODULE.EXPECTED_QUALIFICATION_SCHEMA,
        "astra_analysis_id": MODULE.ASTRA_ANALYSIS_ID,
        "attempt": MODULE.EXPECTED_QUALIFICATION_ATTEMPT,
        "paper_code_discrepancy_register": {
            "alexnet_backend_and_input_size": {
                "paper": "official Caffe bvlc_alexnet with 224 by 224 inputs",
                "state": "recorded_unresolved_for_pipeline_conformance",
                "tagged_code": "bdpy PyTorch AlexNet port with direct 227 by 227 resize",
            }
        },
        "source_pipeline_assets": {
            "public_model_payload_audit": {
                "schema_version": "ep21.public_model_payload_audit.v2",
                "attempt": MODULE.EXPECTED_QUALIFICATION_ATTEMPT,
                "checkpoint_deserialization_performed": False,
                "empirical_or_decoded_payload_accessed": False,
                "ood_or_ep04_payload_accessed": False,
                "evaluator_checkpoint_references": {
                    "alexnet": {
                        "state": (
                            "lab_local_checkpoint_unmanifested_"
                            "public_caffenet_not_substitutable"
                        ),
                        "exact_public_bvlc_alexnet_named_member_verified": False,
                        "caffenet_equivalence_claim_allowed": False,
                        "lab_local_path": lab_path,
                        "public_caffenet_candidate_member": caffenet,
                    }
                },
                "source_requirements": {
                    "evaluator_branches": {
                        "alexnet": {
                            "checkpoint_path": lab_path,
                            "checkpoint_provenance": "lab_local_unchecksummed",
                            "image_size": [227, 227],
                            "model_class": "AlexNet",
                            "layer_map": "alexnet",
                        }
                    }
                },
            }
        },
    }


class BvlcAlexNetAssetAuditTests(unittest.TestCase):
    def test_metadata_fetch_rejects_transport_and_payload_changes(self):
        payload = b"pinned metadata\n"
        digest = hashlib.sha256(payload).hexdigest()

        def valid_opener(request, timeout):
            del timeout
            return FakeResponse(payload, request.full_url)

        self.assertEqual(
            MODULE.fetch_pinned_text(MODULE.README_URL, digest, open_url=valid_opener),
            payload.decode(),
        )

        def status_opener(request, timeout):
            del timeout
            return FakeResponse(payload, request.full_url, status=503)

        def redirect_opener(request, timeout):
            del request, timeout
            return FakeResponse(payload, "https://raw.githubusercontent.com/other")

        with self.assertRaises(ValueError):
            MODULE.fetch_pinned_text(MODULE.README_URL, digest, open_url=status_opener)
        with self.assertRaises(ValueError):
            MODULE.fetch_pinned_text(MODULE.README_URL, digest, open_url=redirect_opener)
        with self.assertRaises(ValueError):
            MODULE.fetch_pinned_text(MODULE.README_URL, "0" * 64, open_url=valid_opener)

        oversized = b"x" * (MODULE.MAX_METADATA_BYTES + 1)

        def oversized_opener(request, timeout):
            del timeout
            return FakeResponse(oversized, request.full_url)

        with self.assertRaises(ValueError):
            MODULE.fetch_pinned_text(
                MODULE.README_URL,
                hashlib.sha256(oversized).hexdigest(),
                open_url=oversized_opener,
            )
        with self.assertRaises(ValueError):
            MODULE._validate_url("https://example.org/readme.md", metadata=True)

    def test_provider_readme_requires_all_exact_fields(self):
        text = """---
name: BAIR/BVLC AlexNet Model
caffemodel: bvlc_alexnet.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
license: unrestricted
sha1: 9116a64c0fbe4459d18f4bb6b56d647b63920377
caffe_commit: 709dc15af4a06bebda027c1eb2b3f3e3375d5077
---
Provider notes.
"""
        parsed = MODULE.parse_provider_readme(text)
        self.assertEqual(parsed["sha1"], MODULE.MODEL_SHA1)
        with self.assertRaises(ValueError):
            MODULE.parse_provider_readme(text.replace("unrestricted", "unknown"))
        with self.assertRaises(ValueError):
            MODULE.parse_provider_readme(text.rsplit("---\n", 1)[0])
        with self.assertRaises(ValueError):
            MODULE.parse_provider_readme(
                text.replace(
                    "license: unrestricted\n",
                    "license: unknown\nlicense: unrestricted\n",
                )
            )

    def test_deploy_parser_requires_shape_layers_and_groups(self):
        blocks = [
            'name: "AlexNet"',
            'input: "data"',
            "input_dim: 10",
            "input_dim: 3",
            "input_dim: 227",
            "input_dim: 227",
        ]
        for name in ("conv1", "conv2", "conv3", "conv4", "conv5"):
            group = "group: 2" if name in {"conv2", "conv4", "conv5"} else ""
            blocks.append(
                f'layer {{\n name: "{name}"\n type: "Convolution"\n {group}\n}}'
            )
        for name in ("fc6", "fc7", "fc8"):
            blocks.append(f'layer {{\n name: "{name}"\n type: "InnerProduct"\n}}')
        text = "\n".join(blocks)
        parsed = MODULE.parse_deploy_prototxt(text)
        self.assertEqual(parsed["input_shape"], [10, 3, 227, 227])
        self.assertEqual(parsed["grouped_convolution_layers"], ["conv2", "conv4", "conv5"])
        legacy_text = (
            text.replace("layer {", "layers {")
            .replace('type: "Convolution"', "type: CONVOLUTION")
            .replace('type: "InnerProduct"', "type: INNER_PRODUCT")
        )
        self.assertEqual(
            MODULE.parse_deploy_prototxt(legacy_text)["input_shape"],
            [10, 3, 227, 227],
        )
        with self.assertRaises(ValueError):
            MODULE.parse_deploy_prototxt(text.replace("input_dim: 227", "input_dim: 224"))
        with self.assertRaises(ValueError):
            MODULE.parse_deploy_prototxt(
                text.replace('type: "Convolution"', 'type: "InnerProduct"', 1)
            )
        with self.assertRaises(ValueError):
            MODULE.parse_deploy_prototxt(
                text.replace(
                    'name: "conv1"\n type: "Convolution"',
                    'name: "conv1"\n type: "Convolution"\n # group: 2',
                )
                .replace('name: "conv2"\n type: "Convolution"\n group: 2',
                         'name: "conv2"\n type: "Convolution"\n # group: 2')
            )

    def test_modern_deploy_input_layer_and_comments_are_structural(self):
        blocks = [
            'name: "AlexNet"',
            '# input_dim: 10',
            'layer {\n name: "data"\n type: "Input"\n '
            'input_param { shape: { dim: 10 dim: 3 dim: 227 dim: 227 } }\n}',
        ]
        for name in ("conv1", "conv2", "conv3", "conv4", "conv5"):
            group = "group: 2" if name in {"conv2", "conv4", "conv5"} else ""
            blocks.append(
                f'layer {{\n name: "{name}"\n type: "Convolution"\n '
                f'{group}\n # group: 9\n}}'
            )
        for name in ("fc6", "fc7", "fc8"):
            blocks.append(f'layer {{\n name: "{name}"\n type: "InnerProduct"\n}}')
        parsed = MODULE.parse_deploy_prototxt("\n".join(blocks))
        self.assertEqual(parsed["input_shape"], [10, 3, 227, 227])
        self.assertEqual(parsed["grouped_convolution_layers"], ["conv2", "conv4", "conv5"])

    def test_write_all_retries_short_writes(self):
        observed = bytearray()

        def short_write(descriptor, payload):
            self.assertEqual(descriptor, 7)
            chunk = bytes(payload[:2])
            observed.extend(chunk)
            return len(chunk)

        MODULE._write_all(7, b"abcdefg", write=short_write)
        self.assertEqual(observed, b"abcdefg")
        with self.assertRaises(OSError):
            MODULE._write_all(7, b"x", write=lambda descriptor, payload: 0)

    def test_qualification_binding_requires_canonical_path_and_sha(self):
        fixture = qualification_fixture()
        encoded = (json.dumps(fixture, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(encoded).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode"
            expected = episode / MODULE.QUALIFICATION_REPORT_RELATIVE
            expected.parent.mkdir(parents=True)
            expected.write_bytes(encoded)
            parsed, facts, binding = MODULE.load_qualification_binding(
                episode,
                expected,
                expected_sha256=digest,
            )
            self.assertEqual(parsed["attempt"], 17)
            self.assertFalse(facts["tagged_checkpoint"]["authenticated"])
            self.assertEqual(binding["sha256"], digest)
            with self.assertRaises(ValueError):
                MODULE.load_qualification_binding(
                    episode,
                    expected,
                    expected_sha256="0" * 64,
                )
            decoy = episode / "decoy.json"
            decoy.write_bytes(encoded)
            with self.assertRaises(ValueError):
                MODULE.load_qualification_binding(
                    episode,
                    decoy,
                    expected_sha256=digest,
                )

    def test_qualification_binding_rejects_symlinked_ancestor(self):
        fixture = qualification_fixture()
        encoded = (json.dumps(fixture, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(encoded).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode"
            (episode / "outputs").mkdir(parents=True)
            external = root / "external" / "attempt-017"
            external.mkdir(parents=True)
            (external / "qualification_report.json").write_bytes(encoded)
            (episode / "outputs" / "source_qualification").symlink_to(
                external.parent,
                target_is_directory=True,
            )
            expected = episode / MODULE.QUALIFICATION_REPORT_RELATIVE
            with self.assertRaises(OSError):
                MODULE.load_qualification_binding(
                    episode,
                    expected,
                    expected_sha256=digest,
                )

    def test_frozen_attempt17_binding_matches_real_artifact(self):
        parsed, facts, binding = MODULE.load_qualification_binding(
            MODULE.EXPECTED_EPISODE_DIR,
            MODULE.EXPECTED_EPISODE_DIR / MODULE.QUALIFICATION_REPORT_RELATIVE,
        )
        self.assertEqual(parsed["attempt"], MODULE.EXPECTED_QUALIFICATION_ATTEMPT)
        self.assertEqual(binding["sha256"], MODULE.EXPECTED_QUALIFICATION_SHA256)
        self.assertFalse(facts["tagged_checkpoint"]["authenticated"])

    def test_report_keeps_three_assets_and_two_decisions_distinct(self):
        facts = MODULE.validate_qualification_facts(qualification_fixture())
        provider = {
            "name": "BAIR/BVLC AlexNet Model",
            "caffemodel": MODULE.MODEL_FILENAME,
            "caffemodel_url": MODULE.MODEL_URL,
            "license": "unrestricted",
            "sha1": MODULE.MODEL_SHA1,
            "caffe_commit": MODULE.BVLC_MODEL_TRAINING_CAFFE_COMMIT,
        }
        deploy = {
            "network_name": "AlexNet",
            "input_shape": [10, 3, 227, 227],
            "feature_layers": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "fc8"],
            "grouped_convolution_layers": ["conv2", "conv4", "conv5"],
        }
        report = MODULE.build_report(
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:01:00+00:00",
            process_id=1,
            qualification_binding={"sha256": MODULE.EXPECTED_QUALIFICATION_SHA256},
            qualification_facts=facts,
            provider=provider,
            deploy=deploy,
            cache={"observed_sha1": MODULE.MODEL_SHA1},
            cache_path=Path(MODULE.MODEL_FILENAME),
        )
        assets = report["asset_distinctions"]
        self.assertEqual(set(assets), {
            "official_paper_cited_bvlc_alexnet",
            "tagged_evaluator_bvlc_alexnet",
            "public_bvlc_reference_caffenet",
        })
        boundary = report["scientific_boundary"]
        self.assertFalse(boundary["tagged_pytorch_checkpoint_authenticated_by_this_audit"])
        self.assertFalse(boundary["official_caffemodel_equivalent_to_tagged_bvlc_alexnet_pt_verified"])
        self.assertFalse(boundary["official_caffemodel_satisfies_tagged_checkpoint_requirement"])
        self.assertFalse(boundary["caffemodel_to_pytorch_conversion_performed"])
        self.assertFalse(boundary["paper_caffe_vs_tagged_pytorch_backend_decision_resolved"])
        self.assertFalse(boundary["paper_text_224_vs_official_deploy_227_decision_resolved"])
        self.assertFalse(boundary["paper_faithful_pipeline_specification_complete"])
        self.assertFalse(boundary["paper_faithful_pipeline_runtime_ready"])

    def test_download_is_atomic_and_reusable(self):
        payload = b"official-model-fixture"
        expected_sha1 = hashlib.sha1(payload).hexdigest()

        def opener(request, timeout):
            del timeout
            return FakeResponse(payload, request.full_url)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            first, path = MODULE.download_or_reuse_model(
                cache,
                expected_size=len(payload),
                expected_sha1=expected_sha1,
                open_url=opener,
            )
            self.assertEqual(first["payload_bytes_downloaded_this_attempt"], len(payload))
            self.assertFalse(first["cache_reused_after_identity_check"])
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(path.stat().st_nlink, 1)
            self.assertEqual(path.stat().st_mode & 0o777, 0o440)

            def forbidden_opener(request, timeout):
                del request, timeout
                raise AssertionError("verified cache should be reused")

            second, _ = MODULE.download_or_reuse_model(
                cache,
                expected_size=len(payload),
                expected_sha1=expected_sha1,
                open_url=forbidden_opener,
            )
            self.assertTrue(second["cache_reused_after_identity_check"])
            self.assertEqual(second["payload_bytes_downloaded_this_attempt"], 0)

    def test_bad_download_is_not_published(self):
        payload = b"wrong-model"

        def opener(request, timeout):
            del timeout
            return FakeResponse(payload, request.full_url)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            with self.assertRaises(ValueError):
                MODULE.download_or_reuse_model(
                    cache,
                    expected_size=len(payload),
                    expected_sha1="0" * 40,
                    open_url=opener,
                )
            self.assertFalse((cache / MODULE.MODEL_FILENAME).exists())
            self.assertEqual(list(cache.glob("*.part.*")), [])

    def test_fresh_download_rehashes_the_stored_inode_before_publication(self):
        payload = b"official-model-fixture"
        expected_sha1 = hashlib.sha1(payload).hexdigest()

        def opener(request, timeout):
            del timeout
            return FakeResponse(payload, request.full_url)

        original_write_all = MODULE._write_all

        def corrupting_write(descriptor, block):
            original_write_all(descriptor, block)
            os.pwrite(descriptor, b"X", 0)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            with mock.patch.object(MODULE, "_write_all", side_effect=corrupting_write):
                with self.assertRaises(ValueError):
                    MODULE.download_or_reuse_model(
                        cache,
                        expected_size=len(payload),
                        expected_sha1=expected_sha1,
                        open_url=opener,
                    )
            self.assertFalse((cache / MODULE.MODEL_FILENAME).exists())

    def test_download_rejects_status_redirect_and_content_length(self):
        payload = b"official-model-fixture"
        expected_sha1 = hashlib.sha1(payload).hexdigest()
        cases = [
            {"status": 503},
            {"url": "http://dl.caffe.berkeleyvision.org/other.caffemodel"},
            {"lengths": [str(len(payload) + 1)]},
            {"lengths": [str(len(payload)), str(len(payload))]},
        ]
        for index, overrides in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                def opener(request, timeout):
                    del timeout
                    options = {
                        "status": overrides.get("status", 200),
                        "lengths": overrides.get("lengths"),
                    }
                    return FakeResponse(
                        payload,
                        overrides.get("url", request.full_url),
                        **options,
                    )

                cache = Path(directory) / "cache"
                with self.assertRaises(ValueError):
                    MODULE.download_or_reuse_model(
                        cache,
                        expected_size=len(payload),
                        expected_sha1=expected_sha1,
                        open_url=opener,
                    )
                self.assertFalse((cache / MODULE.MODEL_FILENAME).exists())

    def test_download_rejects_short_and_oversize_bodies(self):
        cases = [
            (b"abc", 4, hashlib.sha1(b"abcx").hexdigest()),
            (b"abcde", 4, hashlib.sha1(b"abcd").hexdigest()),
        ]
        for payload, expected_size, expected_sha1 in cases:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                def opener(request, timeout):
                    del timeout
                    return FakeResponse(
                        payload,
                        request.full_url,
                        lengths=[str(expected_size)],
                    )

                cache = Path(directory) / "cache"
                with self.assertRaises(ValueError):
                    MODULE.download_or_reuse_model(
                        cache,
                        expected_size=expected_size,
                        expected_sha1=expected_sha1,
                        open_url=opener,
                    )
                self.assertFalse((cache / MODULE.MODEL_FILENAME).exists())

    def test_symlink_cache_entry_is_rejected(self):
        payload = b"official-model-fixture"
        expected_sha1 = hashlib.sha1(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(payload)
            cache = root / "cache"
            cache.mkdir()
            (cache / MODULE.MODEL_FILENAME).symlink_to(target)
            with self.assertRaises(OSError):
                MODULE.download_or_reuse_model(
                    cache,
                    expected_size=len(payload),
                    expected_sha1=expected_sha1,
                )
            self.assertEqual(target.read_bytes(), payload)

    def test_symlink_cache_directory_is_rejected(self):
        payload = b"official-model-fixture"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            cache = root / "cache"
            cache.symlink_to(target, target_is_directory=True)
            with self.assertRaises(OSError):
                MODULE.download_or_reuse_model(
                    cache,
                    expected_size=len(payload),
                    expected_sha1=hashlib.sha1(payload).hexdigest(),
                )

    def test_symlinked_cache_ancestor_cannot_create_outside_path(self):
        payload = b"official-model-fixture"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            ancestor = root / "ancestor"
            ancestor.symlink_to(target, target_is_directory=True)
            cache = ancestor / "new-cache"
            with self.assertRaises(OSError):
                MODULE.download_or_reuse_model(
                    cache,
                    expected_size=len(payload),
                    expected_sha1=hashlib.sha1(payload).hexdigest(),
                )
            self.assertFalse((target / "new-cache").exists())

    def test_fifo_and_hardlink_cache_entries_are_rejected_without_network(self):
        payload = b"official-model-fixture"
        expected_sha1 = hashlib.sha1(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "fifo-cache"
            cache.mkdir()
            os.mkfifo(cache / MODULE.MODEL_FILENAME)
            with self.assertRaises(ValueError):
                MODULE.download_or_reuse_model(
                    cache,
                    expected_size=len(payload),
                    expected_sha1=expected_sha1,
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(payload)
            cache = root / "hardlink-cache"
            cache.mkdir()
            os.link(source, cache / MODULE.MODEL_FILENAME)
            with self.assertRaises(ValueError):
                MODULE.download_or_reuse_model(
                    cache,
                    expected_size=len(payload),
                    expected_sha1=expected_sha1,
                )

    def test_atomic_json_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            MODULE.atomic_json(output, {"ok": True})
            self.assertEqual(output.stat().st_nlink, 1)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                MODULE.atomic_json(output, {"ok": False})

    def test_atomic_json_refuses_symlink_without_changing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("original\n")
            output_dir = root / "output"
            output_dir.mkdir()
            output = output_dir / "report.json"
            output.symlink_to(target)
            with self.assertRaises(FileExistsError):
                MODULE.atomic_json(output, {"ok": False})
            self.assertEqual(target.read_text(), "original\n")


if __name__ == "__main__":
    unittest.main()
