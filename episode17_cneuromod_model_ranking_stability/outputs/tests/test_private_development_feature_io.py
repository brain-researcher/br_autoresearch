"""Full-development delivery checks with invented metadata, pixels and values.

Coordinator Slurm compute only. No acquired image, credential, model checkpoint,
neural value or pilot artifact is read. The persistence check retains all 2423
rows and all sixteen real feature widths; it writes one roughly 143 MiB private
fixture container in the compute job's temporary directory, then removes it.
"""
from contextlib import ExitStack
import errno
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest import mock
import zipfile

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import private_development_feature_io as feature_io
import run_private_development_features as runner


def metadata_fixture():
    concepts, images, provider = [], [], []
    for number in range(720):
        if number < 480:
            role, qualified = "development", 6 if number < 23 else 5
        elif number < 600:
            role, qualified = "support_calibration", 6 if number < 485 else 5
        else:
            role, qualified = "sealed_audit", 6 if number < 606 else 5
        cid, name = str(number + 1), f"invented_concept_{number:03}"
        concept = dict(things_category_nr=cid, uniqueID=name, provider_category_nr=cid,
            role=role, qualified_image_count=str(qualified),
            outer_fold=f"fold-{number // 80 + 1:02}" if role == "development" else "",
            development_block=f"block-{number % 24 + 1:02}" if role == "development" else "",
            audit_block=f"audit-{number % 6 + 1:02}" if role == "sealed_audit" else "")
        concepts.append(concept)
        for exemplar in range(1, 7):
            filename = f"invented_{exemplar:02}.jpg"
            path = f"images_fmri/{name}/{filename}"
            eligible = exemplar <= qualified
            images.append(dict(provider_image_path=path, image_name=filename, concept_name=name,
                things_category_nr=cid, provider_category_nr=cid, provider_exemplar_nr=str(exemplar),
                condition="exp", role=role, outer_fold=concept["outer_fold"],
                development_block=concept["development_block"], audit_block=concept["audit_block"],
                eligible_image=str(eligible), common_to_four=str(eligible),
                failure_reasons="" if eligible else "invented_ineligible"))
            provider.append(dict(image_path=path, condition="exp", things_category_nr=cid,
                                 category_nr=cid, exemplar_nr=str(exemplar)))
    # Deliberately nonnumeric order: the full route must not inherit pilot sorting.
    images.reverse()
    report = dict(status="completed", mode="full", metadata_assignments_complete=True,
        preserved_metadata=dict(concepts=720, images=4320, presentations=49980, runs=833),
        role_counts=dict(development=480, support_calibration=120, sealed_audit=120))
    return report, concepts, images, provider


def pilot_report():
    return dict(analysis_id="stimulus_features", attempt="pilot002",
        status="completed_private_staging", pilot_feature_extraction_complete=True,
        archive_content_identity_verified=True, images_preprocessed=32,
        eligible_development_images=2423, completed_models=list(feature_io.nested.MODEL_IDS),
        audit_pixels_or_features_opened=False, neural_values_read=False, peak_rss_kib=123456,
        cost_projection_coverage=dict(fixed_seconds=7.0, variable_seconds_including_residual=11.0))


class PrivateDevelopmentFeatureIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Actual runtime guard is deliberately not mocked: login execution,
        # changed package versions or excess threads must still refuse.
        feature_io.delivery.features.check_runtime()
        cls.metadata = metadata_fixture()
        cls.design, cls.manifest = feature_io.bind_development_metadata(*cls.metadata)

    def test_complete_metadata_preserves_original_order_and_refuses_role_join_drift(self):
        report, concepts, images, provider = self.metadata
        expected = tuple(row["provider_image_path"] for row in images
                         if row["role"] == "development" and row["eligible_image"] == "True")
        self.assertEqual(self.design.image_ids, expected)
        self.assertEqual(len(expected), 2423)
        self.assertEqual(len(set(self.design.concept_ids)), 480)
        self.assertEqual(self.design.fold_labels, tuple(f"fold-{i:02}" for i in range(6, 0, -1)))
        self.assertEqual(len(self.design.block_labels), 24)
        self.assertTrue(all(type(label) is str for label in self.design.fold_ids + self.design.block_ids))
        self.assertEqual([row["feature_row_zero_based"] for row in feature_io.member_rows(self.design)],
                         list(range(2423)))
        for field, value in (("role", "sealed_audit"), ("outer_fold", "fold-99"),
                             ("provider_exemplar_nr", "99"), ("eligible_image", "False")):
            changed = [dict(row) for row in images]
            index = next(i for i, row in enumerate(changed)
                         if row["provider_image_path"] == expected[0])
            changed[index][field] = value
            with self.subTest(field=field), self.assertRaises((feature_io.FeatureIOError,
                    feature_io.archive.Refusal, feature_io.nested.NestedPredictionError)):
                feature_io.bind_development_metadata(report, concepts, changed, provider)
        with self.assertRaises(feature_io.FeatureIOError):
            feature_io.bind_development_metadata({**report, "metadata_assignments_complete": False},
                                                  concepts, images, provider)

    def test_complete_private_container_roundtrip_binding_and_abort_never_publish_partial(self):
        nested = feature_io.nested
        panels = {}
        for model_index, model in enumerate(nested.MODEL_IDS):
            layers = {layer: (np.arange(2423, dtype=np.float32)[:, None]
                      + np.arange(nested.FEATURE_WIDTHS[layer], dtype=np.float32)[None, :] / 8
                      + model_index * 4096 + layer_index * 512)
                      for layer_index, layer in enumerate(nested.LAYERS)}
            panels[model] = nested.FeaturePanel(self.design, MappingProxyType(layers))
        cap, events = 256 * 1024 * 1024, []
        with tempfile.TemporaryDirectory(prefix="ep17-feature-io-") as directory:
            path = Path(directory) / "features.h5"
            os.chmod(directory, 0o700)
            result = feature_io.save_complete_features(path, self.design, self.manifest, panels,
                                                       byte_cap=cap, check=events.append)
            self.assertGreater(result["bytes"], feature_io.FEATURE_BYTES)
            self.assertTrue(result["partial_cleanup_complete"])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(path.with_name("features.h5.partial").exists())
            self.assertEqual(events.count("after_feature_dataset_write"), 16)
            restored = feature_io.load_complete_features(path, self.design, self.manifest,
                                                         byte_cap=cap, check=events.append)
            self.assertEqual(tuple(restored), nested.MODEL_IDS)
            for model in nested.MODEL_IDS:
                self.assertIs(restored[model].rows, self.design)
                for layer in nested.LAYERS:
                    values = restored[model].layers[layer]
                    np.testing.assert_array_equal(values, panels[model].layers[layer])
                    with self.assertRaises(ValueError):
                        values.flags.writeable = True
            del restored
            unchanged = (path.stat().st_size, path.stat().st_mtime_ns)
            with self.assertRaises(feature_io.FeatureIOError):
                feature_io.save_complete_features(path, self.design, self.manifest, panels,
                                                  byte_cap=cap, check=events.append)
            self.assertEqual((path.stat().st_size, path.stat().st_mtime_ns), unchanged)
            self.assertFalse(path.with_name("features.h5.partial").exists())
            # Exact manifest content is bound, not merely shape/row counts.
            changed_manifest = tuple({**row, "extra_fixture_field": "changed"} for row in self.manifest)
            reads = []
            original_getitem = h5py.Dataset.__getitem__

            def read_dataset(dataset, key):
                reads.append(dataset.name)
                return original_getitem(dataset, key)

            with mock.patch.object(h5py.Dataset, "__getitem__", read_dataset):
                with self.assertRaisesRegex(feature_io.FeatureIOError, "feature_index_"):
                    feature_io.load_complete_features(path, self.design, changed_manifest,
                                                      byte_cap=cap, check=events.append)
            self.assertTrue(all(name == "/index_json" for name in reads))
            # A bad last header must refuse before reading any feature values.
            with h5py.File(path, "r+") as target:
                del target[nested.MODEL_IDS[-1]][nested.LAYERS[-1]]
                target[nested.MODEL_IDS[-1]].create_dataset(nested.LAYERS[-1],
                    shape=(2423, nested.FEATURE_WIDTHS[nested.LAYERS[-1]]), dtype="float64")
            with mock.patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("early value read")):
                with self.assertRaisesRegex(feature_io.FeatureIOError, "feature_dataset_header_changed"):
                    feature_io.load_complete_features(path, self.design, self.manifest,
                                                      byte_cap=cap, check=events.append)
            failed = Path(directory) / "failed"
            failed.mkdir(mode=0o700)
            failed.chmod(0o700)

            def stop_after_first_dataset(operation):
                if operation == "after_feature_dataset_write":
                    raise RuntimeError("invented budget exhaustion")

            with self.assertRaisesRegex(RuntimeError, "budget exhaustion"):
                feature_io.save_complete_features(failed / "features.h5", self.design, self.manifest,
                    panels, byte_cap=cap, check=stop_after_first_dataset)
            self.assertFalse((failed / "features.h5").exists())
            partial = failed / "features.h5.partial"
            self.assertTrue(partial.exists())
            with h5py.File(partial, "r") as source:
                self.assertEqual(source.attrs["complete"], 0)
            with self.assertRaises(FileExistsError):
                feature_io.save_complete_features(failed / "features.h5", self.design, self.manifest,
                    panels, byte_cap=cap, check=events.append)

    def test_exact_development_member_bounds_and_image_lifetime(self):
        image_id = self.design.image_ids[0]
        encoded = io.BytesIO()
        with Image.new("RGB", (2, 2), (10, 20, 30)) as image:
            image.save(encoded, format="JPEG")
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w") as target:
            target.writestr(image_id, encoded.getvalue())
        events = []
        with zipfile.ZipFile(packed, "r") as source:
            with feature_io.exact_image_loader(source, frozenset((image_id,)), image_id,
                                               b"invented-token", events.append) as opened:
                self.assertEqual(opened.size, (2, 2))
                self.assertEqual(opened.format, "JPEG")
                opened.load()
            with self.assertRaises(ValueError):
                opened.getpixel((0, 0))
            self.assertEqual(events, ["before_member_read", "after_member_read", "after_image_close"])
            with self.assertRaisesRegex(RuntimeError, "invented caller failure"):
                with feature_io.exact_image_loader(source, frozenset((image_id,)), image_id,
                                                   b"invented-token", events.append) as interrupted:
                    interrupted.load()
                    raise RuntimeError("invented caller failure")
            with self.assertRaises(ValueError):
                interrupted.getpixel((0, 0))
            with mock.patch.object(source, "getinfo", side_effect=AssertionError("outside lookup")):
                with self.assertRaisesRegex(feature_io.FeatureIOError, "outside_development"):
                    with feature_io.exact_image_loader(source, frozenset((image_id,)),
                            "images_fmri/invented_audit/closed.jpg", b"invented-token", events.append):
                        self.fail("closed role reached decoder")
            with mock.patch.object(feature_io.pilot, "PIXEL_CAP", 3):
                with self.assertRaisesRegex(feature_io.FeatureIOError, "pixel_cap"):
                    with feature_io.exact_image_loader(source, frozenset((image_id,)), image_id,
                                                       b"invented-token", events.append):
                        self.fail("oversized image reached producer")

    def test_pilot_cost_evidence_secret_pipe_and_full_completion_closeout(self):
        projection = runner.pilot_projection(pilot_report())
        self.assertEqual(projection["buffered_seconds_estimate_not_admission"], 2 * (7 + 2423 / 32 * 11) + 60)
        self.assertEqual(projection["batch1_forward_count"], 9692)
        self.assertEqual(projection["full_models_resident_together"], 4)
        for change in ({"images_preprocessed": 31}, {"status": "stopped"},
                       {"neural_values_read": True}, {"completed_models": ["DINO_R50"]}):
            with self.subTest(change=change), self.assertRaises(runner.FullFeatureRefusal):
                runner.pilot_projection({**pilot_report(), **change})
        selected_pilot_report = runner.REPORT_ROOT / "pilot002/pilot_report.json"
        with mock.patch.object(Path, "open", return_value=io.BytesIO(json.dumps(pilot_report()).encode())):
            self.assertEqual(runner._read_projection(selected_pilot_report), projection)
        with mock.patch.object(Path, "open", return_value=io.BytesIO(json.dumps(pilot_report()).encode())):
            with self.assertRaises(runner.FullFeatureRefusal):
                runner._read_projection(runner.PILOT_REPORT)  # pilot002 content cannot stand in for pilot001.
        with mock.patch.object(Path, "open") as unopened:
            with self.assertRaises(runner.FullFeatureRefusal):
                runner._read_projection(Path("/invented-other-episode/pilot002/pilot_report.json"))
            unopened.assert_not_called()
        # Delivery proves fixed metadata precedes own private notice, with only
        # the token in an inherited pipe and no token in argv or environment.
        token, order, inherited = b"invented-full-token", [], []
        arguments = ["--attempt", "full001", "--deadline-epoch", "1234567890",
                     "--rss-limit-kib", "123", "--private-byte-cap", "268435456",
                     "--report-byte-cap", "65536", "--pilot-report", str(selected_pilot_report)]
        args = SimpleNamespace(attempt="full001", pilot_report=selected_pilot_report)

        def notice(path, mode):
            self.assertEqual(path, feature_io.archive.ACQUISITION / "terms/password_images.txt")
            self.assertEqual(order, ["projection", "metadata"])
            order.append("notice")
            return io.BytesIO(b"Password for invented fixture: " + token + b"\n")

        def execute(executable, values):
            self.assertEqual(order, ["projection", "metadata", "notice"])
            self.assertEqual(values[3:-2], arguments)
            self.assertEqual(values[-2], "--private-password-fd")
            self.assertNotIn(token.decode(), " ".join(values))
            self.assertNotIn(token.decode(), " ".join(os.environ.values()))
            descriptor = int(values[-1])
            self.assertTrue(os.get_inheritable(descriptor))
            self.assertEqual(os.read(descriptor, 100), token)
            self.assertEqual(os.read(descriptor, 1), b"")
            inherited.append(descriptor)
            raise OSError("invented exec failure")

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "_read_projection", side_effect=lambda path: order.append("projection")))
            stack.enter_context(mock.patch.object(feature_io, "load_development_metadata", side_effect=lambda: order.append("metadata")))
            stack.enter_context(mock.patch.object(Path, "exists", return_value=False))
            stack.enter_context(mock.patch.object(Path, "open", autospec=True, side_effect=notice))
            stack.enter_context(mock.patch.object(os, "execv", side_effect=execute))
            with self.assertRaisesRegex(OSError, "invented exec failure"):
                runner._deliver_and_exec(args, arguments)
        self.assertEqual(len(inherited), 1)
        with self.assertRaises(OSError) as closed:
            os.fstat(inherited[0])
        self.assertEqual(closed.exception.errno, errno.EBADF)
        # The runner's new orchestration is exercised, not just its helpers.
        # Only already-qualified image/model work and archive auth are mocked.
        for completed, interrupted, expected_status in (
                (2423, False, "completed_private_staging"), (2422, False, "stopped"),
                (2423, True, "stopped")):
            with self.subTest(completed=completed, interrupted=interrupted), tempfile.TemporaryDirectory(prefix="ep17-full-cli-") as directory:
                root = Path(directory)
                archive_path = root / "invented.zip"
                with zipfile.ZipFile(archive_path, "w"):
                    pass
                descriptor, writer = os.pipe()
                os.write(writer, token)
                os.close(writer)
                args = SimpleNamespace(attempt="full001", deadline_epoch=runner.time.time() + 60,
                    rss_limit_kib=4 * 1024 * 1024, private_byte_cap=256 * 1024 * 1024,
                    report_byte_cap=65536, private_password_fd=descriptor, pilot_report=selected_pilot_report)
                sentinel = object()

                def producer(design, manifest, *, image_loader, random_checkpoint_path, resource_check):
                    self.assertIs(design, self.design)
                    self.assertIs(manifest, self.manifest)
                    self.assertEqual(random_checkpoint_path, feature_io.pilot.RANDOM_CHECKPOINT)
                    for operation, count in (("load_backbone", 4), ("preprocess_image", completed),
                                             ("infer_image", completed * 4)):
                        for _ in range(count):
                            resource_check(SimpleNamespace(operation=operation, when="after"))
                    return sentinel

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(runner, "PRIVATE_ROOT", root / "private"))
                    stack.enter_context(mock.patch.object(runner, "REPORT_ROOT", root / "reports"))
                    stack.enter_context(mock.patch.object(runner, "_read_projection", return_value=projection))
                    stack.enter_context(mock.patch.object(feature_io, "load_development_metadata", return_value=(self.design, self.manifest)))
                    stack.enter_context(mock.patch.object(feature_io.archive, "ARCHIVE", archive_path))
                    stack.enter_context(mock.patch.object(feature_io.pilot, "_verify_archive", return_value=("invented-identity", [])))
                    stack.enter_context(mock.patch.object(feature_io.pilot, "_identity", return_value="invented-identity"))
                    stack.enter_context(mock.patch.object(feature_io.delivery, "build_development_features", side_effect=producer))
                    save = stack.enter_context(mock.patch.object(feature_io, "save_complete_features",
                        return_value={"bytes": feature_io.FEATURE_BYTES + 1234, "partial_cleanup_complete": True}))
                    previous_trace = sys.gettrace()
                    stops = []

                    def post_publication_stop(frame, event, _arg):
                        if (frame.f_code is runner._run.__code__ and event == "line"
                                and frame.f_locals.get("report", {}).get("full_development_features_complete") is True):
                            stops.append(True)
                            raise runner.FullFeatureRefusal()
                        return post_publication_stop

                    try:
                        # Deterministic analogue of a signal arriving after
                        # publication; no real signal or timing race in tests.
                        if interrupted:
                            sys.settrace(post_publication_stop)
                        self.assertEqual(runner._run(args), 0 if expected_status == "completed_private_staging" else 1)
                    finally:
                        sys.settrace(previous_trace)
                    self.assertEqual(stops, [True] if interrupted else [])
                report = json.loads((root / "reports/full001/feature_report.json").read_text())
                self.assertEqual(report["status"], expected_status)
                self.assertEqual(report["selected_pilot_report"], str(selected_pilot_report))
                self.assertEqual(report["full_development_features_complete"], completed == 2423)
                for field in ("private_artifacts_releasable", "reader_handoffs_activated",
                              "candidate_scoring_authorized", "audit_pixels_or_features_opened", "neural_values_read"):
                    self.assertIs(report[field], False)
                self.assertIsNone(args.private_password_fd)
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
                if completed == 2423:
                    save.assert_called_once()
                    self.assertIs(save.call_args.args[3], sentinel)
                else:
                    save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
