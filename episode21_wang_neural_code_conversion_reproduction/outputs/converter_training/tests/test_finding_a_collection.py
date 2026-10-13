"""Three collection checks; source-free and compute-allocation-only.

One real full fc8 layer is fabricated; the other854 slots are explicitly
unavailable. This tests storage/composition, not a reduced scientific panel or
a new bootstrap/recovery study. Existing metric/inference tests own arithmetic.
"""
from dataclasses import replace
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_collection as collection

prediction, metrics = collection.prediction, collection.metrics
inference, secondary = collection.inference, collection.secondary
PEOPLE = collection.PARTICIPANTS
IMAGES = tuple("fixture-image-%03d" % index for index in range(224))
TRAINING = {owner: tuple("fixture-%s-unique-%04d" % (owner, index) for index in range(4712))
            for owner in PEOPLE}
KEY = ("content_loss", PEOPLE[1], PEOPLE[0], "fc8")
IDENTITY = dict(role="tau_test", dtype="float32", checkpoint="fixture-checkpoint",
                preprocessing="fixture-preprocessing", extraction_status="complete")


def readonly(value):
    value.setflags(write=False)
    return value


def settings():
    axes = {owner: prediction.NativeAxis(owner, np.array([2, 5, 9, 20]),
        np.array([2, 5, 9, 20]), np.arange(4), np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32)) for owner in PEOPLE}
    objects = {stream: None for stream in collection.STREAMS}
    objects[KEY[:3]] = "fixture-completed-content-model"
    objects[("brain_loss", KEY[1], KEY[2])] = "fixture-completed-brain-model"
    return dict(image_ids=IMAGES, decoder_axes_by_owner=axes,
        decoder_training_image_ids_by_owner=TRAINING, fitted_object_ids=objects,
        feature_identity=IDENTITY, budget=SimpleNamespace(check=lambda: None),
        maximum_bytes=32 * 1024**2, require_admitted=lambda: None)


def values():
    rows = np.arange(224, dtype=np.float32)[:, None]
    units = np.arange(1000, dtype=np.float32)[None, :]
    predicted = rows / np.float32(16) + units / np.float32(32)
    truth = predicted * np.float32(2) + units % np.float32(7)
    return predicted, truth


def blocks(key=KEY, *, scale_change=False):
    predicted, _ = values()
    method, source, target, layer = key
    for first in range(0, 1000, 256):
        ids = np.arange(first, min(first + 256, 1000), dtype=np.int64)
        mean = ids.astype(np.float32) / np.float32(8)
        std = np.ones(len(ids), dtype=np.float32) + (ids % 3).astype(np.float32) / np.float32(2)
        if scale_change:
            std *= np.float32(2)
        decoder = prediction.DecoderBlock(target, layer, None, first, ids, ids,
            np.tile([2, 5, 9, 20], (len(ids), 1)), np.tile(np.arange(4), (len(ids), 1)),
            np.zeros((len(ids), 4), dtype=np.float32), np.zeros(len(ids), dtype=np.float32), mean, std)
        data = readonly(predicted[:, ids].copy())
        block = prediction.FeaturePredictionBlock(method, source, target, target,
            IMAGES, layer, None, ids, ids, data.astype(np.float64), data.astype(np.float64), data)
        yield block, decoder


def truth_provider(*, layer, image_ids, full_layer_C_index, feature_identity):
    _, truth = values()
    chosen = truth[:, full_layer_C_index]
    return collection.TruthBlock(image_ids, layer, full_layer_C_index, chosen,
                                 np.isfinite(chosen), feature_identity)


def seal(writer):
    for key in collection.SLOTS:
        if key != KEY:
            method, source, target, layer = key
            writer.unavailable_layer(method=method, source=source, target=target,
                                     layer=layer, reason="fixture_model_not_available")
    writer.seal_decoded()


def completed_results(writer):
    def provider(**kwargs):
        return writer.block_provider(**kwargs, truth_provider=truth_provider)
    point = inference.evaluate_panel(provider, TRAINING, IMAGES,
        check=lambda: None, result_sink=writer.result_sink)
    intervals = {}
    for metric in ("pattern", "profile"):
        fields = []
        for _, point_name in inference._INTERVAL_FIELDS:
            # Deliberately unavailable intervals exercise honest partial
            # inference reporting; they make no numerical bootstrap claim.
            fields.append({endpoint: inference._interval(estimate,
                np.full((1000,) + np.asarray(estimate).shape, np.nan), lambda: None)
                for endpoint, estimate in getattr(getattr(point, metric), point_name).items()})
        intervals[metric] = inference.MetricIntervals(*fields)
    draws = readonly(np.random.RandomState(42).randint(0, 224, (1000, 224)).astype(np.int64))
    primary = inference.BootstrapResult(point, draws, intervals["pattern"], intervals["profile"])
    incoming = secondary.target_incoming(point, check=lambda: None)
    dyadic = secondary.dyadic_bootstrap(point, check=lambda: None)
    return primary, incoming, dyadic


def _failure_child(mode, directory):
    """Exercise clean cancellation or an explicitly fatal OS file-size cap."""
    maximum_bytes = 1024**2 if mode == "cap" else settings()["maximum_bytes"]
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (maximum_bytes, hard))
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    if mode == "cap":
        # This child owns the fatal policy; its parent inspects the exact
        # terminating signal and retained files, without native EFBIG recovery.
        _, core_hard = resource.getrlimit(resource.RLIMIT_CORE)
        resource.setrlimit(resource.RLIMIT_CORE, (0, core_hard))
    path = Path(directory) / (mode + ".h5")
    writer = None
    with mock.patch.object(collection.assembly, "_require_reader_context",
                           side_effect=lambda budget: budget.check()):
        if mode == "cancel":
            cancelled, closed = [False], []

            def check():
                if cancelled[0]:
                    raise TimeoutError("fixture cancellation")

            def interrupted_blocks():
                try:
                    iterator = blocks()
                    yield next(iterator)
                    cancelled[0] = True
                    yield next(iterator)
                finally:
                    closed.append(True)

            with collection.FindingACollection(path, **{
                    **settings(), "budget": SimpleNamespace(check=check)}) as writer:
                try:
                    writer.write_layer(method=KEY[0], source=KEY[1], target=KEY[2],
                                       layer=KEY[3], blocks=interrupted_blocks())
                except TimeoutError as error:
                    assert str(error) == "fixture cancellation"
                    assert not getattr(error, "__notes__", ())
                else:
                    raise AssertionError("required cancellation was not raised")
                # The outer context has not exited yet: operation failure must
                # already have strongly closed all local HDF5 object IDs.
                assert writer._closed and not writer._h5.id.valid
            assert closed == [True]
        elif mode == "cap":
            # Either native-axis metadata or fc8 values exceed this real cap.
            # An entry marker is not a success marker: SIGXFSZ must stop this
            # process before unsafe recovery or interpreter cleanup can occur.
            print("COLLECTION_FAILURE_CHILD_ENTER cap", flush=True)
            with collection.FindingACollection(path, **{
                    **settings(), "maximum_bytes": maximum_bytes}) as writer:
                writer.write_layer(method=KEY[0], source=KEY[1], target=KEY[2],
                                   layer=KEY[3], blocks=blocks())
            raise AssertionError("required fatal SIGXFSZ resource limit was not reached")
        else:
            raise AssertionError("unknown failure fixture")
    partial = Path(str(path) + ".partial")
    assert partial.exists() and partial.stat().st_size <= maximum_bytes
    assert not path.exists()
    assert not path.with_suffix(".decoded_feature_manifest.json").exists()
    assert not path.with_suffix(".finding_a_report.json").exists()
    print("COLLECTION_FAILURE_CHILD_OK " + mode, flush=True)


class FindingACollectionTests(unittest.TestCase):
    def setUp(self):
        # This fixture process owns and restores both settings. The library
        # merely checks them; inherited hard limits remain unchanged.
        original_signal = signal.getsignal(signal.SIGXFSZ)
        signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
        self.addCleanup(signal.signal, signal.SIGXFSZ, original_signal)
        original = resource.getrlimit(resource.RLIMIT_FSIZE)
        resource.setrlimit(resource.RLIMIT_FSIZE, (settings()["maximum_bytes"], original[1]))
        self.addCleanup(resource.setrlimit, resource.RLIMIT_FSIZE, original)

    def test_complete_roster_exact_float32_replay_point_vectors_and_honest_publication(self):
        self.assertEqual(len(collection.SLOTS), 855)
        self.assertEqual([sum(key[0] == method for key in collection.SLOTS)
                          for method in collection.METHODS], [95, 380, 380])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(collection.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()):
            path = Path(directory) / "finding_a.h5"
            with collection.FindingACollection(path, **settings()) as writer:
                writer.write_layer(method=KEY[0], source=KEY[1], target=KEY[2], layer=KEY[3], blocks=blocks())
                seal(writer)
                request = dict(zip(("method", "source", "target", "layer"), KEY))
                replay = list(writer.block_provider(**request, truth_provider=truth_provider))
                predicted, truth = values()
                np.testing.assert_array_equal(np.concatenate([item.predicted for item in replay], axis=1), predicted)
                np.testing.assert_array_equal(np.concatenate([item.truth for item in replay], axis=1), truth)
                self.assertTrue(all(item.predicted.shape[0] == 224 and item.predicted.dtype == np.float32 for item in replay))
                draw = readonly(np.repeat(np.arange(112, dtype=np.int64), 2))
                result = metrics.score_layer(writer.block_provider(**request, truth_provider=truth_provider),
                    **request, expected_image_ids=IMAGES, decoder_training_image_ids=TRAINING[KEY[2]],
                    check=lambda: None, image_indices=draw)
                np.testing.assert_array_equal(result.image_indices, draw)
                mean = np.arange(1000, dtype=np.float32) / np.float32(8)
                std = np.float32(1) + (np.arange(1000) % 3).astype(np.float32) / np.float32(2)
                x, y = (predicted - mean) / std, (truth - mean) / std
                self.assertAlmostEqual(result.pattern.correlations[0], np.corrcoef(x[draw[0]], y[draw[0]])[0, 1], places=12)
                self.assertAlmostEqual(result.profile.correlations[17], np.corrcoef(predicted[draw, 17], truth[draw, 17])[0, 1], places=12)
                primary, incoming, dyadic = completed_results(writer)
                published = writer.publish(primary, incoming, dyadic)
            self.assertEqual(published, (path, path.with_suffix(".decoded_feature_manifest.json"),
                                         path.with_suffix(".finding_a_report.json")))
            manifest = json.loads(published[1].read_text())
            report = json.loads(published[2].read_text())
            self.assertEqual((len(manifest["slots"]), manifest["decoded_slots"], manifest["unavailable_slots"]), (855, 1, 854))
            self.assertEqual(report["status"], "complete_requested_record")
            self.assertEqual(report["quantitative_finding"], "not_evaluable")
            self.assertEqual(report["comparability"]["reason"], "missing_outcome_independent_margin")
            with h5py.File(path, "r") as handle:
                self.assertTrue(handle.attrs["complete"])
                record = next(item for item in manifest["slots"] if item["status"] == "decoded")
                self.assertEqual(record["logical_shape"], [224, 1000])
                self.assertEqual(record["stored_shape"], [1000, 224])
                self.assertEqual(record["storage_axes"], ["append_order_unit", "image"])
                np.testing.assert_array_equal(handle[record["values"]][:].T, predicted)
                score = report["point_scores"][0]
                np.testing.assert_array_equal(handle[score["profile"]["coordinate"]][:], np.arange(1000))
                self.assertEqual(handle[score["pattern"]["correlations"]].shape, (224,))
                self.assertEqual(handle[score["profile"]["defined_mask"]].shape, (1000,))
                self.assertEqual(handle["resampling/image_draw_indices"].shape, (1000, 224))
            self.assertFalse(Path(str(path) + ".partial").exists())
            with self.assertRaises(FileExistsError):
                collection.FindingACollection(path, **settings())

    def test_admission_owner_native_coordinates_moments_and_common_truth_refusals(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(collection.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()):
            path = Path(directory) / "denied.h5"
            def deny():
                raise PermissionError("fixture external denial")
            with mock.patch.object(Path, "open", side_effect=AssertionError("must admit before I/O")):
                with self.assertRaisesRegex(PermissionError, "external denial"):
                    collection.FindingACollection(path, **{**settings(), "require_admitted": deny})
            self.assertFalse(Path(str(path) + ".partial").exists())
            for defect, message in (("owner", "binding"), ("axis", "native_axis"), ("coordinate", "coordinates")):
                with self.subTest(defect=defect):
                    current = Path(directory) / (defect + ".h5")
                    with collection.FindingACollection(current, **settings()) as writer:
                        item, decoder = next(blocks())
                        if defect == "owner":
                            item = replace(item, decoder_participant=PEOPLE[2])
                        elif defect == "axis":
                            decoder = replace(decoder, selected_native_voxel_ids=decoder.selected_native_voxel_ids + 1)
                        else:
                            item = replace(item, full_layer_Fortran_index=item.full_layer_Fortran_index + 1)
                        with self.assertRaisesRegex(ValueError, message):
                            writer.write_layer(method=KEY[0], source=KEY[1], target=KEY[2], layer=KEY[3], blocks=[(item, decoder)])
                    self.assertTrue(Path(str(current) + ".partial").exists())
                    self.assertFalse(current.exists())
            with collection.FindingACollection(Path(directory) / "moments.h5", **settings()) as writer:
                writer.write_layer(method=KEY[0], source=KEY[1], target=KEY[2], layer=KEY[3], blocks=blocks())
                brain = ("brain_loss", KEY[1], KEY[2], KEY[3])
                with self.assertRaisesRegex(ValueError, "moments_changed"):
                    writer.write_layer(method=brain[0], source=brain[1], target=brain[2], layer=brain[3],
                                       blocks=blocks(brain, scale_change=True))
            with collection.FindingACollection(Path(directory) / "truth.h5", **settings()) as writer:
                writer.write_layer(method=KEY[0], source=KEY[1], target=KEY[2], layer=KEY[3], blocks=blocks())
                seal(writer)
                def changed_truth(**kwargs):
                    result = truth_provider(**kwargs)
                    return replace(result, feature_identity={**IDENTITY, "checkpoint": "other"})
                with self.assertRaisesRegex(ValueError, "common_truth"):
                    list(writer.block_provider(method=KEY[0], source=KEY[1], target=KEY[2], layer=KEY[3],
                                               truth_provider=changed_truth))

    def test_bounded_cancelled_incomplete_and_late_publication_failure_preserve_partial(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(collection.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()):
            for soft in (resource.RLIM_INFINITY, settings()["maximum_bytes"] + 1):
                with self.subTest(refused_soft_limit=soft), \
                        mock.patch.object(collection.resource, "getrlimit", return_value=(soft, resource.RLIM_INFINITY)), \
                        mock.patch.object(collection.h5py, "File", side_effect=AssertionError("cap must precede HDF5 open")):
                    with self.assertRaisesRegex(ValueError, "finite_caller_RLIMIT_FSIZE"):
                        collection.FindingACollection(Path(directory) / "uncapped.h5", **settings())
                self.assertFalse((Path(directory) / "uncapped.h5.partial").exists())
            with mock.patch.object(collection.signal, "getsignal", return_value=signal.SIG_IGN), \
                    mock.patch.object(collection.h5py, "File", side_effect=AssertionError("signal must precede HDF5 open")):
                with self.assertRaisesRegex(ValueError, "caller_SIGXFSZ_default"):
                    collection.FindingACollection(Path(directory) / "ignored_signal.h5", **settings())
            self.assertFalse((Path(directory) / "ignored_signal.h5.partial").exists())
            for mode in ("cancel", "cap"):
                with self.subTest(child=mode):
                    completed = subprocess.run([sys.executable, "-B", "-c",
                        "from test_finding_a_collection import _failure_child; "
                        "import sys; _failure_child(sys.argv[1], sys.argv[2])", mode, directory],
                        cwd=Path(__file__).resolve().parent, capture_output=True,
                        text=True, timeout=30, check=False)
                    expected_exit = -signal.SIGXFSZ if mode == "cap" else 0
                    self.assertEqual(completed.returncode, expected_exit, completed.stdout + completed.stderr)
                    marker = "COLLECTION_FAILURE_CHILD_ENTER cap" if mode == "cap" else "COLLECTION_FAILURE_CHILD_OK cancel"
                    self.assertEqual(completed.stdout.strip(), marker)
                    self.assertNotIn("Exception ignored in", completed.stderr)
                    if mode == "cap":
                        path = Path(directory) / "cap.h5"
                        partial = Path(str(path) + ".partial")
                        self.assertTrue(partial.exists())
                        self.assertLessEqual(partial.stat().st_size, 1024**2)
                        for output in (path, path.with_suffix(".decoded_feature_manifest.json"),
                                       path.with_suffix(".finding_a_report.json")):
                            self.assertFalse(output.exists())
            with collection.FindingACollection(Path(directory) / "incomplete.h5", **settings()) as writer:
                with self.assertRaisesRegex(ValueError, "all_855"):
                    writer.seal_decoded()
            path = Path(directory) / "late.h5"
            with collection.FindingACollection(path, **settings()) as writer:
                writer.write_layer(method=KEY[0], source=KEY[1], target=KEY[2], layer=KEY[3], blocks=blocks())
                seal(writer)
                primary, incoming, dyadic = completed_results(writer)
                changed_groups = dict(primary.point.pattern.group_means)
                changed_groups["content_minus_brain"] = 1.0
                changed_pattern = replace(primary.point.pattern, group_means=changed_groups)
                changed_primary = replace(primary, point=replace(primary.point, pattern=changed_pattern))
                with self.assertRaisesRegex(ValueError, "primary_derived_summary"):
                    writer._bind_results(changed_primary, incoming, dyadic)
                changed_dyadic_groups = dict(dyadic.pattern.group_means)
                changed_dyadic_groups["content_minus_brain"] = 1.0
                changed_dyadic = replace(dyadic, pattern=replace(dyadic.pattern,
                    group_means=changed_dyadic_groups))
                with self.assertRaisesRegex(ValueError, "dyadic_derived_summary"):
                    writer._bind_results(primary, incoming, changed_dyadic)
                original_link = collection.os.link
                def fail_report(source, destination):
                    if Path(destination) == writer.report_path:
                        raise OSError("fixture final report publication failure")
                    return original_link(source, destination)
                with mock.patch.object(collection.os, "link", side_effect=fail_report):
                    with self.assertRaisesRegex(OSError, "report publication"):
                        writer.publish(primary, incoming, dyadic)
            self.assertTrue(path.exists())
            self.assertTrue(writer.manifest_path.exists())
            self.assertFalse(writer.report_path.exists())
            self.assertTrue(Path(str(writer.report_path) + ".partial").exists())
            with self.assertRaises(FileExistsError):
                collection.FindingACollection(path, **settings())


if __name__ == "__main__":
    unittest.main()
