"""Three staged persistence checks; execute only in coordinated Slurm compute.

Reuse the original full-fc8 collection fixture and its intentionally unavailable
interval tables. No new metric/bootstrap qualification, native-cap crash case,
empirical input, model fit or scientific result is introduced by these tests.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_staged_collection as staged
import finding_a_collection as original
import test_finding_a_collection as fixtures
from test_finding_a_batched_inference import equal


def read_arguments():
    args = fixtures.settings()
    return {key: args[key] for key in ("budget", "maximum_bytes", "require_admitted")}


def make_stage(path):
    with staged.StagedFindingACollection(path, **fixtures.settings()) as writer:
        key = fixtures.KEY
        writer.write_layer(method=key[0], source=key[1], target=key[2], layer=key[3], blocks=fixtures.blocks())
        fixtures.seal(writer)
        results = fixtures.completed_results(writer)
        published = writer.publish_decoded_point(results[0].point)
    return published, results


class FindingAStagedCollectionTests(unittest.TestCase):
    def setUp(self):
        fixtures.FindingACollectionTests.setUp(self)

    def test_stage_reopens_readonly_and_finalizes_same_inode_with_original_publication(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(original.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()):
            path, final = Path(directory) / "stage.h5", Path(directory) / "final.h5"
            (stage_path, metadata), results = make_stage(path)
            self.assertEqual(stage_path, path)
            self.assertEqual(metadata, path.with_suffix(".decoded_point.json"))
            historical = metadata.read_bytes()
            record = json.loads(historical)
            self.assertIs(record["complete"], False)
            self.assertIs(record["decoded_point_complete"], True)
            self.assertEqual(record["inference"], "pending")
            self.assertEqual((len(record["records"]), len(record["scores"])), (855, 1))
            for suffix in (".decoded_feature_manifest.json", ".finding_a_report.json"):
                self.assertFalse(path.with_suffix(suffix).exists())
            inode = path.stat().st_dev, path.stat().st_ino
            location = record["scores"][0]["summary"]["profile"]["correlations"]
            decoded_location = next(slot["values"] for slot in record["records"] if slot["status"] == "decoded")
            with h5py.File(path, "r") as source:
                self.assertFalse(source.attrs["complete"])
                self.assertTrue(source.attrs["decoded_point_complete"])
                self.assertNotIn("resampling", source)
                old_values, old_profile = source[decoded_location][...], source[location][...]

            with mock.patch.object(original.prediction, "predict_decoder_block",
                                   side_effect=AssertionError("stage replay must not predict or refit")):
                with staged.open_decoded_point(path, **read_arguments()) as reader:
                    self.assertEqual(reader._state._h5.mode, "r")
                    self.assertEqual(reader.image_ids, fixtures.IMAGES)
                    self.assertEqual(dict(reader.training), fixtures.TRAINING)
                    self.assertEqual(set(reader.axes), set(original.PARTICIPANTS))
                    self.assertEqual(len(reader.records), 855)
                    self.assertEqual(reader.stage_identity["collection_path"], str(path))
                    self.assertEqual(reader.stage_identity["point_path"], str(metadata) + "#point")
                    self.assertEqual(len(reader.stage_identity["fitted_object_ids"]), 45)
                    equal(self, reader.point, results[0].point)
                    with self.assertRaises(TypeError):
                        reader.records[fixtures.KEY]["status"] = "unavailable"
                    with self.assertRaises(TypeError):
                        reader.fitted_object_ids[fixtures.KEY[:3]] = "different"
                    self.assertTrue(all(not axis.mean.flags.writeable for axis in reader.axes.values()))
                    key = fixtures.KEY
                    blocks = list(reader.block_provider(method=key[0], source=key[1], target=key[2],
                        layer=key[3], truth_provider=fixtures.truth_provider))
                    np.testing.assert_array_equal(np.concatenate([block.predicted for block in blocks], axis=1),
                                                  fixtures.values()[0])
                closed = []

                def require_closed():
                    self.assertTrue(reader._state._closed)
                    self.assertFalse(path.with_suffix(".finalization_started.json").exists())
                    closed.append(True)

                published = staged.finalize_decoded_point(path, final, *results,
                    **read_arguments(), require_readers_closed=require_closed)
            self.assertEqual(closed, [True])
            self.assertEqual(published, (final, final.with_suffix(".decoded_feature_manifest.json"),
                                        final.with_suffix(".finding_a_report.json")))
            self.assertEqual((final.stat().st_dev, final.stat().st_ino), inode)
            self.assertEqual((path.stat().st_dev, path.stat().st_ino), inode)
            self.assertEqual(metadata.read_bytes(), historical)
            self.assertFalse(Path(str(final) + ".partial").exists())
            with h5py.File(final, "r") as source:
                self.assertTrue(source.attrs["complete"])
                self.assertIn("resampling", source)
                np.testing.assert_array_equal(source[decoded_location][...], old_values)
                np.testing.assert_array_equal(source[location][...], old_profile)
            manifest = json.loads(published[1].read_text())
            report = json.loads(published[2].read_text())
            self.assertEqual(manifest["hdf5"], str(final))
            self.assertEqual((manifest["decoded_slots"], manifest["unavailable_slots"]), (1, 854))
            self.assertEqual(report["quantitative_finding"], "not_evaluable")
            self.assertEqual(report["comparability"]["status"], "not_evaluable")
            with mock.patch.object(staged.h5py, "File", side_effect=AssertionError("claimed stage reopened")):
                with self.assertRaisesRegex(ValueError, "finalization_already_claimed"):
                    staged.open_decoded_point(path, **read_arguments())

    def test_admission_point_completeness_and_saved_hdf_json_binding_refuse(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(original.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()):
            path = Path(directory) / "denied.h5"
            deny = mock.Mock(side_effect=PermissionError("stage not admitted"))
            with mock.patch.object(staged.h5py, "File", side_effect=AssertionError("admission must precede HDF")):
                with self.assertRaises(PermissionError):
                    staged.StagedFindingACollection(path, **{**fixtures.settings(), "require_admitted": deny})
                with self.assertRaises(PermissionError):
                    staged.open_decoded_point(path, **{**read_arguments(), "require_admitted": deny})
            self.assertFalse(Path(str(path) + ".partial").exists())
            for missing_vectors in (True, False):
                current = Path(directory) / ("point%d.h5" % missing_vectors)
                with staged.StagedFindingACollection(current, **fixtures.settings()) as writer:
                    key = fixtures.KEY
                    writer.write_layer(method=key[0], source=key[1], target=key[2], layer=key[3], blocks=fixtures.blocks())
                    fixtures.seal(writer)
                    point = original.inference.evaluate_panel(
                        lambda **request: writer.block_provider(**request, truth_provider=fixtures.truth_provider),
                        fixtures.TRAINING, fixtures.IMAGES, check=lambda: None,
                        result_sink=None if missing_vectors else writer.result_sink)
                    if not missing_vectors:
                        means = {name: value.copy() for name, value in point.pattern.layer_means.items()}
                        means["content_loss"][18, 1, 0] += .125
                        metric = original.inference._reduce((means, point.pattern.layer_defined_counts,
                            point.pattern.layer_undefined_counts, point.pattern.layer_reasons), lambda: None)
                        point = replace(point, pattern=metric)
                    with self.assertRaisesRegex(ValueError, "point_vectors_required|disagrees_with_saved_vectors"):
                        writer.publish_decoded_point(point)
                    self.assertTrue(writer._closed and writer._failed)
                self.assertFalse(current.exists())
                self.assertTrue(Path(str(current) + ".partial").exists())
                self.assertFalse(current.with_suffix(".decoded_point.json").exists())
            (path, metadata), _ = make_stage(Path(directory) / "binding.h5")
            historical = metadata.read_bytes()
            record = json.loads(historical)
            # Keep a structurally valid metadata record but break its direct
            # pairing to the immutable record retained inside this HDF inode.
            row = next(row for row in record["stage_identity"]["fitted_object_ids"] if row["fitted_object_id"] is not None)
            row["fitted_object_id"] = "a different admitted-model declaration"
            metadata.write_text(json.dumps(record, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "hdf_json_binding"):
                staged.open_decoded_point(path, **read_arguments())
            metadata.write_bytes(historical)
            with mock.patch.object(staged.h5py, "File", side_effect=AssertionError("partial used as stage")):
                with self.assertRaisesRegex(ValueError, "published_stage"):
                    staged.open_decoded_point(Path(str(path) + ".partial"), **read_arguments())
            with staged.open_decoded_point(path, **read_arguments()) as reader:
                # A claimed stage must stop an already-open replay handle at
                # its next callback too; it is not a lease/locking substitute.
                path.with_suffix(".finalization_started.json").write_text("fixture external claim")
                with self.assertRaisesRegex(ValueError, "finalization_already_claimed"):
                    reader.block_provider(method=fixtures.KEY[0], source=fixtures.KEY[1],
                        target=fixtures.KEY[2], layer=fixtures.KEY[3], truth_provider=fixtures.truth_provider)

    def test_reader_closure_gate_and_late_finalization_failure_preserve_stage_and_claim(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(original.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()):
            (path, metadata), results = make_stage(Path(directory) / "stage.h5")
            final = Path(directory) / "final.h5"
            historical, inode = metadata.read_bytes(), path.stat().st_ino
            with self.assertRaisesRegex(PermissionError, "readers remain"):
                staged.finalize_decoded_point(path, final, *results, **read_arguments(),
                    require_readers_closed=mock.Mock(side_effect=PermissionError("readers remain open")))
            self.assertFalse(path.with_suffix(".finalization_started.json").exists())
            self.assertFalse(Path(str(final) + ".partial").exists())
            with staged.open_decoded_point(path, **read_arguments()) as reader:
                equal(self, reader.point, results[0].point)
            real_link = original.os.link

            def fail_report(source, destination):
                if Path(destination) == final.with_suffix(".finding_a_report.json"):
                    raise OSError("fixture final report publication failure")
                return real_link(source, destination)

            with mock.patch.object(original.os, "link", side_effect=fail_report):
                with self.assertRaisesRegex(OSError, "report publication"):
                    staged.finalize_decoded_point(path, final, *results, **read_arguments(),
                                                 require_readers_closed=lambda: None)
            self.assertTrue(path.exists() and final.exists())
            self.assertEqual(path.stat().st_ino, inode)
            self.assertEqual(final.stat().st_ino, inode)
            self.assertEqual(metadata.read_bytes(), historical)
            self.assertTrue(path.with_suffix(".finalization_started.json").exists())
            self.assertTrue(final.with_suffix(".decoded_feature_manifest.json").exists())
            report = final.with_suffix(".finding_a_report.json")
            self.assertFalse(report.exists())
            self.assertTrue(Path(str(report) + ".partial").exists())
            # Neither a failed report nor a new destination enables retry of
            # this already-mutated stage inode.
            with mock.patch.object(staged.h5py, "File", side_effect=AssertionError("failed stage retried")):
                with self.assertRaisesRegex(ValueError, "finalization_already_claimed"):
                    staged.finalize_decoded_point(path, Path(directory) / "retry.h5", *results,
                        **read_arguments(), require_readers_closed=lambda: None)


if __name__ == "__main__":
    unittest.main()
