"""Five bounded source-free integration fixtures; coordinator compute only.

No acquired checkpoints, source archives, credentials, neural data or completed
private artifacts are used. Tiny generated JPEGs and fake backbone outputs
test the adapter's boundaries, not representation or scientific performance.
"""
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr, redirect_stdout
import copy
import csv
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import private_stimulus_feature_pilot as pilot


def selection():
    return [{"feature_row_zero_based": index,
             "provider_image_path": f"images_fmri/concept{index + 1}/image{index + 1}.jpg",
             "image_name": f"image{index + 1}.jpg", "things_category_nr": index + 1,
             "uniqueID": f"concept{index + 1}", "role": "development", "outer_fold": "fold01",
             "development_block": "block01", "eligible_image": True, "member_match_count": ""}
            for index in range(32)]


def image_archive(rows, *, invalid_last=False):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, row in enumerate(rows):
            encoded = io.BytesIO()
            with Image.new("RGB", (8 + index, 7), (index, 20, 40)) as image:
                image.save(encoded, format="PNG" if invalid_last and index == 31 else "JPEG")
            archive.writestr(row["provider_image_path"], encoded.getvalue())
        # This must never be opened, even though it lives in the same ZIP.
        archive.writestr("images_fmri/audit/private.jpg", b"closed audit fixture")
    output.seek(0)
    return output


class Input:
    def __init__(self, index):
        self.index = index

    def unsqueeze(self, axis):
        if axis != 0:
            raise AssertionError("batch axis changed")
        return self


class Vector:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Backbone:
    def __init__(self, number, calls):
        self.number, self.calls = number, calls

    def __call__(self, value):
        self.calls.append((self.number, value.index))
        return {name: Vector(np.full((1, width), self.number * 100 + value.index, dtype=np.float32))
                for name, width in zip(pilot.LAYERS, pilot.WIDTHS)}


@contextmanager
def fixture_run(root, *, rows=None, decode=None, writer=None):
    """Patch external inputs/runtime only; use actual orchestration/files/writer."""
    rows = selection() if rows is None else rows
    archive_path = root / "source-fixture.zip"
    archive_path.write_bytes(b"public artificial archive identity fixture")
    private = root / "private" / "pilot001" / "development"
    report = root / "reports" / "pilot001" / "pilot_report.json"
    random_path = root / "models" / "random-fixture.pth"
    features = SimpleNamespace(check_runtime=lambda: None,
                               runtime_metadata=lambda: {"fixture_backend": True})
    calls = []

    def decoded(_stream, _rows, _password, _features, state, _check):
        state["images_preprocessed"] = 32
        state["image_pixels_read"] = True
        return [Input(index) for index in range(32)]

    with ExitStack() as stack:
        for name, value in (("PRIVATE_OUTPUT", private), ("REPORT_PATH", report),
                            ("RANDOM_CHECKPOINT", random_path)):
            stack.enter_context(mock.patch.object(pilot, name, value))
        stack.enter_context(mock.patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}))
        stack.enter_context(mock.patch.object(pilot.socket, "gethostname", return_value="compute-fixture"))
        stack.enter_context(mock.patch.dict(sys.modules, {"model_features": features}))
        stack.enter_context(mock.patch.object(pilot, "_deadline", side_effect=lambda: nullcontext(lambda: None)))
        stack.enter_context(mock.patch.object(pilot.archive_metadata, "ARCHIVE", archive_path))
        stack.enter_context(mock.patch.object(pilot.archive_metadata, "load_selection", return_value=rows))
        stack.enter_context(mock.patch.object(pilot.archive_metadata, "read_csv", return_value=[
            {"role": "development", "eligible_image": "True"} for _ in range(32)]))
        verification = stack.enter_context(mock.patch.object(pilot, "_verify_archive", return_value=((1,), {})))
        stack.enter_context(mock.patch.object(pilot, "_identity", return_value=(1,)))
        stack.enter_context(mock.patch.object(pilot, "_prepare_random", return_value="created"))
        delivery = stack.enter_context(mock.patch.object(pilot, "_read_password", return_value=b"fixture-secret-sentinel"))
        stack.enter_context(mock.patch.object(pilot, "_decode_inputs", side_effect=decode or decoded))
        stack.enter_context(mock.patch.object(pilot, "_load_backbone", side_effect=lambda _f, mid, _p:
                                             Backbone(pilot.MODELS.index(mid), calls)))
        if writer is not None:
            stack.enter_context(mock.patch.object(pilot, "_write_features", side_effect=writer))
        kwargs = {"password_fd": 999, "private_output_dir": private,
                  "operational_report_path": report, "random_checkpoint_path": random_path}
        yield kwargs, private, report, verification, delivery, calls


class PrivateStimulusPilotTests(unittest.TestCase):
    def test_role_or_order_drift_refuses_before_archive_secret_or_pixels(self):
        invalid_rosters = []
        for field, value in (("role", "sealed_audit"), ("eligible_image", False),
                             ("outer_fold", ""), ("feature_row_zero_based", 0)):
            rows = selection()
            rows[-1][field] = value
            invalid_rosters.append(rows)
        invalid_rosters.append(selection()[:-1])
        for rows in invalid_rosters:
            with self.subTest(field=rows[-1]), tempfile.TemporaryDirectory() as directory:
                with fixture_run(Path(directory), rows=rows) as (kwargs, private, path, verify, delivery, _):
                    result = pilot.run_private_development_feature_pilot(**kwargs)
                    self.assertEqual(result["error_code"], "selection_failed")
                    self.assertFalse(result["image_pixels_read"])
                    self.assertFalse(result["pilot_feature_extraction_complete"])
                    verify.assert_not_called()
                    delivery.assert_not_called()
                    self.assertFalse((private / "image_index.csv").exists())
                    self.assertEqual(json.loads(path.read_text()), result)

    def test_private_descriptor_is_bounded_owned_and_never_echoed(self):
        for payload, expected in ((b"fixture-only-password\r\n", b"fixture-only-password"),
                                  (b"  preserve spaces  \n", b"  preserve spaces  "),
                                  (b"embedded\nnewline", None), (b"x" * (pilot.SECRET_CAP + 1), None),
                                  (b"", None)):
            with self.subTest(length=len(payload)), tempfile.TemporaryFile() as descriptor:
                descriptor.write(payload)
                descriptor.seek(0)
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    if expected is None:
                        with self.assertRaises(pilot.PilotRefusal) as caught:
                            pilot._read_password(descriptor.fileno())
                        self.assertNotIn("fixture-only-password", str(caught.exception))
                    else:
                        self.assertEqual(pilot._read_password(descriptor.fileno()), expected)
                self.assertEqual(out.getvalue() + err.getvalue(), "")
                self.assertFalse(descriptor.closed)
                os.fstat(descriptor.fileno())  # The caller's inherited fd survives.
        for invalid in (0, 1, 2, True, "3"):
            with self.assertRaises(pilot.PilotRefusal):
                pilot._read_password(invalid)
        # Reproduce the observed Oak setgid inheritance locally. Only this
        # call's new directories may be normalized, never an existing parent.
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent.chmod(0o750)
            original_mkdir, original_fchmod = Path.mkdir, os.fchmod
            def inherited_setgid(path, *args, **kwargs):
                original_mkdir(path, *args, **kwargs)
                path.chmod(0o2700)
            with mock.patch.object(Path, "mkdir", inherited_setgid), \
                 mock.patch.object(os, "fchmod", wraps=original_fchmod) as chmod:
                pilot._mkdir_private(parent / "new" / "attempt", exclusive=True)
                self.assertEqual(chmod.call_count, 2)
                pilot._mkdir_private(parent)
                self.assertEqual(chmod.call_count, 2)
            self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o750)
            self.assertEqual(stat.S_IMODE((parent / "new").stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((parent / "new" / "attempt").stat().st_mode), 0o700)

    def test_decode_only_exact_selected_streams_and_no_format_substitution(self):
        rows = selection()
        for invalid_last in (False, True):
            with self.subTest(invalid_last=invalid_last):
                archive = image_archive(rows, invalid_last=invalid_last)
                state = {"selected_member_read_attempted": False, "image_pixels_read": False,
                         "images_preprocessed": 0}
                opened = []
                original = zipfile.ZipFile.open

                def tracked(container, name, *args, **kwargs):
                    actual = name.filename if isinstance(name, zipfile.ZipInfo) else name
                    self.assertIn(actual, {row["provider_image_path"] for row in rows})
                    opened.append(actual)
                    return original(container, name, *args, **kwargs)

                transformed = []
                def preprocess(image):
                    self.assertEqual(image.mode, "RGB")
                    transformed.append(image.size)
                    return Input(len(transformed) - 1)

                with mock.patch.object(zipfile.ZipFile, "open", tracked):
                    if invalid_last:
                        with self.assertRaisesRegex(pilot.PilotRefusal, "format_or_pixel_cap"):
                            pilot._decode_inputs(archive, rows, b"fixture", SimpleNamespace(preprocess=preprocess), state, lambda: None)
                    else:
                        bank = pilot._decode_inputs(archive, rows, b"fixture", SimpleNamespace(preprocess=preprocess), state, lambda: None)
                        self.assertEqual([item.index for item in bank], list(range(32)))
                self.assertEqual(opened, [row["provider_image_path"] for row in rows])
                self.assertEqual(state["images_preprocessed"], 31 if invalid_last else 32)
                self.assertEqual(transformed[0], (8, 7))

    def test_secret_bearing_failure_or_interrupt_preserves_partial_without_release(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt), tempfile.TemporaryDirectory() as directory:
                def failed(_features, _inputs, destination, _checkpoint, state, _check):
                    with pilot._exclusive_file(destination / "partial.h5") as partial:
                        partial.write(b"artificial incomplete private output")
                    state["completed_models"].append(pilot.MODELS[0])
                    if interrupt:
                        raise KeyboardInterrupt("fixture-secret-sentinel: private-value")
                    raise RuntimeError("fixture-secret-sentinel: private-value")

                with fixture_run(Path(directory), writer=failed) as (kwargs, private, path, _, _, _):
                    out, err = io.StringIO(), io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        if interrupt:
                            with self.assertRaisesRegex(KeyboardInterrupt, "private_pilot_interrupted"):
                                pilot.run_private_development_feature_pilot(**kwargs)
                        else:
                            pilot.run_private_development_feature_pilot(**kwargs)
                    raw_report = path.read_text()
                    result = json.loads(raw_report)
                    self.assertNotIn("fixture-secret-sentinel", raw_report + out.getvalue() + err.getvalue())
                    self.assertNotIn("private-value", raw_report)
                    self.assertEqual(result["status"], "interrupted" if interrupt else "stopped")
                    self.assertFalse(result["pilot_feature_extraction_complete"])
                    for field in ("private_artifacts_releasable", "physical_role_isolation_verified",
                                  "reader_handoffs_activated", "source_roles_complete", "candidate_scoring_authorized"):
                        self.assertFalse(result[field])
                    self.assertEqual((private / "partial.h5").read_bytes(), b"artificial incomplete private output")
                    with self.assertRaisesRegex(pilot.PilotRefusal, "attempt_already_exists"):
                        pilot.run_private_development_feature_pilot(**kwargs)
                    self.assertEqual(path.read_text(), raw_report)

    def test_all_four_models_have_identical_ordered_private_rows_and_complete_only_at_end(self):
        with tempfile.TemporaryDirectory() as directory:
            with fixture_run(Path(directory)) as (kwargs, private, path, _, delivery, calls):
                result = pilot.run_private_development_feature_pilot(**kwargs)
                self.assertEqual(result["status"], "completed_private_staging")
                self.assertTrue(result["pilot_feature_extraction_complete"])
                self.assertFalse(result["private_artifacts_releasable"])
                self.assertFalse(result["audit_pixels_or_features_opened"])
                self.assertEqual(calls, [(model, row) for model in range(4) for row in range(32)])
                delivery.assert_called_once_with(999)
                with (private / "image_index.csv").open() as stream:
                    index = list(csv.DictReader(stream))
                self.assertEqual([int(row["feature_row_zero_based"]) for row in index], list(range(32)))
                self.assertEqual([row["provider_image_path"] for row in index],
                                 [row["provider_image_path"] for row in selection()])
                self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o700)
                for model_number, model_id in enumerate(pilot.MODELS):
                    cache = private / (model_id + ".h5")
                    self.assertEqual(stat.S_IMODE(cache.stat().st_mode), 0o600)
                    with h5py.File(cache, "r") as store:
                        self.assertEqual(set(store), set(pilot.LAYERS))
                        for layer, width in zip(pilot.LAYERS, pilot.WIDTHS):
                            self.assertEqual(store[layer].shape, (32, width))
                            self.assertEqual(store[layer].dtype, np.dtype("float32"))
                            np.testing.assert_array_equal(store[layer][:, 0], model_number * 100 + np.arange(32))
                self.assertEqual(json.loads(path.read_text()), result)
                coverage = result["cost_projection_coverage"]
                self.assertGreaterEqual(result["timing_seconds"]["other_preprojection_work"], 0.)
                self.assertAlmostEqual(coverage["fixed_seconds"] + coverage["variable_seconds_including_residual"],
                                       coverage["measured_preprojection_seconds"], places=6)
                self.assertTrue(coverage["includes_selection_index_container_creation_flush_close_fsync"])
                self.assertEqual(result["buffered_full_development_seconds_estimate_not_guarantee"],
                                 2 * (coverage["fixed_seconds"] + coverage["variable_seconds_including_residual"]) + 60)


if __name__ == "__main__":
    unittest.main()
