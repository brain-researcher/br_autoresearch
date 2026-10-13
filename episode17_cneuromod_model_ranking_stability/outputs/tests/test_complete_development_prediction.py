"""Development-only disk routing and bounded publication checks on invented IO.

No acquired artifact is opened. These checks run with one CPU; scientific
response aggregation and checkpoint arithmetic retain their existing tests.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import complete_development_inputs as full
import development_prediction_cost as cost
import parallel_nested_prediction as parallel
import predictor_checkpoint as checkpoint
import run_complete_development_prediction as runner
from test_released_development_inputs import fixture


class CompleteDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full.existing.feature_io.delivery.features.check_runtime()

    def test_complete_reader_uses_only_twelve_development_files_and_closes_on_failure(self):
        design, _, axes, copy_report = fixture()
        existing = full.existing
        entries = existing._copy_entries(copy_report)
        feature_report = dict(analysis_id="stimulus_features", attempt="full001", status="completed_private_staging",
            full_development_features_complete=True, development_images=2423, development_concepts=480,
            model_ids=list(existing.nested.MODEL_IDS), layers=list(existing.nested.LAYERS),
            neural_values_read=False, audit_pixels_or_features_opened=False, feature_file_bytes=777)
        required = 3 * 2423 * sum(len(value) for value in axes.values()) * 8
        for outcome in ("complete", "last_header_failure", "aggregation_failure", "cap_failure"):
            opened, closed, headers, events = [], [], [], []
            manifest, row_maps, features, panels = object(), object(), object(), object()
            case = self

            class Handle:
                def __init__(self, path, mode, *, rdcc_nbytes):
                    case.assertEqual(mode, "r")
                    case.assertEqual(rdcc_nbytes, 1024 * 1024)
                    relative = path.relative_to(existing.COPY_PRIVATE)
                    case.assertEqual(relative.parts[0], "development")
                    case.assertEqual(len(relative.parts), 3)
                    self.key = relative.parts[1], path.stem
                    case.assertIn(self.key, entries)

                def __enter__(self):
                    opened.append(self.key)
                    return self

                def __exit__(self, *_):
                    closed.append(self.key)

            def header(handle, shape, dtype):
                self.assertEqual(shape, entries[handle.key]["shape"])
                self.assertEqual(dtype, "<f4")
                headers.append(handle.key)
                if outcome == "last_header_failure" and len(headers) == 12:
                    raise ValueError("invented private header detail")

            def build(rows, supplied_manifest, supplied_maps, handles, supplied_axes, *, resource_check):
                self.assertIs(rows, design)
                self.assertIs(supplied_manifest, manifest)
                self.assertIs(supplied_maps, row_maps)
                self.assertIs(supplied_axes, axes)
                self.assertEqual(set(handles), set(entries))
                self.assertEqual(len(headers), 12)
                self.assertEqual(closed, [])
                if outcome == "aggregation_failure":
                    raise ValueError("invented private neural detail")
                return panels

            with self.subTest(outcome=outcome), ExitStack() as stack:
                stack.enter_context(mock.patch.object(existing, "_json", side_effect=lambda path, cap, check:
                    feature_report if path.name == "feature_report.json" else copy_report))
                stack.enter_context(mock.patch.object(existing.feature_io, "load_development_metadata", return_value=(design, manifest)))
                stack.enter_context(mock.patch.object(existing, "_row_maps", return_value=row_maps))
                stack.enter_context(mock.patch.object(existing, "_frozen_columns", return_value=axes))
                stack.enter_context(mock.patch.object(existing.feature_io.pilot, "_ordinary_private_file", return_value=SimpleNamespace(st_size=777)))
                stack.enter_context(mock.patch.object(existing, "_ordinary_input_file", return_value=123456))
                stack.enter_context(mock.patch.object(full.h5py, "File", side_effect=Handle))
                stack.enter_context(mock.patch.object(existing.columns, "output_dataset", side_effect=header))
                feature_load = stack.enter_context(mock.patch.object(existing.feature_io, "load_complete_features", return_value=features))
                builder = stack.enter_context(mock.patch.object(existing.responses, "build_development_responses", side_effect=build))
                args = dict(feature_attempt="full001", feature_byte_cap=1024, metadata_byte_cap=1024,
                    response_byte_cap=required - (outcome == "cap_failure"), resource_check=events.append)
                if outcome == "complete":
                    result = full.load_complete_development_inputs(**args)
                    self.assertTrue(result.complete_development_response_release)
                    self.assertIs(result.responses, panels)
                    self.assertIs(result.features, features)
                    self.assertEqual(result.operational_dimensions["response_mean_bytes"], required)
                    self.assertEqual(tuple(opened), tuple(entries))
                    builder.assert_called_once()
                else:
                    with self.assertRaises(existing.DevelopmentInputError) as error:
                        full.load_complete_development_inputs(**args)
                    self.assertNotIn("invented private", str(error.exception))
                    if outcome != "aggregation_failure":
                        builder.assert_not_called()
                self.assertEqual(closed, list(reversed(opened)))
                if outcome == "cap_failure":
                    self.assertEqual(opened, [])
                    feature_load.assert_not_called()
                plan = [event for event in events if event.operation == "plan_complete_response_means"]
                self.assertEqual([event.byte_count for event in plan], [required])

    def test_selected_cost_binding_refuses_incomplete_or_other_feature_attempt(self):
        args = SimpleNamespace(cost_attempt="cost001", feature_attempt="full001", metadata_byte_cap=1024)
        for defect in (None, "feature", "counts", "status"):
            value = dict(analysis_id="prediction_qualification", stage="prediction_cost", attempt="cost001",
                feature_attempt="full001", status="completed", cost_report=dict(status="complete", failure_code=None,
                completed_counts=dict(cost.REQUIRED_COUNTS)))
            if defect == "feature":
                value["feature_attempt"] = "full002"
            elif defect == "counts":
                value["cost_report"]["completed_counts"]["fit_ridge"] -= 1
            elif defect == "status":
                value["status"] = "stopped"
            with self.subTest(defect=defect), mock.patch.object(full.existing, "_json", return_value=value):
                if defect is None:
                    runner._completed_cost(args, full, cost, lambda event: None)
                else:
                    with self.assertRaises(runner.FitRunRefusal):
                        runner._completed_cost(args, full, cost, lambda event: None)

    def test_cli_preserves_completed_failed_and_interrupted_publication_attempts(self):
        cost_report = dict(analysis_id="prediction_qualification", stage="prediction_cost", attempt="cost001",
            feature_attempt="full001", status="completed", cost_report=dict(status="complete", failure_code=None,
            completed_counts=dict(cost.REQUIRED_COUNTS)))
        for outcome in ("complete", "fit_failure", "publication_interruption"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                args = SimpleNamespace(feature_attempt="full001", cost_attempt="cost001", attempt="fit001", workers=1,
                    deadline_epoch=time.time() + 60, rss_limit_kib=3 * 1024 * 1024, metadata_byte_cap=1024,
                    feature_byte_cap=1024, response_byte_cap=1024, checkpoint_byte_cap=1024, report_byte_cap=65536)
                source = SimpleNamespace(design=object(), features=object(), responses=object(), frozen_columns=object(),
                    complete_development_response_release=True, operational_dimensions={"development_images": 2423})
                fitted = object()

                def load(**kwargs):
                    kwargs["resource_check"](full.existing.InputEvent("plan_complete_response_means", byte_count=128))
                    kwargs["resource_check"](full.existing.InputEvent("read_stripe"))
                    return source

                def fit(*values, workers, resource_check):
                    self.assertEqual(values, (source.design, source.features, source.responses, source.frozen_columns))
                    self.assertEqual(workers, 1)
                    resource_check(full.existing.InputEvent("all_features_feasible", when="after"))
                    resource_check(full.existing.InputEvent("inner_error_sums"))
                    if outcome == "fit_failure":
                        raise ValueError("invented private objective")
                    return fitted

                def publish(result, destination, *, max_bytes, resource_check):
                    self.assertIs(result, fitted)
                    self.assertEqual(destination, root / "private/fit001/complete_prediction.zip")
                    self.assertEqual(max_bytes, 1024)
                    destination.write_bytes(b"invented published artifact")
                    if outcome == "publication_interruption":
                        raise RuntimeError("invented private publication detail")
                    return checkpoint.CheckpointWriteResult(destination, True)

                stack.enter_context(mock.patch.object(runner, "REPORT_ROOT", root / "public"))
                stack.enter_context(mock.patch.object(runner, "PRIVATE_ROOT", root / "private"))
                stack.enter_context(mock.patch.object(full.existing, "_json", return_value=cost_report))
                stack.enter_context(mock.patch.object(full, "load_complete_development_inputs", side_effect=load))
                stack.enter_context(mock.patch.object(parallel, "run_parallel_nested_prediction", side_effect=fit))
                publisher = stack.enter_context(mock.patch.object(checkpoint, "save_prediction_checkpoint", side_effect=publish))
                self.assertEqual(runner._run(args), 0 if outcome == "complete" else 1)
                path = root / "public/fit001/fit_report.json"
                preserved = path.read_bytes()
                with self.assertRaises(runner.FitRunRefusal):
                    runner._run(args)
                self.assertEqual(path.read_bytes(), preserved)
                report = json.loads(preserved)
                self.assertEqual(report["status"], "completed" if outcome == "complete" else "stopped")
                self.assertEqual(report["checkpoint_published"], outcome == "complete")
                self.assertEqual(report["checkpoint_path_present"], outcome != "fit_failure")
                self.assertTrue(report["private_inner_scoring_attempted"])
                self.assertEqual(report["response_mean_bytes"], 128)
                self.assertFalse(report["audit_access"])
                self.assertNotIn("invented private", preserved.decode())
                if outcome == "fit_failure":
                    publisher.assert_not_called()
                else:
                    publisher.assert_called_once()


if __name__ == "__main__":
    unittest.main()
