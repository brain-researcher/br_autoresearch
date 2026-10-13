"""Bounded binding/selection/CLI checks; coordinated compute, invented inputs.

No acquired HDF5, role map, operator array, feature bank or report is opened.
The logical HDF selection is instrumented independently of the mean reducer.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import released_development_inputs as loader
import run_development_prediction_cost as runner


def fixture():
    nested = loader.nested
    concepts = tuple(i for i in range(480) for _ in range(6 if i < 23 else 5))
    design = nested.DevelopmentDesign(
        tuple(f"invented_image_{i:04}" for i in range(2423)),
        tuple(f"invented_concept_{i:03}" for i in concepts), ("development",) * 2423,
        tuple(f"fold{i // 80 + 1:02}" for i in concepts),
        tuple(f"block{i % 24 + 1:02}" for i in concepts))
    repeats = {person: tuple((2 * i, 2 * i + 1) for i in range(2423)) for person in loader.endpoint.SUBJECTS}
    axes = {group: (1000 * (i + 1) + 11 * np.arange(20, dtype=np.int64))[::-1].copy()
            for i, group in enumerate(loader.endpoint.GROUPS)}
    report = dict(status="completed", mode="full", full_copy_complete=True,
        all_copy_bytes_equal=True, neural_read_completed=True, errors=[], copy_layout=loader.columns.LAYOUT,
        private_attempt=str(loader.COPY_PRIVATE), public_attempt=str(loader.COPY_PUBLIC),
        role_assignment_attempt=str(loader.original.ASSIGNMENT), products=[])
    for person in loader.endpoint.SUBJECTS:
        width = loader.endpoint.support.PARTICIPANT_COLUMNS[person]
        for stage in loader.endpoint.STAGES:
            entries = []
            for role in loader.original.ROLES:
                entries.append(dict(role=role, copy_bytes_equal=True, empty_role=False, verified_columns=width,
                    array_file=str(loader.COPY_PRIVATE / role / person / f"{stage}.h5"), rows=4846,
                    shape=[4846, 1, 1, width], dtype_str="<f4", array_file_bytes=123456))
            report["products"].append(dict(participant=person, stage=stage, complete=True,
                verified_columns=width, dtype_str="<f4", outputs=entries))
    return design, repeats, axes, report


class ReleasedDevelopmentInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader.feature_io.delivery.features.check_runtime()

    def test_existing_copy_writer_map_binds_with_inherited_setgid_without_chmod(self):
        routed = {}
        for person in loader.endpoint.SUBJECTS:
            rows = []
            for index, role in enumerate(("development", "support_calibration", "development")):
                row = {field: f"invented_{field}_{index}" for field in loader.original.MAP_FIELDS
                       if field != "local_row_index_zero_based"}
                row.update(participant=person, role=role, glmsingle_trial_index_zero_based=str(index))
                rows.append(row)
            routed[person] = rows
        payload = loader.columns.map_payloads(routed)["development"]
        entries = {(person, stage): {"rows": 2}
                   for person in loader.endpoint.SUBJECTS for stage in loader.endpoint.STAGES}
        with tempfile.TemporaryDirectory(prefix="ep17-row-map-") as temporary:
            parent = Path(temporary)
            directory = parent / "development"
            directory.mkdir(mode=0o700)
            directory.chmod(0o700)
            path = directory / "source_index_map.csv"
            path.write_bytes(payload)
            path.chmod(0o600)
            with mock.patch.object(loader, "COPY_PRIVATE", parent), mock.patch.object(
                    loader.copied, "load_full_plan", return_value=(Path("/unopened/source"), routed)):
                for mode in (0o700, 0o2700):
                    parent.chmod(mode)
                    before = tuple(item.lstat().st_mode for item in (parent, directory, path))
                    result = loader._row_maps(entries, len(payload), lambda event: None)
                    self.assertEqual(tuple(item.lstat().st_mode for item in (parent, directory, path)), before)
                    for person in loader.endpoint.SUBJECTS:
                        self.assertEqual([row["participant"] for row in result[person]], [person, person])
                        self.assertEqual([row["local_row_index_zero_based"] for row in result[person]], ["0", "1"])
                        self.assertEqual([row["glmsingle_trial_index_zero_based"] for row in result[person]], ["0", "2"])
                    self.assertEqual(path.read_bytes(), payload)
                changed = deepcopy(routed)
                changed[loader.endpoint.SUBJECTS[0]][0]["outer_fold"] = "changed_fold"
                with mock.patch.object(loader.copied, "load_full_plan", return_value=(Path("/unopened/source"), changed)):
                    with self.assertRaisesRegex(loader.DevelopmentInputError, "not_frozen_projection"):
                        loader._row_maps(entries, len(payload), lambda event: None)

    def test_read_only_input_stat_preserves_private_file_and_directory_requirements(self):
        with tempfile.TemporaryDirectory(prefix="ep17-input-stat-") as temporary:
            parent = Path(temporary)
            parent.chmod(0o2700)
            path = parent / "invented.bin"
            path.write_bytes(b"invented")
            path.chmod(0o600)
            self.assertEqual(loader._ordinary_input_file(path, parent), 8)
            for mode in (0o750, 0o2750, 0o500):
                parent.chmod(mode)
                with self.assertRaisesRegex(loader.DevelopmentInputError, "directory_permissions"):
                    loader._ordinary_input_file(path, parent)
            parent.chmod(0o2700)
            for mode in (0o640, 0o400):
                path.chmod(mode)
                with self.assertRaisesRegex(loader.DevelopmentInputError, "file_representation"):
                    loader._ordinary_input_file(path, parent)
            path.chmod(0o600)
            alias = parent / "alias"
            alias.symlink_to(path)
            with self.assertRaisesRegex(loader.DevelopmentInputError, "file_representation"):
                loader._ordinary_input_file(alias, parent)
            alias.unlink()
            os.link(path, alias)
            with self.assertRaisesRegex(loader.DevelopmentInputError, "file_representation"):
                loader._ordinary_input_file(path, parent)
            alias.unlink()
            alias.symlink_to(parent, target_is_directory=True)
            with self.assertRaisesRegex(loader.DevelopmentInputError, "directory_permissions"):
                loader._ordinary_input_file(alias / path.name, parent)
            with self.assertRaisesRegex(loader.DevelopmentInputError, "outside_attempt"):
                loader._ordinary_input_file(parent.parent / "unopened", parent)
            self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o2700)

    def test_copy_binding_and_full_response_byte_admission_precede_reads(self):
        design, repeats, axes, report = fixture()
        entries = loader._copy_entries(report)
        self.assertEqual(len(entries), 12)
        self.assertTrue(all(entry["role"] == "development" for entry in entries.values()))
        for failure in ("incomplete", "wrong_role_path", "wrong_assignment"):
            changed = deepcopy(report)
            if failure == "incomplete":
                changed["products"][-1]["complete"] = False
            elif failure == "wrong_role_path":
                changed["products"][0]["outputs"][0]["array_file"] = str(loader.COPY_PRIVATE / "sealed_audit/sub-01/B.h5")
            else:
                changed["role_assignment_attempt"] = "/invented_wrong_assignment"
            with self.subTest(failure=failure), self.assertRaises(loader.DevelopmentInputError):
                loader._copy_entries(changed)
        required = 3 * 2423 * sum(len(axis) for axis in axes.values()) * 8
        with mock.patch.object(loader.np, "full") as allocate, mock.patch.object(loader.h5py, "File") as opened:
            with self.assertRaisesRegex(loader.DevelopmentInputError, "allocation_required"):
                loader._read_training_responses(design, repeats, axes, entries,
                    response_byte_cap=required - 1, check=lambda event: None)
            allocate.assert_not_called()
            opened.assert_not_called()

    def test_exact_training_repeats_and_columns_equal_means_leave_other_cells_unread(self):
        design, repeats, axes, report = fixture()
        entries = loader._copy_entries(report)
        train = np.asarray([i for i, fold in enumerate(design.fold_ids) if fold in ("fold03", "fold04", "fold05", "fold06")])
        admitted_rows = np.asarray([row for i in train for row in (2 * i, 2 * i + 1)])
        admitted_columns = {person: sorted({int(column) for group, axis in axes.items() if group[0] == person
                                           for column in sorted(axis)[:16]}) for person in loader.endpoint.SUBJECTS}
        reads, opens, closes, events = [], [], [], []
        case = self

        class Handle:
            def __init__(self, path, mode, *, rdcc_nbytes):
                case.assertEqual(mode, "r")
                case.assertEqual(rdcc_nbytes, loader.HDF_CACHE_BYTES)
                relative = path.relative_to(loader.COPY_PRIVATE)
                case.assertEqual(relative.parts[0], "development")
                case.assertEqual(len(relative.parts), 3)
                self.person, self.stage = relative.parts[1], path.stem
                opens.append((self.person, self.stage))

            def __enter__(self):
                return self

            def __exit__(self, *_):
                closes.append((self.person, self.stage))

            def __getitem__(self, key):
                case.assertEqual(len(key), 4)
                rows, zero1, zero2, column = key
                case.assertEqual((zero1, zero2), (0, 0))
                np.testing.assert_array_equal(rows, admitted_rows)
                case.assertIn(column, admitted_columns[self.person])
                reads.append((self.person, self.stage, column))
                stage_offset = loader.endpoint.STAGES.index(self.stage) / 4
                person_offset = loader.endpoint.SUBJECTS.index(self.person) * 10000
                return (rows * 8 + column / 128 + stage_offset + person_offset).astype("float32")

        def output_dataset(handle, shape, dtype):
            case.assertEqual(shape, entries[handle.person, handle.stage]["shape"])
            case.assertEqual(dtype, "<f4")
            return handle

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(loader.h5py, "File", side_effect=Handle))
            stack.enter_context(mock.patch.object(loader.columns, "output_dataset", side_effect=output_dataset))
            panels, dimensions = loader._read_training_responses(design, repeats, axes, entries,
                response_byte_cap=16 * 1024 * 1024, check=events.append)
        expected_reads = [(person, stage, column) for person in loader.endpoint.SUBJECTS
                          for stage in loader.endpoint.STAGES for column in admitted_columns[person]]
        self.assertEqual(reads, expected_reads)
        self.assertEqual(opens, closes)
        self.assertEqual(len(opens), 12)
        self.assertEqual(events[0].operation, "plan_cost_response_headers")
        self.assertEqual(events[1].operation, "allocate_cost_only_response_headers")
        self.assertEqual(events[0].byte_count, 3 * 2423 * sum(len(axis) for axis in axes.values()) * 8)
        self.assertEqual(dimensions["logical_training_column_reads"], 384)
        self.assertEqual(dimensions["logical_training_response_values"], 384 * len(admitted_rows))
        self.assertIs(dimensions["validation_or_fold01_response_values_returned"], False)
        self.assertIs(dimensions["HDF_physical_cache_may_span_unselected_cells"], True)
        for group in loader.endpoint.GROUPS:
            positions = np.argsort(axes[group])[:16]
            for stage in loader.endpoint.STAGES:
                values = panels[group].stages[stage]
                self.assertFalse(values.flags.writeable)
                expected = (train[:, None] * 16 + 4 + axes[group][positions][None, :] / 128
                    + loader.endpoint.STAGES.index(stage) / 4 + loader.endpoint.SUBJECTS.index(group[0]) * 10000)
                np.testing.assert_array_equal(values[np.ix_(train, positions)], expected)
                allowed = np.zeros(values.shape, dtype=bool)
                allowed[np.ix_(train, positions)] = True
                self.assertTrue(np.isnan(values[~allowed]).all())
                # The existing production finite guard refuses this as a full
                # neural response bank; no new fitting/qualification is run.
                with self.assertRaises(ValueError):
                    loader.nested._frozen_matrix(values, 2423, len(axes[group]), "responses")

    def test_explicit_cost_cli_records_header_plan_and_preserves_completed_or_stopped_attempt(self):
        for outcome in ("complete", "stopped", "raised", "header_cap"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory(prefix="ep17-cost-cli-") as directory:
                args = SimpleNamespace(feature_attempt="full001", attempt="cost001",
                    deadline_epoch=runner.time.time() + 60, rss_limit_kib=7 * 1024 * 1024,
                    metadata_byte_cap=32 * 1024 * 1024, feature_byte_cap=256 * 1024 * 1024,
                    response_byte_cap=4096, report_byte_cap=65536)
                source = SimpleNamespace(design=object(), features=object(), responses=object(),
                    frozen_columns=object(), operational_dimensions={"training_concepts": 320},
                    complete_development_response_release=False)
                planned = 8192 if outcome == "header_cap" else 2048

                def load(**kwargs):
                    self.assertEqual(kwargs["feature_attempt"], "full001")
                    self.assertEqual(kwargs["response_byte_cap"], 4096)
                    kwargs["resource_check"](loader.InputEvent("plan_cost_response_headers", byte_count=planned))
                    kwargs["resource_check"](loader.InputEvent("read_training_column", participant="sub-01",
                                                               stage="B", original_column=123))
                    return source

                def measure(*values, resource_check):
                    self.assertEqual(values, (source.design, source.features, source.responses, source.frozen_columns))
                    resource_check(loader.cost.ResourceEvent("fit_pca"))
                    if outcome == "raised":
                        raise RuntimeError("invented private kernel detail")
                    return loader.cost.PredictionCostReport(status=outcome,
                        failure_code=None if outcome == "complete" else "operation_failed",
                        last_operation="predict_validation", completed_counts={"fit_pca": 16},
                        attempted_counts={"fit_pca": 16}, operation_seconds={"fit_pca": .1},
                        total_wall_seconds=.2, process_cpu_seconds=.1, process_peak_rss_kib=1234,
                        dimensions={"training_concepts": 320})

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(runner, "REPORT_ROOT", Path(directory)))
                    stack.enter_context(mock.patch.object(loader, "load_development_prediction_cost_inputs", side_effect=load))
                    measured = stack.enter_context(mock.patch.object(loader.cost, "measure_development_prediction_cost", side_effect=measure))
                    self.assertEqual(runner._run(args), 0 if outcome == "complete" else 1)
                    path = Path(directory) / "cost001/cost_report.json"
                    before = path.read_bytes()
                    with self.assertRaises(runner.CostRunRefusal):
                        runner._run(args)
                    self.assertEqual(path.read_bytes(), before)
                report = json.loads(before)
                self.assertEqual(report["status"], "completed" if outcome == "complete" else "stopped")
                self.assertEqual(report["response_header_bytes"], planned)
                self.assertEqual(report["feature_attempt"], "full001")
                self.assertGreaterEqual(report["total_pipeline_wall_seconds"], 0)
                self.assertNotIn("invented private kernel detail", before.decode())
                self.assertEqual(report["neural_training_read_attempted"], outcome != "header_cap")
                if outcome == "header_cap":
                    measured.assert_not_called()
                else:
                    measured.assert_called_once()
                if outcome in {"complete", "stopped"}:
                    self.assertEqual(report["cost_report"]["status"], outcome)
                    self.assertEqual(report["loader_dimensions"], {"training_concepts": 320})
                else:
                    self.assertIsNone(report["cost_report"])


if __name__ == "__main__":
    unittest.main()
