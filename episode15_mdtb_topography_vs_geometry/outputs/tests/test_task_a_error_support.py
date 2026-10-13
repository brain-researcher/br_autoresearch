"""Four focused source-fixture checks; run only in root's compute allocation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_task_a_error_support as diagnostic
import development_handoff_reader as reader
from local_functional_source import TASK_A_CONDITIONS
from test_development_handoff_reader import member_fixture, release_fixture


class TaskAErrorSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.summaries = member_fixture()
        cls.a_rows = [row for row in cls.rows if row["session"] == "ses-s1"]
        cls.a_summary = {"ses-s1": cls.summaries["ses-s1"]}

    def test_signed_covariances_spatial_offsets_deletions_and_invisible_shared_effect(self):
        people = np.arange(1., 13.)
        conditions = np.arange(1., 30.)
        pattern = np.tile([-1., 1.], 104)
        for coefficients in ([3., 1., 2., 4.], [1., 1., -1., -1.], [1., 1., 1., 1.]):
            b = np.array(coefficients)
            values = (people[:, None, None, None] * b[None, :, None, None]
                      * conditions[None, None, :, None] * pattern[None, None, None, :])
            # Per-partition/condition/person constants must vanish spatially.
            values += np.arange(12 * 4 * 29).reshape(12, 4, 29, 1)
            original = values.copy()
            result = diagnostic.summarize_partitions(values, check=lambda: None)
            np.testing.assert_array_equal(values, original)
            factors = np.array([(b[0] - b[2]) * (b[1] - b[3]),
                                (b[0] - b[1]) * (b[2] - b[3])])
            expected = 13. * factors[:, None] * conditions[None, :] ** 2
            deleted = []
            for omit in range(12):
                remaining = np.delete(people, omit)
                variance = sum((remaining - sum(remaining) / 11) ** 2) / 10
                deleted.append(variance * factors[:, None] * conditions[None, :] ** 2)
            deleted = np.stack(deleted)
            self.assertEqual([row["condition"] for row in result["condition_summaries"]], list(TASK_A_CONDITIONS))
            for c, row in enumerate(result["condition_summaries"]):
                np.testing.assert_allclose([row["S"], row["N"]], expected[:, c], atol=1e-10, rtol=1e-14)
                for number, name in enumerate(("S", "N")):
                    np.testing.assert_allclose(row[name + "_delete_one_range"],
                        [deleted[:, number, c].min(), deleted[:, number, c].max()], atol=1e-10, rtol=1e-14)
            self.assertFalse(result["independence_established"])
            self.assertFalse(result["scientific_pass_decision"])
            self.assertFalse(result["ranges_are_confidence_intervals"])
        # The last case has real between-person variation, invisible to S/N.
        self.assertTrue(all(row["S"] == row["N"] == 0 for row in result["condition_summaries"]))

    def test_complete_run_arithmetic_retains_multiplicity_and_refuses_bad_role_before_reads(self):
        plans, _ = diagnostic.compile_task_a_plan(self.a_rows, self.a_summary, check=lambda: None)
        conditions = {name: number for number, name in enumerate(TASK_A_CONDITIONS)}
        first = reader.DEVELOPMENT_IDS[0]

        def sample(row):
            offset = 0.
            if row["run"] == 1 and row["cond_name"] == TASK_A_CONDITIONS[0]:
                offset = -10. if row["reg_id"] == 0 else 10.
            return 100 * conditions[row["cond_name"]] + row["run"] + offset + np.arange(208) / 1024

        result = diagnostic.partition_maps(plans[first], sample=sample, check=lambda: None)
        baseline = 100 * np.arange(29)[:, None] + np.arange(208) / 1024
        np.testing.assert_array_equal(result, np.stack([baseline + k for k in (2.5, 6.5, 10.5, 14.5)]))
        bindings, support, _ = release_fixture(self.rows)
        for rows, summary in ((self.a_rows + [self.rows[-1]], self.a_summary),
                              (self.a_rows[:-1], self.a_summary),
                              (self.a_rows, self.summaries)):
            with patch.object(diagnostic.zipfile, "ZipFile") as opened, self.assertRaises(ValueError):
                diagnostic.read_task_a_partitions(release_bindings=bindings, member_rows=rows,
                    condition_multiplicity_by_session=summary, common_support=support, check=lambda: None)
            opened.assert_not_called()

    def test_full_released_mapping_denies_B_and_checks_last_header_before_any_voxel(self):
        for problem in (None, "last_beta_grid", "last_forward_unknown_units",
                        "last_forward_nonmm_units", "support_order"):
            bindings, support, archives = release_fixture(self.rows)
            # Match the actual historical metadata: anatomy records lack
            # units, while the allowlisted live NIfTI headers carry them.
            for binding in bindings:
                for header in binding.anatomy_headers.values():
                    header.pop("xyzt_units")
            original_headers = deepcopy([binding.anatomy_headers for binding in bindings])
            allowed = {pid: {row["beta_member"] for row in self.a_rows
                            if row["participant_id"] == pid and not row["instruction"]}
                       for pid in reader.DEVELOPMENT_IDS}
            for pid, archive in archives.items():
                # The existing fixture raises on any nonselected stream.
                archive.selected = allowed[pid]
            if problem == "last_beta_grid":
                last = archives[reader.DEVELOPMENT_IDS[-1]]
                payload = bytearray(last.beta_bytes)
                struct.pack_into("<f", payload, 292, 999.)
                last.override[sorted(last.selected)[-1]] = bytes(payload)
            elif problem in ("last_forward_unknown_units", "last_forward_nonmm_units"):
                last = archives[reader.DEVELOPMENT_IDS[-1]]
                payload = bytearray(last.forward_bytes)
                payload[123] = 0 if problem == "last_forward_unknown_units" else 1
                last.override[last.forward] = bytes(payload)
            elif problem == "support_order":
                support["source_transform_voxel_ijk"][[0, 1]] = support["source_transform_voxel_ijk"][[1, 0]]
            native = np.tile([10.5, 20.5, 30.5], (208, 1))
            native[-1] = [11, 21, 31]

            def pull(*_):
                for pid, archive in archives.items():
                    self.assertEqual(set(archive.opened), allowed[pid] | {archive.forward})
                return native

            with patch.object(diagnostic.zipfile, "ZipFile", side_effect=lambda path: archives[path.stem]), \
                    patch.object(reader, "_native_coordinates", side_effect=pull) as coordinates:
                if problem is not None:
                    with self.assertRaises(ValueError):
                        diagnostic.read_task_a_partitions(release_bindings=bindings, member_rows=self.a_rows,
                            condition_multiplicity_by_session=self.a_summary, common_support=support, check=lambda: None)
                    coordinates.assert_not_called()
                    self.assertTrue(all(count == 348 for archive in archives.values() for _, count in archive.reads))
                else:
                    maps, report = diagnostic.read_task_a_partitions(release_bindings=bindings, member_rows=self.a_rows,
                        condition_multiplicity_by_session=self.a_summary, common_support=support, check=lambda: None)
                    expected = np.full(208, 4.); expected[-1] = 11.
                    np.testing.assert_array_equal(maps, np.broadcast_to(expected, (12, 4, 29, 208)))
                    self.assertEqual(coordinates.call_count, 12)
                    self.assertEqual(report["mapped_Task_A_beta_members"], 12 * (29 * 16 + 1))
                    self.assertEqual(report["Task_B_streams_opened"], 0)
                    self.assertFalse(report["private_reliability_read"])
            self.assertEqual(original_headers, [binding.anatomy_headers for binding in bindings])

    def test_fresh_attempt_preservation_and_failed_composition_never_publishes_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary)
            runs = outputs / "runs"; runs.mkdir()
            args = dict(manifest_path=outputs / "explicit_manifest.json", deadline_epoch=time.time() + 60,
                        rss_limit_kib=10**9, durable_byte_cap=1024**2)
            with patch.object(diagnostic, "_compute"), patch.object(diagnostic, "OUTPUTS", outputs), \
                    patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}), \
                    patch.object(diagnostic, "run_released_diagnostic", return_value=({"complete_fixture": True}, {})):
                completed = runs / "complete"
                self.assertEqual(diagnostic.execute(output_dir=completed, **args), 0)
                prior = {path.name: path.read_bytes() for path in completed.iterdir()}
                with self.assertRaises(FileExistsError):
                    diagnostic.execute(output_dir=completed, **args)
                self.assertEqual(prior, {path.name: path.read_bytes() for path in completed.iterdir()})
            with patch.object(diagnostic, "_compute"), patch.object(diagnostic, "OUTPUTS", outputs), \
                    patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}), \
                    patch.object(diagnostic, "run_released_diagnostic", side_effect=ValueError("private fixture detail")):
                stopped = runs / "stopped"
                self.assertEqual(diagnostic.execute(output_dir=stopped, **args), 1)
                self.assertFalse((stopped / "diagnostic.json").exists())
                report = json.loads((stopped / "operational.json").read_text())
                self.assertEqual(report["status"], "stopped")
                self.assertFalse(report["diagnostic_published"])
                self.assertNotIn("private fixture detail", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
