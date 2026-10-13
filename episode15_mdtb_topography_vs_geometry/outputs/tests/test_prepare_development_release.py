"""Two focused metadata-release fixtures; no real records or image payloads."""
import csv
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_development_release as preparation
from build_condition_partition_plan import MEMBER_COLUMNS
from development_lopo import DEVELOPMENT_IDS
from test_development_handoff_reader import member_fixture, release_fixture


def metadata_fixture(root):
    rows, summaries = member_fixture()
    bindings, _, _ = release_fixture(rows)
    development = root / "containers" / "development"
    development.mkdir(parents=True)
    workers = [dict(complete=True, containers=[]) for _ in range(4)]
    for index, pid in enumerate(DEVELOPMENT_IDS):
        path = development / (pid + ".zip")
        payload = b"fixture container must only be stat'ed, never opened"
        path.write_bytes(payload)
        workers[index % 4]["containers"].append(dict(participant_id=pid, handoff="development",
            path=str(path), members=1524, container_bytes=len(payload), payload_bytes=1,
            closed_original_CRC_verified=True))
    for index, worker in enumerate(workers):
        worker["containers"] += [dict(handoff="sealed_audit", path=f"/audit_never_touch/{index}/{k}") for k in range(6)]
    materialization = dict(status="complete_unreleased_role_materialization", materialization_complete=True,
        workers_joined=True, completed_containers=36, image_arrays_decoded=False,
        scientific_audit_opened=False, worker_results=workers,
        private_unreleased_output=str(root / "containers"))
    anatomy = dict(forward_transform_headers_share_grid=True, all_voxel_payloads_closed=True,
        participants=[dict(participant_id=item.participant_id, headers=item.anatomy_headers) for item in bindings]
        + [dict(participant_id="sub-02", headers={})])
    metadata = dict(metadata_join_complete=True, all_selected_Task_A_headers_equal_native_cerebellar_mask_grid=True,
        Task_B_image_streams_opened=0, condition_multiplicity_by_session=summaries,
        participants=[dict(participant_id=item.participant_id, beta_members_joined={"ses-s1": 736, "ses-s2": 784},
            selected_Task_A_header_equals_prior_native_cerebellar_mask_grid=True,
            selected_Task_A_header=dict(header=item.task_a_header)) for item in bindings]
        + [dict(participant_id="sub-02")])
    paths = {}
    for name, document in (("materialization_report", materialization), ("anatomical_headers", anatomy),
                           ("source_metadata", metadata)):
        path = root / (name + ".json")
        path.write_text(json.dumps(document))
        paths[name] = path
    path = root / "role_member_plan.tsv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MEMBER_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    paths["role_member_plan"] = path
    path = root / "common_support.npz"
    path.write_bytes(b"existing shared support must be stat'ed, not decoded here")
    paths["common_support"] = path
    return paths, materialization


class PrepareDevelopmentReleaseTests(unittest.TestCase):
    def test_exact_views_only_stat_twelve_development_files_and_never_open_containers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, _ = metadata_fixture(root)
            allowed_stats = {root / "containers" / "development" / (pid + ".zip") for pid in DEVELOPMENT_IDS}
            allowed_stats.add(paths["common_support"])
            observed = []
            original_stat = Path.lstat
            original_open = Path.open

            def stat_only(path, *args, **kwargs):
                self.assertIn(path, allowed_stats)
                observed.append(path)
                return original_stat(path, *args, **kwargs)

            def metadata_only(path, *args, **kwargs):
                self.assertIn(path, {paths[name] for name in paths if name != "common_support"})
                return original_open(path, *args, **kwargs)

            final = root / "release" / "view"
            with patch.object(Path, "lstat", new=stat_only), patch.object(Path, "open", new=metadata_only):
                payloads, report = preparation.build_view_payloads(**paths, final_view=final, check=lambda: None)
            self.assertEqual(set(observed), allowed_stats)
            self.assertEqual(len(observed), 13)
            panel = json.loads(payloads["point_panel_manifest.json"])
            diagnostic = json.loads(payloads["Task_A_diagnostic_manifest.json"])
            self.assertEqual(set(panel), {"release_bindings", "member_rows_path", "condition_multiplicity_by_session", "common_support_path"})
            self.assertEqual([row["participant_id"] for row in panel["release_bindings"]], list(DEVELOPMENT_IDS))
            self.assertEqual(diagnostic["release_bindings"], panel["release_bindings"])
            self.assertEqual(set(diagnostic["condition_multiplicity_by_session"]), {"ses-s1"})
            self.assertEqual(panel["member_rows_path"], str(final / "development_members.tsv"))
            for name, expected_count, sessions in (("development_members.tsv", 18240, {"ses-s1", "ses-s2"}),
                                                    ("Task_A_members.tsv", 8832, {"ses-s1"})):
                rows = list(csv.DictReader(io.StringIO(payloads[name].decode()), delimiter="\t"))
                self.assertEqual(len(rows), expected_count)
                self.assertEqual({row["session"] for row in rows}, sessions)
                self.assertEqual({row["participant_id"] for row in rows}, set(DEVELOPMENT_IDS))
            self.assertNotIn(b"audit_never_touch", b"".join(payloads.values()))
            self.assertFalse(report["container_values_opened"])

    def test_atomic_view_failure_and_existing_attempt_history_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary)
            paths, materialization = metadata_fixture(outputs)
            runs = outputs / "runs"; runs.mkdir()
            args = dict(**paths, deadline_epoch=time.time() + 90, rss_limit_kib=10**9, durable_byte_cap=32 * 1024**2)
            with patch.object(preparation, "_compute"), patch.object(preparation, "OUTPUTS", outputs), \
                    patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}):
                completed = runs / "complete"
                self.assertEqual(preparation.execute(output_dir=completed, **args), 0)
                manifest = completed / "view" / "Task_A_diagnostic_manifest.json"
                before = manifest.read_bytes(), (completed / "operational.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    preparation.execute(output_dir=completed, **args)
                self.assertEqual(before, (manifest.read_bytes(), (completed / "operational.json").read_bytes()))
                changed = deepcopy(materialization)
                changed["worker_results"][0]["containers"][0]["container_bytes"] += 1
                paths["materialization_report"].write_text(json.dumps(changed))
                stopped = runs / "mismatched_size"
                self.assertEqual(preparation.execute(output_dir=stopped, **args), 1)
                self.assertFalse((stopped / "view").exists())
                report = json.loads((stopped / "operational.json").read_text())
                self.assertEqual(report["status"], "stopped")
                self.assertFalse(report["view_published"])
                paths["materialization_report"].write_text(json.dumps(materialization))
                capped = runs / "durable_cap"
                self.assertEqual(preparation.execute(output_dir=capped, **{**args, "durable_byte_cap": 65537}), 1)
                self.assertFalse((capped / "view").exists())


if __name__ == "__main__":
    unittest.main()
