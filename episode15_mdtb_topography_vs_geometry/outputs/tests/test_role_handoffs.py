"""Source-free routing and literal ZIP-copy checks; execute only on compute."""
import argparse
from collections import Counter
import copy
import csv
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import prepare_role_handoffs as handoff


DEVELOPMENT = [f"sub-{x:02}" for x in (8, 12, 14, 15, 17, 18, 20, 21, 24, 25, 27, 31)]
AUDIT = [f"sub-{x:02}" for x in (2, 3, 4, 6, 9, 10, 19, 22, 26, 28, 29, 30)]


def inputs_fixture():
    rows = []
    for pid in sorted(DEVELOPMENT+AUDIT):
        role = "development" if pid in DEVELOPMENT else "audit"
        for session, kind in handoff.SESSIONS.items():
            route = "development" if role == "development" else (
                "audit_calibration" if kind == "A" else "sealed_audit")
            # Synthetic identities exercise routing only, not scientific weights.
            for run in range(1, 17):
                for reg in range(handoff.EXPECTED_BETAS[session]//16):
                    row = dict.fromkeys(handoff.MEMBER_COLUMNS, "")
                    row.update(participant_id=pid, session=session, task_set=kind, run=run,
                        reg_id=reg, participant_role=role, planned_handoff=route,
                        beta_member=handoff.beta_member(pid, session, run, reg))
                    rows.append(row)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=handoff.MEMBER_COLUMNS, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    roles = {"enumeration_complete": True, "participants": 24,
        "development_ids": DEVELOPMENT, "audit_ids": AUDIT}
    return {"report": {"status": "complete_condition_partition_member_plan", "plan_complete": True,
            "member_rows": 36480, "output_files": ["condition_partition_plan.json", "role_member_plan.tsv"]},
        "plan": {"member_rows": 36480, "member_table": "role_member_plan.tsv",
            "development_ids": DEVELOPMENT, "audit_ids": AUDIT}, "roles": roles,
        "members": stream.getvalue().encode(),
        "anatomy": {"participants": [{"participant_id": pid,
            "headers": dict.fromkeys(handoff.ANATOMY, {})} for pid in sorted(DEVELOPMENT+AUDIT)]}}


def empty_report():
    return {"zip_directories_read": 0, "members_inventoried": 0, "members_copied": 0,
        "source_member_open_attempts": 0, "source_member_streams_opened": 0,
        "first_payload_access_attempt_utc": None, "first_payload_access_utc": None,
        "source_bytes_read": 0, "bytes_written": 0, "destination_bytes_verified": 0,
        "active_member": None}


def tiny_archive(payloads):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    stream.seek(0)
    return zipfile.ZipFile(stream)


def row_for(info, kind="A", handoff_role="development"):
    return {"participant_id": "sub-08", "member": info.filename, "kind": kind,
        "handoff": handoff_role, **handoff.member_metadata(info)}


class RoleHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = inputs_fixture()
        cls.records = handoff.expected_members(cls.inputs)

    def test_original_member_ids_and_whole_person_routes_with_fixed_pilot(self):
        self.assertEqual(len(self.records), 36576)
        self.assertEqual(Counter(row["handoff"] for row in self.records), handoff.HANDOFF_COUNTS)
        self.assertEqual(len({row["member"] for row in self.records}), 36576)
        self.assertIn(handoff.beta_member("sub-08", "ses-s1", 1, 0),
                      {row["member"] for row in self.records})
        for row in self.records:
            expected = "development" if row["participant_id"] in DEVELOPMENT else (
                "sealed_audit" if row["kind"] == "B" else "audit_calibration")
            self.assertEqual(row["handoff"], expected)
        selected = handoff.pilot_members(self.records)
        self.assertEqual(Counter(row["kind"] for row in selected), {"A": 736, "anatomy": 4})
        self.assertEqual({row["participant_id"] for row in selected}, {"sub-08"})
        changed = copy.deepcopy(self.inputs)
        changed["plan"]["audit_ids"] = DEVELOPMENT
        with self.assertRaisesRegex(handoff.HandoffFailure, "fixed_disjoint"):
            handoff.expected_members(changed)
        changed = dict(self.inputs)
        changed["members"] = self.inputs["members"].replace(b"audit_calibration", b"development", 1)
        with self.assertRaisesRegex(handoff.HandoffFailure, "identity_or_role"):
            handoff.expected_members(changed)

    def test_inventory_reads_directories_only_and_reports_exact_size_sums(self):
        opened = []
        records = self.records
        class DirectoryOnly:
            def __init__(self, path):
                opened.append(path.name)
                self.pid = path.stem.removeprefix("MDTB_")
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def infolist(self):
                result = []
                for row in records:
                    if row["participant_id"] == self.pid:
                        info = zipfile.ZipInfo(row["member"])
                        info.file_size = 13 if row["kind"] == "anatomy" else 7
                        info.compress_size = info.file_size
                        info.CRC = 123
                        result.append(info)
                return result
            def open(self, *_):
                raise AssertionError("inventory must never open a member stream")
        report = empty_report()
        with patch.object(handoff.zipfile, "ZipFile", DirectoryOnly):
            result = handoff.inventory(Path("/synthetic/source"), records, report)
        self.assertEqual(len(opened), 24)
        self.assertEqual(report["source_member_streams_opened"], 0)
        self.assertEqual(report["members_inventoried"], 36576)
        self.assertEqual(result["total_bytes"], 36480*7+96*13)
        self.assertEqual(result["pilot"]["bytes"], 736*7+4*13)
        terminal = {"mode": "inventory", "status": "inventory_complete", "operation_complete": True}
        selected = handoff.bind_inventory(result, terminal, records, Path("/synthetic/source"))
        self.assertEqual(len(selected), 740)
        terminal["operation_complete"] = False
        with self.assertRaisesRegex(handoff.HandoffFailure, "completed_inventory"):
            handoff.bind_inventory(result, terminal, records, Path("/synthetic/source"))

    def test_traversal_duplicate_and_nonregular_members_are_refused(self):
        for name in ("../escape", "/escape", "a/../../escape", "a\\escape", "a//b", "a/./b"):
            with self.subTest(name=name), self.assertRaises(handoff.HandoffFailure):
                handoff.safe_member(name)
        duplicate = zipfile.ZipInfo("sub-08/file.nii")
        link = zipfile.ZipInfo("sub-08/link.nii")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        for infos, reason in (([duplicate, duplicate], "duplicate"), ([link], "unexpected_zip_member_type")):
            fake = unittest.mock.Mock()
            fake.infolist.return_value = infos
            with self.assertRaisesRegex(handoff.HandoffFailure, reason):
                handoff.directory(fake)

    def test_literal_copy_preserves_all_bytes_closed_crc_and_fresh_paths(self):
        names = [handoff.beta_member("sub-08", "ses-s1", 1, 0), "sub-08/anat/sub-08_T1w.nii"]
        payloads = {names[0]: b"\x00\x80\xff\x7f"*9, names[1]: b"literal anatomy header and opaque bytes\x00"}
        with tiny_archive(payloads) as archive, tempfile.TemporaryDirectory() as directory:
            private = Path(directory)
            rows = [row_for(archive.getinfo(name), kind) for name, kind in zip(names, ("A", "anatomy"))]
            report = empty_report()
            handoff.copy_members(archive, rows, private, report)
            self.assertEqual(report["members_copied"], 2)
            self.assertEqual(report["source_bytes_read"], sum(map(len, payloads.values())))
            self.assertEqual(report["bytes_written"], report["destination_bytes_verified"])
            self.assertIsNotNone(report["first_payload_access_utc"])
            for name, payload in payloads.items():
                destination = private/"development"/name
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
                self.assertFalse(destination.with_name(destination.name+".partial").exists())
            with self.assertRaises(FileExistsError):
                handoff.copy_members(archive, rows, private, empty_report())

    def test_all_selected_metadata_and_role_checks_precede_any_copy(self):
        name = handoff.beta_member("sub-08", "ses-s1", 1, 0)
        with tiny_archive({name: b"abc"}) as archive, tempfile.TemporaryDirectory() as directory:
            original = row_for(archive.getinfo(name))
            for changes, reason in (({"bytes": 4}, "source_changed"), ({"kind": "B"}, "only_sub08"),
                                    ({"handoff": "sealed_audit"}, "only_sub08")):
                report = empty_report()
                with patch.object(archive, "open", side_effect=AssertionError("payload opened")):
                    with self.assertRaisesRegex(handoff.HandoffFailure, reason):
                        handoff.copy_members(archive, [original, {**original, **changes}], Path(directory), report)
                self.assertEqual(report["source_member_open_attempts"], 0)
                self.assertEqual(report["bytes_written"], 0)

    def test_partial_read_is_truthful_and_no_incomplete_file_is_published(self):
        name = handoff.beta_member("sub-08", "ses-s1", 1, 0)
        with tiny_archive({name: b"abcdefghij"}) as archive, tempfile.TemporaryDirectory() as directory:
            row, report = row_for(archive.getinfo(name)), empty_report()
            def stop_after_second_read():
                if report["source_bytes_read"] >= 8:
                    raise TimeoutError("selected deadline")
            with patch.object(handoff, "CHUNK", 4), self.assertRaises(TimeoutError):
                handoff.copy_members(archive, [row], Path(directory), report, stop_after_second_read)
            target = Path(directory)/"development"/name
            self.assertFalse(target.exists())
            self.assertEqual(target.with_name(target.name+".partial").read_bytes(), b"abcd")
            self.assertEqual((report["source_bytes_read"], report["bytes_written"], report["members_copied"]), (8, 4, 0))
            self.assertEqual(report["active_member"], name)
            self.assertEqual(report["active_partial_bytes_on_disk"], 4)
        with tiny_archive({name: b"abcdefghij"}) as archive, tempfile.TemporaryDirectory() as directory:
            row, report = row_for(archive.getinfo(name)), empty_report()
            original_fdopen = os.fdopen
            class PartialWriter:
                def __init__(self, *args):
                    self.stream = original_fdopen(*args)
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    self.stream.close()
                def write(self, chunk):
                    self.stream.write(chunk[:2])
                    raise OSError("write failed after a prefix")
            with patch.object(handoff.os, "fdopen", PartialWriter), self.assertRaises(OSError):
                handoff.copy_members(archive, [row], Path(directory), report)
            self.assertEqual(report["bytes_written"], 0)
            self.assertEqual(report["active_partial_bytes_on_disk"], 2)
            self.assertEqual(report["members_copied"], 0)

    def test_pilot_cap_refuses_before_archive_or_private_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)/"outputs"
            source = Path(directory)/handoff.SOURCE_RELATIVE
            inventory_dir = outputs/"runs/role_handoffs/inventory001"
            inventory_dir.mkdir(parents=True)
            members = [{**row, "bytes": 2, "compressed_bytes": 2, "zip_crc32": 1, "zip_compression": 0}
                       for row in self.records]
            (inventory_dir/"inventory.json").write_text(json.dumps({"inventory_complete": True,
                "source": str(source), "member_count": len(members), "members": members, "pilot": {"bytes": 1480}}))
            (inventory_dir/handoff.REPORT_NAME).write_text(json.dumps({"mode": "inventory",
                "status": "inventory_complete", "operation_complete": True}))
            private = outputs/"cache/private/role_handoffs/pilot001"
            args = argparse.Namespace(mode="pilot", max_seconds=600, max_output_bytes=1479,
                inventory=inventory_dir/"inventory.json", private_dir=private,
                output_dir=outputs/"runs/role_handoffs/pilot001")
            env = {key: "1" for key in handoff.THREAD_ENV}
            env.update(SLURM_JOB_ID="fixture", SLURM_CPUS_PER_TASK="1", OAK=directory)
            with patch.dict(os.environ, env), patch.object(handoff, "OUTPUTS", outputs), \
                    patch.object(handoff.socket, "gethostname", return_value="compute-fixture"), \
                    patch.object(handoff, "load_inputs", return_value=self.inputs), \
                    patch.object(handoff.zipfile, "ZipFile", side_effect=AssertionError("archive opened")):
                result = handoff.run(args)
            self.assertFalse(result["operation_complete"])
            self.assertIn("exceeds_explicit", result["error"])
            self.assertFalse(private.exists())
            self.assertEqual(result["source_member_open_attempts"], 0)
            self.assertFalse(result["reader_activation"])

    def test_terminal_publication_deadline_clears_success_and_release_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)/"outputs"
            args = argparse.Namespace(mode="inventory", max_seconds=120,
                output_dir=outputs/"runs/role_handoffs/inventory001")
            clock = [0.0]
            original_write = handoff.write_json
            def publish(path, value):
                original_write(path, value)
                if path.name == handoff.REPORT_NAME and value["operation_complete"]:
                    clock[0] = 121.0
            document = {"total_bytes": 42, "pilot": {"bytes": 12}, "member_count": 36576}
            env = {key: "1" for key in handoff.THREAD_ENV}
            env.update(SLURM_JOB_ID="fixture", SLURM_CPUS_PER_TASK="1", OAK=directory)
            with patch.dict(os.environ, env), patch.object(handoff, "OUTPUTS", outputs), \
                    patch.object(handoff.socket, "gethostname", return_value="compute-fixture"), \
                    patch.object(handoff, "load_inputs", return_value=self.inputs), \
                    patch.object(handoff, "inventory", return_value=document), \
                    patch.object(handoff.time, "monotonic", side_effect=lambda: clock[0]), \
                    patch.object(handoff, "write_json", side_effect=publish):
                result = handoff.run(args)
            saved = json.loads((args.output_dir/handoff.REPORT_NAME).read_text())
            self.assertFalse(result["operation_complete"])
            self.assertEqual(saved["status"], "technical_handoff_failure")
            self.assertIn("deadline_during_terminal", saved["error"])
            self.assertFalse(saved["reader_activation"])
            self.assertFalse(saved["all_role_handoffs_materialized"])
            self.assertFalse(saved["reader_isolation_established"])
            self.assertEqual(saved["Task_B_streams_opened"], 0)


if __name__ == "__main__":
    unittest.main()
