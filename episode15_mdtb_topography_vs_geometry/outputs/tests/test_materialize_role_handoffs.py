"""Small opaque-byte fixtures for unreleased ordinary role containers."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import materialize_role_handoffs as full
import prepare_role_handoffs as preparation
from test_role_handoffs import inputs_fixture


def fixture(base):
    source, private, pilot, output = [base/name for name in ("source", "private", "pilot", "public")]
    for directory in (source, private, pilot, output):
        directory.mkdir()
    for role in full.ROLES:
        (private/role).mkdir()
    rows, payloads = [], {}
    for pid in ("sub-02", "sub-08"):
        items = (("A", preparation.beta_member(pid, "ses-s1", 1, 0)),
                 ("B", preparation.beta_member(pid, "ses-s2", 1, 0)),
                 ("anatomy", f"{pid}/anat/{pid}_T1w.nii"))
        with zipfile.ZipFile(source/f"MDTB_{pid}.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for kind, name in items:
                payload = (pid+kind).encode()+b"\x00\x80\xff\x7f"
                payloads[name] = payload
                archive.writestr(name, payload)
                role = "development" if pid == "sub-08" else ("sealed_audit" if kind == "B" else "audit_calibration")
                rows.append({"participant_id": pid, "member": name, "kind": kind, "handoff": role,
                             **preparation.member_metadata(archive.getinfo(name))})
                if pid == "sub-08" and kind != "B":
                    path = pilot/"development"/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
    return sorted(rows, key=lambda row: (row["participant_id"], row["member"])), payloads, source, private, pilot, output


class FullRoleHandoffTests(unittest.TestCase):
    def test_whole_participant_byte_packs_are_deterministic(self):
        rows = [{"participant_id": pid, "bytes": size} for pid, size in
                (("p1", 8), ("p2", 7), ("p3", 6), ("p4", 5), ("p5", 4), ("p6", 3), ("p1", 2))]
        packs, sizes = full.balanced_packs(rows)
        self.assertEqual(packs, [["p1"], ["p2"], ["p3", "p6"], ["p4", "p5"]])
        self.assertEqual(sizes, [10, 7, 9, 9])
        self.assertEqual(full.balanced_packs(list(reversed(rows))), (packs, sizes))
        self.assertEqual(Counter(pid for pack in packs for pid in pack), dict.fromkeys(("p1", "p2", "p3", "p4", "p5", "p6"), 1))

    def test_literal_role_containers_reuse_pilot_and_record_real_audit_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, payloads, source, private, pilot, output = fixture(Path(directory))
            original_open = zipfile.ZipFile.open
            source_opened = []
            def tracked(archive, name, *args, **kwargs):
                member = name.filename if isinstance(name, zipfile.ZipInfo) else name
                if archive.filename and Path(archive.filename).parent == source:
                    source_opened.append(member)
                    if member.startswith("sub-08/") and "ses-s2" not in member:
                        raise AssertionError("completed pilot source must not be reread from mixed ZIP")
                return original_open(archive, name, *args, **kwargs)
            stop = threading.Event()
            with patch.object(zipfile.ZipFile, "open", tracked):
                report = full.copy_pack(0, ["sub-02", "sub-08"], rows, source, private, pilot, output,
                                        time.monotonic()+30, stop)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["containers"]), 3)
            self.assertEqual(len(source_opened), 4)
            self.assertEqual(report["roles"]["development"]["pilot_streams_opened"], 2)
            for role in full.ROLES:
                with zipfile.ZipFile(private/role/("sub-08.zip" if role == "development" else "sub-02.zip")) as archive:
                    expected = [row for row in rows if row["handoff"] == role]
                    self.assertEqual(archive.namelist(), [row["member"] for row in expected])
                    for row in expected:
                        self.assertEqual(archive.read(row["member"]), payloads[row["member"]])
                        self.assertEqual(archive.getinfo(row["member"]).compress_type, zipfile.ZIP_STORED)
                counter = report["roles"][role]
                self.assertEqual(counter["acknowledged_payload_bytes_written"], counter["closed_container_bytes_verified"])
            for kind, suffix in (("A", "Task_A"), ("B", "Task_B"), ("anatomy", "anatomy")):
                expected = sum(row["bytes"] for row in rows if row["participant_id"] == "sub-02" and row["kind"] == kind)
                self.assertEqual(report["raw_audit_"+suffix+"_bytes_read"], expected)
                self.assertIsNotNone(report["first_raw_audit_"+suffix+"_byte_utc"])
            self.assertFalse(report["scientific_audit_opened"])
            self.assertFalse(report["reader_activation"])
            for row in rows:
                if row["participant_id"] == "sub-08" and row["kind"] != "B":
                    self.assertEqual((pilot/"development"/row["member"]).read_bytes(), payloads[row["member"]])

    def test_changed_reused_pilot_refuses_publication_without_source_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, source, private, pilot, output = fixture(Path(directory))
            row = next(row for row in rows if row["participant_id"] == "sub-08" and row["kind"] == "anatomy")
            path = pilot/"development"/row["member"]
            path.write_bytes(b"x"*row["bytes"])
            report = full.copy_pack(0, ["sub-08"], rows, source, private, pilot, output,
                                    time.monotonic()+30, threading.Event())
            self.assertFalse(report["complete"])
            self.assertIn("size_CRC_changed", report["error"])
            self.assertFalse((private/"development/sub-08.zip").exists())
            self.assertGreater(report["active_partial_container_bytes_on_disk"], 0)
            self.assertEqual(report["roles"]["development"]["source_open_attempts"], 0)

    def test_returned_read_prefix_and_partial_container_survive_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, source, private, pilot, output = fixture(Path(directory))
            stop = threading.Event()
            original_open = full.open_member
            class StopAfterRead:
                def __init__(self, reader):
                    self.reader = reader
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    self.reader.close()
                def read(self, count):
                    value = self.reader.read(count)
                    stop.set()
                    return value
            def opened(*args):
                reader, kind = original_open(*args)
                return StopAfterRead(reader), kind
            with patch.object(full, "open_member", side_effect=opened), patch.object(full, "CHUNK", 2):
                report = full.copy_pack(0, ["sub-02"], rows, source, private, pilot, output,
                                        time.monotonic()+30, stop)
            self.assertFalse(report["complete"])
            self.assertEqual(report["roles"]["audit_calibration"]["source_bytes_read"], 2)
            self.assertEqual(report["roles"]["audit_calibration"]["acknowledged_payload_bytes_written"], 0)
            self.assertGreater(report["active_partial_container_bytes_on_disk"], 0)
            self.assertFalse((private/"audit_calibration/sub-02.zip").exists())
            saved = json.loads((output/"worker0.json").read_text())
            self.assertFalse(saved["complete"])
            self.assertEqual(saved["active_member"], report["active_member"])

    def test_original_full_inventory_and_completed_pilot_binding(self):
        inputs = inputs_fixture()
        original = preparation.expected_members(inputs)
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)/"outputs"
            source = Path(directory)/preparation.SOURCE_RELATIVE
            rows = [{**row, "bytes": 2, "compressed_bytes": 2, "zip_crc32": 1, "zip_compression": 0}
                    for row in original]
            document = {"inventory_complete": True, "source": str(source), "member_count": len(rows), "members": rows,
                "pilot": {"bytes": 1480}, "bytes_by_handoff": {role: count*2 for role, count in preparation.HANDOFF_COUNTS.items()},
                "total_bytes": len(rows)*2}
            pilot = {"status": "pilot_complete", "mode": "pilot", "operation_complete": True,
                "members_copied": 740, "bytes_written": 1480, "destination_bytes_verified": 1480, "expected_pilot_bytes": 1480,
                "inventory_source": str(outputs/full.INVENTORY_RELATIVE), "private_unreleased_output": str(outputs/full.PILOT_PRIVATE),
                "Task_B_streams_opened": 0, "audit_member_streams_opened": 0}
            def read(path, _):
                if path == outputs/full.INVENTORY_RELATIVE:
                    value = document
                elif path == outputs/full.PILOT_RELATIVE:
                    value = pilot
                else:
                    self.assertEqual(path, (outputs/full.INVENTORY_RELATIVE).parent/preparation.REPORT_NAME)
                    value = {"mode": "inventory", "status": "inventory_complete", "operation_complete": True}
                return json.dumps(value).encode()
            with patch.object(full, "OUTPUTS", outputs), patch.dict(os.environ, OAK=directory), \
                    patch.object(preparation, "load_inputs", return_value=inputs), \
                    patch.object(preparation, "read_bounded", side_effect=read):
                actual, _, _, forecast = full.load_manifest()
                self.assertEqual(actual, rows)
                self.assertLess(forecast["conservative_total_storage_bytes"], full.STORAGE_CAP)
                pilot["operation_complete"] = False
                with self.assertRaisesRegex(preparation.HandoffFailure, "completed_fixed_sub08"):
                    full.load_manifest()

    def exercise_parent(self, directory, fake_worker, clock=None, write=None):
        outputs = Path(directory)/"outputs"
        args = argparse.Namespace(output_dir=outputs/"runs/role_handoffs/full001",
            private_dir=outputs/"cache/private/role_handoffs/full001", max_seconds=30)
        env = {key: "1" for key in preparation.THREAD_ENV}
        env.update(SLURM_JOB_ID="fixture", SLURM_CPUS_PER_TASK="1", OAK=directory)
        rows = [{"participant_id": "sub-08", "bytes": 1}]
        forecast = {"payload_bytes": 36576, "retained_pilot_bytes": 1}
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in (patch.object(full, "OUTPUTS", outputs), patch.object(full, "WORKERS", 1),
                         patch.dict(os.environ, env), patch.object(full.socket, "gethostname", return_value="compute-fixture"),
                         patch.object(full, "load_manifest", return_value=(rows, Path(directory), Path(directory), forecast)),
                         patch.object(full, "copy_pack", side_effect=fake_worker)):
                stack.enter_context(item)
            if clock is not None:
                stack.enter_context(patch.object(full.time, "monotonic", side_effect=lambda: clock[0]))
            if write is not None:
                stack.enter_context(patch.object(preparation, "write_json", side_effect=write))
            return full.run(args), args

    def test_parent_joins_failed_worker_and_restores_signal_handlers(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        entered = threading.Event()
        def failed(*_):
            entered.set()
            raise RuntimeError("selected synthetic worker failure")
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.exercise_parent(directory, failed)
        self.assertTrue(entered.is_set())
        self.assertFalse(result["materialization_complete"])
        self.assertTrue(result["workers_joined"])
        self.assertIn("synthetic worker failure", result["error"])
        self.assertFalse(any(thread.name.startswith("role-copy") for thread in threading.enumerate()))
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)

    def test_parent_terminal_publication_expiry_clears_materialization(self):
        def complete(*_):
            containers = []
            for role, count in preparation.HANDOFF_COUNTS.items():
                for index in range(12):
                    members = count//12
                    containers.append({"handoff": role, "members": members, "payload_bytes": members, "container_bytes": members})
            return {"complete": True, "containers": containers}
        clock = [0.0]
        original_write = preparation.write_json
        def publication(path, value):
            original_write(path, value)
            if value.get("materialization_complete"):
                clock[0] = 31.0
        with tempfile.TemporaryDirectory() as directory:
            result, args = self.exercise_parent(directory, complete, clock, publication)
            saved = json.loads((args.output_dir/"materialization_report.json").read_text())
        self.assertFalse(result["materialization_complete"])
        self.assertEqual(saved["status"], "technical_materialization_failure")
        self.assertIn("deadline_during_terminal_publication", saved["error"])
        self.assertTrue(saved["workers_joined"])
        self.assertFalse(saved["scientific_audit_opened"])
        self.assertFalse(saved["reader_activation"])


if __name__ == "__main__":
    unittest.main()
