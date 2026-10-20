"""Focused future-lifecycle fixtures; compute-only, no Torch or EP21 payloads.

The supplied tiny base uses real h5py/sec2 close. These tests validate lifecycle
failure handling, NOT integration with the scientific runner or its full panel.
"""
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import unittest
from unittest import mock

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import amended_decode_lifecycle as lifecycle_module


class TinyCollection:
    """Real HDF fixture with the original writer's exception/close shape."""
    def __init__(self, path, **unused):
        self.partial = Path(str(path) + ".partial")
        self._h5 = h5py.File(self.partial, "x", driver="sec2")
        self._closed = False
        self._check()

    def _check(self):
        if self._closed:
            raise ValueError("fixture_closed")

    def write(self, tick):
        self._check()
        self._h5.create_dataset("fixture", data=[1, 2, 3])
        tick()
        self._check()

    def close(self):
        if self._closed:
            return
        try:
            self._h5.close()
        finally:
            self._closed = not self._h5.id.valid

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note("fixture suppressed cleanup: " + type(cleanup_error).__name__)


Observed = lifecycle_module.observed_collection_class(TinyCollection)


class DecodeLifecycleTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"SLURM_JOB_ID": "60000001"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.now = 100.

    def lifecycle(self):
        return lifecycle_module.DecodeLifecycle(job_id="60000001",
            collection_path=self.root / "future.h5", record_path=self.root / "closure.json",
            work_stop_epoch=110., hard_stop_epoch=120., clock=lambda: self.now)

    def stop(self):
        self.now = 110.

    def record(self, lifecycle):
        self.assertTrue(lifecycle.record_published)
        return json.loads(lifecycle.record_path.read_text())

    def test_deadline_unwinds_real_hdf_and_publishes_observed_close(self):
        lifecycle = self.lifecycle()
        with self.assertRaises(lifecycle_module.DecodeWorkStopped):
            with lifecycle.run():
                with Observed(lifecycle.collection_path, lifecycle=lifecycle) as writer:
                    writer.write(self.stop)
        record = self.record(lifecycle)
        self.assertFalse(writer._h5.id.valid)
        self.assertTrue(record["hdf_close_completed"] and record["fsync_completed"])
        self.assertEqual(record["operation_error_type"], "DecodeWorkStopped")
        self.assertEqual(record["cleanup_reserve_seconds"], 10.)
        self.assertEqual(record["job_id"], "60000001")
        self.assertEqual(record["partial_path"], str(writer.partial))
        self.assertEqual(record["file_identity"]["inode"], writer.partial.stat().st_ino)
        self.assertFalse(record["historical_certification"])
        with h5py.File(writer.partial, "r") as handle:
            self.assertEqual(handle["fixture"][...].tolist(), [1, 2, 3])

    def test_native_close_failure_stays_failed_after_later_close(self):
        lifecycle = self.lifecycle()
        native_close = h5py.File.close
        attempts = []

        def fail_once(handle):
            attempts.append(True)
            if len(attempts) == 1:
                raise OSError("injected native close failure")
            return native_close(handle)

        with mock.patch.object(h5py.File, "close", fail_once):
            with self.assertRaises(lifecycle_module.DecodeWorkStopped):
                with lifecycle.run():
                    with Observed(lifecycle.collection_path, lifecycle=lifecycle) as writer:
                        writer.write(self.stop)
        self.assertFalse(writer._h5.id.valid)
        record = self.record(lifecycle)
        self.assertEqual(record["close_state"], "failed")
        self.assertFalse(record["hdf_close_completed"] or record["fsync_completed"])
        self.assertEqual(record["cleanup_error_type"], "OSError")

    def test_constructor_timeout_after_hdf_open_is_closed_by_outer_lifecycle(self):
        lifecycle = self.lifecycle()
        stop, entered = self.stop, []

        class ConstructorTimeout(TinyCollection):
            def __init__(self, path, **kwargs):
                super().__init__(path, **kwargs)
                stop()
                self._check()

            def __enter__(self):
                entered.append(True)
                return super().__enter__()

        observed = lifecycle_module.observed_collection_class(ConstructorTimeout)
        with self.assertRaises(lifecycle_module.DecodeWorkStopped):
            with lifecycle.run():
                with observed(lifecycle.collection_path, lifecycle=lifecycle):
                    self.fail("constructor timeout must precede inner context entry")
        self.assertFalse(entered)
        self.assertFalse(lifecycle.writer._h5.id.valid)
        self.assertTrue(lifecycle.partial_path.exists())
        record = self.record(lifecycle)
        self.assertEqual(record["operation_error_type"], "DecodeWorkStopped")
        self.assertTrue(record["hdf_close_completed"] and record["fsync_completed"])

    def test_fsync_failure_cannot_be_certified_by_idempotent_second_close(self):
        lifecycle = self.lifecycle()
        real_fsync = os.fsync
        calls = []

        def fail_first(descriptor):
            calls.append(True)
            if len(calls) == 1:
                raise OSError("injected HDF fsync failure")
            return real_fsync(descriptor)

        with mock.patch.object(lifecycle_module.os, "fsync", side_effect=fail_first):
            with self.assertRaises(lifecycle_module.DecodeWorkStopped):
                with lifecycle.run():
                    with Observed(lifecycle.collection_path, lifecycle=lifecycle) as writer:
                        writer.write(self.stop)
        self.assertTrue(writer._closed)
        record = self.record(lifecycle)
        self.assertEqual(record["close_state"], "failed")
        self.assertFalse(record["hdf_close_completed"] or record["fsync_completed"])

    def test_record_fsync_failure_leaves_no_accepted_record(self):
        lifecycle = self.lifecycle()
        real_fsync = os.fsync
        calls = []

        def fail_second(descriptor):
            calls.append(True)
            if len(calls) == 2:  # Actual HDF synchronization succeeded first.
                raise OSError("injected operational record fsync failure")
            return real_fsync(descriptor)

        with mock.patch.object(lifecycle_module.os, "fsync", side_effect=fail_second):
            with self.assertRaises(OSError):
                with lifecycle.run():
                    with Observed(lifecycle.collection_path, lifecycle=lifecycle):
                        pass
        self.assertEqual(lifecycle.close_state, "succeeded")
        self.assertFalse(lifecycle.record_published or lifecycle.record_path.exists())
        self.assertTrue(Path(str(lifecycle.record_path) + ".partial").exists())

    def test_hard_stop_prevents_closure_record_publication(self):
        lifecycle = self.lifecycle()
        with self.assertRaises(lifecycle_module.DecodeWorkStopped):
            with lifecycle.run():
                with Observed(lifecycle.collection_path, lifecycle=lifecycle) as writer:
                    self.now = 120.
                    writer._check()
        self.assertFalse(lifecycle.record_published or lifecycle.record_path.exists())

    def test_directory_fsync_failure_removes_linked_final_record(self):
        lifecycle = self.lifecycle()
        real_fsync = os.fsync
        linked_before_failure = []

        def fail_directory(descriptor):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                linked_before_failure.append(lifecycle.record_path.exists())
                raise OSError("injected directory fsync failure")
            return real_fsync(descriptor)

        with mock.patch.object(lifecycle_module.os, "fsync", side_effect=fail_directory):
            with self.assertRaises(OSError):
                with lifecycle.run():
                    with Observed(lifecycle.collection_path, lifecycle=lifecycle):
                        pass
        self.assertEqual(linked_before_failure, [True])
        self.assertEqual(lifecycle.close_state, "succeeded")
        self.assertFalse(lifecycle.record_published or lifecycle.record_path.exists())
        self.assertTrue(Path(str(lifecycle.record_path) + ".partial").exists())

    def test_unobserved_close_and_existing_partial_are_not_certified(self):
        lifecycle = self.lifecycle()
        with lifecycle.run():
            with Observed(lifecycle.collection_path, lifecycle=lifecycle) as writer:
                writer._h5.close()  # Deliberately bypass the observer.
                writer._closed = True
        record = self.record(lifecycle)
        self.assertEqual(record["close_state"], "not_attempted")
        self.assertFalse(record["hdf_close_completed"] or record["fsync_completed"])
        with self.assertRaises(FileExistsError):
            self.lifecycle()

    def test_optional_term_is_cooperative_and_handler_is_restored(self):
        lifecycle = self.lifecycle()
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(lifecycle_module.DecodeWorkStopped):
            with lifecycle.run(handle_sigterm=True):
                with Observed(lifecycle.collection_path, lifecycle=lifecycle) as writer:
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    writer._check()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertTrue(self.record(lifecycle)["term_requested"])


if __name__ == "__main__":
    unittest.main()
