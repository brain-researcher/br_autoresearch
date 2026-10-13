"""New continuation adapter checks only; run inside its approved Slurm job."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_channel_continuation as driver


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        driver.prefix.load_libraries()
        cls.np = driver.prefix.np

    def budget(self):
        return driver.Budget({"job_end_epoch": time.time() + 86400}, driver.MAX_SECONDS)

    def test_authority_absence_or_wrong_scope_precedes_payload_and_imports(self):
        with mock.patch.object(driver.prefix, "read_json") as read:
            with self.assertRaisesRegex(driver.Stop, "fixed_authorization_path"):
                driver.authorize("/tmp/unrelated.json")
            read.assert_not_called()
        for approved in (False, 1, "true"):
            with mock.patch.object(driver.prefix, "read_json", return_value=dict(driver.AUTH_EXPECTED, approved=approved)):
                with self.assertRaisesRegex(driver.Stop, "explicit_continuation_authorization"):
                    driver.authorize(driver.AUTH)
        with mock.patch.object(Path, "cwd", return_value=driver.BASE), \
             mock.patch.object(driver.prefix, "read_json", side_effect=FileNotFoundError), \
             mock.patch.object(driver.prefix, "load_libraries") as imports, \
             mock.patch.object(driver, "verify_allocation") as allocation:
            with self.assertRaises(FileNotFoundError):
                driver.run(SimpleNamespace(authorization=driver.AUTH, max_seconds=driver.MAX_SECONDS))
            imports.assert_not_called()
            allocation.assert_not_called()

    def test_24_hour_allocation_guard_and_remaining_time_closure_reserve(self):
        environment = {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "1",
                       "SLURM_MEM_PER_NODE": "7168", "SLURM_JOB_END_TIME": "96400",
                       **{key: "1" for key in driver.prefix.THREADS}}
        line = ("JobId=123 JobState=RUNNING NumCPUs=1 NumNodes=1 NodeList=compute "
                "AllocTRES=cpu=1,mem=7G TimeLimit=1-00:00:00")
        with mock.patch.dict(driver.os.environ, environment, clear=True), \
             mock.patch.object(driver.socket, "gethostname", return_value="compute"), \
             mock.patch.object(driver.time, "time", return_value=10000), \
             mock.patch.object(driver.subprocess, "run", return_value=SimpleNamespace(stdout=line)) as command:
            allocation = driver.verify_allocation()
            self.assertEqual(allocation["time_limit_seconds"], 86400)
            self.assertEqual(driver.Budget(allocation, driver.MAX_SECONDS).seconds, 86100)
            shorter = dict(allocation, job_end_epoch=11200)
            self.assertEqual(driver.Budget(shorter, driver.MAX_SECONDS).seconds, 900)
            for changed, error in ((line.replace("1-00:00:00", "1-00:00:01"), "24_hour_maximum"),
                                   (line.replace("NumCPUs=1", "NumCPUs=2"), "single_CPU"),
                                   (line.replace("mem=7G", "mem=7G,gres/gpu=1"), "resource_mismatch")):
                command.return_value = SimpleNamespace(stdout=changed)
                with self.assertRaisesRegex(driver.Stop, error):
                    driver.verify_allocation()

    def test_exact_remaining_coverage_and_F_to_C_mapping(self):
        np = self.np
        intervals = list(driver.blocks())
        self.assertEqual(len(intervals), 783)
        self.assertEqual([s for begin, end in intervals for s in range(begin, end)], list(range(64, 50176)))
        self.assertTrue(all(end - begin == 64 for begin, end in intervals))
        plane = np.arange(224 * 224, dtype=np.float32).reshape(1, 224, 224)
        arranged = plane.transpose(0, 2, 1).reshape(1, 50176)
        for s in (64, 223, 224, 225, 50175):
            coordinate, full_f, full_c = driver.coordinates(s)
            self.assertEqual(coordinate, (0, s % 224, s // 224))
            self.assertEqual(full_f, 64 * s)
            self.assertEqual(arranged[0, s], full_c)
        with self.assertRaisesRegex(driver.Stop, "no_prefix_replay"):
            driver.coordinates(63)
        with self.assertRaisesRegex(driver.Stop, "no_prefix_replay"):
            list(driver.blocks(0, 64))

    def test_source_helper_dispatch_block_layout_and_partial_solve_preservation(self):
        np = self.np
        bank = np.arange(5 * 128, dtype=np.float32).reshape(5, 128)
        native = np.arange(501)
        prepared = {"branch": "source_greater_than_500_with_native_ID_ties",
                    "right": np.zeros((64, 502)), "correlations": np.zeros((64, 501)),
                    "means": np.arange(64, dtype=np.float32)[None, :],
                    "scales": np.ones((1, 64), dtype=np.float32)}
        def prepare(x, augmented, gram, targets, corrmat):
            self.assertTrue(targets.flags.c_contiguous)
            self.assertEqual(targets.dtype, np.dtype("float32"))
            np.testing.assert_array_equal(targets, bank[:, 64:128])
            return prepared
        def solve(gram, right, correlations, ids):
            self.assertIs(ids, native)
            return np.arange(500), np.ones(500, dtype=np.float32), np.float32(2)
        budget = self.budget()
        fragment = mock.Mock()
        # Start-of-block check plus three successful solve checks; fourth stops.
        with mock.patch.object(driver, "blocks", return_value=iter([(64, 128)])), \
             mock.patch.object(driver.prefix, "prepare_layer_math", side_effect=prepare) as math_call, \
             mock.patch.object(driver.prefix, "solve_selected", side_effect=solve) as solve_call, \
             mock.patch.object(budget, "check", side_effect=[None, None, None, None, driver.Stop("deadline")]):
            with self.assertRaisesRegex(driver.Stop, "deadline"):
                driver.fit_blocks(bank, None, None, None, native, None, fragment, budget)
        self.assertEqual(math_call.call_count, 1)
        self.assertEqual(solve_call.call_count, 3)
        start, rows = fragment.commit.call_args.args
        self.assertEqual(start, 64)
        self.assertEqual(len(rows), 3)
        self.assertEqual([float(row[3]) for row in rows], [0, 1, 2])

    def test_data_precedes_markers_and_committed_rows_cannot_replay(self):
        np = self.np
        row = (np.arange(500), np.ones(500, dtype=np.float32), 2., 3., 4., 0.1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fragment.h5"
            store = driver.Fragment(path, self.budget(), start=64, end=68)
            real_sync = driver.prefix.fsync_h5
            seen = []
            def sync(handle):
                seen.append(handle["state"][:].tolist())
                real_sync(handle)
            with mock.patch.object(driver.prefix, "fsync_h5", side_effect=sync):
                store.commit(64, [row, row])
            self.assertEqual(seen, [[0, 0, 0, 0], [2, 2, 0, 0]])
            self.assertEqual(store.committed_end, 66)
            with self.assertRaisesRegex(driver.Stop, "disjoint_ascending"):
                store.commit(64, [row])
            with mock.patch.object(driver.prefix, "fsync_h5", side_effect=OSError("data flush failure")):
                with self.assertRaisesRegex(OSError, "data flush failure"):
                    store.commit(66, [row])
            np.testing.assert_array_equal(store.file["state"][:], [2, 2, 0, 0])
            store.close()
            with self.assertRaises((FileExistsError, OSError)):
                driver.Fragment(path, self.budget(), start=64, end=68)
            uncertain = driver.Fragment(Path(temporary) / "uncertain.h5", self.budget(), start=64, end=68)
            calls = []
            def fail_marker_sync(handle):
                calls.append(True)
                if len(calls) == 2:
                    raise OSError("marker flush failure")
                real_sync(handle)
            with mock.patch.object(driver.prefix, "fsync_h5", side_effect=fail_marker_sync):
                with self.assertRaisesRegex(OSError, "marker flush failure"):
                    uncertain.commit(64, [row])
            self.assertEqual(uncertain.committed_end, 64)
            self.assertEqual(uncertain.uncertain_publication,
                             {"spatial_Fortran_interval": [64, 65],
                              "phase": "completion_marker_fsync", "error_type": "OSError"})
            # Preserve the potentially published marker; it cannot safely be
            # advertised as remaining unfitted work or silently cleared.
            self.assertEqual(int(uncertain.file["state"][0]), 2)
            uncertain.close()


if __name__ == "__main__":
    unittest.main()
