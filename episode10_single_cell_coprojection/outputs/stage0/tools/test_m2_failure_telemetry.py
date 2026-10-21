"""Generated scalar failures only; no optimizer, recovery campaign or real data."""
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify_v3_m2_optional as qualification


FAMILY = "low_rank_population_recovery"
SOURCE = '''
from types import SimpleNamespace
import numpy as np
LOW_RANK_GRADIENT_INFINITY_TOLERANCE = 1e-5
LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE = 1e-8
ERROR = RuntimeError("PRIVATE_EXCEPTION_TEXT_MUST_NOT_ENTER_JSON")
MODE = "synthetic_qualification"
def _fit_low_rank_impl():
    objective_mode = MODE
    specification = SimpleNamespace(rank=4, rho=.1, strength=.2)
    current = SimpleNamespace(objective=1.2, smooth_objective=1., penalty_value=.2,
        gradient_infinity=.004, gradient_U=np.array([[.002, -.003]]),
        gradient_d=np.array([.004]))
    accepted_evaluation = current
    state = object()
    accepted_state = state
    last_relative_change = 0.
    iteration = 3
    maximum_iterations = 5
    step = .5
    total_line_trials = 8
    directional_derivative = -.01
    production_data = np.array([987654321.])
    raise ERROR
def worker(family, rank):
    return _fit_low_rank_impl()
'''


def load_source(path):
    spec = importlib.util.spec_from_file_location("generated_optimizer_frame", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_failure(source_path, failure_path):
    """Spawn-safe harness exercising the integrated qualification child wrapper."""
    source = load_source(source_path)
    with patch.object(qualification, "_run_recovery_case", source.worker), \
            patch.object(qualification.optional, "__file__", source_path):
        return qualification._run_recovery_case_with_failure_record(FAMILY, 4, Path(failure_path))


class FailureTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ep10_telemetry_fixture_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source_path = self.root / "generated_optimizer.py"
        self.source_path.write_text(SOURCE, encoding="utf-8")
        self.source = load_source(self.source_path)
        self.failure = self.root / "failure.json"

    def test_success_object_and_original_exception_are_preserved(self):
        sentinel = object()
        with patch.object(qualification, "_run_recovery_case", return_value=sentinel):
            returned = qualification._run_recovery_case_with_failure_record(FAMILY, 4, self.failure)
        self.assertIs(returned, sentinel)
        self.assertFalse(self.failure.exists())
        with patch.object(qualification, "_run_recovery_case", self.source.worker), \
                patch.object(qualification.optional, "__file__", str(self.source_path)):
            with self.assertRaises(RuntimeError) as caught:
                qualification._run_recovery_case_with_failure_record(FAMILY, 4, self.failure)
        self.assertIs(caught.exception, self.source.ERROR)
        packet = json.loads(self.failure.read_text())
        self.assertTrue(packet["low_rank_frame_found"])
        self.assertEqual(packet["active_rho"], .1)
        self.assertEqual(packet["gradient_U_infinity"], .003)
        self.assertEqual(packet["gradient_d_infinity"], .004)
        self.assertFalse(packet["gradient_gate"])
        self.assertTrue(packet["relative_objective_gate"])
        self.assertTrue(packet["current_is_accepted_trial"])
        self.assertNotIn("PRIVATE_EXCEPTION", self.failure.read_text())
        self.assertNotIn("987654321", self.failure.read_text())

    def test_spawn_child_records_before_exception_transport(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            future = pool.submit(child_failure, str(self.source_path), str(self.failure))
            with self.assertRaisesRegex(RuntimeError, "PRIVATE_EXCEPTION_TEXT"):
                future.result(timeout=30)
        packet = json.loads(self.failure.read_text())
        self.assertTrue(packet["low_rank_frame_found"])
        self.assertEqual(packet["iteration"], 3)
        self.assertEqual(packet["total_line_trials"], 8)
        self.assertFalse(packet["qualification_pass"])
        self.assertNotIn("PRIVATE_EXCEPTION", self.failure.read_text())

    def test_production_frames_never_emit_diagnostics_or_swallow_failure(self):
        self.source.MODE = "production"
        with patch.object(qualification, "_run_recovery_case", self.source.worker), \
                patch.object(qualification.optional, "__file__", str(self.source_path)):
            with self.assertRaises(RuntimeError) as caught:
                qualification._run_recovery_case_with_failure_record(FAMILY, 4, self.failure)
        self.assertIs(caught.exception, self.source.ERROR)
        self.assertFalse(self.failure.exists())
        self.assertEqual(caught.exception.__notes__,
            ["Failure telemetry could not be preserved; diagnostic type=ValueError"])

    def test_existing_failure_record_is_not_overwritten(self):
        self.failure.write_text("preserved historical failure", encoding="utf-8")
        with patch.object(qualification, "_run_recovery_case") as worker:
            with self.assertRaisesRegex(ValueError, "fresh failure path"):
                qualification._run_recovery_case_with_failure_record(FAMILY, 4, self.failure)
            worker.assert_not_called()
        self.assertEqual(self.failure.read_text(), "preserved historical failure")


if __name__ == "__main__":
    unittest.main()
