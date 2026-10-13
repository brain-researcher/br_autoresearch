"""One bounded check of the explicit no-execution full-plan wrapper branch."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cpe_array import NeurocamCPEArray
from development_worlds import _Grid, _build
from streaming_array_acquisition import StreamingArrayAcquisition
from streaming_development_simulation import run_streaming_development_simulation
from test_development_simulation import LIMIT
from test_streaming_development_simulation import cases


class ConditionalPreflightOnlyTests(unittest.TestCase):
    def test_complete_original_plan_has_same_preflight_without_writer_or_advance(self):
        # Reuse the established two-window orchestration fixture, including
        # the aperture crossing its first storage seam. No saved world reads.
        world = _build("W1", 2001, _Grid(), check=lambda: None,
                       max_elements=LIMIT, include_scales=False)
        with tempfile.TemporaryDirectory() as directory:
            eager, arguments, _, _, _ = cases(world, Path(directory) / "default.h5")
            arguments["physical_actions"] = lambda: iter(eager["physical_actions"])
            ordinary = run_streaming_development_simulation(**arguments)
            self.assertEqual(ordinary["status"], "complete", ordinary.get("failure"))
            self.assertTrue(ordinary["full_record_completed"])
            self.assertFalse(ordinary["preflight_only"])
            expected_path = Path(directory) / "must_not_exist.h5"
            phases = []
            with patch("streaming_development_simulation.StreamingSimulationOutput",
                       side_effect=AssertionError("preflight must not create a writer")) as writer, \
                 patch.object(StreamingArrayAcquisition, "run",
                       side_effect=AssertionError("preflight must not execute the record")) as run, \
                 patch.object(NeurocamCPEArray, "advance_row",
                       side_effect=AssertionError("preflight must not advance analog state")) as advance:
                actual = run_streaming_development_simulation(**dict(arguments,
                    result_path=expected_path, preflight_only=True), on_phase=phases.append)
            writer.assert_not_called()
            run.assert_not_called()
            advance.assert_not_called()
            self.assertFalse(expected_path.exists())
            self.assertEqual(actual["status"], "preflight_complete", actual.get("failure"))
            self.assertTrue(actual["preflight_completed"])
            self.assertTrue(actual["preflight_only"])
            self.assertFalse(actual["full_record_completed"])
            self.assertFalse(actual["electrical_state_advanced"])
            self.assertFalse(actual["nuisance_applied"])
            self.assertEqual(actual["confirmed_physical_time"], 0.)
            self.assertEqual(actual["accounting"]["completed_electrical_intervals"], 0)
            self.assertIsNone(actual["output"])
            self.assertIsNone(actual["readout"])
            self.assertEqual(actual["preflight"], ordinary["preflight"])
            self.assertEqual(actual["preflight"]["windows"], 2)
            self.assertEqual(actual["hold"], ordinary["hold"])
            self.assertTrue(actual["physical_action_factory_replayed"])
            self.assertNotIn("exclusive output initialization", phases)
            self.assertNotIn("persistent electrical/readout execution", phases)
            self.assertEqual(phases[-1],
                "whole-plan preflight complete; electrical execution not selected")


if __name__ == "__main__":
    unittest.main()
