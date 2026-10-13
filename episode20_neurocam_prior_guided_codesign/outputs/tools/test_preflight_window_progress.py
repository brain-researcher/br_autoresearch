"""Two focused checks for completed-window progress; source-prepared, initially UNRUN.

Reuse the existing finite seam fixture. No electrical execution, production
recipe, saved-world payload or new scientific scenario is required.
"""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

import streaming_array_acquisition as acquisition
from streaming_acquisition_readout import StreamingAcquisitionReadout
from test_streaming_array_acquisition import make_fixture, runner_arguments


FIRST = dict(completed_windows=1, physical_planning_frontier_seconds=.5,
    physical_input_events=1, physical_fragments=1, split_intervals=2,
    window_digest_bytes=32)
SECOND = dict(completed_windows=2, physical_planning_frontier_seconds=1.,
    physical_input_events=2, physical_fragments=3, split_intervals=7,
    window_digest_bytes=64)


class PreflightWindowProgressTests(unittest.TestCase):
    def test_exact_completed_prefix_is_owned_and_never_advances_the_array(self):
        circuit, fields, windows, bindings = make_fixture()
        before = circuit.volts, circuit.previous_field_volts
        snapshots, closed, digested, emissions = [], [], [], []
        original_close = StreamingAcquisitionReadout.close_window
        original_digest = acquisition._window_digest

        def close(readout, index):
            result = original_close(readout, index)
            closed.append(index)
            return result

        def digest(window, field_digests):
            result = original_digest(window, field_digests)
            digested.append(window.index)
            return result

        def progress(snapshot):
            count = len(snapshots) + 1
            self.assertEqual(closed, list(range(count)))
            self.assertEqual(digested, list(range(count)))
            self.assertEqual(set(snapshot), set(FIRST))
            for name, value in snapshot.items():
                self.assertIs(type(value), float if name == "physical_planning_frontier_seconds" else int)
            snapshots.append(dict(snapshot))
            # The observer owns this dict; it is not a cursor or report alias.
            snapshot.clear()

        args = runner_arguments(circuit, fields, windows, bindings,
            raw_sink=emissions.append, observation_sink=emissions.append,
            interval_sink=emissions.append)
        with patch.object(circuit, "advance_row", side_effect=AssertionError("preflight advanced")) as advance, \
                patch.object(StreamingAcquisitionReadout, "close_window", close), \
                patch.object(acquisition, "_window_digest", side_effect=digest):
            runner = acquisition.StreamingArrayAcquisition(**args, on_preflight_progress=progress)
        self.assertEqual(snapshots, [FIRST, SECOND])
        self.assertEqual(runner.status, "ready")
        self.assertEqual(runner.physical_time, 0.)
        self.assertEqual(runner.preflight_summary["windows"], 2)
        self.assertEqual(runner.preflight_summary["split_intervals"], 7)
        self.assertEqual(runner.preflight_summary["window_digest_bytes"], 64)
        self.assertEqual((circuit.volts, circuit.previous_field_volts), before)
        self.assertFalse(emissions)
        advance.assert_not_called()

        # Omission of the optional observer preserves the original result.
        ordinary, fields, windows, bindings = make_fixture()
        with patch.object(ordinary, "advance_row", side_effect=AssertionError("preflight advanced")):
            default = acquisition.StreamingArrayAcquisition(
                **runner_arguments(ordinary, fields, windows, bindings))
        self.assertEqual(default.preflight_summary, runner.preflight_summary)

    def test_refusals_retain_only_completed_windows_and_never_imply_finalization(self):
        # The first digest refuses after window close: no event may escape.
        # A finalization refusal follows both valid windows: even that complete
        # planning frontier must not cause the constructor to return a runner.
        for failure in ("first_digest", "late_field", "late_close",
                        "late_digest", "finalize", "callback"):
            with self.subTest(failure=failure):
                circuit, fields, windows, bindings = make_fixture()
                before = circuit.volts, circuit.previous_field_volts
                snapshots, emissions = [], []
                latest = None
                args = runner_arguments(circuit, fields, windows, bindings,
                    raw_sink=emissions.append, observation_sink=emissions.append,
                    interval_sink=emissions.append)
                original_close = StreamingAcquisitionReadout.close_window
                original_digest = acquisition._window_digest

                def progress(snapshot):
                    nonlocal latest
                    latest = dict(snapshot)
                    snapshots.append(latest)
                    if failure == "callback":
                        raise TimeoutError("observer budget refusal")

                def digest(window, field_digests):
                    target = 0 if failure == "first_digest" else 1
                    if failure in ("first_digest", "late_digest") and window.index == target:
                        raise TimeoutError("digest budget refusal")
                    return original_digest(window, field_digests)

                def close(readout, index):
                    if failure == "late_close" and index == 1:
                        raise TimeoutError("window close budget refusal")
                    return original_close(readout, index)

                if failure == "late_field":
                    args["field_provider"] = lambda key: (
                        fields[key] if key == 0 else np.full((64, 64), np.nan))
                expected_error = ValueError if failure == "late_field" else TimeoutError
                returned = None
                with ExitStack() as stack:
                    advance = stack.enter_context(patch.object(circuit, "advance_row",
                        side_effect=AssertionError("preflight advanced")))
                    stack.enter_context(patch.object(acquisition, "_window_digest", side_effect=digest))
                    stack.enter_context(patch.object(StreamingAcquisitionReadout, "close_window", close))
                    if failure == "finalize":
                        stack.enter_context(patch.object(StreamingAcquisitionReadout, "finalize",
                            side_effect=TimeoutError("final publication budget refusal")))
                    with self.assertRaises(expected_error):
                        returned = acquisition.StreamingArrayAcquisition(
                            **args, on_preflight_progress=progress)
                expected = [] if failure == "first_digest" else (
                    [FIRST, SECOND] if failure == "finalize" else [FIRST])
                self.assertEqual(snapshots, expected)
                self.assertEqual(latest, expected[-1] if expected else None)
                self.assertIsNone(returned)
                self.assertEqual((circuit.volts, circuit.previous_field_volts), before)
                self.assertFalse(emissions)
                advance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
