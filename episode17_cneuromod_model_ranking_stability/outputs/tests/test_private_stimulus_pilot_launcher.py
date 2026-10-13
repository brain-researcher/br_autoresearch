"""Three focused launcher checks using invented notices and mocked pilot work.

Coordinator compute only. No provider file, image, model, neural array or real
credential is opened. Mocked exec never replaces the test process. Real pipes
exercise descriptor transfer and closure without writing a credential file.
"""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import errno
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_private_stimulus_feature_pilot as launcher


def fake_pilot(selection=None, run=None):
    private = mock.Mock()
    private.exists.return_value = False
    private.is_symlink.return_value = False
    private.parent.exists.return_value = False
    report = mock.Mock()
    report.parent.exists.return_value = False
    return SimpleNamespace(
        archive_metadata=SimpleNamespace(ACQUISITION=Path("/ep17-invented-acquisition")),
        PRIVATE_OUTPUT=private,
        REPORT_PATH=report,
        RANDOM_CHECKPOINT=object(),
        pilot_paths=mock.Mock(return_value=(private, report)),
        _selection=selection or mock.Mock(),
        run_private_development_feature_pilot=run or mock.Mock(),
    )


class PrivateStimulusPilotLauncherTests(unittest.TestCase):
    def assert_closed(self, descriptor):
        with self.assertRaises(OSError) as caught:
            os.fstat(descriptor)
        self.assertEqual(caught.exception.errno, errno.EBADF)

    def test_selection_notice_and_secret_only_inherited_pipe_precede_exec(self):
        token = b"invented-EP17-token-17"
        notice = b"Invented research notice.\nPassword for this fixture: " + token + b"\n"
        events, inherited, inspected = [], [], []
        pilot = fake_pilot(selection=lambda: events.append("selection"))

        def open_notice(path, mode):
            self.assertEqual(path, pilot.archive_metadata.ACQUISITION / "terms/password_images.txt")
            self.assertEqual(mode, "rb")
            self.assertEqual(events, ["selection"])
            events.append("notice")
            return io.BytesIO(notice)

        def inspect_exec(executable, arguments):
            self.assertEqual(events, ["selection", "notice"])
            events.append("exec")
            self.assertEqual(executable, sys.executable)
            self.assertEqual(arguments[:6], [sys.executable, "-B",
                str(Path(launcher.__file__).resolve()), "--attempt", "pilot002", launcher.CHILD_OPTION])
            self.assertEqual(len(arguments), 7)
            self.assertNotIn(token.decode(), " ".join(arguments))
            self.assertNotIn(token.decode(), " ".join(os.environ.values()))
            descriptor = int(arguments[6])
            inherited.append(descriptor)
            self.assertGreaterEqual(descriptor, 3)
            self.assertTrue(os.get_inheritable(descriptor))
            self.assertEqual(os.read(descriptor, launcher.NOTICE_BYTES + 1), token)
            self.assertEqual(os.read(descriptor, 1), b"")  # Writer closed before exec.
            inspected.append(True)
            raise OSError("invented private exception " + token.decode())

        output, errors = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"private_stimulus_feature_pilot": pilot}))
            stack.enter_context(mock.patch.dict(os.environ, {"SLURM_JOB_ID": "invented"}, clear=True))
            stack.enter_context(mock.patch.object(launcher, "_require_compute"))
            stack.enter_context(mock.patch.object(Path, "open", autospec=True, side_effect=open_notice))
            execute = stack.enter_context(mock.patch.object(os, "execv", side_effect=inspect_exec))
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(redirect_stderr(errors))
            self.assertEqual(launcher.main(["--attempt", "pilot002"]), 1)
        self.assertEqual(events, ["selection", "notice", "exec"])
        execute.assert_called_once()
        self.assertEqual(json.loads(output.getvalue()), {"status": "stopped", "phase": "private_delivery"})
        self.assertEqual(errors.getvalue(), "")
        self.assertNotIn(token.decode(), output.getvalue())
        self.assertEqual(len(inherited), 1)
        self.assertEqual(inspected, [True])  # main must not swallow a failed callback assertion.
        self.assert_closed(inherited[0])
        pilot.run_private_development_feature_pilot.assert_not_called()
        pilot.pilot_paths.assert_called_once_with("pilot002")

    def test_ambiguous_or_failed_notice_refuses_without_exec_or_private_output(self):
        token = b"invented-format-token"
        for value in (token, token + b"\r\n", b"Password for fixture: " + token + b"\n",
                      b"archive password = " + token + b"\n"):
            self.assertEqual(launcher.password_from_notice(value), token)
        self.assertEqual(launcher.password_from_notice(
            b"Password use requires the invented agreement.\n\nPassword for fixture.zip, another.zip,\n"
            b"last_fixture.zip: " + token + b"\n"), token)
        invalid = (b"", b"No explicit field here.", b"Password: first\nPassword: second\n",
                   b"Password: \"quoted\"\n", b"Password: two words\n", b"\xff",
                   b"Password for first archive\none\nPassword for second archive\ntwo\n",
                   b"Password: first\nPassword for archive\nsecond phrase\n",
                   b"Password for archive\n\"quoted phrase\"\n",
                   b"Password for archive\nfirst line\nsecond line\n",
                   b"x" * (launcher.NOTICE_BYTES + 1))
        for value in invalid:
            with self.subTest(size=len(value)):
                with self.assertRaises(launcher.DeliveryRefusal):
                    launcher.password_from_notice(value)

        for failure in ("selection", "missing", "ambiguous", "existing_attempt"):
            with self.subTest(failure=failure):
                selection = mock.Mock(side_effect=RuntimeError("invented private detail")
                                      if failure == "selection" else None)
                pilot = fake_pilot(selection=selection)
                pilot.PRIVATE_OUTPUT.parent.exists.return_value = failure == "existing_attempt"
                output, errors = io.StringIO(), io.StringIO()
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.dict(sys.modules, {"private_stimulus_feature_pilot": pilot}))
                    stack.enter_context(mock.patch.object(launcher, "_require_compute"))
                    opened = stack.enter_context(mock.patch.object(Path, "open", autospec=True))
                    if failure == "missing":
                        opened.side_effect = FileNotFoundError("invented private detail")
                    else:
                        opened.return_value = io.BytesIO(b"Password: one\nPassword: two\n")
                    execute = stack.enter_context(mock.patch.object(os, "execv"))
                    pipe = stack.enter_context(mock.patch.object(os, "pipe"))
                    stack.enter_context(redirect_stdout(output))
                    stack.enter_context(redirect_stderr(errors))
                    self.assertEqual(launcher.main([]), 1)
                selection.assert_called_once()
                if failure in {"selection", "existing_attempt"}:
                    opened.assert_not_called()
                else:
                    opened.assert_called_once_with(
                        pilot.archive_metadata.ACQUISITION / "terms/password_images.txt", "rb")
                execute.assert_not_called()
                pipe.assert_not_called()
                pilot.run_private_development_feature_pilot.assert_not_called()
                self.assertEqual(json.loads(output.getvalue()), {"status": "stopped", "phase": "private_delivery"})
                self.assertEqual(errors.getvalue(), "")
                self.assertNotIn("invented private detail", output.getvalue())

    def test_pilot_requires_both_completion_fields_and_always_closes_descriptor(self):
        cases = (
            ({"status": "completed_private_staging", "pilot_feature_extraction_complete": True}, 0),
            ({"status": "completed_private_staging", "pilot_feature_extraction_complete": False}, 1),
            ({"status": "stopped", "pilot_feature_extraction_complete": True}, 1),
            (RuntimeError("invented private pilot exception"), 1),
        )
        for result, expected in cases:
            with self.subTest(expected=expected, raises=isinstance(result, Exception)):
                read_fd, write_fd = os.pipe()
                os.close(write_fd)
                run = mock.Mock(side_effect=result) if isinstance(result, Exception) else mock.Mock(return_value=result)
                pilot = fake_pilot(run=run)
                output, errors = io.StringIO(), io.StringIO()
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.dict(sys.modules, {"private_stimulus_feature_pilot": pilot}))
                    stack.enter_context(mock.patch.object(launcher, "_require_compute"))
                    stack.enter_context(redirect_stdout(output))
                    stack.enter_context(redirect_stderr(errors))
                    self.assertEqual(launcher.main(["--attempt", "pilot002", launcher.CHILD_OPTION, str(read_fd)]), expected)
                run.assert_called_once_with(password_fd=read_fd,
                    private_output_dir=pilot.PRIVATE_OUTPUT,
                    operational_report_path=pilot.REPORT_PATH,
                    random_checkpoint_path=pilot.RANDOM_CHECKPOINT, attempt="pilot002")
                self.assert_closed(read_fd)
                self.assertEqual(json.loads(output.getvalue()), {
                    "status": "completed" if expected == 0 else "stopped", "phase": "fixed_pilot"})
                self.assertEqual(errors.getvalue(), "")
                self.assertNotIn("invented private", output.getvalue())
                pilot._selection.assert_not_called()

        # Actual callable guards must preserve prior attempts before any output
        # creation, credential read or runtime/model import. No real file opens.
        import private_stimulus_feature_pilot as actual_pilot
        selected_private, selected_report = actual_pilot.pilot_paths("pilot002")
        self.assertEqual(selected_private.parent.name, "pilot002")
        self.assertEqual(selected_report.parent.name, "pilot002")
        self.assertEqual(actual_pilot.pilot_paths(), (actual_pilot.PRIVATE_OUTPUT, actual_pilot.REPORT_PATH))
        with mock.patch.object(Path, "exists", return_value=True), \
                mock.patch.object(actual_pilot, "_mkdir_private") as mkdir:
            with self.assertRaisesRegex(actual_pilot.PilotRefusal, "attempt_already_exists"):
                actual_pilot.run_private_development_feature_pilot(password_fd=99,
                    private_output_dir=selected_private, operational_report_path=selected_report,
                    random_checkpoint_path=actual_pilot.RANDOM_CHECKPOINT, attempt="pilot002")
            with self.assertRaisesRegex(actual_pilot.PilotRefusal, "fixed_pilot_paths_required"):
                actual_pilot.run_private_development_feature_pilot(password_fd=99,
                    private_output_dir=actual_pilot.PRIVATE_OUTPUT, operational_report_path=selected_report,
                    random_checkpoint_path=actual_pilot.RANDOM_CHECKPOINT, attempt="pilot002")
            mkdir.assert_not_called()
        for invalid in ("../pilot001", "pilot002/extra", "pilot2", "full002"):
            with self.assertRaises(actual_pilot.PilotRefusal):
                actual_pilot.pilot_paths(invalid)


if __name__ == "__main__":
    unittest.main()
