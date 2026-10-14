"""One fresh-attempt/device wiring check; no empirical inputs or CUDA kernels."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_content_cuda_l40s_measurement as entry
import test_content_cuda_admission as old


class L40SAdmissionTests(unittest.TestCase):
    def test_fresh_stage_device_refusal_and_complete_measurement_preserve_old_attempts(self):
        original = entry.original
        with old.fixture("cache-002") as f:
            root = f.run.parent
            run, proposal = root / "l40s", root / "l40s.json"
            run.mkdir()
            (run / "allocation.txt").write_text(old.allocation())
            scratch = f.scratch.parent / entry.SCRATCH_NAME
            scratch.mkdir()
            preserved = [(f.proposal, f.proposal.read_bytes()),
                         (f.run / "allocation.txt", (f.run / "allocation.txt").read_bytes())]
            preserved.extend((path, content) for _, path, content, _ in f.histories)
            preserved.extend((stage.profile_path, content) for stage, _, _, content in f.histories)
            with mock.patch.object(entry, "RUN_DIR", run), mock.patch.object(entry, "PROFILE", proposal), \
                    mock.patch.dict(os.environ, {"TMPDIR": str(scratch)}), \
                    mock.patch.object(f.torch.cuda, "get_device_properties", return_value=SimpleNamespace(
                        total_memory=48 * 1024**3, name=entry.DEVICE_NAME)) as properties, \
                    mock.patch.object(original, "_load_operations") as operations:
                selected = entry.stage()
                self.assertTrue(selected.cached)
                self.assertEqual(selected.fixed["feature_read_mode"], "bounded_channel_two_pass_cache")
                self.assertNotIn(selected.name, original.STAGES)
                self.assertTrue(all(selected.run_dir != original._stage(name).run_dir for name in original.STAGES))
                valid = {**selected.fixed, "authorized": True, "approval_source": "fixture_only",
                         "resources": dict(original.RESOURCES), "input_scope": dict(original.INPUT_SCOPE)}
                proposal.write_text(json.dumps({**valid, "authorized": False}))
                with mock.patch.object(original, "_allocation") as allocation:
                    with self.assertRaisesRegex(original.AdmissionError, "profile_not_activated"):
                        entry.admit()
                    allocation.assert_not_called()
                operations.assert_not_called()
                proposal.write_text(json.dumps(valid))
                properties.return_value = SimpleNamespace(total_memory=48 * 1024**3, name="unselected GPU")
                with self.assertRaisesRegex(original.AdmissionError, "selected_L40S_device_required"):
                    entry.admit()
                operations.assert_not_called()
                properties.return_value = SimpleNamespace(total_memory=48 * 1024**3, name=entry.DEVICE_NAME)
                context, budget = entry.admit()
                self.assertEqual(context.allocation["deadline_epoch"], 1_999_999_880 + 6300)
                options = old.options(run, "cache-002")
                arguments = dict(source=original.SOURCE, target=original.TARGET, budget=budget)
                entry.resolver(context, operation=selected.fixed["operation"],
                               execution_options=options, **arguments)
                with self.assertRaisesRegex(original.AdmissionError, "fixed_CUDA_measurement_options_required"):
                    entry.resolver(context, operation=selected.fixed["operation"],
                                   execution_options={**options, "updates": 3}, **arguments)
                with self.assertRaisesRegex(original.AdmissionError, "operation_not_authorized"):
                    entry.resolver(context, operation="train_and_publish_content_pair",
                                   execution_options=options, **arguments)
                reserved_partial = run / "reserved_final_model.pt.partial"
                reserved_partial.write_text("preserve prior attempt")
                with self.assertRaisesRegex(original.AdmissionError, "existing_attempt_no_retry"):
                    entry.run_once()
                self.assertEqual(reserved_partial.read_text(), "preserve prior attempt")
                self.assertFalse((run / "claim.json").exists())
                operations.assert_not_called()
                reserved_partial.unlink()
                bound = object()

                def bind(actual_context, *, require_admitted_context, source, target, budget):
                    require_admitted_context(actual_context, operation="bind_content_inputs",
                        source=source, target=target, budget=budget, execution_options=None)
                    return bound

                def measure(actual_context, *, require_admitted_context, prepare_inputs, budget, reserved_final_path):
                    self.assertEqual(reserved_final_path, run / "reserved_final_model.pt")
                    require_admitted_context(actual_context, operation=selected.fixed["operation"],
                        source=original.SOURCE, target=original.TARGET, budget=budget,
                        execution_options=old.options(run, "cache-002"))
                    self.assertIs(prepare_inputs(), bound)
                    return dict(status="complete_measurement", completed_updates=4,
                        model_published=False, production_fit_completed=False)

                operations.return_value = (bind, measure)
                self.assertEqual(entry.run_once(), 0)
                report = json.loads((run / "report.json").read_text())
                self.assertEqual(report["stage"], "l40s-001")
                self.assertEqual(report["completed_updates"], 4)
                self.assertFalse(report["model_published"])
                self.assertFalse((run / "reserved_final_model.pt").exists())
                with self.assertRaisesRegex(original.AdmissionError, "existing_attempt_no_retry"):
                    entry.run_once()
                operations.assert_called_once_with(selected)
                for path, content in preserved:
                    self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
