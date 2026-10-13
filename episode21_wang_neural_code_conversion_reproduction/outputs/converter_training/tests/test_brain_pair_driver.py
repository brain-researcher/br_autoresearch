"""Three composition checks; run only in coordinated Slurm compute.

Reuse the tiny completed-fit artifact fixture and mock only numerical fitting.
No CV rerun, empirical data, scheduler or execution authority is exercised.
"""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import brain_loss as brain
import brain_pair_driver as driver
import empirical_brain_inputs as inputs
import final_brain_artifact as final
import test_final_brain_artifact as artifact_fixtures


class BrainPairDriverTests(unittest.TestCase):
    def fixture(self, resolver=None):
        helper = artifact_fixtures.FinalBrainArtifactTests()
        fit, budget = helper.fixture(), helper.budget()
        responses = []
        for person, axis in ((fit.summary.source, fit.source_native_voxel_ids),
                             (fit.summary.target, fit.target_native_voxel_ids)):
            values = np.zeros((897, len(axis)), dtype=np.float32)
            values.setflags(write=False)
            responses.append(brain.TauTrainResponses(person, fit.image_ids, axis, values))
        identity = budget.allocation["job_id"], budget.allocation["hostname"]
        bound = inputs.BoundBrainInputs(object(), resolver or (lambda *args, **kw: None), budget,
            identity, brain.make_common_folds(fit.image_ids), *responses, (3,) * 897, (4,) * 897)
        return bound, fit

    def test_exact_pair_inputs_and_separate_operation_join_final_publisher(self):
        events = []
        def resolver(context, **kwargs):
            self.assertEqual(kwargs["operation"], "train_and_publish_brain_pair")
            self.assertEqual((kwargs["source"], kwargs["target"]), brain.PAIR_ORDER[5])
            self.assertEqual(dict(kwargs["execution_options"]), {"final_path": path, "maximum_bytes": 100_000})
            events.append("admit")
        bound, fit = self.fixture(resolver)
        def fitted(source, target, folds, **kwargs):
            self.assertIs(source, bound.source)
            self.assertIs(target, bound.target)
            self.assertIs(folds, bound.folds)
            self.assertEqual(kwargs["source_native_ids"], fit.source_native_voxel_ids)
            self.assertEqual(kwargs["target_native_ids"], fit.target_native_voxel_ids)
            self.assertIs(kwargs["check"], bound.budget.check)
            self.assertFalse(path.exists())
            events.append("fit")
            return fit
        actual_prepare, actual_write = final.prepare_final_brain_artifact, final.write_final_brain_artifact
        def prepare(*args, **kwargs):
            self.assertEqual(kwargs["expected_pair"], brain.PAIR_ORDER[5])
            self.assertEqual(kwargs["authenticated_tau_train_image_ids"], fit.image_ids)
            self.assertEqual(kwargs["source_native_ids"], fit.source_native_voxel_ids)
            self.assertEqual(kwargs["target_native_ids"], fit.target_native_voxel_ids)
            events.append("prepare")
            return actual_prepare(*args, **kwargs)
        def write(*args, **kwargs):
            self.assertEqual(events[-1], "admit")
            events.append("publish")
            return actual_write(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.pt"
            with mock.patch.object(brain, "fit_brain_pair", side_effect=fitted) as fitting, \
                    mock.patch.object(final, "prepare_final_brain_artifact", side_effect=prepare), \
                    mock.patch.object(final, "write_final_brain_artifact", side_effect=write):
                result = driver.train_and_publish_brain_pair(bound, final_path=path, maximum_bytes=100_000)
            fitting.assert_called_once()
            self.assertEqual(events, ["admit", "fit", "admit", "prepare", "admit", "publish"])
            self.assertEqual(result.path, path)
            self.assertEqual(result.pair_index, 5)
            self.assertEqual(result.summary, fit.summary)
            restored = final.load_final_brain_artifact(path, budget=bound.budget, maximum_bytes=100_000)
            np.testing.assert_array_equal(restored["matrix"].numpy(), fit.matrix)

    def test_bind_only_wrong_allocation_or_existing_output_refuses_before_fit(self):
        for defect in ("bind_only", "allocation", "final", "partial", "relative_path", "missing_parent", "byte_cap"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "final.pt"
                def resolver(context, **kwargs):
                    if defect == "bind_only" and kwargs["operation"] != "bind_brain_inputs":
                        raise PermissionError("bind-only context")
                bound, fit = self.fixture(resolver)
                if defect == "allocation":
                    bound = replace(bound, allocation_identity=("different-job", "different-host"))
                elif defect in ("final", "partial"):
                    (path if defect == "final" else path.with_name("final.pt.partial")).write_bytes(b"preserved")
                selected_path = Path("relative.pt") if defect == "relative_path" else path
                if defect == "missing_parent":
                    selected_path = path.parent / "absent" / "final.pt"
                cap = 0 if defect == "byte_cap" else 100_000
                expected = PermissionError if defect == "bind_only" else FileExistsError if defect in ("final", "partial") else ValueError
                with mock.patch.object(brain, "fit_brain_pair") as fitting, \
                        mock.patch.object(final, "write_final_brain_artifact") as publication:
                    with self.assertRaises(expected):
                        driver.train_and_publish_brain_pair(bound, final_path=selected_path, maximum_bytes=cap)
                fitting.assert_not_called()
                publication.assert_not_called()
                if defect in ("final", "partial"):
                    self.assertEqual((path if defect == "final" else path.with_name("final.pt.partial")).read_bytes(), b"preserved")

    def test_fit_incomplete_cancellation_or_publication_error_propagates_without_retry(self):
        for defect in ("numerical", "deadline", "incomplete", "wrong_pair", "revoked", "write"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                admissions = []
                def resolver(context, **kwargs):
                    admissions.append(True)
                    if defect == "revoked" and len(admissions) == 3:
                        raise PermissionError("operation closed before publication")
                bound, fit = self.fixture(resolver)
                def fitted(*args, **kwargs):
                    if defect == "numerical":
                        raise brain.BrainLossNotEvaluable("required CV cell unavailable")
                    if defect == "deadline":
                        raise TimeoutError("deadline")
                    if defect == "incomplete":
                        return replace(fit, summary=replace(fit.summary, completed_fits=50))
                    if defect == "wrong_pair":
                        return replace(fit, summary=replace(fit.summary, source=fit.summary.target,
                                                           target=fit.summary.source))
                    return fit
                expected = {"numerical": brain.BrainLossNotEvaluable, "deadline": TimeoutError,
                            "incomplete": ValueError, "wrong_pair": ValueError,
                            "revoked": PermissionError, "write": OSError}[defect]
                path = Path(temporary) / "final.pt"
                with mock.patch.object(brain, "fit_brain_pair", side_effect=fitted) as fitting, \
                        mock.patch.object(final, "write_final_brain_artifact", side_effect=OSError("publication failed")) as publication:
                    with self.assertRaises(expected):
                        driver.train_and_publish_brain_pair(bound, final_path=path, maximum_bytes=100_000)
                fitting.assert_called_once()
                self.assertEqual(publication.call_count, 1 if defect == "write" else 0)
                self.assertFalse(path.exists())
                self.assertFalse(path.with_name("final.pt.partial").exists())


if __name__ == "__main__":
    unittest.main()
