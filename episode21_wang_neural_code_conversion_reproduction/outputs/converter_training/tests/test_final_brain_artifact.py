"""Three publication checks; run only in coordinated Slurm compute.

Fabricated completed fits use all 897 image identities and tiny complete native
axes. No fitting, empirical payload, tau values, decoder or scheduler is read.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import socket
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import brain_loss as brain
import final_brain_artifact as final


class FinalBrainArtifactTests(unittest.TestCase):
    def fixture(self):
        ids = tuple(f"image-{index:04d}" for index in range(897))
        scores = np.full((len(brain.ALPHAS), 5), 2., dtype=np.float64)
        # Unequal fold counts must select 640; equal-fold weighting would tie.
        scores[0], scores[1] = [1., 0., 0., 0., 0.], [0., 0., 1., 0., 0.]
        alpha, aggregate = brain._choose_alpha(scores)
        summary = brain.BrainPairSummary(*brain.PAIR_ORDER[5], True,
            "all_50_CV_fits_and_selected_final_refit_complete", alpha, aggregate,
            tuple(tuple(float(value) for value in row) for row in scores), completed_fits=51)
        return brain.BrainLossFit(summary, ids, (12, 3, 8), (900, 4),
            np.arange(8, dtype=np.float32).reshape(4, 2) / 8,
            np.array([[.1, -.2, .3]], dtype=np.float32), np.array([[2., 3., 4.]], dtype=np.float32),
            np.array([[-.4, .5]], dtype=np.float32), np.array([[5., 6.]], dtype=np.float32))

    def prepare(self, fit, **overrides):
        kwargs = dict(expected_pair=brain.PAIR_ORDER[5], authenticated_tau_train_image_ids=fit.image_ids[::-1],
                      source_native_ids=(12, 3, 8), target_native_ids=(900, 4), check=lambda: None)
        kwargs.update(overrides)
        return final.prepare_final_brain_artifact(fit, **kwargs)

    def budget(self):
        return SimpleNamespace(allocation={"job_id": os.environ.get("SLURM_JOB_ID", ""),
            "hostname": socket.gethostname().split(".")[0], "job_end_epoch": time.time() + 120},
            deadline=time.monotonic() + 60, check=lambda: None)

    def test_final_mapping_complete_native_axes_CV_and_moments_roundtrip(self):
        fit = self.fixture()
        artifact = self.prepare(fit)
        matrix = fit.matrix.copy()
        fit.matrix[:] = 99
        np.testing.assert_array_equal(artifact["matrix"].numpy(), matrix)
        for name in ("source_mean", "source_std", "target_mean", "target_std"):
            np.testing.assert_array_equal(artifact[name].numpy(), getattr(fit, name))
        self.assertEqual(artifact["source_native_voxel_ids"], (12, 3, 8))
        self.assertEqual(artifact["target_native_voxel_ids"], (900, 4))
        self.assertEqual(artifact["training_image_ids"], fit.image_ids)
        self.assertEqual(artifact["validation_indices"], brain.make_common_folds(fit.image_ids).validation_indices)
        self.assertEqual(artifact["summary"]["selected_alpha"], 640)
        self.assertEqual(artifact["summary"]["fold_mse_by_alpha"], fit.summary.fold_mse_by_alpha)
        self.assertEqual(artifact["summary"]["validation_mse"], fit.summary.validation_mse)
        self.assertEqual(artifact["canonical_pair_order"], brain.PAIR_ORDER)
        self.assertEqual(artifact["pair_index"], 5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.pt"
            final.write_final_brain_artifact(path, artifact, budget=self.budget(), maximum_bytes=100_000)
            self.assertFalse(path.with_name("final.pt.partial").exists())
            with mock.patch.object(final.torch, "load", wraps=torch.load) as loader:
                restored = final.load_final_brain_artifact(path, budget=self.budget(), maximum_bytes=100_000)
            self.assertEqual(loader.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
            np.testing.assert_array_equal(restored["matrix"].numpy(), matrix)
            self.assertEqual(restored["summary"], artifact["summary"])

    def test_incomplete_misbound_or_nonfinite_models_are_refused(self):
        fit = self.fixture()
        summaries = (
            replace(fit.summary, available=False), replace(fit.summary, completed_fits=50),
            replace(fit.summary, fold_mse_by_alpha=fit.summary.fold_mse_by_alpha[:-1]),
            replace(fit.summary, selected_alpha=620),
            replace(fit.summary, validation_mse=(0.,) * 10),
        )
        for summary in summaries:
            with self.subTest(summary=summary), self.assertRaises((ValueError, brain.BrainLossNotEvaluable)):
                self.prepare(replace(fit, summary=summary))
        for changed in (replace(fit, image_ids=fit.image_ids[:-1]),
                        replace(fit, source_native_voxel_ids=(3, 12, 8)),
                        replace(fit, matrix=fit.matrix[:-1]),
                        replace(fit, target_std=np.zeros((1, 2), dtype=np.float32)),
                        replace(fit, source_mean=np.full((1, 3), np.nan, dtype=np.float32))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.prepare(changed)
        with self.assertRaisesRegex(ValueError, "pair_binding"):
            self.prepare(fit, expected_pair=brain.PAIR_ORDER[0])
        foreign_ids = ("foreign",) + fit.image_ids[1:]
        with self.assertRaisesRegex(ValueError, "authenticated_897"):
            self.prepare(fit, authenticated_tau_train_image_ids=foreign_ids)
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            self.prepare(fit, check=mock.Mock(side_effect=TimeoutError("deadline")))

    def test_publication_preserves_existing_and_partial_files_and_obeys_byte_cap(self):
        artifact = self.prepare(self.fixture())
        for existing in ("final", "partial", "dangling_final"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "final.pt"
                partial = path.with_name("final.pt.partial")
                if existing == "dangling_final":
                    path.symlink_to(Path(temporary) / "missing")
                else:
                    (path if existing == "final" else partial).write_bytes(b"preserved")
                with self.assertRaises(FileExistsError):
                    final.write_final_brain_artifact(path, artifact, budget=self.budget(), maximum_bytes=100_000)
                if existing == "partial":
                    self.assertFalse(path.exists())
                    self.assertEqual(partial.read_bytes(), b"preserved")
                else:
                    self.assertFalse(partial.exists())
                    if existing == "final":
                        self.assertEqual(path.read_bytes(), b"preserved")
                    else:
                        self.assertTrue(path.is_symlink())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bounded.pt"
            with self.assertRaises((ValueError, RuntimeError)):
                final.write_final_brain_artifact(path, artifact, budget=self.budget(), maximum_bytes=512)
            self.assertFalse(path.exists())
            self.assertLessEqual(path.with_name("bounded.pt.partial").stat().st_size, 512)


if __name__ == "__main__":
    unittest.main()
