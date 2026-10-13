"""Four final-checkpoint fixtures; run only in coordinated Slurm compute.

Tiny fabricated models/completion objects test serialization and publication,
not scientific fitting or admission. No empirical fragment, response, feature,
tau/OOD data or existing checkpoint is opened. The complete decoder catalog's
already tested metadata validator is stubbed only for these in-memory fixtures.
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
import final_content_artifact as final
import decoder_interface as interface
import schedule


class TinyConverter(torch.nn.Module):
    """Literal tagged parameter layout at tiny widths; no forward fit is run."""
    def __init__(self, input_width, output_width):
        super().__init__()
        hidden = input_width // 2
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden), torch.nn.InstanceNorm1d(1), torch.nn.LeakyReLU(.2),
            torch.nn.Linear(hidden, hidden), torch.nn.InstanceNorm1d(1), torch.nn.LeakyReLU(.2),
            torch.nn.Linear(hidden, output_width))


class FinalContentArtifactTests(unittest.TestCase):
    def fixture(self):
        widths = dict(zip(schedule.PARTICIPANTS, (9, 11, 8, 10, 7)))
        # Use a later pair: its saved RNG states include preceding constructors,
        # so these checks do not endorse resetting every pair to seed42.
        initial = schedule.capture_pair_initializations(widths, TinyConverter)[5]
        model = schedule.initialize_pair(initial, TinyConverter)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.125)  # Stand-in final update, distinct from initialization.
        def normalization(participant, width):
            native = 100 + 2 * np.arange(width + 2, dtype=np.int64)
            mask = np.ones(width + 2, dtype=bool)
            mask[[1, width]] = False
            return final.assembly.Normalization(participant, native, native[mask], mask,
                np.linspace(-1, 1, width + 2, dtype=np.float32),
                np.where(mask, np.float32(2), np.float32(0)),
                np.asarray([f"{participant}_image_{i:04d}" for i in range(final.TRAINING_ROWS)]),
                np.ones(final.TRAINING_ROWS, dtype=np.int64))
        source = normalization(initial.source, initial.input_width)
        target = normalization(initial.target, initial.output_width)
        source_training = interface.SourceTraining(interface.native_axis(source), tuple(source.training_image_ids.tolist()),
            np.zeros((final.TRAINING_ROWS, initial.input_width), dtype=np.float32),
            final.assembly.BASE / "outputs/steward_provisioning/regular_vgg19_sealed" / initial.source)
        states = tuple(final.assembly.required_state(person) for person in schedule.PARTICIPANTS)
        manifest = final.assembly.CompleteDecoderManifest((), states)
        pair = interface.ContentPair(manifest, source_training, final.assembly.required_state(initial.target),
                                     target, interface.native_axis(target))
        completion = final.CompletedContentFit(1024, 4712, 1.25, final.LEARNING_RATE / 512, "sparse")
        return model, completion, pair, initial, schedule.make_channel_schedule()

    def prepare(self, fixture):
        with mock.patch.object(final.assembly, "verify_catalog"):
            return final.prepare_final_content_artifact(*fixture, check=lambda: None)

    def budget(self):
        # Tests use the actual compute shell. No scheduler lookup or fabricated
        # login-node exemption is introduced by the fixture budget.
        return SimpleNamespace(allocation={"job_id": os.environ.get("SLURM_JOB_ID", ""),
            "hostname": socket.gethostname().split(".")[0], "job_end_epoch": time.time() + 120},
            deadline=time.monotonic() + 60, check=lambda: None)

    def test_final_CPU_weights_axes_rows_schedule_and_seed42_roundtrip(self):
        fixture = self.fixture()
        artifact = self.prepare(fixture)
        expected = {key: value.clone() for key, value in fixture[0].state_dict().items()}
        with torch.no_grad():
            for parameter in fixture[0].parameters():
                parameter.add_(10)
        # Snapshot is the supplied final update, neither initialization nor a
        # reference to the model the caller may subsequently modify.
        for name, value in expected.items():
            self.assertTrue(torch.equal(artifact["state_dict"][name], value))
            self.assertEqual(artifact["state_dict"][name].device.type, "cpu")
            self.assertFalse(artifact["state_dict"][name].requires_grad)
        self.assertEqual(artifact["source_training_image_ids"], fixture[2].source.image_ids)
        self.assertEqual(len(artifact["source_training_image_ids"]), 4712)
        self.assertEqual(artifact["initialization"]["seed"], 42)
        self.assertEqual(artifact["initialization"]["pair_index"], 5)
        self.assertTrue(torch.equal(artifact["initialization"]["cpu_rng_before"], fixture[3].cpu_rng_before))
        self.assertEqual(artifact["channel_schedule"]["seed"], 42)
        self.assertEqual(artifact["channel_schedule"]["channels_by_layer"], fixture[4].channels_by_layer)
        self.assertEqual(artifact["target_decoder_manifest"], final.assembly.MANIFEST)
        for label, axis in (("source_axis", fixture[2].source.axis), ("target_axis", fixture[2].target_axis)):
            for name in ("full_native_ids", "eligible_native_ids", "eligible_full_columns", "mean", "std"):
                np.testing.assert_array_equal(artifact[label][name].numpy(), getattr(axis, name))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.pt"
            final.write_final_content_artifact(path, artifact, budget=self.budget(), maximum_bytes=2_000_000)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name("final.pt.partial").exists())
            actual_load = torch.load
            with mock.patch.object(final.torch, "load", wraps=actual_load) as loader:
                restored = final.load_final_content_artifact(path, budget=self.budget(), maximum_bytes=2_000_000)
            self.assertEqual(loader.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
            for name, value in expected.items():
                self.assertTrue(torch.equal(restored["state_dict"][name], value))
            self.assertEqual(restored["source_training_image_ids"], artifact["source_training_image_ids"])

    def test_incomplete_nonfinite_or_misbound_final_artifacts_are_refused(self):
        fixture = self.fixture()
        for completion in (replace(fixture[1], iterations=1023), replace(fixture[1], rows=4711)):
            with self.subTest(completion=completion), self.assertRaises(ValueError):
                self.prepare((fixture[0], completion, *fixture[2:]))
        wrong_source = replace(fixture[2].source, feature_root=fixture[2].source.feature_root.parent / fixture[3].target)
        with self.assertRaisesRegex(ValueError, "references_required"):
            self.prepare((*fixture[:2], replace(fixture[2], source=wrong_source), *fixture[3:]))
        overlap_source = replace(fixture[2].source, image_ids=tuple(fixture[2].target_normalization.training_image_ids.tolist()))
        with self.assertRaisesRegex(ValueError, "overlap_forbidden"):
            self.prepare((*fixture[:2], replace(fixture[2], source=overlap_source), *fixture[3:]))
        with torch.no_grad():
            next(fixture[0].parameters()).reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite_final"):
            self.prepare(fixture)

    def test_publication_is_exclusive_and_storage_limited(self):
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
                    final.write_final_content_artifact(path, artifact, budget=self.budget(), maximum_bytes=2_000_000)
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
            # PyTorch may wrap a writer exception while closing its zip writer;
            # the invariant is preserved partial bytes and no final publication.
            with self.assertRaises((ValueError, RuntimeError)):
                final.write_final_content_artifact(path, artifact, budget=self.budget(), maximum_bytes=512)
            self.assertFalse(path.exists())
            self.assertLessEqual(path.with_name("bounded.pt.partial").stat().st_size, 512)

    def test_fsync_failures_preserve_outputs_and_never_report_success(self):
        artifact = self.prepare(self.fixture())
        # First fsync failure is before publication, second before partial
        # removal, third after removal. Every failure propagates; none replaces
        # or deletes the already published final file to conceal uncertainty.
        for failure_call in (1, 2, 3):
            with self.subTest(failure_call=failure_call), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "final.pt"
                actual_fsync = os.fsync
                calls = []
                def fsync(descriptor):
                    calls.append(descriptor)
                    if len(calls) == failure_call:
                        raise OSError(5, "fixture fsync failure")
                    return actual_fsync(descriptor)
                with mock.patch.object(final.os, "fsync", side_effect=fsync), self.assertRaises(OSError):
                    final.write_final_content_artifact(path, artifact, budget=self.budget(), maximum_bytes=2_000_000)
                partial = path.with_name("final.pt.partial")
                self.assertEqual(partial.is_file(), failure_call != 3)
                self.assertEqual(path.exists(), failure_call >= 2)
                if path.exists() and partial.exists():
                    self.assertEqual(path.stat().st_ino, partial.stat().st_ino)


if __name__ == "__main__":
    unittest.main()
