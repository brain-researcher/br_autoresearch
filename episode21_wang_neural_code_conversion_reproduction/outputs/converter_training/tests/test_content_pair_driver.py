"""Focused orchestration fixtures; run only in coordinator-owned Slurm compute.

All arrays/models are tiny fabricated inputs. The successful fixture uses the
actual 1,024-step loop with its separately tested loss arithmetic replaced by a
tiny step, then actual final-only artifact publication. No empirical
responses, decoder coefficients, features or public-source archive are opened.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import content_pair_driver as driver
import decoder_interface as interface
import final_content_artifact as final
import schedule
import train_content as training
import test_final_content_artifact as artifact_fixtures


class ContentPairDriverTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        widths = dict(zip(schedule.PARTICIPANTS, (9, 11, 8, 10, 7)))
        normalizations = {}
        for participant, width in widths.items():
            native = 100 + 2 * np.arange(width + 2, dtype=np.int64)
            mask = np.ones(width + 2, dtype=bool)
            mask[[1, width]] = False
            normalizations[participant] = interface.assembly.Normalization(
                participant, native, native[mask], mask,
                np.linspace(-1, 1, width + 2, dtype=np.float32),
                np.where(mask, np.float32(2), np.float32(0)),
                np.asarray([f"{participant}_image_{i:04d}" for i in range(4712)]),
                np.ones(4712, dtype=np.int64))
        states = tuple(interface.assembly.required_state(person) for person in schedule.PARTICIPANTS)
        manifest = interface.assembly.CompleteDecoderManifest((), states)
        budget = artifact_fixtures.FinalContentArtifactTests().budget()
        events, constructors, admitted = [], [], []
        budget.check = lambda: events.append("check")

        def factory(input_width, output_width):
            self.assertEqual(events[-1], "check")
            constructors.append((input_width, output_width))
            events.append("construct")
            return artifact_fixtures.TinyConverter(input_width, output_width)

        def state_context(actual_manifest, participant, actual_budget):
            self.assertIs(actual_manifest, manifest)
            self.assertIs(actual_budget, budget)
            admitted.append(participant)
            return interface.assembly.required_state(participant), normalizations[participant]

        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(mock.patch.object(interface, "_state_context", side_effect=state_context))
            stack.enter_context(mock.patch.object(interface.assembly, "verify_catalog"))
            loader = stack.enter_context(mock.patch.object(schedule, "load_pinned_converter_factory",
                                                           return_value=factory))
            panel = driver.prepare_content_panel(manifest, budget=budget)
            source_name, target = schedule.PAIR_ORDER[5]
            axis = panel.axes[schedule.PARTICIPANTS.index(source_name)]
            values = np.arange(4712 * widths[source_name], dtype=np.float32).reshape(4712, -1) / 1000
            values.setflags(write=False)
            source = interface.SourceTraining(axis,
                tuple(normalizations[source_name].training_image_ids.tolist()), values,
                interface.assembly.BASE / "outputs/steward_provisioning/regular_vgg19_sealed" / source_name)
            yield SimpleNamespace(panel=panel, source=source, target=target, budget=budget,
                path=Path(directory) / "final.pt", events=events, constructors=constructors,
                admitted=admitted, normalizations=normalizations, loader=loader)

    def run_pair(self, fixture, **changes):
        options = dict(target=fixture.target, budget=fixture.budget, device="cpu",
            block_units=7, row_chunk_size=37, backend="sparse", final_path=fixture.path,
            maximum_bytes=2_000_000)
        options.update(changes)
        return driver.train_and_publish_content_pair(fixture.panel, fixture.source, **options)

    def test_full_canonical_initialization_joins_exact_roles_then_publishes_final(self):
        with self.fixture() as fixture:
            self.assertEqual(fixture.admitted, list(schedule.PARTICIPANTS))
            widths = {axis.participant: len(axis.eligible_native_ids) for axis in fixture.panel.axes}
            expected_pairs = [(widths[source], widths[target]) for source, target in schedule.PAIR_ORDER]
            self.assertEqual(fixture.constructors, expected_pairs)
            self.assertEqual(len(fixture.panel.initializations), 20)
            fixture.loader.assert_called_once_with()
            self.assertEqual(len(fixture.panel.channel_schedule.channels_by_layer), 19)
            before = fixture.source.normalized_X.copy()
            expected_initial = schedule.initialize_pair(fixture.panel.initializations[5],
                                                        artifact_fixtures.TinyConverter)
            expected_final = {}
            provider = SimpleNamespace(close=mock.Mock(side_effect=lambda: fixture.events.append("provider_closed")))
            steps = []
            actual_fit = training.fit_content_pair

            def step(model, optimizer, values, active, actual_provider, **limits):
                self.assertEqual(values.shape[0], 4712)
                self.assertEqual(len(active), 19)
                self.assertEqual(sum(item.units for item in active), 151096)
                self.assertIs(actual_provider, provider)
                optimizer.zero_grad(set_to_none=True)
                for parameter in model.parameters():
                    parameter.grad = torch.zeros_like(parameter)
                optimizer.step()
                with torch.no_grad():
                    next(model.parameters()).add_(.0001)
                steps.append(True)
                return SimpleNamespace(loss=torch.tensor(1.25))

            def fit(model, values, channels, actual_provider, **limits):
                self.assertFalse(fixture.path.exists())
                self.assertFalse(fixture.path.with_name("final.pt.partial").exists())
                self.assertIs(channels, fixture.panel.channel_schedule)
                self.assertIs(actual_provider, provider)
                self.assertEqual(values.shape, (4712, widths[fixture.source.axis.participant]))
                np.testing.assert_array_equal(values.numpy(), before)
                self.assertEqual(limits["max_block_units"], 7)
                self.assertEqual(limits["row_chunk_size"], 37)
                self.assertEqual(limits["backend"], "sparse")
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, expected_initial.state_dict()[name]), name)
                completion = actual_fit(model, values, channels, actual_provider, **limits)
                expected_final.update({name: value.clone() for name, value in model.state_dict().items()})
                fixture.events.append("fit_completed")
                return completion

            actual_write = final.write_final_content_artifact
            def publish(*args, **kwargs):
                self.assertIn("fit_completed", fixture.events)
                self.assertIn("provider_closed", fixture.events)
                fixture.events.append("publish")
                return actual_write(*args, **kwargs)

            with mock.patch.object(training, "fit_content_pair", side_effect=fit) as fit_call, \
                    mock.patch.object(training, "content_loss_step", side_effect=step), \
                    mock.patch.object(interface, "content_block_provider", return_value=provider) as provider_call, \
                    mock.patch.object(final, "write_final_content_artifact", side_effect=publish):
                result = self.run_pair(fixture)
            fit_call.assert_called_once()
            provider.close.assert_called_once_with()
            self.assertEqual(provider_call.call_args.kwargs, {"budget": fixture.budget, "block_units": 7})
            bound_pair = provider_call.call_args.args[0]
            self.assertIs(bound_pair.source, fixture.source)
            self.assertEqual(bound_pair.target_axis.participant, fixture.target)
            self.assertFalse(set(bound_pair.source.image_ids).intersection(
                bound_pair.target_normalization.training_image_ids.tolist()))
            self.assertEqual(result.pair_index, 5)
            self.assertEqual(result.path, fixture.path)
            self.assertEqual(result.completion.iterations, 1024)
            self.assertEqual(len(steps), 1024)
            self.assertEqual(fixture.constructors, expected_pairs + [expected_pairs[5]])
            np.testing.assert_array_equal(fixture.source.normalized_X, before)
            artifact = final.load_final_content_artifact(fixture.path, budget=fixture.budget,
                                                         maximum_bytes=2_000_000)
            for name, value in expected_final.items():
                self.assertTrue(torch.equal(artifact["state_dict"][name], value), name)
            self.assertEqual(artifact["source_training_image_ids"], fixture.source.image_ids)
            self.assertEqual(artifact["initialization"]["pair_index"], 5)
            self.assertTrue(torch.equal(artifact["initialization"]["cpu_rng_before"],
                                       fixture.panel.initializations[5].cpu_rng_before))

    def test_misjoined_rows_axes_or_target_overlap_refuse_before_model_or_fit(self):
        for defect in ("row_order", "row_count", "source_axis", "feature_role", "target_axis", "overlap"):
            with self.subTest(defect=defect), self.fixture() as fixture:
                if defect == "row_order":
                    fixture.source = replace(fixture.source, image_ids=fixture.source.image_ids[::-1])
                elif defect == "row_count":
                    fixture.source = replace(fixture.source, normalized_X=fixture.source.normalized_X[:-1])
                elif defect == "source_axis":
                    fixture.source = replace(fixture.source, axis=replace(fixture.source.axis,
                        mean=fixture.source.axis.mean + np.float32(1)))
                elif defect == "feature_role":
                    fixture.source = replace(fixture.source,
                        feature_root=fixture.source.feature_root.parent / fixture.target)
                elif defect == "target_axis":
                    target = fixture.normalizations[fixture.target]
                    fixture.normalizations[fixture.target] = replace(target, mean=target.mean + np.float32(1))
                else:
                    target = fixture.normalizations[fixture.target]
                    fixture.normalizations[fixture.target] = replace(target,
                        training_image_ids=np.asarray(fixture.source.image_ids))
                with mock.patch.object(training, "fit_content_pair") as fit_call, \
                        mock.patch.object(final, "write_final_content_artifact") as publisher, \
                        self.assertRaises(ValueError):
                    self.run_pair(fixture)
                self.assertEqual(len(fixture.constructors), 20)
                fit_call.assert_not_called()
                publisher.assert_not_called()
                self.assertFalse(fixture.path.exists())

    def test_interruption_incomplete_fit_and_publication_failure_never_return_success(self):
        from content_pair_measurement import _MeasurementComplete
        for defect in ("fit_interrupt", "measurement_stop", "close_failure", "incomplete",
                       "publication", "existing_final", "existing_partial"):
            with self.subTest(defect=defect), self.fixture() as fixture:
                if defect.startswith("existing"):
                    existing = fixture.path if defect == "existing_final" else fixture.path.with_name("final.pt.partial")
                    existing.write_bytes(b"preserved")
                complete = training.CompletedContentFit(1024, 4712, 1.25,
                    training.LEARNING_RATE / 512, "sparse")
                stop = _MeasurementComplete()
                provider = SimpleNamespace(close=mock.Mock(
                    side_effect=OSError("provider close failed") if defect == "close_failure" else None))
                factory = mock.Mock(return_value=provider)
                def fit(*args, **kwargs):
                    self.assertIs(args[3], provider)
                    if defect == "fit_interrupt":
                        raise KeyboardInterrupt("interrupted full fit")
                    if defect in ("measurement_stop", "close_failure"):
                        raise stop
                    return replace(complete, iterations=1023) if defect == "incomplete" else complete
                with mock.patch.object(training, "fit_content_pair", side_effect=fit) as fit_call, \
                        mock.patch.object(interface, "content_block_provider") as ordinary_provider, \
                        mock.patch.object(final, "write_final_content_artifact", side_effect=OSError("write failure")) as publisher:
                    exception = {"fit_interrupt": KeyboardInterrupt, "incomplete": ValueError,
                                 "measurement_stop": _MeasurementComplete, "close_failure": OSError,
                                 "publication": OSError}.get(defect, FileExistsError)
                    with self.assertRaises(exception) as raised:
                        self.run_pair(fixture, provider_factory=factory)
                    ordinary_provider.assert_not_called()
                if defect == "measurement_stop":
                    self.assertIs(raised.exception, stop)
                elif defect == "close_failure":
                    self.assertIs(raised.exception.__context__, stop)
                self.assertEqual(fit_call.call_count, 0 if defect.startswith("existing") else 1)
                self.assertEqual(factory.call_count, fit_call.call_count)
                self.assertEqual(provider.close.call_count, fit_call.call_count)
                if fit_call.call_count:
                    self.assertIs(factory.call_args.args[0].source, fixture.source)
                    self.assertEqual(factory.call_args.args[0].target_axis.participant, fixture.target)
                    self.assertEqual(factory.call_args.kwargs, {"budget": fixture.budget, "block_units": 7})
                self.assertEqual(publisher.call_count, 1 if defect == "publication" else 0)
                if defect.startswith("existing"):
                    self.assertEqual(existing.read_bytes(), b"preserved")
                    self.assertEqual(len(fixture.constructors), 20)
                else:
                    self.assertFalse(fixture.path.exists())


if __name__ == "__main__":
    unittest.main()
