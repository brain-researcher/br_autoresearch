"""Three source-free full-model composition checks; compute only, UNRUN.

Existing artifact and numerical prediction tests are reused. Tiny native axes
and mocked already-validated final models test the new full roster, independent
identity joins, per-model admission and release without fitting or payloads.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_full_models as models


TEST_IDS = tuple(f"test-{number:03d}" for number in range(224))
TAU_IDS = tuple(f"tau-{number:03d}" for number in range(897))


class Artifact(dict):
    """Weak-referenceable model stand-in; no real checkpoint or model tensors."""


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        base = Path(directory).resolve()
        owners = models.PARTICIPANTS
        widths = {owner: 5 + index for index, owner in enumerate(owners)}
        eligible = {**widths, "sub-01": widths["sub-01"] - 1}
        axes, responses, training = {}, {}, {}
        for index, owner in enumerate(owners):
            full = np.arange(widths[owner], dtype=np.int64) * 3 + 100 * index
            columns = np.arange(eligible[owner], dtype=np.int64)
            mean, std = np.full(eligible[owner], index, np.float32), np.ones(eligible[owner], np.float32)
            axes[owner] = models.prediction.NativeAxis(owner, full, full[columns], columns, mean, std)
            values = np.full((224, widths[owner]), index + 10, np.float32)
            values.setflags(write=False)
            responses[owner] = models.prediction.TauTestResponses(owner, TEST_IDS, tuple(full), values)
            training[owner] = tuple(f"unique-{owner}-{number:04d}" for number in range(4712))
        context = object()
        budget = SimpleNamespace(allocation={"job_id": "fixture-job", "hostname": "fixture-node"},
                                 check=lambda: None)
        calls, admissions, references, factories = [], [], [], []
        state = SimpleNamespace(defect=None, bad_method="content_loss", bad_index=0,
                                missing=False, move=False, revoke=None)

        def resolver(actual, **kwargs):
            if (actual is not context or kwargs["budget"] is not budget
                    or kwargs["execution_options"] is not None):
                raise AssertionError("original context/budget/options not forwarded")
            operation = kwargs["operation"]
            pair = kwargs["source"], kwargs["target"]
            if operation in (models.BIND, models.inputs.BIND):
                if pair != (None, None):
                    raise AssertionError("binding operation must not imply one pair")
            elif operation not in (models.READ_CONTENT, models.READ_BRAIN) or pair not in models.PAIR_ORDER:
                raise AssertionError("unexpected model operation")
            if operation == models.READ_BRAIN and "sub-01" in pair:
                raise AssertionError("unavailable original brain pair was accessed")
            admissions.append((operation, pair))
            if state.revoke == (operation, pair):
                raise PermissionError("model read revoked")

        identity = "fixture-job", "fixture-node"
        admitted = SimpleNamespace(context=context, resolver=resolver, budget=budget, identity=identity,
            check=lambda: resolver(context, operation=models.inputs.BIND, source=None, target=None,
                                   budget=budget, execution_options=None))
        bound = models.inputs.BoundFullFindingAInputs(context, resolver, budget, identity, owners,
            TEST_IDS, MappingProxyType(responses), MappingProxyType({owner: (1,) * 224 for owner in owners}),
            MappingProxyType({}), None, admitted)
        kwargs = dict(authenticated_tau_train_image_ids=TAU_IDS, execution_context=context,
                      require_admitted_context=resolver, budget=budget)

        def load(method, path, *, budget, maximum_bytes):
            if any(reference() is not None for reference in references):
                raise AssertionError("previous model artifact survived into next model load")
            matches = [(index, source, target) for index, (source, target) in enumerate(models.PAIR_ORDER)
                       if not (method == "brain_loss" and "sub-01" in (source, target))
                       and path == models._model_path(method, source, target)]
            if len(matches) != 1:
                raise AssertionError("unexpected original model path")
            index, source, target = matches[0]
            expected_operation = models.READ_CONTENT if method == "content_loss" else models.READ_BRAIN
            if (expected_operation, (source, target)) not in admissions:
                raise AssertionError("model opened before its distinct admission")
            if maximum_bytes != (1073741824 if method == "content_loss" else 805306368):
                raise AssertionError("original artifact byte limit changed")
            calls.append((method, index, path))
            affected = (method, index) == (state.bad_method, state.bad_index)
            if affected and state.missing:
                raise FileNotFoundError("expected final model is missing")
            if method == "content_loss":
                result = Artifact(initialization={"pair_index": index, "source": source, "target": target,
                    "input_width": eligible[source], "output_width": eligible[target]},
                    completion={"backend": "sparse"}, source_training_image_ids=training[source],
                    target_decoder_training_image_ids=training[target], source_axis=axes[source], target_axis=axes[target])
            else:
                result = Artifact(pair_index=index, summary={"source": source, "target": target},
                    training_image_ids=TAU_IDS, source_native_voxel_ids=tuple(axes[source].full_native_ids),
                    target_native_voxel_ids=tuple(axes[target].full_native_ids))
            if affected:
                defect = state.defect
                if defect == "pair_index":
                    (result["initialization"] if method == "content_loss" else result)["pair_index"] = index + 1
                elif defect == "source":
                    result["initialization"]["source"] = target
                elif defect == "width":
                    result["initialization"]["input_width"] += 1
                elif defect == "backend":
                    result["completion"]["backend"] = "dense"
                elif defect == "training":
                    key = "source_training_image_ids" if method == "content_loss" else "training_image_ids"
                    result[key] = tuple(reversed(result[key]))
                elif defect == "target_training":
                    result["target_decoder_training_image_ids"] = training[source]
                elif defect in ("source_axis", "target_axis"):
                    if method == "content_loss":
                        original = result[defect]
                        result[defect] = replace(original, mean=original.mean + np.float32(1))
                    else:
                        key = defect.replace("axis", "native_voxel_ids")
                        result[key] = tuple(reversed(result[key]))
                if state.move:
                    budget.allocation["job_id"] = "moved-job"
            references.append(weakref.ref(result))
            return result

        def prepared(method, responses, axis, *, index=None):
            source, target = responses.participant, None if method == "within_source" else axis.participant
            value = (models.PARTICIPANTS.index(source) + 1 if index is None else
                     index + (100 if method == "content_loss" else 200))
            matrix = np.full((224, len(axis.eligible_native_ids)), value, np.float32)
            matrix.setflags(write=False)
            return models.prediction.PreparedDecoderInput(method, source, target, axis, TEST_IDS, matrix)

        def within(response, axis, *, decoder_training_image_ids, expected_test_image_ids, check):
            check()
            if decoder_training_image_ids != training[response.participant] or expected_test_image_ids != TEST_IDS:
                raise AssertionError("within-source training/test join changed")
            return prepared("within_source", response, axis)

        factory = object()

        def content(response, artifact, axis, *, expected_test_image_ids, converter_factory, check):
            check()
            if expected_test_image_ids != TEST_IDS or converter_factory is not factory:
                raise AssertionError("content test identities or pinned factory changed")
            return prepared("content_loss", response, axis, index=artifact["initialization"]["pair_index"])

        def brain(response, artifact, axis, *, target_decoder_training_image_ids, expected_test_image_ids, check):
            check()
            if target_decoder_training_image_ids != training[axis.participant] or expected_test_image_ids != TEST_IDS:
                raise AssertionError("brain target training/test identities changed")
            return prepared("brain_loss", response, axis, index=artifact["pair_index"])

        def factory_loader():
            factories.append(True)
            return factory

        replacements = {"BASE": base, "NATIVE_VOXELS": widths, "ELIGIBLE_VOXELS": eligible,
            "_require_reader_context": lambda budget: budget.check(),
            "_brain_disposition": lambda check: (check(), models.first.BRAIN_REASON)[1],
            "_artifact_axis": lambda record, check: (check(), record)[1],
            "_load_content": lambda path, **kwargs: load("content_loss", path, **kwargs),
            "_load_brain": lambda path, **kwargs: load("brain_loss", path, **kwargs),
            "_prepare_within": within, "_prepare_content": content, "_prepare_brain": brain,
            "_converter_factory": factory_loader}
        for name, replacement in replacements.items():
            stack.enter_context(mock.patch.object(models, name, replacement))
        yield SimpleNamespace(base=base, bound=bound, axes=axes, training=training, kwargs=kwargs,
            state=state, budget=budget, calls=calls, admissions=admissions, references=references,
            factories=factories, eligible=eligible, responses=responses)


class FullFindingAModelsTests(unittest.TestCase):
    def test_full_original_roster_paths_prepared_arrays_and_sequential_release(self):
        self.assertEqual(models.NATIVE_VOXELS["sub-01"], 12295)
        self.assertEqual(models.ELIGIBLE_VOXELS["sub-01"], 12294)
        self.assertEqual((models.CONTENT_BYTES, models.BRAIN_BYTES), (1073741824, 805306368))
        with fixture() as env:
            result = models.prepare_full_models(env.bound, env.axes, env.training, **env.kwargs)
            self.assertEqual((len(result.prepared), len(result.fitted_object_ids), len(result.unavailable_reasons)),
                             (37, 45, 8))
            self.assertEqual(Counter(key[0] for key in result.prepared),
                             {"within_source": 5, "content_loss": 20, "brain_loss": 12})
            self.assertEqual(Counter(method for method, _, _ in env.calls), {"content_loss": 20, "brain_loss": 12})
            self.assertEqual(len(env.factories), 20)
            self.assertTrue(all(reference() is None for reference in env.references))
            self.assertEqual(result.image_ids, TEST_IDS)
            for method, source, target in models.STREAMS:
                key = method, source, target
                if key in result.unavailable_reasons:
                    self.assertEqual(method, "brain_loss")
                    self.assertIn("sub-01", (source, target))
                    self.assertIsNone(result.fitted_object_ids[key])
                    self.assertNotIn(key, result.prepared)
                    continue
                actual = result.prepared[key]
                owner = source if target is None else target
                self.assertEqual(actual.decoder_axis.participant, owner)
                self.assertEqual(actual.values.shape, (224, env.eligible[owner]))
                self.assertFalse(actual.values.flags.writeable)
                if method == "within_source":
                    value = models.PARTICIPANTS.index(source) + 1
                    self.assertTrue(result.fitted_object_ids[key].endswith("#participant=" + source))
                else:
                    index = models.PAIR_ORDER.index((source, target))
                    value = index + (100 if method == "content_loss" else 200)
                    stem = f"sub{source[-2:]}_to_sub{target[-2:]}"
                    if method == "content_loss":
                        run = "cuda001" if index == 4 else stem + "_cuda001"
                        expected = f"outputs/runs/content_pair_training/{run}/final_content_pair.pt"
                    else:
                        run = stem + ("_mem16_002" if index == 5 else "_001")
                        expected = f"outputs/runs/brain_pair_training/{run}/final_brain_pair.pt"
                    self.assertEqual(result.fitted_object_ids[key], expected)
                np.testing.assert_array_equal(actual.values, np.full(actual.values.shape, value, np.float32))
            observed = "brain_loss", "sub-03", "sub-01"
            self.assertEqual(result.unavailable_reasons[observed], models.first.BRAIN_REASON)
            inferred = [value for key, value in result.unavailable_reasons.items() if key != observed]
            self.assertEqual(len(inferred), 7)
            self.assertTrue(all("source_inferred_not_executed" in value and "full12295_native_axis" in value
                                for value in inferred))
            with self.assertRaises(TypeError):
                result.prepared[observed] = None
            result.check()
            env.budget.allocation["hostname"] = "different-node"
            with self.assertRaisesRegex(ValueError, "cannot_move_allocations"):
                result.check()

    def test_admission_and_all_owner_role_support_precede_model_or_diagnostic_reads(self):
        with fixture() as env, mock.patch.object(models, "_brain_disposition") as diagnostic:
            denied = dict(env.kwargs, require_admitted_context=mock.Mock(side_effect=PermissionError("closed")))
            with self.assertRaises(PermissionError):
                models.prepare_full_models(env.bound, env.axes, env.training, **denied)
            diagnostic.assert_not_called()
            self.assertEqual(env.calls, [])
        for defect in ("last_axis", "last_training_overlap", "tau_test_overlap", "tau_count",
                       "source01_dropped_native", "foreign_context", "missing_owner"):
            with self.subTest(defect=defect), fixture() as env, \
                    mock.patch.object(models, "_brain_disposition") as diagnostic:
                axes, training, bound, kwargs = dict(env.axes), dict(env.training), env.bound, dict(env.kwargs)
                if defect == "last_axis":
                    axes["sub-07"] = axes["sub-06"]
                elif defect == "last_training_overlap":
                    training["sub-07"] = training["sub-01"]
                elif defect == "tau_test_overlap":
                    kwargs["authenticated_tau_train_image_ids"] = (TEST_IDS[0],) + TAU_IDS[1:]
                elif defect == "tau_count":
                    kwargs["authenticated_tau_train_image_ids"] = TAU_IDS[:-1]
                elif defect == "source01_dropped_native":
                    original = env.responses["sub-01"]
                    env.responses["sub-01"] = replace(original, native_voxel_ids=original.native_voxel_ids[:-1],
                                                      values=original.values[:, :-1])
                elif defect == "foreign_context":
                    bound = replace(bound, execution_context=object())
                else:
                    training.pop("sub-07")
                with self.assertRaises(ValueError):
                    models.prepare_full_models(bound, axes, training, **kwargs)
                diagnostic.assert_not_called()
                self.assertEqual(env.calls, [])

    def test_model_identity_failures_missing_publication_and_revocation_never_fallback(self):
        defects = {"content_loss": ("pair_index", "source", "width", "backend", "training",
                                    "target_training", "source_axis", "target_axis"),
                   "brain_loss": ("pair_index", "training", "source_axis", "target_axis")}
        for method, cases in defects.items():
            for defect in cases:
                with self.subTest(method=method, defect=defect), fixture() as env:
                    env.state.bad_method, env.state.bad_index = method, 0 if method == "content_loss" else 5
                    env.state.defect = defect
                    with self.assertRaises(ValueError):
                        models.prepare_full_models(env.bound, env.axes, env.training, **env.kwargs)
                    self.assertEqual(env.calls[-1][:2], (env.state.bad_method, env.state.bad_index))
                    self.assertTrue(all(reference() is None for reference in env.references))
        for failure in ("late_missing_content", "late_missing_brain", "revoked_before_open", "moved_after_open"):
            with self.subTest(failure=failure), fixture() as env:
                expected_error = FileNotFoundError
                if failure.startswith("late_missing"):
                    env.state.bad_method = "content_loss" if failure.endswith("content") else "brain_loss"
                    env.state.bad_index, env.state.missing = 19, True
                elif failure == "revoked_before_open":
                    env.state.revoke = (models.READ_BRAIN, ("sub-03", "sub-05"))
                    expected_error = PermissionError
                else:
                    env.state.move = True
                    expected_error = ValueError
                with self.assertRaises(expected_error):
                    models.prepare_full_models(env.bound, env.axes, env.training, **env.kwargs)
                if failure.startswith("late_missing"):
                    self.assertEqual(env.calls[-1][:2], (env.state.bad_method, 19))
                elif failure == "revoked_before_open":
                    self.assertNotIn(("brain_loss", 5), [call[:2] for call in env.calls])
                else:
                    self.assertEqual(len(env.calls), 1)
                self.assertTrue(all(reference() is None for reference in env.references))


if __name__ == "__main__":
    unittest.main()
