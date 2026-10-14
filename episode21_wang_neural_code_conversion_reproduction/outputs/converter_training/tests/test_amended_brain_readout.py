"""Three changed-interface checks; coordinated compute only, no real assets.

Reuse existing prediction/full-model fixtures. The numerical bridge test uses
actual distinct-artifact validation and all224 invented test rows. The roster
tests mock model I/O/numerical prediction while exercising real identity and
admission joins, sequential lifetimes, all20 fresh paths and refusal behavior.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock
import weakref

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import amended_brain_readout as bridge
import amended_full_models as models
import final_brain_constant_artifact as artifacts
import finding_a_prediction as prediction
import test_finding_a_full_models as model_fixtures
import test_finding_a_prediction as prediction_fixtures


def amended_prediction_fixture():
    env = prediction_fixtures.fixture()
    value = deepcopy(env.brain)
    value.update(kind=artifacts.KIND, schema_version=artifacts.SCHEMA_VERSION,
                 normalization=dict(artifacts.NORMALIZATION))
    for axis in ("source", "target"):
        mean, raw = value.pop(axis + "_mean"), value.pop(axis + "_std")
        coordinate = (len(value["source_native_voxel_ids"]) - 1 if axis == "source" else
            value["target_native_voxel_ids"].index(int(env.target_axis.eligible_native_ids[0])))
        raw[0, coordinate] = 0
        scale = raw.clone()
        scale[0, coordinate] = 1
        mask = raw == 0
        state = dict(mean=mean, raw_sd=raw, effective_scale=scale,
                     replacement_mask=mask, replacement_count=1, fitting_images=897)
        value[axis + "_normalization"] = state
        value["fold_" + axis + "_normalization"] = tuple(
            {**deepcopy(state), "fitting_images": 897 - count} for count in (180, 180, 179, 179, 179))
        # A completely centered source column has a zero ridge row; a constant
        # target has a zero column including the penalized bias.
        if axis == "source":
            value["matrix"][coordinate, :] = 0
        else:
            value["matrix"][:, coordinate] = 0
    return env, value


@contextmanager
def amended_model_fixture():
    with model_fixtures.fixture() as env, ExitStack() as stack:
        original_resolver = env.kwargs["require_admitted_context"]
        def resolver(context, **kwargs):
            operation, pair = kwargs["operation"], (kwargs["source"], kwargs["target"])
            if operation not in (models.BIND, models.READ_BRAIN):
                return original_resolver(context, **kwargs)
            if (context is not env.kwargs["execution_context"] or kwargs["budget"] is not env.budget
                    or kwargs["execution_options"] is not None):
                raise AssertionError("amended context/budget/options not forwarded")
            if ((operation == models.BIND and pair != (None, None))
                    or (operation == models.READ_BRAIN and pair not in models.PAIR_ORDER)):
                raise AssertionError("unexpected amended operation/pair")
            env.admissions.append((operation, pair))
            if env.state.revoke == (operation, pair):
                raise PermissionError("amended model read revoked")
        old = env.bound
        admitted = SimpleNamespace(context=old.execution_context, resolver=resolver,
            budget=old.budget, identity=old.allocation_identity,
            check=lambda: resolver(old.execution_context, operation=models.original.inputs.BIND,
                source=None, target=None, budget=old.budget, execution_options=None))
        env.bound = replace(old, require_admitted_context=resolver, _admission=admitted)
        env.kwargs = {**env.kwargs, "require_admitted_context": resolver}

        def load(path, *, budget, maximum_bytes):
            if any(reference() is not None for reference in env.references):
                raise AssertionError("previous artifact survived into next amended model load")
            matches = [(index, source, target) for index, (source, target) in enumerate(models.PAIR_ORDER)
                       if path == models._model_path("brain_loss", source, target)]
            if len(matches) != 1:
                raise AssertionError("not one of all20 fresh amended model paths")
            index, source, target = matches[0]
            if (models.READ_BRAIN, (source, target)) not in env.admissions:
                raise AssertionError("amended model opened before admission")
            if budget is not env.budget or maximum_bytes != 805306368:
                raise AssertionError("model budget/cap changed")
            env.calls.append(("brain_loss", index, path))
            affected = (env.state.bad_method, env.state.bad_index) == ("brain_loss", index)
            if affected and env.state.missing:
                raise FileNotFoundError("required amended model missing")
            value = model_fixtures.Artifact(kind=artifacts.KIND, normalization=dict(artifacts.NORMALIZATION),
                pair_index=index, summary={"source": source, "target": target},
                training_image_ids=model_fixtures.TAU_IDS,
                source_native_voxel_ids=tuple(env.axes[source].full_native_ids),
                target_native_voxel_ids=tuple(env.axes[target].full_native_ids))
            if affected:
                if env.state.defect == "old_kind":
                    value["kind"] = "ep21_final_brain_pair"
                elif env.state.defect == "pair_index":
                    value["pair_index"] += 1
                elif env.state.defect == "training":
                    value["training_image_ids"] = value["training_image_ids"][::-1]
                elif env.state.defect == "target_axis":
                    value["target_native_voxel_ids"] = value["target_native_voxel_ids"][::-1]
            env.references.append(weakref.ref(value))
            return value

        def prepare(response, artifact, axis, *, target_decoder_training_image_ids,
                    expected_test_image_ids, check):
            check()
            if (target_decoder_training_image_ids != env.training[axis.participant]
                    or expected_test_image_ids != model_fixtures.TEST_IDS):
                raise AssertionError("amended decoder/test identity changed")
            values = np.full((224, len(axis.eligible_native_ids)), 200 + artifact["pair_index"], np.float32)
            values.setflags(write=False)
            return prediction.PreparedDecoderInput("brain_loss", response.participant, axis.participant,
                axis, expected_test_image_ids, values)
        for name, replacement in (("BASE", env.base), ("_load_brain", load),
                ("_prepare_brain", prepare), ("_require_reader_context", lambda budget: budget.check())):
            stack.enter_context(mock.patch.object(models, name, replacement))
        yield env


class AmendedBrainReadoutTests(unittest.TestCase):
    def test_physical_decoder_bridge_keeps_all_rows_native_identity_and_role_boundaries(self):
        env, artifact = amended_prediction_fixture()
        kwargs = dict(target_decoder_training_image_ids=env.content["target_decoder_training_image_ids"],
                      expected_test_image_ids=prediction_fixtures.TEST_IDS, check=lambda: None)
        source, target = artifact["source_normalization"], artifact["target_normalization"]
        tau = (env.responses.values - source["mean"].numpy()) / source["effective_scale"].numpy()
        standardized = np.hstack((tau, np.ones((224, 1)))) @ artifact["matrix"].numpy()
        physical = standardized * target["effective_scale"].numpy() + target["mean"].numpy()
        expected = ((physical[:, env.target_axis.eligible_full_columns] - env.target_axis.mean)
                    / env.target_axis.std).astype(np.float32)
        actual = bridge.prepare_brain(env.responses, artifact, env.target_axis, **kwargs)
        self.assertEqual(actual.image_ids, prediction_fixtures.TEST_IDS)
        self.assertEqual(actual.method, "brain_loss")
        self.assertEqual(actual.values.shape, (224, len(env.target_axis.eligible_native_ids)))
        self.assertFalse(actual.values.flags.writeable)
        np.testing.assert_allclose(actual.values, expected, rtol=0, atol=2e-6)
        np.testing.assert_array_equal(actual.decoder_axis.eligible_native_ids, env.target_axis.eligible_native_ids)
        # Reorder every target-normalization record with the mapping. Identity,
        # rather than the replacement mask or old positional index, selects it.
        reordered = deepcopy(artifact)
        order = np.roll(np.arange(len(artifact["target_native_voxel_ids"])), 2)
        reordered["target_native_voxel_ids"] = tuple(artifact["target_native_voxel_ids"][i] for i in order)
        indices = torch.as_tensor(order, dtype=torch.long)
        reordered["matrix"] = artifact["matrix"][:, indices].clone()
        for state in reordered["fold_target_normalization"] + (reordered["target_normalization"],):
            for name in ("mean", "raw_sd", "effective_scale", "replacement_mask"):
                state[name] = state[name][:, indices].clone()
        remapped = bridge.prepare_brain(env.responses, reordered, env.target_axis, **kwargs)
        np.testing.assert_allclose(remapped.values, actual.values, rtol=0, atol=2e-6)
        for changed in (replace(env.responses, image_ids=env.responses.image_ids[::-1]),
                        replace(env.responses, native_voxel_ids=env.responses.native_voxel_ids[::-1]),
                        replace(env.responses, role="tau_train")):
            with self.assertRaises(ValueError):
                bridge.prepare_brain(changed, artifact, env.target_axis, **kwargs)
        with self.assertRaises(ValueError):
            bridge.prepare_brain(env.responses, env.brain, env.target_axis, **kwargs)
        overlap = (artifact["training_image_ids"][0],) + prediction_fixtures.TEST_IDS[1:]
        with self.assertRaisesRegex(ValueError, "overlap"):
            bridge.prepare_brain(replace(env.responses, image_ids=overlap), artifact, env.target_axis,
                **{**kwargs, "expected_test_image_ids": overlap})
        decoder_overlap = (prediction_fixtures.TEST_IDS[0],) + kwargs["target_decoder_training_image_ids"][1:]
        with self.assertRaisesRegex(ValueError, "overlap"):
            bridge.prepare_brain(env.responses, artifact, env.target_axis,
                **{**kwargs, "target_decoder_training_image_ids": decoder_overlap})

    def test_complete_amended_roster_uses_all20_fresh_models_and_sequential_release(self):
        with amended_model_fixture() as env:
            result = models.prepare_full_models(env.bound, env.axes, env.training, **env.kwargs)
            self.assertEqual(result.model_policy, "exact_constant_brain_comparator_all20_fresh_fits")
            self.assertEqual((len(result.prepared), len(result.fitted_object_ids), len(result.unavailable_reasons)),
                             (45, 45, 0))
            self.assertEqual(Counter(key[0] for key in result.prepared),
                             {"within_source": 5, "content_loss": 20, "brain_loss": 20})
            self.assertEqual(Counter(method for method, _, _ in env.calls), {"content_loss": 20, "brain_loss": 20})
            self.assertTrue(all(reference() is None for reference in env.references))
            self.assertEqual(len(env.factories), 20)
            for index, (source, target) in enumerate(models.PAIR_ORDER):
                key = "brain_loss", source, target
                expected = (f"outputs/runs/brain_constant_training/pair{index:03d}_"
                    f"{source.replace('-', '')}_to_{target.replace('-', '')}_001/final_brain_constant_pair.pt")
                self.assertEqual(result.fitted_object_ids[key], expected)
                self.assertEqual(result.prepared[key].values.shape, (224, env.eligible[target]))
                self.assertFalse(result.prepared[key].values.flags.writeable)
            with self.assertRaises(TypeError):
                result.fitted_object_ids[("brain_loss", "sub-03", "sub-01")] = "original.pt"
            result.check()

    def test_absent_misbound_or_revoked_models_never_become_partial_or_original_fallback(self):
        for defect in ("old_kind", "pair_index", "training", "target_axis", "missing", "revoked", "support"):
            with self.subTest(defect=defect), amended_model_fixture() as env:
                env.state.bad_method, env.state.bad_index = "brain_loss", 19
                env.state.defect = defect
                if defect == "missing":
                    env.state.missing = True
                elif defect == "revoked":
                    env.state.revoke = (models.READ_BRAIN, models.PAIR_ORDER[19])
                elif defect == "support":
                    env.kwargs["authenticated_tau_train_image_ids"] = model_fixtures.TAU_IDS[:-1]
                error = FileNotFoundError if defect == "missing" else PermissionError if defect == "revoked" else ValueError
                with self.assertRaises(error):
                    models.prepare_full_models(env.bound, env.axes, env.training, **env.kwargs)
                self.assertTrue(all(reference() is None for reference in env.references))
                if defect == "support":
                    self.assertEqual(env.calls, [])
                elif defect == "revoked":
                    self.assertNotIn(("brain_loss", 19), [call[:2] for call in env.calls])
                else:
                    self.assertEqual(env.calls[-1][:2], ("brain_loss", 19))


if __name__ == "__main__":
    unittest.main()
