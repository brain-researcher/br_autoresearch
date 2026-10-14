"""Two integration checks on existing fixtures; compute/coordinator only.

No empirical files, recipe selection, resource admission or new recovery work.
Actual event/EEG bridge and PCA are reused. The full36-split test replaces only
iterative multiresponse fitting with the existing independent small dense oracle.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
from continuous_sequence_model import EventTerm, JointSequenceDesign, SequenceDesign
from prepare_continuous_eeg import prepare_participant_sequences
import production_core_reference as core
from production_continuous_designs import TrainingDesignBuilder, TemporalBasis
import test_continuous_tuning as prior
import test_prepare_continuous_eeg as bridge
from test_production_baseline_controller import FixtureBudget
from training_transforms import CORE_IMAGE, CommonSensorSpace


def recipe():
    # Invented fixture values, never a selected scientific recipe.
    return dict(recipe_id="fixture_core_only", image_panel="core_image", image_rank=64,
        temporal_basis="raised_cosine_12", nuisance_recipe=core.NUISANCE_RECIPE,
        intercept_penalty=.1, nuisance_penalty=.1, fit_weight_scale="mean_sequences",
        fit_weight_multiplier=1., normalization="ratio_of_equal_sequence_means",
        covariance_centered=True, covariance_denominator="sum_weights",
        covariance_target="trace_over_61_identity", shrinkage_strength=.1,
        baseline_tie_break=core.BASELINE_TIE, solver=dict(atol=1e-10, btol=1e-10, conlim=1e12, maxiter=100),
        metadata_byte_cap=16*1024**2, npz_byte_cap=16*1024**2, working_array_byte_cap=16*1024**3)


def real_join_fixture():
    arguments = bridge.fixture()
    # Preserve the existing fixture's clocks/segments/spans, but supply the
    # full1854x12 inventory needed by this production join, rather than309
    # identities repeated within each block in the original bridge-only check.
    source, frozen = arguments["main_source_events"], arguments["main_frozen_join"]
    for index, (original, joined) in enumerate(zip(source, frozen)):
        block, concept = divmod(index, 1854)
        canonical = f"concept_{concept:04d}/image_{block:02d}.jpg"
        original.update(objectnumber=str(concept), stim="images/"+canonical)
        joined.update(concept_number=str(concept), canonical_image=canonical,
                      archive_member="exact/archive/"+canonical)
    sequences = prepare_participant_sequences(**arguments, read_samples=bridge.raw_reader)
    names = ("objectnumber", "stimnumber", "blocksequencenumber", "sequencenumber",
             "withinsequencenumber", "presentationnumber", "sample")
    fields = core.EVENT_FIELDS[:7]
    events = {field: np.asarray([int(row[name]) for row in source], dtype=np.int64)
              for field, name in zip(fields, names)}
    events["sample_zero_based"] = events["sample_one_based"] - 1
    events["target_flag"] = (np.arange(22248) % 7 == 0).astype(np.int64)
    events["response_flag"] = (np.arange(22248) % 11 == 0).astype(np.int64)
    events["concept_by_block_image_row"] = np.arange(22248, dtype=np.int32).reshape(12,1854).T.copy()
    core._event_arrays(events)
    ids = tuple(f"{index:032x}" for index in range(22248))
    routing = {row["canonical_image"]: identity for row, identity in zip(frozen, ids)}
    return sequences, events, ids, routing


class ProductionCoreReferenceTests(unittest.TestCase):
    def test_full_join_training_only_five_family_PCA_and_identical_core_design(self):
        sequences, events, ids, routing = real_join_fixture()
        base = np.arange(22248, dtype=float)
        raw = np.column_stack((np.sin(base*.019), np.cos(base*.023), base % 17))
        class Images:
            def __init__(self):
                self.shift = False
                self.calls = []
            def rows(self, family, indices):
                self.calls.append((family, np.asarray(indices).copy()))
                value = raw[indices].copy()
                if self.shift:
                    value[~np.isin(events["block"][indices], split.training_blocks)] += 500
                return value
        images = Images()
        inputs = SimpleNamespace(recipe=recipe(), channel_order=bridge.CHANNELS, main_ids=ids,
            main_to_feature=np.arange(22248), routing=routing, images=images, check=lambda:None)
        person = core.CoreParticipant(inputs,"fixture",sequences,MappingProxyType(events))
        split = folds._split(0,1)
        training = tuple(item for item in sequences if item.block in split.training_blocks)
        first = person.prepare_training(training,split=split,budget_check=lambda:None)
        images.shift = True
        second = person.prepare_training(training,split=split,budget_check=lambda:None)
        for a,b in zip(first.build_designs.image_transform.families, second.build_designs.image_transform.families):
            np.testing.assert_array_equal(a.mean,b.mean)
            np.testing.assert_array_equal(a.population_sd,b.population_sd)
            np.testing.assert_array_equal(a.loadings,b.loadings)
            self.assertEqual(a.training_unit_ids,b.training_unit_ids)
            self.assertEqual(a.requested_rank,64)
            self.assertFalse(a.loadings.flags.writeable)
        self.assertEqual(first.build_designs.image_transform.family_order,CORE_IMAGE)
        for family, indices in images.calls[::2]:
            self.assertIn(family,CORE_IMAGE)
            self.assertEqual(len(indices),len(split.training_blocks)*1854)
            self.assertEqual(set(events["block"][indices]),set(split.training_blocks))
        first_builder = first.build_designs
        sequence = sequences[0]
        view = folds.SequencePredictors(sequence.sequence_id,sequence.block,sequence.within_block,
            sequence.sample_times_seconds,sequence.preparation_input)
        actual = first_builder(view)
        # Reuse the already-qualified original all-model builder as a literal
        # three-model design oracle. No all-model fits or empirical data.
        old = TrainingDesignBuilder(participant_id="fixture",events=MappingProxyType(events),
            false_groups=np.asarray([(events["concept"]+i+1)%1854 for i in range(8)]),
            predictor_metadata=first_builder.predictor_metadata,split=split,recipe={},
            image_transform=first_builder.image_transform,label_transform=None,
            label_kernel=SimpleNamespace(training_features=np.ones((1854,1))),
            image_features=first_builder.image_features,label_features=np.ones((1854,1)),
            rows_by_sequence=first_builder.rows_by_sequence,basis=TemporalBasis("raised_cosine_12"),check=lambda:None)
        expected = old(view)
        self.assertEqual(set(actual),set(core.CORE_MODELS))
        self.assertEqual((actual["nuisance"].shape[1],actual["reduced_task_nuisance"].shape[1]),(88,64))
        for name in core.CORE_MODELS:
            np.testing.assert_array_equal(actual[name].nuisance,expected[name].nuisance)
            coefficients = np.arange(actual[name].shape[1],dtype=float)/1000.
            np.testing.assert_array_equal(JointSequenceDesign([actual[name]])@coefficients,
                                          JointSequenceDesign([expected[name]])@coefficients)
        self.assertFalse(hasattr(first_builder,"sequences"))
        with self.assertRaises(ValueError):
            first_builder(replace(view, sample_times_seconds=view.sample_times_seconds+.001))
        bad = dict(events, sample_zero_based=events["sample_zero_based"].copy())
        bad["sample_zero_based"][0] += 1
        with self.assertRaises(ValueError):
            core.CoreParticipant(inputs,"fixture",sequences,MappingProxyType(bad)).verify_sequences()
        for field in ("nuisance_penalty","solver","covariance_target"):
            value = recipe()
            value.pop(field)
            with self.assertRaises(ValueError):
                core.validate_recipe(value)

    def test_all36_splits_whitener_checkpoint_and_complete_cohort_only(self):
        calls, inner, outer = [],[],[]
        selected = recipe()
        def prepare(training, *, split, budget_check):
            calls.append(split)
            self.assertEqual({row.block for row in training},set(split.training_blocks))
            # One frozen callable is shared by every physical sequence.
            def basis(lags):
                values = np.zeros((len(lags),12))
                values[:,0] = 1.
                return values
            def build(view):
                self.assertFalse(hasattr(view,"sensor_samples"))
                nuisance = np.zeros((len(view.sample_times_seconds),88))
                nuisance[:,0] = 1.
                # A fixed12-wide kernel with one nonzero coordinate lets the
                # independent dense oracle remain small and well conditioned.
                term = EventTerm(np.array([0.]),np.array([[1+view.preparation_input%7]]),basis,12,(0.,.8))
                return {"nuisance":SequenceDesign(view.sequence_id,view.sample_times_seconds,nuisance,{}),
                    "reduced_task_nuisance":SequenceDesign(view.sequence_id,view.sample_times_seconds,nuisance[:,:64],{}),
                    "image":SequenceDesign(view.sequence_id,view.sample_times_seconds,nuisance,{"image":term})}
            return tuning.TuningPreparation(tuple(row.sequence_id for row in training),build)
        person = SimpleNamespace(participant_id="fixture",sequences=prior.fixture(),
            inputs=SimpleNamespace(channel_order=prior.CHANNELS),prepare_training=prepare)
        progress = dict(inner_splits_completed=0,outer_splits_completed=0)
        with patch.object(folds,"_fit",side_effect=prior.dense_fit) as fitted, \
             patch.object(tuning,"_score",wraps=tuning._score) as scored, \
             patch.object(tuning,"_capacity",side_effect=AssertionError("No slot capacity in core")):
            result = core.fit_person(person,selected,check=lambda:None,retain_inner=inner.append,
                retain_outer=outer.append,progress=progress)
        self.assertEqual(len(calls),len(set(calls)))
        self.assertEqual((len(calls),len(inner),len(outer),fitted.call_count),(36,30,6,438))
        self.assertTrue(all(call.args[1] in core.CORE_MODELS for call in scored.call_args_list))
        for index, row in enumerate(outer):
            mean = np.mean([entry.normalized_image_losses for entry in inner[index*5:(index+1)*5]],axis=0)
            self.assertEqual(row.result.image_index,min(range(13),key=lambda j:(mean[j],-tuning.RIDGE_GRID[j],j)))
            self.assertEqual(set(row.fits),set(core.CORE_MODELS))
            self.assertEqual(row.whitener.covariance_centered,True)
            self.assertAlmostEqual(row.whitener.covariance_denominator,1.)
            np.testing.assert_allclose(row.whitener.shrinkage_target,
                np.eye(61)*np.trace(row.whitener.empirical_covariance)/61.,rtol=1e-12,atol=1e-15)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            budget = FixtureBudget()
            # Exact selected arrays survive the existing non-pickle serializer;
            # the production builder identity check remains active in _save_outer.
            core._split_record(root/"outer.json",dict(whitener=outer[0].whitener,
                fits=outer[0].fits,penalties=outer[0].penalties),budget)
            record = json.loads((root/"outer.json").read_text())["record"]
            with np.load(root/"outer.npz",allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays[record["fits"]["image"]["coefficients"]["array"]],
                                              outer[0].fits["image"].coefficients)
            people = [f"sub-{i:02d}" for i in range(1,31)]
            settings = dict(schema=core.SCHEMA,recipe_id=core.digest(selected),participants=people)
            core._publish_json(root/"settings.json",settings,budget)
            core._publish_json(root/"inputs_complete.json",dict(status="complete_core_reference_inputs",
                settings_id=core.digest(settings)),budget)
            for person_id in people[:-1]:
                directory=root/person_id
                directory.mkdir()
                value=dict(result,participant_id=person_id,status="complete_core_reference_person",
                    settings_id=core.digest(settings),inner_splits=30,outer_splits=6,models=list(core.CORE_MODELS),
                    scientific_release_authorized=False,concept_or_false_scores_computed=False,search_credit=0)
                core._publish_json(directory/"complete.json",value,budget)
                core._publish_json(directory/"operational.json",dict(status=value["status"],settings_id=value["settings_id"]),budget)
            # The bounded metadata reader rejects a missing participant before
            # opening the file, using its existing categorical ValueError.
            with self.assertRaisesRegex(ValueError, "^bounded_metadata_file_required$"):
                core.finalize(root,budget=budget)
            self.assertFalse((root/"core_reference.complete.json").exists())
            directory=root/people[-1]
            directory.mkdir()
            value=dict(value,participant_id=people[-1])
            core._publish_json(directory/"complete.json",value,budget)
            core._publish_json(directory/"operational.json",dict(status=value["status"],settings_id=value["settings_id"]),budget)
            report=core.finalize(root,budget=budget)
            self.assertEqual(report["participant_count"],30)
            final=json.loads((root/"core_reference.complete.json").read_text())
            self.assertFalse(final["scientific_release_authorized"])
            self.assertEqual(final["search_credit"],0)
            self.assertEqual(final["delta_C"],max(.001,.05*final["G"]))
            original=(root/"core_reference.complete.json").read_bytes()
            with self.assertRaises(FileExistsError):
                core.finalize(root,budget=budget)
            self.assertEqual((root/"core_reference.complete.json").read_bytes(),original)


if __name__ == "__main__":
    unittest.main()
