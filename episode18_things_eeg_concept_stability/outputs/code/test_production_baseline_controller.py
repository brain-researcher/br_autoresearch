"""Two composition checks on existing small fixtures, never empirical inputs.

The coordinator runs these in compute. The complete 72-sequence fixture and
independent dense solve are reused; no recovery simulation or new campaign.
The plan below is invented test metadata, never a selected scientific recipe.
"""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
import production_baseline_controller as controller
from streamed_tuning_adapter import StreamedResources
import test_continuous_tuning as prior
from test_production_continuous_designs import recipe as invented_recipe
from training_transforms import CORE_IMAGE, EXPANDED_IMAGE, LABEL


class FixtureBudget:
    """Only fixture metadata writes; not an empirical resource admission."""
    record = 16*1024**2
    durable = 64*1024**2
    written = 0

    def check(self):
        pass

    def require_compute_threads(self):
        pass

    def admit_write(self,amount):
        if self.written + amount > self.durable:
            raise MemoryError("fixture_output_cap")


def plan_fixture(release):
    base = invented_recipe()
    base["label_ranks"] = {name:64 for name in LABEL}
    core = dict(base,recipe_id="fixture_core",image_panel="core_image",
        image_ranks={name:64 for name in CORE_IMAGE},temporal_basis="raised_cosine_12")
    # Every one of four temporal levels appears three times in the first12;
    # image rank and bandwidth each appear four times. The next24 are distinct
    # one-bandwidth challenges to an earlier complete coverage tuple.
    coverage = [(temporal,rank,(temporal+rank)%3) for temporal in range(4) for rank in range(3)]
    remainder = [value for value in itertools.product(range(4),range(3),range(3)) if value not in coverage]
    bases = tuple(controller.LEVELS["temporal_basis"])
    ranks,bandwidths = (32,64,128),(.5,1.,2.)
    trials = []
    for index,(temporal,rank,bandwidth) in enumerate(coverage+remainder):
        changed = index >= len(coverage)
        reference = coverage.index((temporal,rank,(temporal+rank)%3))+1
        recipe = dict(base,recipe_id=f"fixture_{index+1:02d}",image_panel="expanded_image",
            image_ranks={name:ranks[rank] for name in EXPANDED_IMAGE},
            temporal_basis=bases[temporal],label_kernel_bandwidth=bandwidths[bandwidth])
        trials.append(dict(trial_id=f"trial_{index+1:02d}",recipe=recipe,
            kind="falsifier" if changed else "ordinary",falsifier_of=f"trial_{reference:02d}" if changed else None,
            changed_factor="label_kernel_bandwidth" if changed else None))
    return dict(schema=controller.PLAN_SCHEMA,semantics=dict(controller.SEMANTICS),
        release_manifest=str(release),core_calibration_recipe=core,
        streamed_resources=dict(chunk_rows=2,qr_block_size=2,array_byte_cap=16*1024**2),trials=trials)


def completed(plan,trial,objective,people=()):
    return dict(status="complete_30_person_baseline_trial",plan_id=controller.digest(plan),
        trial_id=trial["trial_id"],recipe_id=controller.digest(trial["recipe"]),participant_count=30,
        participants=list(people),inner_splits=900,outer_splits=180,objective=objective,
        concept_or_false_scores_computed=False,audit_access=False)


def write(path,value):
    path.write_text(json.dumps(value),encoding="utf-8")


class ProductionBaselineControllerTests(unittest.TestCase):
    def test_all36_splits_match_original_baseline_and_core_stays_unranked(self):
        prepared,covariances,selections,inner,outer = [],[],[],[],[]
        recipe = prior.recipe(covariances,selections)
        person = SimpleNamespace(participant_id="fixture",sequences=prior.fixture(),
            inputs=SimpleNamespace(channel_order=prior.CHANNELS),prepare_training=prior.preparation(prepared))
        resources = StreamedResources(2,2,16*1024**2)
        # Exact streamed-vs-dense algebra is checked separately. Here the
        # dense oracle isolates controller composition and exposure/order.
        def capacity(cache,selected,check,*,resources):
            return tuning._capacity(cache,selected,check)
        with patch.object(controller,"make_tuning_recipe",return_value=recipe), \
             patch.object(controller,"solver_limits",return_value=prior.LIMITS), \
             patch.object(controller,"capacity_for_split",side_effect=capacity) as called, \
             patch.object(folds,"_fit",side_effect=prior.dense_fit) as fitted, \
             patch.object(tuning,"_score",wraps=tuning._score) as scored:
            actual = controller.baseline_participant(person,{},resources,check=lambda:None,
                retain_inner=inner.append,retain_outer=outer.append)
            self.assertEqual(called.call_count,36)
            self.assertEqual(fitted.call_count,30*29+6*5)
            self.assertTrue(all(call.args[1] in folds.BASELINE_MODELS for call in scored.call_args_list))
        self.assertEqual((len(prepared),len(set(prepared)),len(covariances),len(selections)),(36,36,36,6))
        self.assertEqual((actual.inner_splits,actual.outer_splits,len(inner),len(outer)),(30,6,30,6))
        for value in inner:
            self.assertEqual(value.raw_sequence_losses["image"].shape,(13,12))
            self.assertEqual(value.raw_sequence_losses["label_capacity_matched"].shape,(13,12))
            self.assertEqual(set(value.raw_sequence_losses),set(folds.BASELINE_MODELS))
        with patch.object(folds,"_fit",side_effect=prior.dense_fit):
            expected = tuning.run_tuned_participant(participant_id="fixture",sequences=prior.fixture(),
                frozen_channel_order=prior.CHANNELS,phase="baseline",recipe=prior.recipe([],[]),
                prepare_training=prior.preparation([]),solver_limits=prior.LIMITS,
                retain_fit=lambda value:None,budget_check=lambda:None)
        for name in folds.BASELINE_MODELS:
            self.assertEqual(actual.normalized_participant_losses[name],expected.normalized_participant_losses[name])
        self.assertEqual([value.result.choice for value in outer],[value.choice for value in expected.outer])
        self.assertEqual([value.result.match for value in outer],[value.match for value in expected.outer])
        core_inner,core_outer = [],[]
        with patch.object(controller,"make_tuning_recipe",return_value=prior.recipe([],[])), \
             patch.object(controller,"solver_limits",return_value=prior.LIMITS), \
             patch.object(controller,"capacity_for_split",side_effect=AssertionError("core cannot require added-slot EDF")), \
             patch.object(folds,"_fit",side_effect=prior.dense_fit) as fitted, \
             patch.object(tuning,"_score",wraps=tuning._score) as scored:
            core = controller.baseline_participant(person,{},resources,check=lambda:None,
                retain_inner=core_inner.append,retain_outer=core_outer.append,core_reference=True)
            self.assertEqual(fitted.call_count,30*14+6*3)
            self.assertTrue(all(call.args[1] in controller.CORE_MODELS for call in scored.call_args_list))
        self.assertEqual((core.phase,core.inner_splits,core.outer_splits),("core_reference",30,6))
        self.assertEqual(set(core.normalized_participant_losses),set(controller.CORE_MODELS))
        for index,value in enumerate(core_outer):
            mean = np.mean([entry.normalized_image_losses for entry in core_inner[index*5:(index+1)*5]],axis=0)
            choice = min(range(13),key=lambda i:(mean[i],-tuning.RIDGE_GRID[i],i))
            self.assertEqual(value.result.image_index,choice)
            self.assertEqual(set(value.fits),set(controller.CORE_MODELS))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/"inner.json"
            controller._split_record(target,inner[0],FixtureBudget())
            saved = json.loads(target.read_text())
            reference = saved["record"]["raw_sequence_losses"]["image"]["array"]
            with np.load(target.with_suffix(".npz"),allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays[reference],inner[0].raw_sequence_losses["image"])
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                controller._split_record(target,inner[0],FixtureBudget())
            self.assertEqual(target.read_bytes(),original)

    def test_complete_cohort_order_search_minima_and_immutable_partial_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            people = [f"sub-{i:02d}" for i in range(1,31)]
            development,release,planfile = root/"development.json",root/"release.json",root/"plan.json"
            write(development,dict(role="development",participants=people))
            write(release,dict(development_manifest=str(development)))
            plan = controller.validate_plan(plan_fixture(release))
            write(planfile,plan)
            constant = [completed(plan,trial,1.) for trial in plan["trials"]]
            self.assertFalse(controller.search_status(plan,constant[:21])["search_can_freeze"])
            status = controller.search_status(plan,constant[:22])
            self.assertEqual((status["coverage_at"],status["postcoverage_trials"],status["postcoverage_falsifiers"]),(12,10,10))
            self.assertTrue(status["search_can_freeze"])
            self.assertEqual(status["selected_trial_id"],"trial_01")
            improving = [completed(plan,trial,float(36-i)) for i,trial in enumerate(plan["trials"])]
            unfinished = controller.search_status(plan,improving)
            self.assertTrue(unfinished["plan_incomplete"])
            self.assertFalse(unfinished["search_can_freeze"])
            changed = deepcopy(plan)
            changed["trials"][13]["recipe"] = dict(changed["trials"][12]["recipe"],recipe_id="not_a_new_trial")
            with self.assertRaisesRegex(ValueError,"duplicate_recipe"):
                controller.validate_plan(changed)
            changed = deepcopy(plan)
            changed["trials"][12]["changed_factor"] = "image_rank"
            with self.assertRaisesRegex(ValueError,"exactly_one_registered_factor"):
                controller.validate_plan(changed)
            search,budget = root/"search",FixtureBudget()
            controller.initialize_search(planfile,search,budget=budget)
            with self.assertRaises(ValueError):
                controller._next_trial(search,plan,"trial_01",budget)
            core_recipe = controller.digest(plan["core_calibration_recipe"])
            for person in people:
                directory = search/(controller.CORE_ID+"__"+person)
                directory.mkdir()
                record = dict(status="complete_participant_core_calibration",plan_id=controller.digest(plan),
                    trial_id=controller.CORE_ID,recipe_id=core_recipe,participant_id=person,
                    inner_splits=30,outer_splits=6,participant_losses=dict(nuisance=1.,reduced_task_nuisance=1.1,image=.8),
                    concept_or_false_scores_computed=False,audit_access=False)
                if person != people[-1]:
                    write(directory/"complete.json",record)
            with self.assertRaises(ValueError):
                controller.finish_trial(search,controller.CORE_ID,budget=budget)
            self.assertFalse((search/(controller.CORE_ID+".complete.json")).exists())
            write(directory/"complete.json",record)
            core_status = controller.finish_trial(search,controller.CORE_ID,budget=budget)
            self.assertEqual(core_status["valid_trials"],0)
            core_path = search/(controller.CORE_ID+".complete.json")
            before = core_path.read_bytes()
            core = json.loads(before)
            self.assertAlmostEqual(core["meaningful_margins"]["G"],.2)
            self.assertAlmostEqual(core["meaningful_margins"]["T"],.1)
            self.assertEqual(core["valid_trial_credit"],0)
            with self.assertRaises(ValueError):
                controller.finish_trial(search,controller.CORE_ID,budget=budget)
            self.assertEqual(before,core_path.read_bytes())
            with self.assertRaisesRegex(ValueError,"fixed_prospective_order"):
                controller._next_trial(search,plan,"trial_02",budget)
            # Trusted complete-cohort markers exercise the persistence state
            # machine; the separate numerical method above produces real fits.
            for trial in plan["trials"][:22]:
                write(search/(trial["trial_id"]+".complete.json"),completed(plan,trial,1.,people))
            frozen = controller.freeze_baseline(search,budget=budget)
            self.assertEqual(frozen["selected_trial_id"],"trial_01")
            fixed = json.loads((search/"baseline_fixed.json").read_text())
            self.assertFalse(fixed["scientific_release_authorized"])
            self.assertFalse(fixed["concept_or_false_scores_computed"])
            self.assertEqual(fixed["core_calibration_digest"],controller.digest(core))
            with self.assertRaisesRegex(ValueError,"baseline_already_fixed"):
                controller._settings(search,budget)


if __name__ == "__main__":
    unittest.main()
