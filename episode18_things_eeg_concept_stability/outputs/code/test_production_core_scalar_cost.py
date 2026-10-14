"""Two focused first-grid orchestration checks using existing small fixtures."""
from dataclasses import asdict, replace
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
import core_reference_paired_inner as paired
import production_core_reference as core
import production_core_scalar_cost as cost
from test_core_reference_paired_inner import person_fixture
from test_production_baseline_controller import FixtureBudget
from test_production_core_reference import recipe
from test_continuous_tuning import dense_fit
from training_transforms import CommonSensorSpace


class CoreScalarCostTests(unittest.TestCase):
    def test_complete_original_grid_before_either_validation_side_and_identical_private_results(self):
        admission = FixtureBudget()
        admission.threads = 1
        admission.deadline = 1791850000+7140
        environment = {"SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "15360",
                       "SLURM_JOB_START_TIME": "1791850000", "SLURM_JOB_END_TIME": "1791857203"}
        with patch.dict(os.environ, environment, clear=True), patch.object(cost.time, "time", return_value=1791850060):
            cost._allocation(admission, 4)  # Realistic Slurm actual interval has three extra seconds.
            admission.deadline += 1
            with self.assertRaisesRegex(ValueError, "unchanged_episode_and_two_hour"):
                cost._allocation(admission, 4)
        selected = recipe()
        person = person_fixture([])
        person.participant_id = "sub-07"
        sensor = CommonSensorSpace(person.inputs.channel_order)
        limits = folds.SolverLimits(**selected["solver"])
        convention = SimpleNamespace(normalization=selected["normalization"])
        check = lambda: None
        with patch.object(folds, "_fit", side_effect=dense_fit):
            sequences = folds._roster(person.sequences, sensor, check)
            first = core._prepare_core(sequences, folds._split(0,1), sensor, selected,
                person.prepare_training, limits, check, lambda value: None)
            second = paired._counterpart(first, sequences, sensor, convention, check)
            expected = paired._inner_pair(first, second, convention, limits, check)
        completed, scalar_events, image_scores = [], [], []
        prepared_id = cost._prepared_state_signature(first, sensor, check)
        score = tuning._score
        def checked_score(cache, model, fit, budget_check):
            if model == "image":
                self.assertEqual(completed, list(range(13)))
                image_scores.append((cache.split.outer_index, cache.split.inner_index))
            return score(cache, model, fit, budget_check)
        def fitted(design, targets, weights, penalties, actual_limits, actual_check,
                   *, checkpoint, binding, workers, record_byte_cap, progress):
            index = len(completed)
            self.assertEqual(workers, 4)
            self.assertEqual(actual_limits, limits)
            self.assertEqual(binding, {"settings_id": "fixture_settings", "recipe_id": core.digest(selected),
                "participant_id": "sub-07", "split": asdict(folds._split(0,1)),
                "model": "image", "penalty_index": index, "ridge": float(tuning.RIDGE_GRID[index]),
                "prepared_state_sha256": prepared_id})
            self.assertEqual(checkpoint.name, "image_penalty_%02d.npz" % index)
            self.assertLessEqual(record_byte_cap, selected["npz_byte_cap"])
            self.assertGreater(record_byte_cap, design.shape[1]*61*8)
            np.testing.assert_array_equal(penalties, tuning._grid_penalty(first, "image", index))
            answer = dense_fit(design, targets, weights, penalties, actual_limits, actual_check)
            completed.append(index)
            progress({"column": 60, "completed_scalars": 61, "required_scalars": 61,
                      "stop_code": 2, "iterations": 1, "elapsed_seconds": .01})
            return answer
        with tempfile.TemporaryDirectory() as temporary, patch.object(folds, "_fit", side_effect=dense_fit), \
                patch.object(tuning, "_score", side_effect=checked_score):
            private = Path(temporary)
            actual = cost.fit_first_pair(person, selected, private_dir=private, budget=FixtureBudget(),
                workers=4, settings_id="fixture_settings", on_stage=lambda value: None,
                on_scalar=lambda index, row: scalar_events.append((index, row)), scalar_fit=fitted)
            self.assertEqual(completed, list(range(13)))
            self.assertEqual(len(scalar_events), 13)
            self.assertEqual(image_scores, [(0,1)]*13+[(1,0)]*13)
            for before, after in zip(expected, actual):
                self.assertEqual(after.split, before.split)
                self.assertEqual(after.training_sequence_ids, before.training_sequence_ids)
                self.assertEqual(after.heldout_sequence_ids, before.heldout_sequence_ids)
                np.testing.assert_array_equal(after.image_sequence_losses, before.image_sequence_losses)
                np.testing.assert_array_equal(after.nuisance_sequence_losses, before.nuisance_sequence_losses)
                np.testing.assert_array_equal(after.normalized_image_losses, before.normalized_image_losses)
                self.assertTrue((private / ("inner_%02d_%02d.json" %
                    (after.split.outer_index, after.split.inner_index))).exists())
                with patch.object(cost.controller, "_split_record") as no_overwrite:
                    cost._retain_inner(private, after, FixtureBudget())
                no_overwrite.assert_not_called()
            tiny = FixtureBudget()
            tiny.durable = 1048576
            with patch.object(cost, "_checkpoint_fit") as no_solve, \
                    self.assertRaisesRegex(MemoryError, "fixture_output_cap"):
                cost.fit_first_pair(person, selected, private_dir=private, budget=tiny,
                    workers=4, settings_id="fixture_settings", on_stage=lambda value: None,
                    on_scalar=lambda index, row: None)
            no_solve.assert_not_called()

    def test_explicit_driver_continuation_reuses_completed_scalars_and_keeps_closed_scores_private(self):
        import core_reference_scalar_checkpoint as checkpoint
        with patch("builtins.open", side_effect=[FileNotFoundError(),
                   BytesIO(b"Pss: 2 kB\nRss: 900 kB\nPss: 3 kB\n")]):
            self.assertEqual(cost._owned_process_pss(os.getpid()), 5*1024)
        with patch("builtins.open", side_effect=[FileNotFoundError(),
                   BytesIO(b"x"*(16*1024**2+1))]), \
                self.assertRaisesRegex(ValueError, "owned_process_PSS_record_byte_cap"):
            cost._owned_process_pss(os.getpid())
        selected = recipe()
        person = person_fixture([])
        person.participant_id = "sub-07"
        inputs = SimpleNamespace(people=("sub-07",), settings={"original": "fixture"}, recipe=selected,
            images=SimpleNamespace(check=None), load_participant=lambda name: person, check=None)
        solved, dense, current, interrupt = [], {}, [None], [True]
        score = tuning._score
        def checked_score(cache, model, fit, check):
            if model == "image":
                self.assertEqual(len(solved), 793)
            return score(cache, model, fit, check)
        def solved_column(column, design, targets, weights, penalties, solver):
            index = current[0]
            if index not in dense:
                dense[index] = dense_fit(design, targets, weights, penalties,
                                         folds.SolverLimits(**solver), lambda: None).coefficients
            solved.append((index, column))
            return column, dense[index][:, column], 2, 1, .01, 1048576
        def actual_checkpoint(design, targets, weights, penalties, limits, check, **options):
            current[0] = options["binding"]["penalty_index"]
            report = options["progress"]
            def progress(event):
                report({**event, "private_coefficients": "NEVER_PUBLIC"})
                if interrupt[0] and event.get("event") == "scalar_completed" and event["completed_scalars"] == 3:
                    raise TimeoutError("NEVER_PUBLIC_ERROR_OR_DATA")
            return checkpoint.fit_checkpointed(design, targets, weights, penalties, limits, check,
                                               **{**options, "progress": progress})
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(cost, "_allocation"), patch.object(core, "bind_initialized", return_value=inputs), \
                patch.object(folds, "_fit", side_effect=dense_fit), \
                patch.object(cost, "_checkpoint_fit", side_effect=actual_checkpoint), \
                patch.object(checkpoint, "_solve", side_effect=solved_column), \
                patch.object(tuning, "_score", side_effect=checked_score):
            parent = Path(temporary)
            root, private, public, resumed = parent/"old", parent/"private", parent/"public", parent/"resumed"
            root.mkdir()
            old = root / "unchanged_partial"
            old.write_bytes(b"old attempt preserved")
            budget = FixtureBudget()
            budget.deadline = 1791926990
            result = cost.run_cost(root, private, public, budget=budget, workers=1,
                                   aggregate_pss_limit_bytes=12*1024**3)
            self.assertEqual(result["status"], "incomplete_first_symmetric_core_grid")
            self.assertEqual(result["error_class"], "TimeoutError")
            self.assertEqual(result["completed_image_scalar_fits"], 3)
            self.assertEqual(result["completed_penalties"], 0)
            self.assertEqual(result["completed_image_validation_sides"], 0)
            self.assertFalse(result["scientific_release_authorized"])
            self.assertFalse((private/"inner_00_01.json").exists())
            self.assertTrue((private/"image_penalty_00.npz").exists())
            text = (public/"operational.json").read_text()+(public/"scalar_progress.jsonl").read_text()
            self.assertNotIn("NEVER_PUBLIC", text)
            self.assertNotIn("private_coefficients", text)
            self.assertEqual(json.loads((public/"operational.json").read_text()), result)
            self.assertEqual(old.read_bytes(), b"old attempt preserved")
            with self.assertRaisesRegex(ValueError, "explicit_checkpoint_continuation_or_fresh"):
                cost.run_cost(root, private, resumed, budget=budget, workers=1,
                              aggregate_pss_limit_bytes=12*1024**3)
            original_settings = (private/"settings.json").read_bytes()
            changed = json.loads(original_settings)
            changed["settings_id"] = "wrong original binding"
            (private/"settings.json").write_text(json.dumps(changed))
            with patch.object(inputs, "load_participant") as load:
                with self.assertRaisesRegex(ValueError, "original_checkpoint_stage_binding_changed"):
                    cost.run_cost(root, private, resumed, budget=budget, workers=1,
                                  aggregate_pss_limit_bytes=12*1024**3, resume_checkpoints=True)
                load.assert_not_called()
            (private/"settings.json").write_bytes(original_settings)
            original_sequences = person.sequences
            changed_sequences = []
            for row in original_sequences:
                if row.block == 2:
                    values = row.sensor_samples.copy()
                    values[0, 0] += .001
                    row = replace(row, sensor_samples=values)
                changed_sequences.append(row)
            person.sequences = tuple(changed_sequences)
            mismatch_budget = FixtureBudget()
            mismatch_budget.deadline = 1791926990
            refused = cost.run_cost(root, private, parent/"different_prepared_data", budget=mismatch_budget,
                workers=1, aggregate_pss_limit_bytes=12*1024**3, resume_checkpoints=True)
            self.assertEqual(refused["status"], "incomplete_first_symmetric_core_grid")
            self.assertEqual(refused["error_class"], "ValueError")
            self.assertEqual(len(solved), 3)
            person.sequences = original_sequences
            interrupt[0] = False
            resumed_budget = FixtureBudget()
            resumed_budget.deadline = 1791926990
            finished = cost.run_cost(root, private, resumed, budget=resumed_budget, workers=1,
                                    aggregate_pss_limit_bytes=12*1024**3, resume_checkpoints=True)
            self.assertEqual(finished["status"], "complete_first_symmetric_core_grid")
            self.assertEqual(finished["reused_image_scalar_fits"], 3)
            self.assertEqual((finished["completed_image_scalar_fits"], finished["completed_image_validation_sides"]), (793,2))
            self.assertEqual(len(solved), len(set(solved)))
            self.assertEqual(set(solved), {(i,j) for i in range(13) for j in range(61)})
            self.assertEqual((private/"settings.json").read_bytes(), original_settings)
            self.assertEqual((public/"operational.json").read_text()+(public/"scalar_progress.jsonl").read_text(), text)
            self.assertTrue((private/"complete.json").exists())
            with self.assertRaisesRegex(ValueError, "already_completed_grid_refuses_continuation"):
                cost.run_cost(root, private, parent/"redundant", budget=resumed_budget, workers=1,
                              aggregate_pss_limit_bytes=12*1024**3, resume_checkpoints=True)


if __name__ == "__main__":
    unittest.main()
