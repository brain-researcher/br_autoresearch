"""Two original-serializer continuation checks; coordinator compute only.

The existing tiny 72-sequence/61-contrast dense oracle supplies all36 original
splits once. No empirical inputs, resource selection or scientific recovery
campaign. The serialized five-family PCA fixture has one effective component,
matching the existing fixture's 100-column image design exactly.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import continuous_fold_runner as folds
import core_reference_split_resume as resume
import production_core_reference as core
import test_continuous_tuning as prior
from test_core_reference_paired_inner import person_fixture
from test_production_baseline_controller import FixtureBudget
from test_production_core_reference import recipe
from training_transforms import CORE_IMAGE, CommonSensorSpace, FeaturePanelTransform, fit_feature_transform


def initialized_fixture(root):
    selected = recipe()
    person = person_fixture([])
    settings = dict(schema=core.SCHEMA, recipe=selected, recipe_id=core.digest(selected),
                    participants=[person.participant_id])
    person.inputs.root, person.inputs.settings = root, settings
    directory = root/person.participant_id
    directory.mkdir()
    budget = FixtureBudget()
    core._publish_json(root/"settings.json", settings, budget)
    core._publish_json(root/"inputs_complete.json", dict(status="complete_core_reference_inputs",
        settings_id=core.digest(settings), participants={person.participant_id:{}}), budget)
    core._publish_json(directory/"settings.json", dict(status="incomplete_core_reference_person",
        participant_id=person.participant_id, settings_id=core.digest(settings), cost_only=True,
        scientific_release_authorized=False, audit_access=False), budget)
    return person, directory, budget


def checkpoint_outer(directory, fitted, budget):
    transforms = tuple(fit_feature_transform(
        np.asarray([[0.],[1.],[2.]]) if index == 0 else np.zeros((3,1)),
        training_unit_ids=("fixture_0","fixture_1","fixture_2"), requested_rank=64)
        for index in range(5))
    panel = FeaturePanelTransform("core_image", CORE_IMAGE, transforms)
    # This builder is retained metadata only; never used to build or fit a
    # design. The already-computed dense fixture has precisely one image PC.
    builder = core.CoreDesignBuilder("fixture", None, None, panel, None, {}, lambda:None)
    prepared = replace(fitted.preparation, build_designs=builder)
    core._save_outer(directory, replace(fitted, preparation=prepared), budget)


class CoreSplitResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inner, cls.outer = [], []
        with patch.object(folds, "_fit", side_effect=prior.dense_fit) as fitted:
            cls.original = core.fit_person(person_fixture([]), recipe(), check=lambda:None,
                retain_inner=cls.inner.append, retain_outer=cls.outer.append,
                progress=dict(inner_splits_completed=0, outer_splits_completed=0))
        if fitted.call_count != 438:
            raise AssertionError("Complete original oracle required")

    def test_interrupted_original_checkpoints_resume_exactly_without_rewriting_predecessor(self):
        missing = {(0,1),(1,0),(2,3)}
        with tempfile.TemporaryDirectory() as temporary:
            person, directory, budget = initialized_fixture(Path(temporary))
            for row in self.inner:
                key = (row.split.outer_index,row.split.inner_index)
                path = directory/("inner_%02d_%02d.json" % key)
                if key not in missing or key == (0,1):
                    core._split_record(path,row,budget)
            # A killed JSON publication leaves a complete but unauthoritative
            # NPZ. The resumed work must neither adopt nor overwrite it.
            (directory/"inner_00_01.json").unlink()
            (directory/"inner_01_00.npz.partial").write_bytes(b"unfinished fixture")
            for row in self.outer[3:]:
                checkpoint_outer(directory,row,budget)
            original_bytes = {path.name:path.read_bytes() for path in directory.iterdir()}
            new_inner, new_outer, progress = [], [], {}
            with patch.object(folds, "_fit", side_effect=prior.dense_fit) as fitted:
                actual = resume.fit_person(person,recipe(),predecessor_directory=directory,
                    initialized_settings_id=core.digest(person.inputs.settings),
                    metadata_byte_cap=16*1024**2,array_byte_cap=16*1024**2,
                    check=lambda:None,retain_inner=new_inner.append,retain_outer=new_outer.append,
                    progress=progress)
            self.assertEqual(actual,self.original)
            self.assertEqual(fitted.call_count,37)  # Two training groups + three outer refits.
            self.assertEqual(progress["actual_new_multivariate_fits"],37)
            self.assertEqual(progress["original_logical_multivariate_fits"],438)
            self.assertEqual((progress["reused_inner_splits_completed"],progress["reused_outer_splits_completed"]),(27,3))
            self.assertEqual((progress["new_inner_splits_completed"],progress["new_outer_splits_completed"]),(3,3))
            self.assertEqual((progress["inner_splits_completed"],progress["outer_splits_completed"]),(30,6))
            self.assertEqual({(row.split.outer_index,row.split.inner_index) for row in new_inner},missing)
            self.assertEqual([row.result.split.outer_index for row in new_outer],[0,1,2])
            expected_inner = {(row.split.outer_index,row.split.inner_index):row for row in self.inner}
            for row in new_inner:
                expected = expected_inner[(row.split.outer_index,row.split.inner_index)]
                np.testing.assert_array_equal(row.image_sequence_losses,expected.image_sequence_losses)
                np.testing.assert_array_equal(row.nuisance_sequence_losses,expected.nuisance_sequence_losses)
            for row in new_outer:
                expected = self.outer[row.result.split.outer_index]
                for name in core.CORE_MODELS:
                    np.testing.assert_array_equal(row.fits[name].coefficients,expected.fits[name].coefficients)
            self.assertEqual({path.name:path.read_bytes() for path in directory.iterdir()},original_bytes)
            self.assertEqual(len(progress["reused_inner_records"]),27)
            self.assertEqual(len(progress["reused_outer_records"]),3)
            self.assertTrue(all(Path(row["array_path"]).parent == directory
                for row in progress["reused_inner_records"]+progress["reused_outer_records"]))

    def test_inconsistent_outer_identity_or_array_descriptor_refused_before_any_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            person, directory, budget = initialized_fixture(Path(temporary))
            kwargs = dict(predecessor_directory=directory,
                initialized_settings_id=core.digest(person.inputs.settings),
                metadata_byte_cap=16*1024**2,array_byte_cap=16*1024**2,
                sequences=folds._roster(person.sequences,CommonSensorSpace(prior.CHANNELS),lambda:None),
                sensor=CommonSensorSpace(prior.CHANNELS),check=lambda:None)
            first = self.inner[0]
            path = directory/"inner_00_01.json"
            core._split_record(path,first,budget)
            checkpoint_outer(directory,self.outer[0],budget)
            with patch.object(folds,"_fit",side_effect=AssertionError("No fitting during checkpoint admission")):
                with self.assertRaisesRegex(ValueError,"completed_outer_requires_all_five_predecessor_inner_records"):
                    resume.load_completed_splits(person,recipe(),**kwargs)
                # Complete all five predecessors before validating a saved
                # outer's tuple fields, coefficient widths and choice.
                for row in self.inner[1:5]:
                    core._split_record(directory/("inner_%02d_%02d.json" %
                        (row.split.outer_index,row.split.inner_index)),row,budget)
                inner,outer,_,_ = resume.load_completed_splits(person,recipe(),**kwargs)
                self.assertEqual((len(inner),len(outer)),(5,1))
                self.assertEqual(outer[0],self.original["outer"][0])
                original = path.read_bytes()
                metadata = json.loads(original)
                modifications = [
                    lambda value:value.update(array_file="unselected.npz"),
                    lambda value:value.update(array_bytes=value["array_bytes"]+1),
                    lambda value:value["record"]["image_sequence_losses"].update(shape=[13,11]),
                    lambda value:value["record"]["image_sequence_losses"].update(dtype="<f4"),
                ]
                for change in modifications:
                    value = json.loads(original)
                    change(value)
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        resume.load_completed_splits(person,recipe(),**kwargs)
                    path.write_bytes(original)
                # A declared final JSON with missing NPZ cannot be counted as
                # an interrupted unpublished split or silently recomputed.
                archive = path.with_suffix(".npz")
                stored = archive.read_bytes()
                archive.unlink()
                with self.assertRaisesRegex(ValueError,"complete_exact_split_archive_required"):
                    resume.load_completed_splits(person,recipe(),**kwargs)
                archive.write_bytes(stored)
                with self.assertRaisesRegex(ValueError,"original_initialized_settings_person_recipe_binding_differs"):
                    resume.load_completed_splits(person,recipe(),**dict(kwargs,initialized_settings_id="wrong"))
                with self.assertRaisesRegex(ValueError,"explicit_original_root_person_predecessor_required"):
                    resume.load_completed_splits(person,recipe(),**dict(kwargs,predecessor_directory=directory/"other"))
                # Outer choice is an exact consequence of ordered inner
                # curves, never trusted merely because the archive is complete.
                outer_path = directory/"outer_00.json"
                value = json.loads(outer_path.read_text())
                value["record"]["result"]["image_index"] = (value["record"]["result"]["image_index"]+1)%13
                outer_path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,"saved_outer_choice_differs_from_original_inner_selection"):
                    resume.load_completed_splits(person,recipe(),**kwargs)


if __name__ == "__main__":
    unittest.main()
