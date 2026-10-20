"""Small algebra and frozen-fit checks; run only in a coordinated Slurm job."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import repeated_image_control as control
import repeated_signal_diagnostic as diagnostic
from continuous_tuning import RIDGE_GRID
from core_reference_direct_ridge import DirectCoreRidge
from test_repeated_image_control import recipe, tiny_sequences
from training_transforms import CommonSensorSpace


def fixture():
    """Actual small ridge algebra, not a simulated scientific population."""
    sequences, designs = tiny_sequences()
    for item in sequences:
        item.sample_times_seconds = np.arange(24) / 25.
    person = SimpleNamespace(participant_id="sub-fixture", sequences=sequences,
        image_ids=tuple("image%03d" % index for index in range(200)))
    binding = {"schema": control.SCHEMA, "participant_id": person.participant_id,
        "data_role": "development", "recipe": recipe(), "identity_recipe": control.IDENTITY_RECIPE,
        "image_ids": list(person.image_ids), "channel_order": list(sequences[0].channel_order),
        "sequence_support": [{"sequence_id": item.sequence_id, "ordinal": item.ordinal,
            "samples": 24, "retained_samples": int(np.count_nonzero(item.retained_mask))}
            for item in sequences]}
    sensor = CommonSensorSpace(sequences[0].channel_order)
    backend, records = DirectCoreRidge(check=lambda: None), []
    for parity, direction in enumerate(control.DIRECTIONS):
        training, heldout = sequences[parity::2], sequences[1-parity::2]
        state = control._prepare_training(training, designs, sensor, recipe(), backend, lambda: None)
        index = 5 + parity
        fit = control._identity_fit(state, index, recipe(), backend, lambda: None)
        nuisance = control._score(heldout, designs, sensor, state, "nuisance",
                                  state.nuisance_fit.coefficients, backend, lambda: None)
        raw = control._score(heldout, designs, sensor, state, "identity",
                             fit.coefficients, backend, lambda: None)
        means = np.full(13, 2.); means[index] = 1.
        records.append({"status": "complete_direction", "binding": binding, "direction": direction,
            "training_sequence_ids": [item.sequence_id for item in training],
            "heldout_sequence_ids": [item.sequence_id for item in heldout],
            "inner_validations_completed": 6, "inner_mean_normalized_identity_losses": means.tolist(),
            "selected_penalty_index": index, "selected_ridge_penalty": float(RIDGE_GRID[index]),
            "identity_sequence_losses": raw.tolist(), "nuisance_sequence_losses": nuisance.tolist(),
            "gain": float(1 - control._normal(raw, nuisance)),
            "training_whitening": {"sequence_ids": list(state.sequence_ids),
                "shrinkage_strength": state.whitener.shrinkage_strength,
                "regularized_covariance_trace": float(np.trace(state.whitener.regularized_covariance))}})
        backend.clear()
    completed = {"schema": control.SCHEMA, "status": "complete_repeated_image_control",
        "participant_id": person.participant_id, "data_role": "development", "binding": binding,
        "gain": float(np.mean([record["gain"] for record in records])), "directions": records,
        "image_identity_count": 200, "physical_sequence_count": 12,
        "concept_or_false_models_computed": False, "audit_access": False}
    return person, designs, completed


def run_fixture(person, designs, completed):
    with patch.object(control, "_roster", return_value=person.sequences), \
         patch.object(control, "_sequence_designs", side_effect=lambda item: designs[item.sequence_id]):
        return diagnostic.run_participant(person, recipe=recipe(), completed_control=completed,
            budget_check=lambda: None, progress=lambda stage: None)


class RepeatedSignalDiagnosticTests(unittest.TestCase):
    def test_projection_on_irregular_retained_times_conserves_known_energy(self):
        times = (np.arange(45, dtype=float) / 37.)**2
        times = times[np.arange(45) % 7 != 0]
        basis = diagnostic._polynomial_basis(times)
        slow = basis @ np.array([[1., 2.], [3., -1.], [.2, .7], [4., -3.]])
        fast = np.column_stack((np.sin(times * 17), np.cos(times * 29)))
        fast -= basis @ (basis.T @ fast)
        energies = diagnostic._energies(slow + fast, basis)
        self.assertAlmostEqual(energies["polynomial"], float(np.mean(slow**2)), places=13)
        self.assertAlmostEqual(energies["orthogonal"], float(np.mean(fast**2)), places=13)
        self.assertAlmostEqual(energies["total"], energies["polynomial"] + energies["orthogonal"], places=13)
        self.assertLess(diagnostic._energies(slow, basis)["orthogonal"], 1e-27)
        self.assertLess(diagnostic._energies(fast, basis)["polynomial"], 1e-27)

    def test_original_total_denominator_and_equal_sequence_weighting_are_preserved(self):
        totals = np.array([2., 4., 6., 8., 10., 12.])
        polynomials = np.array([.2, 3., 1., 7., 3., 9.])
        rows = [{"sequence_id": str(i), "retained_samples": 20 + i,
            "nuisance": {"total": total, "polynomial": slow, "orthogonal": total-slow},
            "identity": {"total": total*.75, "polynomial": total*.1, "orthogonal": total*.65}}
            for i, (total, slow) in enumerate(zip(totals, polynomials))]
        original = {"direction": "odd_to_even", "training_sequence_ids": list("abcdef"),
            "heldout_sequence_ids": list("ghijkl"), "selected_penalty_index": 5,
            "selected_ridge_penalty": .1, "gain": .25,
            "identity_sequence_losses": (totals*.75).tolist(), "nuisance_sequence_losses": totals.tolist()}
        result = diagnostic._summarize_direction(rows, original)
        self.assertEqual(result["original_total_nuisance_denominator"], 7.)
        self.assertAlmostEqual(result["nuisance_polynomial_energy_fraction"], float(polynomials.mean()/totals.mean()))
        self.assertNotAlmostEqual(result["nuisance_polynomial_energy_fraction"], float(np.mean(polynomials/totals)))
        self.assertAlmostEqual(result["polynomial_gain_contribution"] + result["orthogonal_gain_contribution"], .25)
        broken = deepcopy(original); broken["identity_sequence_losses"][0] += .01
        with self.assertRaisesRegex(ValueError, "losses_do_not_reproduce"):
            diagnostic._summarize_direction(rows, broken)

    def test_exact_saved_outer_fits_reproduce_R_without_inner_retuning(self):
        person, designs, completed = fixture()
        original_prepare, original_identity = control._prepare_training, control._identity_fit
        seen_training, selected = [], []
        def prepare(items, *args):
            seen_training.append(tuple(item.sequence_id for item in items))
            return original_prepare(items, *args)
        def identity(state, index, *args):
            selected.append(index)
            return original_identity(state, index, *args)
        with patch.object(control, "_prepare_training", side_effect=prepare), \
             patch.object(control, "_identity_fit", side_effect=identity), \
             patch.object(control, "_fit_direction", side_effect=AssertionError("inner retuning forbidden")), \
             patch.object(control, "_load_inner", side_effect=AssertionError("inner access forbidden")):
            result = run_fixture(person, designs, completed)
        self.assertEqual(selected, [5, 6])
        self.assertEqual(seen_training, [tuple(item.sequence_id for item in person.sequences[::2]),
                                         tuple(item.sequence_id for item in person.sequences[1::2])])
        self.assertEqual(result["status"], "complete_repeated_signal_diagnostic")
        self.assertFalse(result["inner_retuning_performed"])
        self.assertFalse(result["primary_R_changed"])
        self.assertAlmostEqual(result["gain_recomputed_R"], completed["gain"], places=12)
        self.assertAlmostEqual(result["polynomial_gain_contribution"] + result["orthogonal_gain_contribution"],
                               completed["gain"], places=12)
        self.assertEqual(len(result["directions"]), 2)
        for direction, original in zip(result["directions"], completed["directions"]):
            self.assertEqual(len(direction["sequence_rows"]), 6)
            self.assertEqual(direction["training_whitening"], original["training_whitening"])

    def test_missing_direction_changed_saved_penalty_or_changed_EEG_is_rejected(self):
        person, designs, completed = fixture()
        broken = deepcopy(completed); broken["directions"] = broken["directions"][:1]
        with self.assertRaisesRegex(ValueError, "both_completed_original"):
            run_fixture(person, designs, broken)
        broken = deepcopy(completed); broken["directions"][0]["selected_penalty_index"] = 4
        with self.assertRaisesRegex(ValueError, "saved_selection"):
            run_fixture(person, designs, broken)
        # Preserve the layout while changing EEG: reuse must fail its actual
        # loss reproduction rather than accept an unchanged metadata binding.
        person.sequences[1].sensor_samples = person.sequences[1].sensor_samples * 1.1
        with self.assertRaisesRegex(ValueError, "losses_do_not_reproduce"):
            run_fixture(person, designs, completed)


if __name__ == "__main__":
    unittest.main()
