"""Focused array-only control checks; no empirical fitting or inference test."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from calibration_falsifiers import (
    TASK_A_HALF_RUNS, VARIANTS, TaskACalibrationPacket, TaskBObservation,
    condition_relabeling, exploratory_falsifier_points, fit_calibration_falsifiers,
    identity_pairings, relabel_task_a_packet, validate_task_a_packets,
)
from development_lopo import (
    DEVELOPMENT_IDS, M1_VARIANTS, M2_VARIANTS, M3_VARIANTS, M4_VARIANTS,
    CalibrationReference, DevelopmentParticipant, FoldContext, PairedIdentityContext,
    M1Prediction, M2Prediction, M3Prediction, M4Prediction, SharedM3Source,
    fit_calibration_panel, fit_development_lopo, validate_source_context,
)
from local_functional_source import LocalFunctionalSource, TASK_A_CONDITIONS
from registered_endpoint_metrics import (
    CONDITIONS, crossvalidated_geometries, geometry_covariance, map_covariance,
    project_gain_direction,
)


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError("forbidden array conversion")


class CalibrationFalsifierTests(unittest.TestCase):
    def setUp(self):
        self.voxels = tuple(range(208))
        x = np.arange(208, dtype=float)
        self.u = np.sin(2*np.pi*x/208)
        self.v = np.cos(2*np.pi*x/208)
        self.B = np.stack((self.u, self.v), axis=-1)/np.sqrt(104)
        self.mean = (2+np.arange(29)/30)[:, None]*self.u
        self.mean += (np.arange(29)/70)[:, None]*self.v
        self.noise = (np.arange(29)/90+.1)[:, None]*self.v
        self.packets = tuple(TaskACalibrationPacket(
            person, TASK_A_CONDITIONS, self.voxels,
            (1+index/25)*self.mean,
            np.stack(((1+index/25)*self.mean+self.noise,
                      (1+index/25)*self.mean-self.noise)), TASK_A_HALF_RUNS,
        ) for index, person in enumerate(DEVELOPMENT_IDS))
        pattern = np.arange(1, 19)[:, None]*self.u/18+self.v[None]/3
        self.records = tuple(DevelopmentParticipant(
            packet.participant_id, TASK_A_CONDITIONS, self.voxels, packet.task_a_maps,
            np.stack([(1+index/20+part/7)*pattern+(index+part**2)*self.v/40
                      for part in range(4)]),
        ) for index, packet in enumerate(self.packets))
        self.source_calls, self.calibration_calls, self.variant_calls = [], [], []

    def shared(self, training, context):
        validate_source_context(context)
        ids = tuple(record.participant_id for record in training)
        self.assertEqual(ids, context.training_ids)
        self.assertNotIn(context.target_id, ids)
        self.assertNotIn(context.deleted_id, ids)
        if isinstance(context, PairedIdentityContext):
            self.assertNotIn(context.donor_id, ids)
        self.source_calls.append(context)
        parts = np.mean([record.task_b_partitions for record in training], axis=0)
        return SharedM3Source(
            ids, TASK_A_CONDITIONS, self.voxels,
            np.mean([record.task_a_maps for record in training], axis=0),
            parts.mean(axis=0), crossvalidated_geometries(parts[None])[0].mean(axis=0),
            1., self.B, np.ones(29)/29, True, 1e-8, 1e-10, parts,
        )

    def calibrate(self, source, task_a, context):
        self.assertIsInstance(task_a, np.ndarray)
        self.assertEqual(task_a.shape, (29, 208))
        gain = float(np.sum(source.task_a_maps*task_a)/np.sum(source.task_a_maps**2))
        self.calibration_calls.append((id(source), context, gain))
        return {"O": np.eye(2), "gain": gain}

    def geometry(self, source, numerical, context):
        return numerical["gain"]**2*source.K0

    def variants(self, cls, names):
        def callback(source, task_a, calibration, context):
            gain = calibration["gain"] if isinstance(calibration, dict) else calibration
            self.variant_calls.append((id(source), context, gain, names))
            values = []
            for index, name in enumerate(names):
                maps = gain*(source.G-source.G.mean(axis=-1, keepdims=True))
                geometry = gain**2*source.K0
                # Cheap nonisometric stand-ins verify paired scoring/data flow,
                # not production M1/M2/M4 optimizer or certificate correctness.
                if cls is not M3Prediction:
                    offset = float(np.mean(task_a[:, :4]))*(index+1)/20
                    maps = maps+offset*self.v
                    geometry = geometry+offset*np.eye(18)
                values.append(cls(name, maps, geometry, {"common_gain": gain}))
            return tuple(values)
        return callback

    def callbacks(self):
        return dict(calibrate_m3=self.calibrate, certify_m3=lambda *args: None,
                    predict_m3_geometry=self.geometry, check=lambda: None,
                    predict_m1=self.variants(M1Prediction, M1_VARIANTS),
                    predict_m2=self.variants(M2Prediction, M2_VARIANTS),
                    predict_m3_variants=self.variants(M3Prediction, M3_VARIANTS),
                    predict_m4=self.variants(M4Prediction, M4_VARIANTS))

    def references(self, deleted_id=None):
        if deleted_id is None:
            run = fit_development_lopo(self.records, fit_shared=self.shared,
                                      include_participant_deletions=False, **self.callbacks()).full
            return tuple(fold.calibration_reference for fold in run.folds)
        references = []
        for packet in self.packets:
            if packet.participant_id == deleted_id:
                continue
            training = tuple(record for record in self.records
                             if record.participant_id not in (packet.participant_id, deleted_id))
            context = FoldContext(packet.participant_id, tuple(record.participant_id for record in training),
                                  deleted_id)
            source = self.shared(training, context)
            panel = fit_calibration_panel(source, packet.task_a_maps, context, **self.callbacks())
            references.append(CalibrationReference(context, source, panel, packet.task_a_maps))
        return tuple(references)

    def fit(self, references, deleted_id=None, packets=None, records=None, **overrides):
        callbacks = self.callbacks()
        callbacks.update(overrides)
        return fit_calibration_falsifiers(
            self.packets if packets is None else packets,
            self.records if records is None else records, references,
            deleted_id=deleted_id, fit_shared=self.shared, **callbacks,
        )

    def observations(self, deleted_id=None):
        return tuple(TaskBObservation(record.participant_id, CONDITIONS, self.voxels, record.task_b_partitions)
                     for record in self.records if record.participant_id != deleted_id)

    def test_packet_axes_and_real_half_mean_refuse_before_conversion(self):
        poisoned = tuple(replace(packet, task_a_maps=Poison(), task_a_halves=Poison())
                         for packet in self.packets)
        for changed in (
                (replace(poisoned[0], participant_id="sub-01"),)+poisoned[1:],
                poisoned[:-1]+(replace(poisoned[-1], voxel_ids=tuple(reversed(self.voxels))),),
                poisoned[:-1]+(replace(poisoned[-1], task_a_condition_ids=tuple(reversed(TASK_A_CONDITIONS))),),
                poisoned[:-1]+(replace(poisoned[-1], half_run_ids=(tuple(range(1, 9)),)*2),)):
            with self.assertRaises(ValueError):
                validate_task_a_packets(changed, DEVELOPMENT_IDS, self.voxels, numerical_atol=1e-10)
        validated = validate_task_a_packets(self.packets, DEVELOPMENT_IDS, self.voxels, numerical_atol=1e-10)
        self.assertFalse(validated[0].task_a_maps.flags.writeable)
        inconsistent = (replace(self.packets[0], task_a_halves=self.packets[0].task_a_halves+1),)+self.packets[1:]
        with self.assertRaisesRegex(ValueError, "arithmetic mean"):
            validate_task_a_packets(inconsistent, DEVELOPMENT_IDS, self.voxels, numerical_atol=1e-10)
        self.assertFalse(hasattr(validated[0], "task_b_partitions"))

    def test_source_reliability_strata_fixed_rest_and_mean_halves_move_together(self):
        context = FoldContext(DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[1:], None)
        mapping, info = condition_relabeling(self.packets[1:], context, self.voxels, numerical_atol=1e-10)
        again, _ = condition_relabeling(self.packets[1:], context, self.voxels, numerical_atol=1e-10)
        self.assertEqual(mapping, again)
        self.assertEqual(mapping[-1], 28)
        self.assertTrue(all(index != moved for index, moved in enumerate(mapping[:-1])))
        self.assertEqual(tuple(sorted(mapping)), tuple(range(29)))
        for members in info["strata"]:
            self.assertEqual(len(members), 7)
            indices = {TASK_A_CONDITIONS.index(name) for name in members}
            self.assertEqual({mapping[index] for index in indices}, indices)
        expected = []
        for packet in self.packets[1:]:
            expected.append([np.corrcoef(packet.task_a_halves[0, row], packet.task_a_halves[1, row])[0, 1]
                             for row in range(29)])
        np.testing.assert_allclose(info["source_median_half_correlations"][:-1],
                                   np.median(expected, axis=0)[:-1], atol=1e-14)
        self.assertIsNone(info["source_median_half_correlations"][-1])
        relabeled = relabel_task_a_packet(self.packets[0], mapping)
        self.assertEqual(relabeled.task_a_condition_ids, TASK_A_CONDITIONS)
        np.testing.assert_array_equal(relabeled.task_a_maps, self.packets[0].task_a_maps[list(mapping)])
        np.testing.assert_array_equal(relabeled.task_a_halves, self.packets[0].task_a_halves[:, list(mapping)])
        constant_rest = []
        for packet in self.packets[1:]:
            mean, halves = packet.task_a_maps.copy(), packet.task_a_halves.copy()
            mean[-1] = 0.
            halves[:, -1] = 0.
            constant_rest.append(replace(packet, task_a_maps=mean, task_a_halves=halves))
        constant_mapping, constant_info = condition_relabeling(
            constant_rest, context, self.voxels, numerical_atol=1e-10)
        self.assertEqual(mapping, constant_mapping)
        self.assertEqual(constant_info["rest_reliability"], "not_used")
        self.assertIsNone(constant_info["source_median_half_correlations"][-1])
        bad = replace(self.packets[1], task_a_maps=np.ones((29, 208)), task_a_halves=np.ones((2, 29, 208)))
        with self.assertRaisesRegex(ValueError, "undefined source"):
            condition_relabeling((bad,)+self.packets[2:], context, self.voxels, numerical_atol=1e-10)
        # Excluded target payload is refused by identity before its conversion.
        with self.assertRaisesRegex(ValueError, "participant identities"):
            condition_relabeling((replace(self.packets[0], task_a_maps=Poison()),)+self.packets[2:],
                                 context, self.voxels, numerical_atol=1e-10)

    def test_identity_cycle_deletion_and_source_exclusions_are_explicit(self):
        pairs = dict(identity_pairings())
        self.assertEqual(set(pairs), set(DEVELOPMENT_IDS))
        self.assertEqual(set(pairs.values()), set(DEVELOPMENT_IDS))
        self.assertTrue(all(person != donor for person, donor in pairs.items()))
        for deleted in DEVELOPMENT_IDS:
            nested = dict(identity_pairings(deleted))
            self.assertNotIn(deleted, nested)
            self.assertNotIn(deleted, nested.values())
            for recipient, donor in nested.items():
                self.assertEqual(donor, pairs[pairs[recipient]] if pairs[recipient] == deleted else pairs[recipient])
        recipient = DEVELOPMENT_IDS[0]
        donor = pairs[recipient]
        ids = tuple(person for person in DEVELOPMENT_IDS if person not in (recipient, donor))
        context = PairedIdentityContext(recipient, ids, None, donor)
        recipe = object.__new__(LocalFunctionalSource)
        # Production source rejects a leaked donor tuple before opening arrays.
        invalid = tuple(replace(record, task_a_maps=Poison(), task_b_partitions=Poison())
                        for record in self.records if record.participant_id != recipient)
        with self.assertRaisesRegex(ValueError, "training records"):
            recipe.fit_shared(invalid, context)
        with self.assertRaisesRegex(ValueError, "exclusions"):
            validate_source_context(replace(context, training_ids=DEVELOPMENT_IDS[1:]))

    def test_full_control_reuses_true_panel_and_refits_one_gain_for_each_packet(self):
        references = self.references()
        self.source_calls.clear()
        self.calibration_calls.clear()
        self.variant_calls.clear()
        run = self.fit(references)
        self.assertEqual(run.participant_ids, DEVELOPMENT_IDS)
        self.assertEqual(len(self.source_calls), 12)
        self.assertEqual(len(self.calibration_calls), 36)
        self.assertEqual(len(self.variant_calls), 36*4)
        for ref, condition, identity in zip(references, run.condition_folds, run.identity_folds):
            self.assertIs(condition.source, ref.source)
            self.assertIs(condition.true_panel, ref.panel)
            self.assertEqual(len(identity.context.training_ids), 10)
            self.assertIsNot(identity.source, ref.source)
            self.assertNotEqual(identity.true_panel.numerical_fit["gain"], identity.control_panel.numerical_fit["gain"])
            for panel in (condition.control_panel, identity.true_panel, identity.control_panel):
                for field in ("m1_predictions", "m2_predictions", "m3_variant_predictions", "m4_predictions"):
                    self.assertTrue(all(value.diagnostics["common_gain"] == panel.numerical_fit["gain"]
                                        for value in getattr(panel, field)))
        for calibration, calls in zip(self.calibration_calls,
                                      [self.variant_calls[start:start+4] for start in range(0, len(self.variant_calls), 4)]):
            self.assertTrue(all(call[:3] == calibration for call in calls))
        self.assertFalse(np.array_equal(run.condition_folds[0].true_panel.m3_maps,
                                       run.condition_folds[0].control_panel.m3_maps))

    def test_nested_deleted_payload_stays_closed_and_sources_have_nine_people(self):
        deleted = DEVELOPMENT_IDS[0]
        refs = self.references(deleted)
        packets = (replace(self.packets[0], task_a_maps=Poison(), task_a_halves=Poison()),)+self.packets[1:]
        records = (replace(self.records[0], task_a_maps=Poison(), task_b_partitions=Poison()),)+self.records[1:]
        self.source_calls.clear()
        run = self.fit(refs, deleted, packets, records)
        self.assertEqual(len(self.source_calls), 11)
        self.assertTrue(all(len(fold.context.training_ids) == 10 for fold in run.condition_folds))
        self.assertTrue(all(len(fold.context.training_ids) == 9 for fold in run.identity_folds))
        self.assertTrue(all(deleted not in (fold.context.target_id, fold.context.donor_id, *fold.context.training_ids)
                            for fold in run.identity_folds))
        for fold in run.condition_folds:
            training_packets = tuple(packet for packet in self.packets if packet.participant_id in fold.context.training_ids)
            _, expected = condition_relabeling(training_packets, fold.context, self.voxels, numerical_atol=1e-10)
            self.assertEqual(fold.diagnostics, expected)
        points = exploratory_falsifier_points(run, self.observations(deleted))
        self.assertEqual(points["biological_n"], 11)

    def test_paired_residual_contrasts_use_own_source_and_recipient_b_only_at_scoring(self):
        run = self.fit(self.references())
        points = exploratory_falsifier_points(run, self.observations())
        self.assertEqual(points["biological_n"], 12)
        self.assertFalse(points["permutation_inference"])
        self.assertEqual(tuple(points["identity"]["variants"]), VARIANTS)
        folds = run.identity_folds
        observed = np.stack([record.task_b_partitions for record in self.records])
        halves = (observed[:, (0, 2)]+observed[:, (1, 3)])/2
        true = np.stack([fold.true_panel.m1_predictions[0].maps for fold in folds])
        donor = np.stack([fold.control_panel.m1_predictions[0].maps for fold in folds])
        expected = map_covariance(halves[:, 0]-donor, halves[:, 1]-donor)-map_covariance(
            halves[:, 0]-true, halves[:, 1]-true)
        self.assertAlmostEqual(points["identity"]["variants"][M1_VARIANTS[0]]["control_minus_true"]["R_map"], expected)
        K0 = np.stack([fold.source.K0 for fold in folds])
        geometry = crossvalidated_geometries(observed)
        true_g = np.stack([fold.true_panel.m1_predictions[0].geometry for fold in folds])
        donor_g = np.stack([fold.control_panel.m1_predictions[0].geometry for fold in folds])
        residual_true = project_gain_direction(geometry-true_g[:, None], K0[:, None])
        residual_donor = project_gain_direction(geometry-donor_g[:, None], K0[:, None])
        expected_g = geometry_covariance(residual_donor[:, 0], residual_donor[:, 1])-geometry_covariance(
            residual_true[:, 0], residual_true[:, 1])
        self.assertAlmostEqual(points["identity"]["variants"][M1_VARIANTS[0]]["control_minus_true"]["R_geometry"], expected_g)
        for label in ("condition", "identity"):
            for name in M3_VARIANTS:
                self.assertAlmostEqual(points[label]["variants"][name]["control_minus_true"]["R_geometry"], 0., places=12)
        observations = self.observations()
        poisoned = tuple(replace(value, task_b_partitions=Poison()) for value in observations)
        invalid = poisoned[:-1]+(replace(poisoned[-1], participant_id=DEVELOPMENT_IDS[0]),)
        with self.assertRaisesRegex(ValueError, "ordered observed-B"):
            exploratory_falsifier_points(run, invalid)
        with self.assertRaisesRegex(ValueError, "all control targets"):
            exploratory_falsifier_points(replace(run, identity_folds=run.identity_folds[:-1]), observations)

    def test_missing_variants_gain_or_fit_refusal_cannot_create_reduced_result(self):
        references = self.references()
        incomplete = replace(references[0], panel=replace(references[0].panel, m4_predictions=()))
        with self.assertRaisesRegex(ValueError, "sixteen"):
            self.fit((incomplete,)+references[1:])
        def no_gain(source, target, context):
            return {"O": np.eye(2), "gain": -1.}
        with self.assertRaisesRegex(ValueError, "positive finite global gain"):
            self.fit(references, calibrate_m3=no_gain)
        def failed(source, target, context):
            raise ValueError("unidentified control operator")
        with self.assertRaisesRegex(ValueError, "unidentified control"):
            self.fit(references, calibrate_m3=failed)
        with self.assertRaises(TypeError):
            self.fit(references, predict_m4=None)


if __name__ == "__main__":
    unittest.main()
