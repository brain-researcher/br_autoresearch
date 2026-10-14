"""Affected integration checks only; no empirical reads, fits or activation."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import paired_stretch_variants as original_stretch
import proposed_development_point_panel as proposed


class ProspectivePanelIntegrationTests(unittest.TestCase):
    def _callback_fixture(self, specification):
        # Exercise the actual callback and numerical dispatch, while replacing
        # only the already separately checked solvers and the operator result.
        # Deterministic nonzero centered maps retain the callback's gain/join
        # checks; no empirical payload or numerical optimization is needed.
        vector = np.linspace(-1., 1., 188)
        task_a = np.tile(vector, (29, 1))
        partitions = np.tile(vector, (4, 18, 1))
        O, C = np.eye(16), np.zeros((16, 16))
        reference = SimpleNamespace(weights=np.ones(29), B=np.zeros((188, 16)),
                                    _source=Mock(), certify_m3=Mock())
        base = object.__new__(original_stretch.PairedStretchVariants)
        base.reference, base.check, base.support_size = reference, Mock(), 188
        base.specifications, base.operator_certificate = (specification,), Mock()
        source = SimpleNamespace(task_a_maps=task_a, G=partitions.mean(axis=0),
            geometry_state=SimpleNamespace(source_partitions=partitions))
        objective = SimpleNamespace(O=O, gain=1., matrix=Mock(return_value=C),
            tangent_singular_values=Mock(return_value=np.ones(specification.parameter_count)))
        numerical = dict(C=C, coefficients=np.zeros(specification.parameter_count), O=O, gain=1.,
                         weighted_task_a_squared_error=0.)
        return SimpleNamespace(wrapper=proposed.ProposedPairedStretchVariants(base), base=base,
            source=source, target=task_a.copy(), reference_fit=dict(O=O, gain=1.),
            context=SimpleNamespace(training_ids=("fixture",)), objective=objective,
            numerical=numerical, specification=specification, C=C, O=O)

    def _callback_patches(self, stack, fixture, curvature_failure):
        stack.enter_context(patch.object(proposed, "PairedM4TaskAObjective", return_value=fixture.objective))
        curvature = stack.enter_context(patch.object(proposed, "proposed_curvature_certificate",
                                                     side_effect=curvature_failure))
        global_fits = {name: stack.enter_context(patch.object(proposed, name,
            return_value=(fixture.numerical, dict(solution_within_frozen_gap=True,
                unique_global_minimum_sufficient_profile=True,
                full_domain_positive_curvature_asserted=False))))
            for name in ("fit_diagonal_global", "fit_block_global")}
        legacy_fit = stack.enter_context(patch.object(proposed, "fit_m4_task_a"))
        predictions = {name: stack.enter_context(patch.object(proposed, name))
                       for name in ("apply_m4", "apply_m3", "crossvalidated_geometries", "M4Prediction")}
        return curvature, global_fits, legacy_fit, predictions

    def _call_callback(self, fixture):
        return fixture.wrapper(fixture.source, fixture.target, fixture.reference_fit, fixture.context)

    def test_global_fallback_operator_rejection_stops_before_prediction(self):
        for index, specification in enumerate(original_stretch.stretch_specifications()):
            with self.subTest(variant=specification.name):
                fixture = self._callback_fixture(specification)
                curvature_failure = original_stretch.M4CertificateUnverified("curvature unresolved", {})
                operator_failure = ValueError("actual operator fixture rejection")
                fixture.base.operator_certificate.side_effect = operator_failure
                selected = "fit_diagonal_global" if index == 0 else "fit_block_global"
                with ExitStack() as stack:
                    curvature, global_fits, legacy_fit, predictions = self._callback_patches(
                        stack, fixture, curvature_failure)
                    order = Mock()
                    order.attach_mock(curvature, "curvature")
                    order.attach_mock(global_fits[selected], "global_fit")
                    order.attach_mock(fixture.base.operator_certificate, "operator")
                    with self.assertRaises(ValueError) as raised:
                        self._call_callback(fixture)
                self.assertIs(raised.exception, operator_failure)
                self.assertEqual([call[0] for call in order.mock_calls], ["curvature", "global_fit", "operator"])
                global_fits[selected].assert_called_once_with(
                    fixture.objective, fixture.specification, check=fixture.base.check)
                for name, operation in global_fits.items():
                    if name != selected:
                        operation.assert_not_called()
                fixture.base.operator_certificate.assert_called_once()
                supplied = fixture.base.operator_certificate.call_args.args
                np.testing.assert_array_equal(supplied[0], fixture.O)
                np.testing.assert_array_equal(supplied[1], fixture.C)
                self.assertIs(supplied[2], fixture.specification)
                legacy_fit.assert_not_called()
                for operation in predictions.values():
                    operation.assert_not_called()

    def test_unknown_family_cannot_use_any_global_fallback(self):
        canonical = original_stretch.stretch_specifications()[1]
        unknown = original_stretch.StretchSpecification("unknown_m4_family",
            canonical.modes, canonical.spectral_norm_bound)
        fixture = self._callback_fixture(unknown)
        failure = original_stretch.M4CertificateUnverified("unknown-family curvature unresolved", {})
        with ExitStack() as stack:
            _, global_fits, legacy_fit, predictions = self._callback_patches(stack, fixture, failure)
            with self.assertRaises(original_stretch.M4CertificateUnverified) as raised:
                self._call_callback(fixture)
        self.assertIs(raised.exception, failure)
        for operation in global_fits.values():
            operation.assert_not_called()
        legacy_fit.assert_not_called()
        fixture.base.operator_certificate.assert_not_called()
        for operation in predictions.values():
            operation.assert_not_called()

    def test_unresolved_global_fit_propagates_before_operator_or_prediction(self):
        for index, specification in enumerate(original_stretch.stretch_specifications()):
            with self.subTest(variant=specification.name):
                fixture = self._callback_fixture(specification)
                curvature_failure = original_stretch.M4CertificateUnverified("curvature unresolved", {})
                global_failure = original_stretch.M4CertificateUnverified("exterior remains unexcluded", {})
                selected = "fit_diagonal_global" if index == 0 else "fit_block_global"
                with ExitStack() as stack:
                    _, global_fits, legacy_fit, predictions = self._callback_patches(
                        stack, fixture, curvature_failure)
                    global_fits[selected].side_effect = global_failure
                    with self.assertRaises(original_stretch.M4CertificateUnverified) as raised:
                        self._call_callback(fixture)
                self.assertIs(raised.exception, global_failure)
                global_fits[selected].assert_called_once()
                for name, operation in global_fits.items():
                    if name != selected:
                        operation.assert_not_called()
                legacy_fit.assert_not_called()
                fixture.base.operator_certificate.assert_not_called()
                for operation in predictions.values():
                    operation.assert_not_called()

    def test_global_fit_pairing_changes_refuse_before_operator_or_prediction(self):
        for index in (0, 2):
            for field in ("C", "O", "gain"):
                with self.subTest(family=index, changed=field):
                    fixture = self._callback_fixture(original_stretch.stretch_specifications()[index])
                    if field == "gain":
                        fixture.numerical[field] = 2.
                    else:
                        fixture.numerical[field] = fixture.numerical[field]+np.eye(16)
                    failure = original_stretch.M4CertificateUnverified("curvature unresolved", {})
                    with ExitStack() as stack:
                        _, _, legacy_fit, predictions = self._callback_patches(stack, fixture, failure)
                        with self.assertRaisesRegex(ValueError, "must remain paired"):
                            self._call_callback(fixture)
                    legacy_fit.assert_not_called()
                    fixture.base.operator_certificate.assert_not_called()
                    for operation in predictions.values():
                        operation.assert_not_called()

    def _assert_failed_optimizer_uses_global_fit_before_operator(self, failure_type):
        for index, specification in enumerate(original_stretch.stretch_specifications()):
            with self.subTest(variant=specification.name):
                fixture = self._callback_fixture(specification)
                operator_failure = ValueError("actual operator fixture rejection")
                fixture.base.operator_certificate.side_effect = operator_failure
                selected = "fit_diagonal_global" if index == 0 else "fit_block_global"
                with ExitStack() as stack:
                    curvature, global_fits, legacy_fit, predictions = self._callback_patches(
                        stack, fixture, None)
                    curvature.return_value = {"qualified_curvature_fixture": True}
                    legacy_fit.side_effect = failure_type("fixture local optimizer failure")
                    order = Mock()
                    order.attach_mock(curvature, "curvature")
                    order.attach_mock(legacy_fit, "local_fit")
                    order.attach_mock(global_fits[selected], "global_fit")
                    order.attach_mock(fixture.base.operator_certificate, "operator")
                    with self.assertRaises(ValueError) as raised:
                        self._call_callback(fixture)
                self.assertIs(raised.exception, operator_failure)
                self.assertEqual([call[0] for call in order.mock_calls],
                                 ["curvature", "local_fit", "global_fit", "operator"])
                global_fits[selected].assert_called_once_with(
                    fixture.objective, specification, check=fixture.base.check)
                for name, operation in global_fits.items():
                    if name != selected:
                        operation.assert_not_called()
                for operation in predictions.values():
                    operation.assert_not_called()

    def test_unconverged_local_optimizer_uses_qualified_global_fit_before_operator(self):
        self._assert_failed_optimizer_uses_global_fit_before_operator(proposed.M4OptimizerUnconverged)

    def test_nonfinite_optimizer_trial_uses_qualified_global_fit_before_operator(self):
        self._assert_failed_optimizer_uses_global_fit_before_operator(proposed.M4OptimizerTrialNonfinite)

    def test_local_solution_gap_uses_global_fit_before_actual_operator(self):
        for index, specification in enumerate(original_stretch.stretch_specifications()):
            with self.subTest(variant=specification.name):
                fixture = self._callback_fixture(specification)
                operator_failure = ValueError("actual operator fixture rejection")
                fixture.base.operator_certificate.side_effect = operator_failure
                selected = "fit_diagonal_global" if index == 0 else "fit_block_global"
                gap_failure = original_stretch.M4OptimalityGapUnverified("local gap too large", {})
                with ExitStack() as stack:
                    curvature, global_fits, legacy_fit, predictions = self._callback_patches(
                        stack, fixture, None)
                    curvature.return_value = {"qualified_curvature_fixture": True}
                    legacy_fit.return_value = fixture.numerical
                    solution = stack.enter_context(patch.object(proposed, "_solution_certificate",
                        side_effect=gap_failure))
                    order = Mock()
                    for name, operation in (("curvature", curvature), ("local_fit", legacy_fit),
                            ("solution", solution), ("global_fit", global_fits[selected]),
                            ("operator", fixture.base.operator_certificate)):
                        order.attach_mock(operation, name)
                    with self.assertRaises(ValueError) as raised:
                        self._call_callback(fixture)
                self.assertIs(raised.exception, operator_failure)
                self.assertEqual([call[0] for call in order.mock_calls],
                    ["curvature", "local_fit", "solution", "global_fit", "operator"])
                global_fits[selected].assert_called_once_with(
                    fixture.objective, specification, check=fixture.base.check)
                for name, operation in global_fits.items():
                    if name != selected:
                        operation.assert_not_called()
                for operation in predictions.values():
                    operation.assert_not_called()

    def test_other_solution_certificate_refusals_do_not_retry(self):
        fixture = self._callback_fixture(original_stretch.stretch_specifications()[1])
        for failure in (original_stretch.M4CertificateUnverified("tangent rank unresolved", {}),
                        ValueError("paired C differs"), ArithmeticError("invalid objective")):
            with self.subTest(refusal=str(failure)), ExitStack() as stack:
                _, global_fits, legacy_fit, predictions = self._callback_patches(stack, fixture, None)
                legacy_fit.return_value = fixture.numerical
                stack.enter_context(patch.object(proposed, "_solution_certificate", side_effect=failure))
                with self.assertRaises(type(failure)) as raised:
                    self._call_callback(fixture)
                self.assertIs(raised.exception, failure)
                for operation in global_fits.values():
                    operation.assert_not_called()
                fixture.base.operator_certificate.assert_not_called()
                for operation in predictions.values():
                    operation.assert_not_called()

    def test_real_solution_certificate_marks_only_valid_excess_gap_for_retry(self):
        fixture = self._callback_fixture(original_stretch.stretch_specifications()[1])
        fixture.objective.final_stretch_diagnostics = Mock()
        fixture.objective.value_gradient = Mock(return_value=(1., np.ones(8)))
        curvature = dict(hessian_lower_bound=1., objective_scale=1.)
        with self.assertRaises(original_stretch.M4OptimalityGapUnverified) as raised:
            original_stretch._solution_certificate(fixture.objective, fixture.specification,
                fixture.numerical, curvature)
        diagnostic = raised.exception.diagnostics
        self.assertGreater(diagnostic["first_order_gap"], diagnostic["required_gap_upper_bound"])
        fixture.objective.tangent_singular_values.return_value = np.r_[np.ones(7), 0.]
        with self.assertRaises(original_stretch.M4CertificateUnverified) as raised:
            original_stretch._solution_certificate(fixture.objective, fixture.specification,
                fixture.numerical, curvature)
        self.assertNotIsInstance(raised.exception, original_stretch.M4OptimalityGapUnverified)

    def test_other_arithmetic_failures_are_not_hidden_by_global_retry(self):
        fixture = self._callback_fixture(original_stretch.stretch_specifications()[1])
        failure = ArithmeticError("fixture invalid returned stretch")
        with ExitStack() as stack:
            curvature, global_fits, legacy_fit, predictions = self._callback_patches(
                stack, fixture, None)
            curvature.return_value = {"qualified_curvature_fixture": True}
            legacy_fit.side_effect = failure
            with self.assertRaises(ArithmeticError) as raised:
                self._call_callback(fixture)
        self.assertIs(raised.exception, failure)
        for operation in global_fits.values():
            operation.assert_not_called()
        fixture.base.operator_certificate.assert_not_called()
        for operation in predictions.values():
            operation.assert_not_called()

    def _fixture(self):
        check = Mock()
        base = object.__new__(original_stretch.PairedStretchVariants)
        base.reference, base.check = object(), check
        base.support_size, base.specifications = 188, original_stretch.stretch_specifications()
        calibration = dict(calibrate_m3=Mock(), certify_m3=Mock(), predict_m3_geometry=Mock(),
            predict_m1=Mock(), predict_m2=Mock(), predict_m3_variants=Mock(), predict_m4=base)
        controls = dict(fit_m0a=Mock(), predict_random_bases=Mock(), predict_matched_warps=Mock())
        records, packets, observations = (tuple(object() for _ in range(12)) for _ in range(3))
        references = tuple(object() for _ in range(12))
        ordinary = SimpleNamespace(full=SimpleNamespace(
            folds=tuple(SimpleNamespace(calibration_reference=x) for x in references)), deletions=())
        return SimpleNamespace(check=check, base=base, calibration=calibration,
            controls=controls, records=records, packets=packets, observations=observations,
            references=references, ordinary=ordinary, fit_shared=Mock(),
            coordinates=object(), voxel_ids=tuple(range(188)), development=object(),
            support=object(), header=object(), falsifiers=object())

    def _patches(self, stack, fixture):
        names = ("fit_development_lopo", "fit_calibration_falsifiers", "exploratory_points",
                 "exploratory_falsifier_points", "m2_alignment_points")
        mocks = {name: stack.enter_context(patch.object(proposed, name)) for name in names}
        mocks["fit_development_lopo"].return_value = fixture.ordinary
        mocks["fit_calibration_falsifiers"].return_value = fixture.falsifiers
        joined = stack.enter_context(patch.object(proposed, "_joined_arrays", return_value=(
            fixture.records, fixture.packets, fixture.observations, fixture.coordinates, fixture.voxel_ids)))
        factory = Mock(return_value=(fixture.fit_shared, fixture.calibration, fixture.controls))
        # Replace only the inactive adapter's module reference during this
        # fixture. No production module globals are monkeypatched.
        stack.enter_context(patch.object(proposed, "original_panel",
                                         SimpleNamespace(_fixed_callbacks=factory)))
        return mocks, joined, factory

    def test_same_adapter_reaches_all_ordinary_and_falsifier_calls(self):
        fixture = self._fixture()
        with ExitStack() as stack:
            calls, joined, factory = self._patches(stack, fixture)
            result = proposed.fit_development_point_panel(fixture.development,
                common_support=fixture.support, forward_header=fixture.header, check=fixture.check)
        joined.assert_called_once_with(fixture.development, fixture.support, fixture.header, fixture.check)
        factory.assert_called_once_with(fixture.coordinates, fixture.voxel_ids, fixture.check)
        ordinary = calls["fit_development_lopo"].call_args
        falsifiers = calls["fit_calibration_falsifiers"].call_args
        self.assertEqual(ordinary.args, (fixture.records,))
        self.assertEqual(falsifiers.args, (fixture.packets, fixture.records, fixture.references))
        self.assertFalse(ordinary.kwargs["include_participant_deletions"])
        self.assertIsNone(falsifiers.kwargs["deleted_id"])
        wrapper = ordinary.kwargs["predict_m4"]
        self.assertIs(type(wrapper), proposed.ProposedPairedStretchVariants)
        self.assertIs(falsifiers.kwargs["predict_m4"], wrapper)
        self.assertIs(wrapper.reference, fixture.base.reference)
        self.assertIs(wrapper.check, fixture.base.check)
        self.assertIs(wrapper.specifications, fixture.base.specifications)
        self.assertEqual(wrapper.support_size, 188)
        self.assertIs(wrapper.operator_certificate.__self__, fixture.base)
        for call in (ordinary, falsifiers):
            self.assertIs(call.kwargs["fit_shared"], fixture.fit_shared)
            self.assertIs(call.kwargs["check"], fixture.check)
            self.assertEqual(call.kwargs["support_size"], 188)
            for name, value in fixture.calibration.items():
                if name != "predict_m4":
                    self.assertIs(call.kwargs[name], value)
        for name, value in fixture.controls.items():
            self.assertIs(ordinary.kwargs[name], value)
        self.assertIs(fixture.calibration["predict_m4"], fixture.base)
        self.assertIs(result.ordinary_fit, fixture.ordinary)
        self.assertIs(result.calibration_falsifier_fit, fixture.falsifiers)
        calls["exploratory_points"].assert_called_once_with(fixture.ordinary.full)
        calls["exploratory_falsifier_points"].assert_called_once_with(
            fixture.falsifiers, fixture.observations, support_size=188)
        calls["m2_alignment_points"].assert_called_once_with(fixture.ordinary.full, support_size=188)

    def test_numerical_refusal_has_no_fallback_or_partial_scoring(self):
        fixture = self._fixture()
        failure = original_stretch.M4CertificateUnverified("fixture refusal", {})
        with ExitStack() as stack:
            calls, _, _ = self._patches(stack, fixture)
            calls["fit_development_lopo"].side_effect = failure
            with self.assertRaises(original_stretch.M4CertificateUnverified) as raised:
                proposed.fit_development_point_panel(fixture.development,
                    common_support=fixture.support, forward_header=fixture.header, check=fixture.check)
        self.assertIs(raised.exception, failure)
        calls["fit_development_lopo"].assert_called_once()
        for name in ("fit_calibration_falsifiers", "exploratory_points",
                     "exploratory_falsifier_points", "m2_alignment_points"):
            calls[name].assert_not_called()
        with self.assertRaises(TypeError):
            proposed.ProposedPairedStretchVariants(lambda: None)


if __name__ == "__main__":
    unittest.main()
