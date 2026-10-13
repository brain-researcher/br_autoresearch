"""Exact existing operator-schema binding on invented inputs; no study reads.

These checks do not run uncertainty draws. The separate outcome-free evaluator
command performs one full unmocked composition under coordinator admission.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import released_endpoint_family as core
from test_endpoint_evaluator import fixture as endpoint_fixture


def fixture():
    supports, _, _ = endpoint_fixture()
    arrays, groups = {}, {}
    for group, support in supports.items():
        name = "_".join(group)
        for field in core.FIELDS:
            arrays[name + "_" + field] = getattr(support, field).copy()
        decisions = {}
        for edge, component in core.EDGE_COMPONENTS:
            strata = support.beta_strata[component]
            arrays[name + "_" + edge + "_low_columns"] = strata.low_columns.copy()
            arrays[name + "_" + edge + "_high_columns"] = strata.high_columns.copy()
            decisions[edge] = dict(applicable=True, reason="qualified_under_frozen_gain_rule",
                axis_key=name + "_A_all", gain_key=name + "_" + edge + "_gain",
                beta_primary_endpoints_removed=False, frozen_columns_removed=False)
        groups[name] = dict(beta_explanations=decisions,
            endpoint_calibration_prerequisites={key: True for key in core.endpoint.PRIMARY},
            full_endpoint_eligibility_requires_separate_development_checks=True)
    applicability = dict(source_private_calibration=str(core.operators.SOURCE_PRIVATE), groups=groups,
        H_component_calibration_applicability={key: True for key in core.endpoint.H_COMPONENTS},
        development_forecast_fitted=False, candidate_scores_consumed=False,
        worker_release_or_final_endpoint_eligibility_established=False)
    return arrays, applicability


class ReleasedEndpointFamilyTests(unittest.TestCase):
    def test_producer_keys_preserve_all_axes_weights_and_explicit_inapplicability(self):
        arrays, applicability = fixture()
        family = core.bind_released_endpoint_family(arrays, applicability)
        self.assertEqual(family.primary_endpoint_ids, core.endpoint.PRIMARY)
        self.assertEqual(family.h_component_ids, core.endpoint.H_COMPONENTS)
        self.assertEqual(family.directed_edges, core.endpoint.EDGES)
        for group, value in family._supports.items():
            prefix = "_".join(group) + "_"
            np.testing.assert_array_equal(value.columns, arrays[prefix + "A_all"])
            np.testing.assert_array_equal(value.anatomical_weights, arrays[prefix + "A_N_weights"])
            np.testing.assert_array_equal(value.reliable_weights, arrays[prefix + "R_N_weights"])
            for component in core.endpoint.H_COMPONENTS:
                self.assertEqual(tuple(len(part) for part in value.strata[component]), (256, 256))
        group = core.endpoint.GROUPS[0]
        name = "_".join(group)
        before = family._supports[group].anatomical_weights.copy()
        arrays[name + "_A_N_weights"][:] = 0
        np.testing.assert_array_equal(family._supports[group].anatomical_weights, before)
        with self.assertRaises(ValueError):
            family._supports[group].anatomical_weights.setflags(write=True)
        arrays, applicability = fixture()
        changed = applicability["groups"][name]
        changed["beta_explanations"]["B_to_C"].update(applicable=False, reason="undefined_calibration_gain")
        applicability["H_component_calibration_applicability"]["H_BC"] = False
        arrays.pop(name + "_B_to_C_low_columns")
        arrays.pop(name + "_B_to_C_high_columns")
        family = core.bind_released_endpoint_family(arrays, applicability)
        self.assertEqual(family.h_component_ids, ("H_CD",))
        self.assertEqual(family.primary_endpoint_ids, core.endpoint.PRIMARY)
        self.assertEqual(family.directed_edges, core.endpoint.EDGES)
        self.assertEqual(family.calibration_inapplicability["gain_reasons"][group, "H_BC"],
                         "undefined_calibration_gain")
        arrays[name + "_V_NC"] = np.array([], dtype=np.int64)
        arrays[name + "_V_NC_ceiling_denominator"] = np.array([], dtype=float)
        changed["endpoint_calibration_prerequisites"].update(D_VNC_raw=False, D_VNC_normalized=False)
        family = core.bind_released_endpoint_family(arrays, applicability)
        self.assertEqual(family.primary_endpoint_ids, core.endpoint.PRIMARY[:5])
        self.assertEqual(family.calibration_inapplicability["ceiling_empty_groups"], (group,))
        self.assertEqual(family.directed_edges, core.endpoint.EDGES[:4])

    def test_wrong_group_axis_decision_or_retuned_support_is_refused(self):
        for defect in ("source", "group", "axis", "disposition", "global", "removed", "endpoint", "support"):
            arrays, applicability = fixture()
            name = "_".join(core.endpoint.GROUPS[0])
            group = applicability["groups"][name]
            if defect == "source":
                applicability["source_private_calibration"] = "/invented/wrong_source"
            elif defect == "group":
                del applicability["groups"][name]
            elif defect == "axis":
                group["beta_explanations"]["B_to_C"]["axis_key"] = "sub-other_A_all"
            elif defect == "disposition":
                group["beta_explanations"]["B_to_C"].update(applicable=False, reason="unknown")
            elif defect == "global":
                applicability["H_component_calibration_applicability"]["H_BC"] = False
            elif defect == "removed":
                group["beta_explanations"]["B_to_C"]["frozen_columns_removed"] = True
            elif defect == "endpoint":
                group["endpoint_calibration_prerequisites"]["B_Aall_raw"] = False
            else:
                arrays[name + "_R_N_weights"][0] += .01
            with self.subTest(defect=defect), self.assertRaises((core.EndpointInputError, core.endpoint.EndpointError)):
                core.bind_released_endpoint_family(arrays, applicability)

    def test_disk_bridge_opens_only_named_derived_artifacts_and_checks_caps_first(self):
        arrays, applicability = fixture()
        with tempfile.TemporaryDirectory(prefix="ep17-endpoint-inputs-") as temporary:
            root = Path(temporary)
            private, public = root / "private", root / "public"
            private.mkdir(mode=0o700)
            private.chmod(0o2700)
            public.mkdir()
            metadata = private / "applicability.json"
            metadata.write_text(json.dumps(applicability))
            metadata.chmod(0o600)
            data = private / "operators.npz"
            np.savez(data, **arrays)
            data.chmod(0o600)
            report = dict(analysis_id="calibration_support", status="completed", errors=[],
                operator_artifacts_complete=True, participant_ROI_groups_processed=8,
                beta_group_edges_processed=16, private_attempt=str(private),
                source_private_calibration=str(core.operators.SOURCE_PRIVATE))
            (public / "operator_report.json").write_text(json.dumps(report))
            events, paths = [], []
            ordinary = core.existing._ordinary_input_file

            def checked(path, parent):
                self.assertEqual(parent, private)
                self.assertIn(path, (metadata, data))
                paths.append(path)
                return ordinary(path, parent)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(core.operators, "PRIVATE", private))
                stack.enter_context(mock.patch.object(core.operators, "PUBLIC", public))
                stack.enter_context(mock.patch.object(core.existing, "_ordinary_input_file", side_effect=checked))
                family = core.load_released_endpoint_family(metadata_byte_cap=65536, array_byte_cap=2 * 1024**2,
                                                            resource_check=events.append)
                self.assertEqual(family.component_ids, core.endpoint.PRIMARY + core.endpoint.H_COMPONENTS)
                self.assertEqual(paths, [metadata, data])
                self.assertEqual(events[-1].operation, "bind_calibration_endpoint_family")
                self.assertEqual(events[-1].when, "after")
                self.assertEqual(sum(event.operation == "read_one_calibration_support_vector" for event in events), 80)
                with mock.patch.object(core.np, "load") as opened:
                    with self.assertRaisesRegex(core.EndpointInputError, "array_byte_cap"):
                        core.load_released_endpoint_family(metadata_byte_cap=65536, array_byte_cap=1,
                                                           resource_check=lambda event: None)
                    opened.assert_not_called()
            self.assertEqual(metadata.read_text(), json.dumps(applicability))


if __name__ == "__main__":
    unittest.main()
