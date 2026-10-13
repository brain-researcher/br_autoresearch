#!/usr/bin/env python3
"""Synthetic and adversarial tests for the payload-free EP21 adapter."""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib.util
import json
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("execution_adapter.py")
SPEC = importlib.util.spec_from_file_location("execution_adapter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


class AdapterTests(unittest.TestCase):
    def pair_record(self, *, pair_id="fixture-pair-07", source="participant-A", target="participant-B", ordinal=7):
        core = {
            "pair_id": pair_id,
            "source_id": source,
            "target_id": target,
            "ordinal": ordinal,
        }
        return {**core, "pair_table_sha256": adapter.metadata_sha256([core])}

    def context(
        self,
        mode="image_average_before_fit",
        pair_record=None,
        alpha_scope="fixed_predeclared",
    ):
        record = pair_record or self.pair_record()
        pair_core = {
            field: record[field]
            for field in ("pair_id", "source_id", "target_id", "ordinal")
        }
        selected = alpha_scope == "tau_train_only"
        alpha_declaration = {
            "policy_id": (
                "synthetic-tau-train-selection"
                if selected
                else "synthetic-fixed-alpha"
            ),
            "policy_sha256": "4" * 64,
            "selection_scope": alpha_scope,
            "allowed_fit_role": "tau_train",
            "fixed_before_tau_test": True,
            "tau_test_or_ood_used": False,
            "resolved_value": 1.0,
            "selection_receipt_sha256": "5" * 64 if selected else None,
        }
        return adapter.make_synthetic_context(
            fixture_id=f"fixture-{mode}",
            pair_table_records=[pair_core],
            settings={
                "content_feature_layers": ["conv1-1", "fc8"],
                "brain_loss_repetition_mode": mode,
                "brain_loss_alpha_bindings": {
                    pair_core["pair_id"]: alpha_declaration
                },
            },
        )

    def bundle(
        self,
        mode="image_average_before_fit",
        content_count=4,
        alpha_scope="fixed_predeclared",
    ):
        pair_record = self.pair_record()
        context = self.context(mode, pair_record, alpha_scope)
        pair = adapter.bind_pair(context, pair_record)
        source_unique = [
            {"row_id": f"source-row-{i}", "image_id": f"source-image-{i}"}
            for i in range(content_count)
        ]
        target_unique = [
            {"row_id": f"target-row-{i}", "image_id": f"target-image-{i}"}
            for i in range(content_count + 1)
        ]
        tau_train = ["tau-image-0", "tau-image-1", "tau-image-2"]
        tau_test = ["test-image-0", "test-image-1"]
        source_tau = []
        target_tau = []
        for image_index, image_id in enumerate(tau_train):
            for repetition in range(2):
                source_tau.append(
                    {
                        "row_id": f"source-tau-{image_index}-{repetition}",
                        "image_id": image_id,
                        "local_repetition_id": f"source-local-{repetition}",
                        "cross_subject_alignment_key": f"presentation-{repetition}",
                    }
                )
                reversed_rep = 1 - repetition
                target_tau.append(
                    {
                        "row_id": f"target-tau-{image_index}-{reversed_rep}",
                        "image_id": image_id,
                        "local_repetition_id": f"target-local-{reversed_rep}",
                        "cross_subject_alignment_key": f"presentation-{reversed_rep}",
                    }
                )
        roles = adapter.bind_structural_roles(
            context,
            pair,
            manifest_sha256="1" * 64,
            regular_only_exclusion_receipt_sha256="2" * 64,
            cross_subject_alignment_receipt_sha256=(
                "3" * 64 if mode == "authenticated_cross_subject_pairing" else None
            ),
            source_unique_rows=source_unique,
            target_unique_rows=target_unique,
            source_feature_rows_by_layer={
                "conv1-1": [
                    {"row_id": f"conv-row-{i}", "image_id": row["image_id"]}
                    for i, row in enumerate(source_unique)
                ],
                "fc8": [
                    {"row_id": f"fc-row-{i}", "image_id": row["image_id"]}
                    for i, row in enumerate(source_unique)
                ],
            },
            tau_train_ordered_ids=tau_train,
            tau_test_ordered_ids=tau_test,
            source_tau_presentations=source_tau,
            target_tau_presentations=target_tau,
        )
        worker = adapter.bind_worker(context, max_workers=1, explicit_worker_limit=1)
        response = adapter.make_ordered_rows(
            roles,
            participant_id=pair.record["source_id"],
            role="source_unique_train",
            row_ids=[row["row_id"] for row in source_unique],
            image_ids=[row["image_id"] for row in source_unique],
            shape=[content_count, 12],
        )
        features = {
            "conv1-1": adapter.make_feature_rows(
                roles,
                layer_id="conv1-1",
                row_ids=[f"conv-row-{i}" for i in range(content_count)],
                image_ids=[row["image_id"] for row in source_unique],
                shape=[content_count, 8, 4, 4],
            ),
            "fc8": adapter.make_feature_rows(
                roles,
                layer_id="fc8",
                row_ids=[f"fc-row-{i}" for i in range(content_count)],
                image_ids=[row["image_id"] for row in source_unique],
                shape=[content_count, 16],
            ),
        }
        source_rows = adapter.make_ordered_rows(
            roles,
            participant_id=pair.record["source_id"],
            role="tau_train",
            row_ids=[row["row_id"] for row in source_tau],
            image_ids=[row["image_id"] for row in source_tau],
            shape=[len(source_tau), 20],
        )
        target_rows = adapter.make_ordered_rows(
            roles,
            participant_id=pair.record["target_id"],
            role="tau_train",
            row_ids=[row["row_id"] for row in target_tau],
            image_ids=[row["image_id"] for row in target_tau],
            shape=[len(target_tau), 24],
        )
        selected = alpha_scope == "tau_train_only"
        alpha = adapter.bind_alpha(
            context,
            pair,
            policy_id=(
                "synthetic-tau-train-selection"
                if selected
                else "synthetic-fixed-alpha"
            ),
            policy_sha256="4" * 64,
            selection_scope=alpha_scope,
            allowed_fit_role="tau_train",
            fixed_before_tau_test=True,
            tau_test_or_ood_used=False,
            resolved_value=1.0,
            selection_receipt_sha256="5" * 64 if selected else None,
        )
        return {
            "context": context,
            "pair": pair,
            "roles": roles,
            "worker": worker,
            "response": response,
            "features": features,
            "source_rows": source_rows,
            "target_rows": target_rows,
            "alpha": alpha,
            "source_unique": source_unique,
            "target_unique": target_unique,
            "source_tau": source_tau,
            "target_tau": target_tau,
        }

    def plans(self, mode="image_average_before_fit", content_count=4):
        bundle = self.bundle(mode, content_count)
        content = adapter.plan_content(
            bundle["context"], bundle["pair"], bundle["roles"], bundle["worker"],
            response_rows=bundle["response"], feature_rows_by_layer=bundle["features"],
        )
        brain = adapter.plan_brain_loss(
            bundle["context"], bundle["pair"], bundle["roles"], bundle["worker"],
            source_rows=bundle["source_rows"], target_rows=bundle["target_rows"],
            alpha=bundle["alpha"],
        )
        summary = self.summarize(bundle, content, brain)
        return bundle, content, brain, summary

    def summarize(self, bundle, content, brain, alpha=None):
        return adapter.summarize_plans(
            bundle["context"], bundle["pair"], bundle["roles"], bundle["worker"],
            content, brain, alpha or bundle["alpha"],
            response_rows=bundle["response"],
            feature_rows_by_layer=bundle["features"],
            source_rows=bundle["source_rows"],
            target_rows=bundle["target_rows"],
        )

    def export(self, bundle, content, brain, summary, alpha=None):
        return adapter.export_summary(
            summary,
            bundle["context"], bundle["pair"], bundle["roles"], bundle["worker"],
            content, brain, alpha or bundle["alpha"],
            response_rows=bundle["response"],
            feature_rows_by_layer=bundle["features"],
            source_rows=bundle["source_rows"],
            target_rows=bundle["target_rows"],
        )

    def assert_code(self, code, operation):
        with self.assertRaises(adapter.AdapterValidationError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_source_is_standard_library_metadata_only(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertEqual(
            roots,
            {"__future__", "hashlib", "json", "math", "re", "dataclasses", "typing"},
        )
        for forbidden in ("torch", "numpy", "ProcessPoolExecutor", "open(", "Path(", "device="):
            self.assertNotIn(forbidden, source)

    def test_synthetic_context_cannot_be_empirical(self):
        context = self.context()
        self.assertFalse(context.record["empirical_capable"])
        self.assert_code(
            "E_EMPIRICAL_CONTEXT_UNAVAILABLE",
            lambda: adapter.bind_empirical_context(
                verified_lock_receipt=None, authorization_receipt=None
            ),
        )
        self.assert_code(
            "E_EMPIRICAL_CONTEXT_UNAVAILABLE",
            lambda: adapter.bind_empirical_context(
                verified_lock_receipt={"path": "wrong"},
                authorization_receipt={"synthetic": True},
            ),
        )

    def test_nested_inputs_are_copied_into_immutable_bytes(self):
        settings = {
            "content_feature_layers": ["conv1-1", "fc8"],
                "brain_loss_repetition_mode": "image_average_before_fit",
                "brain_loss_alpha_bindings": {
                    "immutable-pair": {
                        "policy_id": "synthetic-fixed-alpha",
                        "policy_sha256": "4" * 64,
                        "selection_scope": "fixed_predeclared",
                        "allowed_fit_role": "tau_train",
                        "fixed_before_tau_test": True,
                        "tau_test_or_ood_used": False,
                        "resolved_value": 1.0,
                        "selection_receipt_sha256": None,
                    }
                },
        }
        context = adapter.make_synthetic_context(
            fixture_id="immutable-fixture",
            pair_table_records=[
                {
                    "pair_id": "immutable-pair",
                    "source_id": "immutable-source",
                    "target_id": "immutable-target",
                    "ordinal": 0,
                }
            ],
            settings=settings,
        )
        settings["content_feature_layers"].append("mutated")
        self.assertEqual(context.content_layers, ("conv1-1", "fc8"))
        self.assertIsInstance(context.canonical_record_json, bytes)

    def test_direct_construction_and_post_construction_tamper_fail(self):
        context = self.context()
        self.assert_code(
            "E_SEAL",
            lambda: adapter.SyntheticFixtureContext(
                context.canonical_record_json, "0" * 64
            ),
        )
        tampered = adapter.SyntheticFixtureContext.__new__(adapter.SyntheticFixtureContext)
        object.__setattr__(tampered, "canonical_record_json", context.canonical_record_json)
        object.__setattr__(tampered, "seal", "0" * 64)
        pair = self.pair_record()
        self.assert_code("E_SEAL", lambda: adapter.bind_pair(tampered, pair))

    def test_pair_id_is_verbatim_and_wrong_table_or_direction_fails(self):
        pair_record = self.pair_record(pair_id="opaque-pair-z")
        context = self.context(pair_record=pair_record)
        pair = adapter.bind_pair(context, pair_record)
        self.assertEqual(pair.record["pair_id"], "opaque-pair-z")
        wrong_table = dict(pair_record, pair_table_sha256="b" * 64)
        self.assert_code("E_PAIR_TABLE", lambda: adapter.bind_pair(context, wrong_table))
        forged = dict(pair_record, pair_id="unlisted-pair", ordinal=99)
        self.assert_code("E_PAIR_MEMBERSHIP", lambda: adapter.bind_pair(context, forged))
        same = self.pair_record(source="participant-A", target="participant-A")
        self.assert_code("E_PAIR_DIRECTION", lambda: self.context(pair_record=same))

    def test_worker_rejects_bool_zero_string_and_oversubscription(self):
        context = self.context()
        for workers, cpus, code in (
            (True, 1, "E_INTEGER"),
            (0, 1, "E_INTEGER"),
            ("1", 1, "E_INTEGER"),
            (2, 1, "E_WORKER_ALLOCATION"),
            (1, 2, "E_WORKER_ALLOCATION"),
        ):
            with self.subTest(workers=workers, cpus=cpus):
                self.assert_code(
                    code,
                    lambda workers=workers, cpus=cpus: adapter.bind_worker(
                        context, max_workers=workers, explicit_worker_limit=cpus
                    ),
                )

    def test_every_structural_role_pair_is_disjoint(self):
        base = self.bundle()
        roles = {
            "source": copy.deepcopy(base["source_unique"]),
            "target": copy.deepcopy(base["target_unique"]),
            "train": list(base["roles"].record["tau_train_ordered_ids"]),
            "test": list(base["roles"].record["tau_test_ordered_ids"]),
        }
        names = list(roles)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                mutated = copy.deepcopy(roles)
                left_image = mutated[left][0]["image_id"] if left in {"source", "target"} else mutated[left][0]
                if right in {"source", "target"}:
                    mutated[right][0]["image_id"] = left_image
                else:
                    mutated[right][0] = left_image
                source_tau = copy.deepcopy(base["source_tau"])
                target_tau = copy.deepcopy(base["target_tau"])
                feature_rows = copy.deepcopy(
                    base["roles"].record["source_feature_rows_by_layer"]
                )
                for layer_rows in feature_rows.values():
                    for index, row in enumerate(layer_rows):
                        row["image_id"] = mutated["source"][index]["image_id"]
                if right == "train":
                    original = roles["train"][0]
                    for rows in (source_tau, target_tau):
                        for row in rows:
                            if row["image_id"] == original:
                                row["image_id"] = left_image
                with self.subTest(left=left, right=right):
                    self.assert_code(
                        "E_ROLE_OVERLAP",
                        lambda mutated=mutated, source_tau=source_tau, target_tau=target_tau: adapter.bind_structural_roles(
                            base["context"], base["pair"],
                            manifest_sha256="1" * 64,
                            regular_only_exclusion_receipt_sha256="2" * 64,
                            cross_subject_alignment_receipt_sha256=None,
                            source_unique_rows=mutated["source"],
                            target_unique_rows=mutated["target"],
                            source_feature_rows_by_layer=feature_rows,
                            tau_train_ordered_ids=mutated["train"],
                            tau_test_ordered_ids=mutated["test"],
                            source_tau_presentations=source_tau,
                            target_tau_presentations=target_tau,
                        ),
                    )

    def test_content_allocates_two_nonplanning_sizes(self):
        for count in (3, 7):
            with self.subTest(count=count):
                bundle, content, brain, summary = self.plans(content_count=count)
                self.assertEqual(content.record["source_row_count"], count)
                self.assertTrue(
                    all(item["dynamic_buffer_shape"][0] == count for item in content.record["feature_rows"])
                )
                self.assertEqual(
                    self.export(bundle, content, brain, summary)["content_row_count"],
                    count,
                )
                self.assertNotEqual(count, 4712)

    def test_same_count_wrong_or_permuted_response_rows_fail(self):
        base = self.bundle()
        original = base["response"].record
        cases = (
            ([*original["row_ids"][:-1], "wrong-row"], original["image_ids"]),
            (list(reversed(original["row_ids"])), list(reversed(original["image_ids"]))),
        )
        for row_ids, image_ids in cases:
            rows = adapter.make_ordered_rows(
                base["roles"], participant_id=base["pair"].record["source_id"],
                role="source_unique_train", row_ids=row_ids, image_ids=image_ids,
                shape=original["shape"],
            )
            self.assert_code(
                "E_CONTENT_RESPONSE_ALIGNMENT",
                lambda rows=rows: adapter.plan_content(
                    base["context"], base["pair"], base["roles"], base["worker"],
                    response_rows=rows, feature_rows_by_layer=base["features"],
                ),
            )

    def test_same_count_permuted_feature_images_fail(self):
        base = self.bundle()
        original = base["features"]["fc8"].record
        bad = dict(base["features"])
        bad["fc8"] = adapter.make_feature_rows(
            base["roles"], layer_id="fc8", row_ids=original["row_ids"],
            image_ids=list(reversed(original["image_ids"])), shape=original["shape"],
        )
        self.assert_code(
            "E_CONTENT_FEATURE_ALIGNMENT",
            lambda: adapter.plan_content(
                base["context"], base["pair"], base["roles"], base["worker"],
                response_rows=base["response"], feature_rows_by_layer=bad,
            ),
        )

    def test_same_count_substituted_feature_row_id_fails(self):
        base = self.bundle()
        original = base["features"]["fc8"].record
        bad = dict(base["features"])
        bad["fc8"] = adapter.make_feature_rows(
            base["roles"], layer_id="fc8",
            row_ids=[*original["row_ids"][:-1], "substituted-feature-row"],
            image_ids=original["image_ids"], shape=original["shape"],
        )
        self.assert_code(
            "E_CONTENT_FEATURE_ALIGNMENT",
            lambda: adapter.plan_content(
                base["context"], base["pair"], base["roles"], base["worker"],
                response_rows=base["response"], feature_rows_by_layer=bad,
            ),
        )

    def test_missing_or_extra_repeat_fails_against_authenticated_manifest(self):
        base = self.bundle()
        original = base["source_rows"].record
        cases = (
            (original["row_ids"][:-1], original["image_ids"][:-1]),
            ([*original["row_ids"], "extra-row"], [*original["image_ids"], original["image_ids"][0]]),
        )
        for row_ids, image_ids in cases:
            rows = adapter.make_ordered_rows(
                base["roles"], participant_id=base["pair"].record["source_id"],
                role="tau_train", row_ids=row_ids, image_ids=image_ids,
                shape=[len(row_ids), 20],
            )
            self.assert_code(
                "E_BRAIN_PRESENTATION_ALIGNMENT",
                lambda rows=rows: adapter.plan_brain_loss(
                    base["context"], base["pair"], base["roles"], base["worker"],
                    source_rows=rows, target_rows=base["target_rows"], alpha=base["alpha"],
                ),
            )

    def test_image_average_groups_cover_every_presentation(self):
        bundle, content, brain, summary = self.plans("image_average_before_fit")
        self.assertEqual(brain.record["source_group_indices"], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(
            sorted(index for group in brain.record["target_group_indices"] for index in group),
            list(range(6)),
        )
        self.assertEqual(self.export(bundle, content, brain, summary)["brain_fit_unit_count"], 3)

    def test_paired_mode_uses_cross_subject_keys_not_local_repetitions(self):
        bundle, content, brain, summary = self.plans("authenticated_cross_subject_pairing")
        self.assertEqual(brain.record["source_aligned_indices"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(brain.record["target_aligned_indices"], [1, 0, 3, 2, 5, 4])
        self.assertEqual(self.export(bundle, content, brain, summary)["brain_fit_unit_count"], 6)

    def test_equal_local_repetition_does_not_rescue_wrong_alignment_key(self):
        base = self.bundle("authenticated_cross_subject_pairing")
        target_tau = copy.deepcopy(base["target_tau"])
        target_tau[0]["local_repetition_id"] = base["source_tau"][0]["local_repetition_id"]
        target_tau[0]["cross_subject_alignment_key"] = "different-key"
        self.assert_code(
            "E_ALIGNMENT_MISMATCH",
            lambda: adapter.bind_structural_roles(
                base["context"], base["pair"], manifest_sha256="1" * 64,
                regular_only_exclusion_receipt_sha256="2" * 64,
                cross_subject_alignment_receipt_sha256="3" * 64,
                source_unique_rows=base["source_unique"], target_unique_rows=base["target_unique"],
                source_feature_rows_by_layer=base["roles"].record["source_feature_rows_by_layer"],
                tau_train_ordered_ids=base["roles"].record["tau_train_ordered_ids"],
                tau_test_ordered_ids=base["roles"].record["tau_test_ordered_ids"],
                source_tau_presentations=base["source_tau"], target_tau_presentations=target_tau,
            ),
        )

    def test_alpha_rejects_test_policy_role_and_pair_override(self):
        base = self.bundle()
        common = {
            "policy_sha256": "4" * 64,
            "selection_scope": "fixed_predeclared",
            "allowed_fit_role": "tau_train",
            "fixed_before_tau_test": True,
            "tau_test_or_ood_used": False,
            "resolved_value": 1.0,
            "selection_receipt_sha256": None,
        }
        self.assert_code(
            "E_ALPHA_LEAKAGE",
            lambda: adapter.bind_alpha(
                base["context"], base["pair"], policy_id="select-on-tau-test", **common
            ),
        )
        self.assert_code(
            "E_ALPHA_LEAKAGE",
            lambda: adapter.bind_alpha(
                base["context"], base["pair"], policy_id="select.on.tau.test", **common
            ),
        )
        wrong_role = dict(common, allowed_fit_role="tau_test")
        self.assert_code(
            "E_ALPHA_ROLE",
            lambda: adapter.bind_alpha(
                base["context"], base["pair"], policy_id="policy", **wrong_role
            ),
        )
        changed_value = dict(common, resolved_value=2.0)
        self.assert_code(
            "E_ALPHA_BINDING",
            lambda: adapter.bind_alpha(
                base["context"], base["pair"],
                policy_id="synthetic-fixed-alpha", **changed_value
            ),
        )
        changed_numeric_representation = dict(common, resolved_value=1)
        self.assert_code(
            "E_ALPHA_BINDING",
            lambda: adapter.bind_alpha(
                base["context"], base["pair"],
                policy_id="synthetic-fixed-alpha",
                **changed_numeric_representation,
            ),
        )
        other_record = self.pair_record(pair_id="other-pair", ordinal=8)
        other_context = self.context(pair_record=other_record)
        other_pair = adapter.bind_pair(other_context, other_record)
        other_alpha = adapter.bind_alpha(
            other_context, other_pair, policy_id="synthetic-fixed-alpha", **common
        )
        self.assert_code(
            "E_ALPHA_BINDING",
            lambda: adapter.plan_brain_loss(
                base["context"], base["pair"], base["roles"], base["worker"],
                source_rows=base["source_rows"], target_rows=base["target_rows"], alpha=other_alpha,
            ),
        )

    def test_alpha_rejects_leakage_receipt_and_numeric_edge_cases(self):
        base = self.bundle()
        common = {
            "policy_id": "synthetic-policy",
            "policy_sha256": "4" * 64,
            "selection_scope": "fixed_predeclared",
            "allowed_fit_role": "tau_train",
            "fixed_before_tau_test": True,
            "tau_test_or_ood_used": False,
            "resolved_value": 1.0,
            "selection_receipt_sha256": None,
        }
        cases = (
            ({"fixed_before_tau_test": False}, "E_ALPHA_LEAKAGE"),
            ({"tau_test_or_ood_used": True}, "E_ALPHA_LEAKAGE"),
            ({"resolved_value": 0.0}, "E_ALPHA_VALUE"),
            ({"resolved_value": float("nan")}, "E_ALPHA_VALUE"),
            ({"selection_receipt_sha256": "5" * 64}, "E_ALPHA_RECEIPT"),
            ({"selection_scope": "tau_train_only"}, "E_SHA256"),
        )
        for overrides, code in cases:
            values = dict(common, **overrides)
            with self.subTest(overrides=overrides):
                self.assert_code(
                    code,
                    lambda values=values: adapter.bind_alpha(
                        base["context"], base["pair"], **values
                    ),
                )

    def test_tau_train_selected_alpha_receipt_survives_plan_and_summary(self):
        base = self.bundle(alpha_scope="tau_train_only")
        selected = adapter.bind_alpha(
            base["context"], base["pair"],
            policy_id="synthetic-tau-train-selection",
            policy_sha256="4" * 64,
            selection_scope="tau_train_only",
            allowed_fit_role="tau_train",
            fixed_before_tau_test=True,
            tau_test_or_ood_used=False,
            resolved_value=1.0,
            selection_receipt_sha256="5" * 64,
        )
        brain = adapter.plan_brain_loss(
            base["context"], base["pair"], base["roles"], base["worker"],
            source_rows=base["source_rows"], target_rows=base["target_rows"],
            alpha=selected,
        )
        content = adapter.plan_content(
            base["context"], base["pair"], base["roles"], base["worker"],
            response_rows=base["response"], feature_rows_by_layer=base["features"],
        )
        summary = self.summarize(base, content, brain, selected)
        self.assertEqual(
            brain.record["alpha_binding_record"]["selection_scope"],
            "tau_train_only",
        )
        self.assertEqual(self.export(base, content, brain, summary, selected)["brain_fit_unit_count"], 3)

    def test_paired_mode_rejects_missing_or_duplicate_alignment_authority(self):
        base = self.bundle("authenticated_cross_subject_pairing")
        roles = base["roles"].record
        common = {
            "manifest_sha256": "1" * 64,
            "regular_only_exclusion_receipt_sha256": "2" * 64,
            "source_unique_rows": roles["source_unique_rows"],
            "target_unique_rows": roles["target_unique_rows"],
            "source_feature_rows_by_layer": roles["source_feature_rows_by_layer"],
            "tau_train_ordered_ids": roles["tau_train_ordered_ids"],
            "tau_test_ordered_ids": roles["tau_test_ordered_ids"],
            "source_tau_presentations": roles["source_tau_presentations"],
            "target_tau_presentations": roles["target_tau_presentations"],
        }
        self.assert_code(
            "E_ALIGNMENT_RECEIPT",
            lambda: adapter.bind_structural_roles(
                base["context"], base["pair"],
                cross_subject_alignment_receipt_sha256=None, **common
            ),
        )
        duplicated = copy.deepcopy(roles["target_tau_presentations"])
        duplicated[1]["cross_subject_alignment_key"] = duplicated[0]["cross_subject_alignment_key"]
        duplicated[1]["image_id"] = duplicated[0]["image_id"]
        bad = dict(common, target_tau_presentations=duplicated)
        self.assert_code(
            "E_ALIGNMENT_KEY",
            lambda: adapter.bind_structural_roles(
                base["context"], base["pair"],
                cross_subject_alignment_receipt_sha256="3" * 64, **bad
            ),
        )

    def test_wrong_participant_role_and_cross_context_worker_fail(self):
        base = self.bundle()
        response = base["response"].record
        wrong_participant = adapter.make_ordered_rows(
            base["roles"], participant_id=base["pair"].record["target_id"],
            role="source_unique_train", row_ids=response["row_ids"],
            image_ids=response["image_ids"], shape=response["shape"],
        )
        self.assert_code(
            "E_CONTENT_RESPONSE_ALIGNMENT",
            lambda: adapter.plan_content(
                base["context"], base["pair"], base["roles"], base["worker"],
                response_rows=wrong_participant, feature_rows_by_layer=base["features"],
            ),
        )
        other_record = self.pair_record(pair_id="other-pair", ordinal=8)
        other_context = self.context(pair_record=other_record)
        foreign_worker = adapter.bind_worker(
            other_context, max_workers=1, explicit_worker_limit=1
        )
        self.assert_code(
            "E_WORKER_BINDING",
            lambda: adapter.plan_content(
                base["context"], base["pair"], base["roles"], foreign_worker,
                response_rows=base["response"], feature_rows_by_layer=base["features"],
            ),
        )

    def test_duck_typed_plans_alpha_and_summary_are_rejected(self):
        base, content, brain, _ = self.plans()

        class Fake:
            record = {"synthetic_only": False, "empirical_capable": True}
            seal = "a" * 64

        self.assert_code(
            "E_TYPE",
            lambda: self.summarize(base, Fake(), brain),
        )
        self.assert_code("E_TYPE", lambda: self.export(base, content, brain, Fake()))
        self.assert_code(
            "E_TYPE",
            lambda: adapter.plan_brain_loss(
                base["context"], base["pair"], base["roles"], base["worker"],
                source_rows=base["source_rows"], target_rows=base["target_rows"],
                alpha=Fake(),
            ),
        )

    def test_resealed_self_consistent_plan_tampering_is_rederived(self):
        base, content, _, _ = self.plans("authenticated_cross_subject_pairing")
        content_record = content.record
        content_record["source_row_count"] = 999
        content_record["response_shape"][0] = 999
        for feature in content_record["feature_rows"]:
            feature["dynamic_buffer_shape"][0] = 999
        raw_content = json.dumps(content_record, sort_keys=True, separators=(",", ":")).encode()
        resealed_content = adapter.ContentPlan(
            raw_content, adapter._seal("ContentPlan", raw_content)
        )
        brain = adapter.plan_brain_loss(
            base["context"], base["pair"], base["roles"], base["worker"],
            source_rows=base["source_rows"], target_rows=base["target_rows"],
            alpha=base["alpha"],
        )
        self.assert_code(
            "E_CONTENT_PLAN_BINDING",
            lambda: self.summarize(base, resealed_content, brain),
        )
        brain_record = brain.record
        brain_record["target_aligned_indices"][0], brain_record["target_aligned_indices"][1] = (
            brain_record["target_aligned_indices"][1],
            brain_record["target_aligned_indices"][0],
        )
        raw_brain = json.dumps(brain_record, sort_keys=True, separators=(",", ":")).encode()
        resealed_brain = adapter.BrainPlan(raw_brain, adapter._seal("BrainPlan", raw_brain))
        valid_content = adapter.plan_content(
            base["context"], base["pair"], base["roles"], base["worker"],
            response_rows=base["response"], feature_rows_by_layer=base["features"],
        )
        self.assert_code(
            "E_BRAIN_PLAN_BINDING",
            lambda: self.summarize(base, valid_content, resealed_brain),
        )

    def test_subclasses_are_rejected_by_exact_type_consumers(self):
        base, content, brain, _ = self.plans()

        class ContentSubclass(adapter.ContentPlan):
            pass

        subclass = ContentSubclass.__new__(ContentSubclass)
        object.__setattr__(
            subclass, "canonical_record_json", content.canonical_record_json
        )
        object.__setattr__(subclass, "seal", content.seal)
        self.assert_code(
            "E_TYPE",
            lambda: self.summarize(base, subclass, brain),
        )

    def test_content_plan_constructor_rejects_invalid_tail_dimensions(self):
        _, content, _, _ = self.plans()
        for field, value in (
            ("response_shape", -1),
            ("response_shape", 1.5),
            ("response_shape", True),
            ("dynamic_buffer_shape", -1),
            ("dynamic_buffer_shape", 1.5),
            ("dynamic_buffer_shape", True),
        ):
            record = content.record
            if field == "response_shape":
                record[field][-1] = value
            else:
                record["feature_rows"][0][field][-1] = value
            raw = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            with self.subTest(field=field, value=value):
                self.assert_code(
                    "E_CONTENT_PLAN",
                    lambda raw=raw: adapter.ContentPlan(
                        raw, adapter._seal("ContentPlan", raw)
                    ),
                )

    def test_brain_plan_constructor_rejects_bool_and_float_indices(self):
        _, _, brain, _ = self.plans()
        for value in (True, 0.0):
            record = brain.record
            record["source_group_indices"][0][0] = value
            raw = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            with self.subTest(value=value):
                self.assert_code(
                    "E_BRAIN_PLAN",
                    lambda raw=raw: adapter.BrainPlan(
                        raw, adapter._seal("BrainPlan", raw)
                    ),
                )

    def test_brain_plan_rejects_invalid_index_containers_and_ranges(self):
        _, _, brain, _ = self.plans()
        mutations = (
            ("source_group_indices", None, None),
            ("target_group_indices", None, False),
            ("source_aligned_indices", None, ""),
            ("target_aligned_indices", None, {}),
            ("source_group_indices", (0, 0), -1),
            ("source_group_indices", (0, 0), 999),
            ("source_group_indices", (0, 0), "0"),
            ("source_group_indices", (0, 1), 0),
        )
        for field, position, value in mutations:
            record = brain.record
            if position is None:
                record[field] = value
            else:
                record[field][position[0]][position[1]] = value
            raw = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            with self.subTest(field=field, position=position, value=value):
                self.assert_code(
                    "E_BRAIN_PLAN",
                    lambda raw=raw: adapter.BrainPlan(
                        raw, adapter._seal("BrainPlan", raw)
                    ),
                )

    def test_ordered_and_feature_rows_reject_invalid_dimensions(self):
        base = self.bundle()
        response = base["response"].record
        self.assert_code(
            "E_SHAPE",
            lambda: adapter.make_ordered_rows(
                base["roles"],
                participant_id=base["pair"].record["source_id"],
                role="source_unique_train",
                row_ids=response["row_ids"],
                image_ids=response["image_ids"],
                shape=[len(response["row_ids"]), 0],
            ),
        )
        feature = base["features"]["fc8"].record
        self.assert_code(
            "E_SHAPE",
            lambda: adapter.make_feature_rows(
                base["roles"],
                layer_id="fc8",
                row_ids=feature["row_ids"],
                image_ids=feature["image_ids"],
                shape=[len(feature["row_ids"]), True],
            ),
        )

    def test_arbitrary_row_seals_cannot_survive_plan_regeneration(self):
        base, content, brain, _ = self.plans()
        for field in ("response_rows_seal", "feature_rows_seal"):
            record = content.record
            if field == "response_rows_seal":
                record[field] = "9" * 64
            else:
                record["feature_rows"][0]["rows_seal"] = "9" * 64
            raw = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            resealed = adapter.ContentPlan(
                raw, adapter._seal("ContentPlan", raw)
            )
            with self.subTest(field=field):
                self.assert_code(
                    "E_CONTENT_PLAN_BINDING",
                    lambda resealed=resealed: self.summarize(
                        base, resealed, brain
                    ),
                )
        for field in ("source_rows_seal", "target_rows_seal"):
            record = brain.record
            record[field] = "9" * 64
            raw = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            resealed = adapter.BrainPlan(raw, adapter._seal("BrainPlan", raw))
            with self.subTest(field=field):
                self.assert_code(
                    "E_BRAIN_PLAN_BINDING",
                    lambda resealed=resealed: self.summarize(
                        base, content, resealed
                    ),
                )

    def test_resealed_summary_cannot_export_without_matching_authorities(self):
        base, content, brain, summary = self.plans()
        record = json.loads(summary.canonical_record_json.decode("utf-8"))
        record["content_row_count"] = 999
        record["brain_fit_unit_count"] = 999
        raw = json.dumps(
            record, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        resealed = adapter.PlanSummary(
            raw, adapter._seal("PlanSummary", raw)
        )
        self.assert_code(
            "E_SUMMARY_BINDING",
            lambda: self.export(base, content, brain, resealed),
        )
        for field in ("content_plan_seal", "brain_plan_seal", "roles_seal"):
            record = json.loads(summary.canonical_record_json.decode("utf-8"))
            record[field] = "9" * 64
            raw = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            resealed = adapter.PlanSummary(
                raw, adapter._seal("PlanSummary", raw)
            )
            with self.subTest(field=field):
                self.assert_code(
                    "E_SUMMARY_BINDING",
                    lambda resealed=resealed: self.export(
                        base, content, brain, resealed
                    ),
                )

    def test_frozen_plan_tamper_is_revalidated_by_summary(self):
        bundle, content, brain, summary = self.plans()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            content.seal = "0" * 64
        tampered_record = content.record
        tampered_record["source_row_count"] = 999
        tampered = adapter.ContentPlan.__new__(adapter.ContentPlan)
        object.__setattr__(tampered, "canonical_record_json", json.dumps(tampered_record, sort_keys=True, separators=(",", ":")).encode())
        object.__setattr__(tampered, "seal", content.seal)
        self.assert_code(
            "E_CONTENT_PLAN",
            lambda: self.summarize(bundle, tampered, brain),
        )
        exported = self.export(bundle, content, brain, summary)
        exported["content_row_count"] = 999
        self.assertNotEqual(
            exported["content_row_count"],
            self.export(bundle, content, brain, summary)["content_row_count"],
        )

    def test_mutating_original_structures_does_not_change_binding(self):
        base = self.bundle()
        source_image = base["roles"].record["source_unique_rows"][0]["image_id"]
        tau_image = base["roles"].record["source_tau_presentations"][0]["image_id"]
        base["source_unique"][0]["image_id"] = "mutated"
        base["source_tau"][0]["image_id"] = "mutated"
        self.assertEqual(base["roles"].record["source_unique_rows"][0]["image_id"], source_image)
        self.assertEqual(base["roles"].record["source_tau_presentations"][0]["image_id"], tau_image)

    def test_planners_accept_no_payload_device_or_executor_arguments(self):
        base = self.bundle()
        for keyword in ("payload", "device", "executor"):
            with self.subTest(keyword=keyword):
                with self.assertRaises(TypeError):
                    adapter.plan_content(
                        base["context"], base["pair"], base["roles"], base["worker"],
                        response_rows=base["response"], feature_rows_by_layer=base["features"],
                        **{keyword: object()},
                    )


if __name__ == "__main__":
    unittest.main()
