from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import numpy as np
import ep12_executor.full_search_null as full_search_module

from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.fitted_null import (
    FittedNullDesign,
    TypePooledExpansionPlan,
    freeze_null_seed_manifest,
    null_seed_binding,
)
from ep12_executor.full_search_null import (
    CALLBACK_IDENTITY_SCHEMA,
    CALLBACK_PROTOCOL,
    FINAL_ARTIFACT_SCHEMA,
    MODEL_TRIPLET,
    REQUEUE_EXIT_CODE,
    RESTART_SENTINEL_BYTES,
    FullSearchNullError,
    FullSearchNullRequeueRequested,
    FullSearchNullRuntime,
    FullSearchNullTrialTimeout,
    _aggregate_file_inventory,
    _copy_authenticated_aggregate_tree,
    _invoke_fit_score,
    _write_once_json,
    aggregate_full_search_null,
    freeze_final_result,
    freeze_fit_score_result,
    freeze_procedure_lock,
    run_null_replicate,
)
from ep12_executor.null_controller import freeze_null_controller_program
from ep12_executor.policy import EpisodePolicy, digest_bytes, digest_object
from ep12_executor.scientific_models import FittedPredictiveModel, helmert_ilr_basis


EPISODE_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = EPISODE_ROOT / "SEARCH_POLICY.yaml"
PLAN_PATH = EPISODE_ROOT / "outputs/executor/OBSERVED_POST36_TAIL_PLAN.json"
SELECTION_SHA256 = "ab" * 32
ROLE_SHA256 = "cd" * 32


def _design() -> FittedNullDesign:
    return FittedNullDesign.create(
        neuron_ids=[101, 102, 103, 104],
        focal_types=["A", "A", "A", "A"],
        sides=["L", "L", "R", "R"],
        covariates=np.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=float),
        raw_counts=np.asarray(
            [[12, 4, 2, 1], [10, 5, 1, 2], [11, 3, 4, 1], [13, 2, 3, 2]],
            dtype=np.int64,
        ),
        raw_vocabulary=[
            "typed::x", "typed::y", "untyped_status::Traced::complete",
            "missing_annotation",
        ],
        collapsed_vocabulary=[
            "typed::x", "other_typed", "untyped", "missing_annotation",
        ],
        raw_to_collapsed=[0, 1, 2, 3],
        collapsed_typed_bins=["typed::x", "other_typed"],
    )


def _model(design: FittedNullDesign) -> FittedPredictiveModel:
    dimension = len(design.collapsed_vocabulary) - 1
    basis = helmert_ilr_basis(len(design.collapsed_vocabulary))
    coefficients = np.zeros((design.covariates.shape[1] + 1, dimension))
    coefficients[0] = np.asarray([1.0, 0.0, -0.5, -0.5]) @ basis
    return FittedPredictiveModel(
        model_id="T",
        basis=basis,
        nuisance_coefficients=coefficients,
        latent_offsets=np.zeros((1, dimension)),
        log_weights=np.zeros(1),
        prior_means=np.zeros((1, dimension)),
        prior_covariances=np.eye(dimension)[None, :, :] * 1e-10,
        prior_weights=np.ones(1),
        integration_draws=1,
        integration_seed=7,
        parameter_count=int(coefficients.size),
        diagnostics={},
        pseudocount=0.5,
    )


def _program(policy: EpisodePolicy) -> dict[str, Any]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    non_trial = [row["falsifier_id"] for row in plan["non_trial_required_gates"]]
    countable = plan["required_falsifier_classification"][
        "countable_observed_trial_ids"
    ]
    return freeze_null_controller_program(
        policy=policy,
        coverage_configurations=coverage_configurations(policy),
        post36_tail_plan=plan,
        countable_falsifier_ids=countable,
        non_trial_required_falsifier_ids=non_trial,
    )


class SyntheticCallback:
    def __init__(
        self,
        *,
        artifact_root: Path,
        selection_sha256: str,
        supported_falsifiers: Sequence[str],
        improve_at: int | None = None,
        fail_once_at: int | None = None,
        restart_sentinel: Path | None = None,
        request_restart_at: int | None = None,
        increasing_objective: bool = False,
    ) -> None:
        self.artifact_root = artifact_root
        self.improve_at = improve_at
        self.fail_once_at = fail_once_at
        self.restart_sentinel = restart_sentinel
        self.request_restart_at = request_restart_at
        self.increasing_objective = increasing_objective
        self.failed = False
        self.calls: Counter[int] = Counter()
        self._identity = {
            "schema_version": CALLBACK_IDENTITY_SCHEMA,
            "protocol": CALLBACK_PROTOCOL,
            "callback_id": "synthetic_adaptive_driver_test",
            "callback_version": "1",
            "implementation_sha256": "ef" * 32,
            "selection_program_sha256": selection_sha256,
            "supported_model_triplet": list(MODEL_TRIPLET),
            "supported_falsifier_ids": list(supported_falsifiers),
            "development_only": True,
            "final_connectivity_accessed": False,
            "nested_null_searches_supported": False,
        }

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    def fit_score(
        self,
        *,
        sample: Any,
        trial: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        ordinal = int(trial["valid_trial_ordinal"])
        if self.fail_once_at == ordinal and not self.failed:
            self.failed = True
            raise RuntimeError("synthetic interruption")
        self.calls[ordinal] += 1
        value = (
            float(ordinal)
            if self.increasing_objective
            else 100.0 if self.improve_at == ordinal else 10.0 - ordinal / 100.0
        )
        objective = {
            "left_to_right": value,
            "right_to_left": value,
            "bilateral": value,
            "bilateral_floor": value,
            "direction_disagreement": 0.0,
            "types_both_above_margin": 1,
        }
        artifact = {
            "schema_version": "ep12.synthetic_callback_artifact.v1",
            "replicate_index": sample.replicate_index,
            "replicate_id": sample.replicate_id,
            "trial_id": trial["trial_id"],
            "configuration_sha256": trial["configuration_sha256"],
            "objective": objective,
            "complete_T_U_M_result": True,
            "executed_complete_T_U_M_fit_count": 1,
            "executed_model_family_fit_count": len(MODEL_TRIPLET),
            "development_only": True,
            "final_connectivity_accessed": False,
        }
        artifact["artifact_sha256"] = digest_object(artifact)
        relative = Path("callback_artifacts") / sample.replicate_id / (
            f"{ordinal:03d}_{trial['configuration_sha256']}.json"
        )
        path = self.artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n"
        if path.exists():
            if path.read_bytes() != encoded:
                raise RuntimeError("synthetic callback artifact changed")
        else:
            path.write_bytes(encoded)
        if (
            self.restart_sentinel is not None
            and self.request_restart_at == ordinal
        ):
            self.restart_sentinel.write_bytes(RESTART_SENTINEL_BYTES)
        return freeze_fit_score_result(
            replicate_index=sample.replicate_index,
            replicate_id=sample.replicate_id,
            trial=trial,
            objective=objective,
            executed_complete_T_U_M_fit_count=1,
            fit_score_artifact_path=relative.as_posix(),
            fit_score_artifact_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def finalize_search(
        self,
        *,
        sample: Any,
        schedule: Mapping[str, Any],
        trial_results: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        eligible = [
            (trial, result)
            for trial, result in zip(schedule["trials"], trial_results)
            if trial["promotion_eligible"]
        ]
        selected_trial, selected_result = max(
            eligible, key=lambda pair: float(pair[1]["selection_statistic"])
        )
        artifact_core = {
            "schema_version": FINAL_ARTIFACT_SCHEMA,
            "replicate_index": sample.replicate_index,
            "replicate_id": sample.replicate_id,
            "executed_trial_count": len(trial_results),
            "realized_schedule_sha256": schedule["schedule_sha256"],
            "selected_trial_id": selected_trial["trial_id"],
            "selected_configuration_sha256": selected_trial[
                "configuration_sha256"
            ],
            "selected_statistic": float(selected_result["selection_statistic"]),
            "selection_program_sha256": SELECTION_SHA256,
            "complete_controller_rerun": True,
            "initial_ledger_empty": True,
            "development_only": True,
            "final_connectivity_accessed": False,
        }
        artifact = {
            **artifact_core,
            "artifact_sha256": digest_object(artifact_core),
        }
        relative = Path("callback_artifacts") / sample.replicate_id / "selection.json"
        path = self.artifact_root / relative
        encoded = json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n"
        if path.exists():
            if path.read_bytes() != encoded:
                raise RuntimeError("synthetic final selection artifact changed")
        else:
            path.write_bytes(encoded)
        return freeze_final_result(
            replicate_index=sample.replicate_index,
            replicate_id=sample.replicate_id,
            executed_trial_count=len(trial_results),
            selected_trial_id=selected_trial["trial_id"],
            selected_configuration_sha256=selected_trial["configuration_sha256"],
            selected_statistic=float(selected_result["selection_statistic"]),
            selection_program_sha256=SELECTION_SHA256,
            final_selection_artifact_path=relative.as_posix(),
            final_selection_artifact_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def _runtime(
    policy: EpisodePolicy,
    *,
    callback: SyntheticCallback,
    program: Mapping[str, Any],
    durable_output_root: Path | None = None,
) -> FullSearchNullRuntime:
    design = _design()
    model = _model(design)
    generators = {("A", "L"): model, ("A", "R"): model}
    expansion = TypePooledExpansionPlan.build(design)
    binding = null_seed_binding(
        design, generators, expansion, selection_sha256=SELECTION_SHA256
    )
    seeds = freeze_null_seed_manifest(master_seed=12345, binding=binding)
    lock = freeze_procedure_lock(
        policy=policy,
        controller_program=program,
        design=design,
        generators=generators,
        expansion_plan=expansion,
        seed_manifest=seeds,
        selection_sha256=SELECTION_SHA256,
        development_role_manifest_sha256=ROLE_SHA256,
        callback_identity=callback.identity(),
    )
    return FullSearchNullRuntime(
        design=design,
        generators=generators,
        expansion_plan=expansion,
        seed_manifest=seeds,
        selection_sha256=SELECTION_SHA256,
        development_role_manifest_sha256=ROLE_SHA256,
        controller_program=program,
        procedure_lock=lock,
        callback=callback,
        durable_output_root=durable_output_root,
    )


class FullSearchNullAdaptiveTests(unittest.TestCase):
    def test_atomic_json_retries_a_cross_node_temp_name_collision(self) -> None:
        real_open = os.open
        collided = False

        def collision_once(
            path: str,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal collided
            if not collided and flags & os.O_EXCL:
                collided = True
                raise FileExistsError(path)
            if dir_fd is None:
                return real_open(path, flags, mode)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "shared.json"
            with patch.object(full_search_module.os, "open", side_effect=collision_once):
                _write_once_json(
                    target, {"shared": True}, root=Path(temporary)
                )
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")), {"shared": True}
            )
            self.assertTrue(collided)

    def test_atomic_steps_reject_symlinked_components_without_outside_write(
        self,
    ) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        for linked_component in ("replicates", "null_001", "steps"):
            with self.subTest(linked_component=linked_component), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                root = base / "scratch"
                outside = base / "outside"
                root.mkdir()
                outside.mkdir()
                if linked_component == "replicates":
                    (root / "replicates").symlink_to(
                        outside, target_is_directory=True
                    )
                else:
                    replicate_root = root / "replicates"
                    replicate_root.mkdir()
                    if linked_component == "null_001":
                        (replicate_root / "null_001").symlink_to(
                            outside, target_is_directory=True
                        )
                    else:
                        null_root = replicate_root / "null_001"
                        null_root.mkdir()
                        (null_root / "steps").symlink_to(
                            outside, target_is_directory=True
                        )
                callback = SyntheticCallback(
                    artifact_root=root,
                    selection_sha256=SELECTION_SHA256,
                    supported_falsifiers=program["countable_falsifier_ids"],
                )
                with self.assertRaisesRegex(
                    FullSearchNullError, "symlink|not a directory"
                ):
                    run_null_replicate(
                        _runtime(policy, callback=callback, program=program),
                        policy=policy,
                        replicate_index=0,
                        output_dir=root,
                    )
                self.assertEqual(list(outside.rglob("*")), [])

    def test_manifest_copy_excludes_stale_atomic_temporaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            stage = root / "stage"
            source.mkdir()
            payload = b'{"authenticated":true}\n'
            (source / "artifact.json").write_bytes(payload)
            manifest_core = {
                "schema_version": "ep12.executed_full_search_null_manifest.v3",
                "files": {
                    "artifact.json": hashlib.sha256(payload).hexdigest(),
                },
            }
            manifest = {
                **manifest_core,
                "manifest_sha256": digest_object(manifest_core),
            }
            (source / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            stale_root = source / ".ep12_atomic_tmp"
            stale_root.mkdir()
            (stale_root / ("0" * 16 + "." + "1" * 16 + "." + "2" * 16 + ".tmp")).write_bytes(
                b"stale partial write"
            )

            _copy_authenticated_aggregate_tree(
                source=source, destination=stage, manifest=manifest
            )

            self.assertEqual((stage / "artifact.json").read_bytes(), payload)
            self.assertEqual(
                (stage / "manifest.json").read_text(encoding="utf-8"),
                (source / "manifest.json").read_text(encoding="utf-8"),
            )
            self.assertFalse(any(
                path.is_file()
                for path in stage.rglob("*")
                if ".ep12_atomic_tmp" in path.parts
            ))

    def test_isolated_trial_timeout_terminates_work(self) -> None:
        class SlowCallback:
            @staticmethod
            def fit_score(**_: Any) -> Mapping[str, Any]:
                time.sleep(0.5)
                return {"unexpected": True}

        started = time.monotonic()
        with self.assertRaisesRegex(FullSearchNullError, "wall-time limit"):
            _invoke_fit_score(
                SlowCallback(),
                sample=None,
                trial={},
                context={},
                timeout_seconds=0.03,
                isolate=True,
            )
        self.assertLess(time.monotonic() - started, 0.4)

    def test_no_improvement_executes_dynamic_52_and_replay_calls_nothing(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            callback = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
            )
            runtime = _runtime(policy, callback=callback, program=program)
            checkpoint = run_null_replicate(
                runtime, policy=policy, replicate_index=0, output_dir=root
            )
            self.assertEqual(checkpoint["executed_trial_count"], 52)
            self.assertEqual(checkpoint["stop_reason"], "patience_exhausted")
            self.assertEqual(sum(callback.calls.values()), 52)
            callback.calls.clear()
            replayed = run_null_replicate(
                runtime, policy=policy, replicate_index=0, output_dir=root
            )
            self.assertEqual(replayed["checkpoint_sha256"], checkpoint["checkpoint_sha256"])
            self.assertEqual(callback.calls, Counter())

    def test_step_checkpoint_resume_after_interruption_is_exact(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
                fail_once_at=25,
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                run_null_replicate(
                    _runtime(policy, callback=first, program=program),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                )
            self.assertEqual(
                len(list((root / "replicates/null_001/steps").glob("*.json"))),
                24,
            )
            sample_receipt_path = root / "replicates/null_001/sample_receipt.json"
            original_sample_receipt = sample_receipt_path.read_bytes()
            sample_receipt_path.write_bytes(original_sample_receipt + b"tamper")
            with self.assertRaisesRegex(FullSearchNullError, "artifact already differs"):
                run_null_replicate(
                    _runtime(policy, callback=first, program=program),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                )
            sample_receipt_path.write_bytes(original_sample_receipt)
            resumed = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
            )
            checkpoint = run_null_replicate(
                _runtime(policy, callback=resumed, program=program),
                policy=policy,
                replicate_index=0,
                output_dir=root,
            )
            self.assertEqual(checkpoint["executed_trial_count"], 52)
            self.assertEqual(min(resumed.calls), 25)
            self.assertEqual(sum(resumed.calls.values()), 28)

    def test_restart_request_stops_at_boundary_and_resume_is_byte_exact(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)

        def deterministic_invoke(
            callback: Any,
            *,
            sample: Any,
            trial: Mapping[str, Any],
            context: Mapping[str, Any],
            timeout_seconds: float,
            isolate: bool,
        ) -> tuple[Mapping[str, Any], float]:
            del timeout_seconds, isolate
            return callback.fit_score(
                sample=sample, trial=trial, context=context
            ), 1.25

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            baseline_root = parent / "baseline"
            resumed_root = parent / "resumed"
            baseline_root.mkdir()
            resumed_root.mkdir()
            baseline_callback = SyntheticCallback(
                artifact_root=baseline_root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
            )
            sentinel = resumed_root / "restart.request"
            interrupted_callback = SyntheticCallback(
                artifact_root=resumed_root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
                restart_sentinel=sentinel,
                request_restart_at=17,
            )
            with patch.object(
                full_search_module,
                "_invoke_fit_score",
                side_effect=deterministic_invoke,
            ):
                baseline = run_null_replicate(
                    _runtime(policy, callback=baseline_callback, program=program),
                    policy=policy,
                    replicate_index=0,
                    output_dir=baseline_root,
                )
                with self.assertRaises(FullSearchNullRequeueRequested) as caught:
                    run_null_replicate(
                        _runtime(
                            policy, callback=interrupted_callback, program=program
                        ),
                        policy=policy,
                        replicate_index=0,
                        output_dir=resumed_root,
                        restart_sentinel=sentinel,
                    )
                self.assertEqual(caught.exception.completed_trial_count, 17)
                self.assertEqual(
                    len(list(
                        (resumed_root / "replicates/null_001/steps").glob("*.json")
                    )),
                    17,
                )
                self.assertFalse(
                    (resumed_root / "replicates/null_001/checkpoint.json").exists()
                )
                self.assertFalse(
                    (resumed_root / "callback_artifacts/null_001/selection.json").exists()
                )
                prefix = {
                    path.relative_to(resumed_root): path.read_bytes()
                    for path in (
                        resumed_root / "replicates/null_001/steps"
                    ).glob("*.json")
                }
                sentinel.unlink()
                resumed_callback = SyntheticCallback(
                    artifact_root=resumed_root,
                    selection_sha256=SELECTION_SHA256,
                    supported_falsifiers=program["countable_falsifier_ids"],
                )
                resumed = run_null_replicate(
                    _runtime(policy, callback=resumed_callback, program=program),
                    policy=policy,
                    replicate_index=0,
                    output_dir=resumed_root,
                    restart_sentinel=sentinel,
                )

            self.assertEqual(resumed["checkpoint_sha256"], baseline["checkpoint_sha256"])
            self.assertEqual(min(resumed_callback.calls), 18)
            self.assertEqual(sum(resumed_callback.calls.values()), 35)
            self.assertEqual(
                prefix,
                {
                    relative: (resumed_root / relative).read_bytes()
                    for relative in prefix
                },
            )
            baseline_files = {
                path.relative_to(baseline_root): path.read_bytes()
                for path in baseline_root.rglob("*")
                if path.is_file()
            }
            resumed_files = {
                path.relative_to(resumed_root): path.read_bytes()
                for path in resumed_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(resumed_files, baseline_files)

    def test_restart_sentinel_tamper_fails_before_a_trial(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sentinel = root / "restart.request"
            sentinel.write_bytes(b"tampered\n")
            callback = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
            )
            with self.assertRaisesRegex(FullSearchNullError, "payload is invalid"):
                run_null_replicate(
                    _runtime(policy, callback=callback, program=program),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                    restart_sentinel=sentinel,
                )
            self.assertEqual(callback.calls, Counter())

            sentinel.unlink()
            sentinel.symlink_to(root / "missing-restart-request")
            with self.assertRaisesRegex(FullSearchNullError, "restart sentinel"):
                run_null_replicate(
                    _runtime(policy, callback=callback, program=program),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                    restart_sentinel=sentinel,
                )
            self.assertEqual(callback.calls, Counter())

    def test_cumulative_wall_budget_survives_requeue_and_fails_closed(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)

        def budgeted_invoke(
            callback: Any,
            *,
            sample: Any,
            trial: Mapping[str, Any],
            context: Mapping[str, Any],
            timeout_seconds: float,
            isolate: bool,
        ) -> tuple[Mapping[str, Any], float]:
            del isolate
            ordinal = int(trial["valid_trial_ordinal"])
            if ordinal <= 10:
                self.assertAlmostEqual(timeout_seconds, 12.0 * 3600.0)
                return callback.fit_score(
                    sample=sample, trial=trial, context=context
                ), 11.0 * 3600.0
            self.assertEqual(ordinal, 11)
            self.assertAlmostEqual(timeout_seconds, 10.0 * 3600.0)
            raise FullSearchNullTrialTimeout(
                "trial exceeded the frozen wall-time limit"
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            durable_outputs = root / "durable_outputs"
            durable_outputs.mkdir()
            durable_root = durable_outputs / "full_search_null"
            sentinel = root / "restart.request"
            first = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
                restart_sentinel=sentinel,
                request_restart_at=5,
            )
            with (
                patch.object(
                    full_search_module,
                    "_EPISODE_OUTPUTS_ROOT",
                    durable_outputs,
                ),
                patch.object(
                    full_search_module,
                    "_invoke_fit_score",
                    side_effect=budgeted_invoke,
                ),
            ):
                with self.assertRaises(FullSearchNullRequeueRequested):
                    run_null_replicate(
                        _runtime(
                            policy,
                            callback=first,
                            program=program,
                            durable_output_root=durable_root,
                        ),
                        policy=policy,
                        replicate_index=0,
                        output_dir=root,
                        restart_sentinel=sentinel,
                    )
            self.assertEqual(sum(first.calls.values()), 5)
            self.assertEqual(
                len(list(
                    (root / "replicates/null_001/steps").glob("step_*.json")
                )),
                5,
            )

            sentinel.unlink()
            resumed = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
            )
            with (
                patch.object(
                    full_search_module,
                    "_EPISODE_OUTPUTS_ROOT",
                    durable_outputs,
                ),
                patch.object(
                    full_search_module,
                    "_invoke_fit_score",
                    side_effect=budgeted_invoke,
                ),
                self.assertRaisesRegex(
                    FullSearchNullError, "exhausted the cumulative wall budget"
                ),
            ):
                run_null_replicate(
                    _runtime(
                        policy,
                        callback=resumed,
                        program=program,
                        durable_output_root=durable_root,
                    ),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                    restart_sentinel=sentinel,
                )
            self.assertEqual(sorted(resumed.calls), [6, 7, 8, 9, 10])
            self.assertFalse(
                (root / "replicates/null_001/checkpoint.json").exists()
            )
            self.assertFalse(
                (root / "callback_artifacts/null_001/selection.json").exists()
            )
            receipt_path = (
                root / "replicates/null_001/wall_budget_exhaustion.json"
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            durable_receipt_path = (
                durable_outputs
                / "full_search_null_incomplete_searches"
                / "null_001"
                / "wall_budget_exhaustion.json"
            )
            self.assertEqual(
                durable_receipt_path.read_bytes(), receipt_path.read_bytes()
            )
            receipt_core = {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
            self.assertEqual(receipt["receipt_sha256"], digest_object(receipt_core))
            self.assertEqual(receipt["terminal_status"], "incomplete_search")
            self.assertEqual(receipt["executed_trial_count"], 10)
            self.assertEqual(
                receipt["cumulative_persisted_trial_wall_seconds"],
                110.0 * 3600.0,
            )
            self.assertEqual(
                receipt["callback_timeout_seconds"], 10.0 * 3600.0
            )
            self.assertFalse(receipt["finalization_performed"])
            self.assertFalse(receipt["aggregate_eligible"])

            never_called = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
            )
            with (
                patch.object(
                    full_search_module,
                    "_EPISODE_OUTPUTS_ROOT",
                    durable_outputs,
                ),
                patch.object(
                    full_search_module,
                    "_invoke_fit_score",
                    side_effect=AssertionError("exhausted search reran a callback"),
                ),
                self.assertRaisesRegex(
                    FullSearchNullError, "incomplete search"
                ),
            ):
                run_null_replicate(
                    _runtime(
                        policy,
                        callback=never_called,
                        program=program,
                        durable_output_root=durable_root,
                    ),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                )
            self.assertEqual(never_called.calls, Counter())
            with (
                patch.object(
                    full_search_module,
                    "_EPISODE_OUTPUTS_ROOT",
                    durable_outputs,
                ),
                self.assertRaisesRegex(
                    FullSearchNullError, "exhausted.*incomplete"
                ),
            ):
                aggregate_full_search_null(
                    _runtime(
                        policy,
                        callback=never_called,
                        program=program,
                        durable_output_root=durable_root,
                    ),
                    policy=policy,
                    output_dir=root,
                )

    def test_wall_budget_stop_precedes_coincident_patience(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)

        def boundary_invoke(
            callback: Any,
            *,
            sample: Any,
            trial: Mapping[str, Any],
            context: Mapping[str, Any],
            timeout_seconds: float,
            isolate: bool,
        ) -> tuple[Mapping[str, Any], float]:
            del isolate
            ordinal = int(trial["valid_trial_ordinal"])
            prior = 8000.0 * min(ordinal - 1, 51)
            expected_timeout = min(12.0 * 3600.0, 120.0 * 3600.0 - prior)
            self.assertAlmostEqual(timeout_seconds, expected_timeout)
            duration = 24000.0 if ordinal == 52 else 8000.0
            return callback.fit_score(
                sample=sample, trial=trial, context=context
            ), duration

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            callback = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
            )
            with (
                patch.object(
                    full_search_module,
                    "_invoke_fit_score",
                    side_effect=boundary_invoke,
                ),
                self.assertRaisesRegex(
                    FullSearchNullError, "exhausted the cumulative wall budget"
                ),
            ):
                run_null_replicate(
                    _runtime(policy, callback=callback, program=program),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                )
            receipt = json.loads((
                root / "replicates/null_001/wall_budget_exhaustion.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(receipt["executed_trial_count"], 52)
            self.assertEqual(
                receipt["controller_status_at_boundary"]["reason"],
                "patience_exhausted",
            )
            self.assertFalse(
                (root / "replicates/null_001/checkpoint.json").exists()
            )

    def test_exact_108h_plus_12h_timeout_is_durable_and_fail_closed(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        per_trial_seconds = 12.0 * 3600.0

        def exact_boundary_invoke(
            callback: Any,
            *,
            sample: Any,
            trial: Mapping[str, Any],
            context: Mapping[str, Any],
            timeout_seconds: float,
            isolate: bool,
        ) -> tuple[Mapping[str, Any], float]:
            del isolate
            ordinal = int(trial["valid_trial_ordinal"])
            self.assertAlmostEqual(timeout_seconds, per_trial_seconds)
            if ordinal <= 9:
                return callback.fit_score(
                    sample=sample, trial=trial, context=context
                ), per_trial_seconds
            self.assertEqual(ordinal, 10)
            raise FullSearchNullTrialTimeout(
                "trial exceeded the frozen wall-time limit"
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            durable_outputs = root / "durable_outputs"
            durable_outputs.mkdir()
            durable_root = durable_outputs / "full_search_null"
            callback = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
            )

            def make_runtime(active_callback: SyntheticCallback) -> FullSearchNullRuntime:
                return _runtime(
                    policy,
                    callback=active_callback,
                    program=program,
                    durable_output_root=durable_root,
                )

            with (
                patch.object(
                    full_search_module,
                    "_EPISODE_OUTPUTS_ROOT",
                    durable_outputs,
                ),
                patch.object(
                    full_search_module,
                    "_invoke_fit_score",
                    side_effect=exact_boundary_invoke,
                ),
                self.assertRaisesRegex(
                    FullSearchNullError, "exhausted the cumulative wall budget"
                ),
            ):
                run_null_replicate(
                    make_runtime(callback),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                )
            self.assertEqual(sorted(callback.calls), list(range(1, 10)))
            scratch_receipt_path = (
                root / "replicates/null_001/wall_budget_exhaustion.json"
            )
            durable_receipt_path = (
                durable_outputs
                / "full_search_null_incomplete_searches"
                / "null_001"
                / "wall_budget_exhaustion.json"
            )
            scratch_bytes = scratch_receipt_path.read_bytes()
            self.assertEqual(durable_receipt_path.read_bytes(), scratch_bytes)
            receipt = json.loads(scratch_bytes)
            self.assertEqual(
                receipt["cumulative_persisted_trial_wall_seconds"],
                108.0 * 3600.0,
            )
            self.assertEqual(
                receipt["remaining_cumulative_wall_seconds_at_boundary"],
                per_trial_seconds,
            )
            self.assertEqual(
                receipt["callback_timeout_seconds"], per_trial_seconds
            )
            self.assertEqual(receipt["terminal_status"], "incomplete_search")
            self.assertEqual(
                receipt["receipt_sha256"],
                digest_object({
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_sha256"
                }),
            )
            self.assertFalse(
                (root / "replicates/null_001/checkpoint.json").exists()
            )
            self.assertFalse(
                (root / "callback_artifacts/null_001/selection.json").exists()
            )

            never_called = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
            )
            with (
                patch.object(
                    full_search_module,
                    "_EPISODE_OUTPUTS_ROOT",
                    durable_outputs,
                ),
                patch.object(
                    full_search_module,
                    "_invoke_fit_score",
                    side_effect=AssertionError("exact-boundary search retried"),
                ),
                self.assertRaisesRegex(FullSearchNullError, "incomplete search"),
            ):
                run_null_replicate(
                    make_runtime(never_called),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                )
            self.assertEqual(never_called.calls, Counter())

            out_of_range = dict(receipt)
            out_of_range["callback_timeout_seconds"] = per_trial_seconds + 1.0
            out_of_range["receipt_sha256"] = digest_object({
                key: value
                for key, value in out_of_range.items()
                if key != "receipt_sha256"
            })
            out_of_range_bytes = (
                json.dumps(out_of_range, indent=2, sort_keys=True).encode()
                + b"\n"
            )
            scratch_receipt_path.write_bytes(out_of_range_bytes)
            durable_receipt_path.write_bytes(out_of_range_bytes)
            with (
                patch.object(
                    full_search_module,
                    "_EPISODE_OUTPUTS_ROOT",
                    durable_outputs,
                ),
                self.assertRaisesRegex(
                    FullSearchNullError, "differs from authenticated steps"
                ),
            ):
                run_null_replicate(
                    make_runtime(never_called),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                )
            scratch_receipt_path.write_bytes(scratch_bytes)
            durable_receipt_path.write_bytes(scratch_bytes)

            with (
                patch.object(
                    full_search_module,
                    "_EPISODE_OUTPUTS_ROOT",
                    durable_outputs,
                ),
                self.assertRaisesRegex(
                    FullSearchNullError, "exhausted.*incomplete"
                ),
            ):
                aggregate_full_search_null(
                    make_runtime(never_called),
                    policy=policy,
                    output_dir=root,
                )

    def test_maximum_stop_precedes_coincident_wall_boundary(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)

        def maximum_invoke(
            callback: Any,
            *,
            sample: Any,
            trial: Mapping[str, Any],
            context: Mapping[str, Any],
            timeout_seconds: float,
            isolate: bool,
        ) -> tuple[Mapping[str, Any], float]:
            del isolate
            ordinal = int(trial["valid_trial_ordinal"])
            self.assertAlmostEqual(
                timeout_seconds,
                min(
                    12.0 * 3600.0,
                    120.0 * 3600.0 - 4500.0 * (ordinal - 1),
                ),
            )
            return callback.fit_score(
                sample=sample, trial=trial, context=context
            ), 4500.0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            callback = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
                increasing_objective=True,
            )
            with patch.object(
                full_search_module,
                "_invoke_fit_score",
                side_effect=maximum_invoke,
            ):
                checkpoint = run_null_replicate(
                    _runtime(policy, callback=callback, program=program),
                    policy=policy,
                    replicate_index=0,
                    output_dir=root,
                )
            self.assertEqual(checkpoint["executed_trial_count"], 96)
            self.assertEqual(checkpoint["stop_reason"], "valid_trials_max_reached")
            self.assertFalse((
                root / "replicates/null_001/wall_budget_exhaustion.json"
            ).exists())

    def test_cli_maps_boundary_request_to_dedicated_exit_code(self) -> None:
        request = FullSearchNullRequeueRequested(
            replicate_index=7, completed_trial_count=19
        )
        with (
            patch.object(EpisodePolicy, "load", return_value=object()),
            patch.object(full_search_module, "_load_runtime", return_value=object()),
            patch.object(
                full_search_module, "run_null_replicate", side_effect=request
            ),
        ):
            result = full_search_module.main([
                "--runtime-provider", "synthetic:runtime",
                "--policy", "/tmp/policy.yaml",
                "--output-dir", "/tmp/output",
                "--replicate-index", "7",
                "--restart-sentinel", "/tmp/restart.request",
            ])
        self.assertEqual(result, REQUEUE_EXIT_CODE)

    def test_tail_outcome_changes_stop_and_artifact_tamper_fails(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            callback = SyntheticCallback(
                artifact_root=root,
                selection_sha256=SELECTION_SHA256,
                supported_falsifiers=program["countable_falsifier_ids"],
                improve_at=49,
            )
            runtime = _runtime(policy, callback=callback, program=program)
            checkpoint = run_null_replicate(
                runtime, policy=policy, replicate_index=0, output_dir=root
            )
            self.assertEqual(checkpoint["executed_trial_count"], 65)
            report_path = root / "full_search_null_report.json"
            report_path.write_text("{}\n", encoding="utf-8")
            inventory = _aggregate_file_inventory(root, [checkpoint])
            self.assertIn(
                checkpoint["trial_results"][0]["fit_score_artifact_path"],
                inventory,
            )
            unexpected = root / "unexpected.json"
            unexpected.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                FullSearchNullError, "aggregate artifact inventory mismatch"
            ):
                _aggregate_file_inventory(root, [checkpoint])
            unexpected.unlink()
            final_artifact = root / checkpoint["final_result"][
                "final_selection_artifact_path"
            ]
            original_final_artifact = final_artifact.read_bytes()
            final_artifact.write_bytes(original_final_artifact + b"tamper")
            with self.assertRaisesRegex(FullSearchNullError, "selection artifact"):
                run_null_replicate(
                    runtime, policy=policy, replicate_index=0, output_dir=root
                )
            final_artifact.write_bytes(original_final_artifact)
            first_artifact = root / checkpoint["trial_results"][0][
                "fit_score_artifact_path"
            ]
            first_artifact.write_bytes(first_artifact.read_bytes() + b"tamper")
            with self.assertRaisesRegex(FullSearchNullError, "missing or changed"):
                run_null_replicate(
                    runtime, policy=policy, replicate_index=0, output_dir=root
                )
            first_artifact.write_bytes(
                first_artifact.read_bytes().removesuffix(b"tamper")
            )
            first_artifact.unlink()
            with self.assertRaisesRegex(FullSearchNullError, "missing or changed"):
                run_null_replicate(
                    runtime, policy=policy, replicate_index=0, output_dir=root
                )


if __name__ == "__main__":
    unittest.main()
