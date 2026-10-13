"""Prospective EP21 lock and a feature-only context; neural execution stays closed.

Top-level imports are lightweight. The contract-required converter fixture is
called only in the approved allocation, before any real image access.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import zipfile

from public_runtime_harness import (BASE, PUBLIC_MEMBERS, require,
                                   validate_passed_report)

AUTH = BASE / "outputs/regular_vgg19_export_authorization_2026-10-02.json"
LOCK = BASE / "outputs/protocol_lock.yaml"
CONFORMANCE = BASE / "outputs/pipeline_conformance/attempt-008/conformance_report.json"
RESOURCES = {"jobs": 1, "gpus": 1, "cpus": 4, "memory_gib": 16,
             "minutes": 480, "durable_bytes": 1_600_000_000_000,
             "scratch_bytes": 340_000_000_000}
PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
LAYERS = ("conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2",
          "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3", "conv4_4",
          "conv5_1", "conv5_2", "conv5_3", "conv5_4", "fc6", "fc7", "fc8")
SHAPES = ([(64, 224, 224)] * 2 + [(128, 112, 112)] * 2 + [(256, 56, 56)] * 4
          + [(512, 28, 28)] * 4 + [(512, 14, 14)] * 4
          + [(4096,), (4096,), (1000,)])
CLOSED = ("allow_ood", "allow_neural_values", "allow_fitting", "allow_scoring",
          "allow_reconstruction", "allow_external_site_execution")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def publish_json(path, record):
    """JSON is also YAML; publish small metadata atomically without overwrite."""
    path = Path(path)
    require(not path.exists(), "do_not_overwrite_protocol_artifact")
    encoded = (json.dumps(record, indent=2) + "\n").encode()
    require(len(encoded) < 10_000_000, "protocol_metadata_cap_exceeded")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("xb") as f:
        f.write(encoded); f.flush(); os.fsync(f.fileno())
    pending.chmod(0o400)
    pending.rename(path)


def validate_authorization(auth):
    require(auth.get("schema_version") == "ep21.feature_only_execution_authorization.v1"
            and auth.get("episode_id") == "ep21" and auth.get("approved") is True
            and auth.get("purpose") == "regular_vgg19_derivative_export",
            "real_feature_only_authorization_required")
    resources = auth.get("resources", {})
    require({k: v for k, v in resources.items() if k != "minutes"}
            == {k: v for k, v in RESOURCES.items() if k != "minutes"}
            and type(resources.get("minutes")) is int
            and 30 <= resources["minutes"] <= RESOURCES["minutes"],
            "feature_resource_approval_mismatch")
    require(auth.get("allow_protocol_lock_completion") is True
            and auth.get("allow_regular_image_values_after_valid_lock") is True
            and auth.get("includes_public_runtime_conformance") is True,
            "approved_combined_stage_scope_required")
    require(all(auth.get(k) is False for k in CLOSED)
            and auth.get("neural_execution_authorized") is False,
            "feature_only_boundary_required")
    require(auth.get("permitted_image_count") == 24681
            and auth.get("permitted_image_roles") == ["subject_unique", "tau_train", "tau_test"],
            "regular_only_authorization_required")


def image_roles():
    """Read only the authenticated structural CSV, never the image container."""
    source = BASE / "inputs/laion_regular_feature_handoff"
    metadata = read_json(BASE / "outputs/steward_provisioning/source_structural_manifest.json")
    export = read_json(BASE / "outputs/steward_provisioning/regular_stimuli/export_manifest.json")
    data = (source / "source_index.csv").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(digest == export["sha256"]["source_index.csv"], "image_alignment_receipt_mismatch")
    rows = list(csv.DictReader(data.decode().splitlines()))
    require(len(rows) == 24681 and len({r["image_id"] for r in rows}) == 24681,
            "complete_regular_image_support_required")
    groups = {p: [] for p in PARTICIPANTS}
    groups.update({"tau_train": [], "tau_test": []})
    for ordinal, row in enumerate(rows):
        require(row["role"] in ("subject_unique", "tau_train", "tau_test"), "forbidden_image_role")
        key = row["participant"] if row["role"] == "subject_unique" else row["role"]
        require(key in groups, "participant_mismatch")
        groups[key].append((ordinal, row))
    require(metadata["participants"] == list(PARTICIPANTS), "participant_contract_mismatch")
    for p in PARTICIPANTS:
        require(len(groups[p]) == 4712 and set(r["image_id"] for _, r in groups[p])
                == set(metadata["unique_image_ids"][p]), "unique_role_identity_mismatch")
    split = metadata["provider_split"]["variant"]
    for role, key, count in (("tau_train", "train_ids", 897), ("tau_test", "test_ids", 224)):
        require(len(groups[role]) == count and set(r["image_id"] for _, r in groups[role])
                == set(split[key]), "authenticated_tau_identity_mismatch")
    require(export["status"] == "complete_regular_only_stimulus_export"
            and export["image_count"] == 24681 and export["ood_image_values_read"] is False,
            "regular_only_provider_export_required")
    # The sealed provider export already authenticated this large payload.
    # Reuse its identity; do not rehash or decode images to prepare a lock.
    return rows, groups, {"source_index_sha256": digest,
        "image_count": len(rows), "HDF5_row_rule": "source_index_CSV_ordinal",
        "provider_stimulus_sha256": export["sha256"]["task-images_stimuli.h5"],
        "role_image_ids": {k: [r["image_id"] for _, r in v] for k, v in groups.items()},
        "provider_code_commit": metadata["provider_code_commit"]}, metadata


def tagged_content_loss_reduction(source):
    """Bind the authenticated training caller to its exact loss constructor.

    Checking the extracted loss helper alone does not fix its caller-supplied
    reduction. Accept only the pinned caller's one direct sum-MSE assignment
    and its unchanged sixth positional argument to chunk_process_batch.
    """
    tree = ast.parse(source)
    callers = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "converter_training"]
    require(len(callers) == 1, "tagged_loss_caller_required")
    caller = callers[0]
    assignments = [node for node in caller.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "criterion_mse"]
    stores = [node for node in ast.walk(caller) if isinstance(node, ast.Name)
              and node.id == "criterion_mse" and isinstance(node.ctx, ast.Store)]
    mutations = [node for node in ast.walk(caller) if isinstance(node, ast.Attribute)
                 and isinstance(node.value, ast.Name) and node.value.id == "criterion_mse"
                 and isinstance(node.ctx, (ast.Store, ast.Del))]
    require(len(assignments) == 1 and len(stores) == 1 and not mutations,
            "tagged_loss_criterion_assignment_required")
    constructor = assignments[0].value
    require(isinstance(constructor, ast.Call)
            and ast.dump(constructor.func) == ast.dump(ast.parse("torch.nn.MSELoss", mode="eval").body)
            and not constructor.args and len(constructor.keywords) == 1
            and constructor.keywords[0].arg == "reduction"
            and isinstance(constructor.keywords[0].value, ast.Constant)
            and constructor.keywords[0].value.value == "sum",
            "tagged_sum_MSE_constructor_required")
    calls = [node for node in ast.walk(caller) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "chunk_process_batch"]
    require(len(calls) == 1 and len(calls[0].args) == 8 and not calls[0].keywords
            and isinstance(calls[0].args[5], ast.Name)
            and calls[0].args[5].id == "criterion_mse"
            and assignments[0].lineno < calls[0].lineno,
            "tagged_loss_caller_criterion_argument_required")
    return "sum"


def converter_conformance():
    """Only the outstanding tagged architecture/loss fixture, inside Slurm."""
    require(os.environ.get("SLURM_JOB_ID"), "Slurm_required_for_converter_fixture")
    cpu_text = os.environ.get("SLURM_CPUS_PER_TASK", "")
    require(cpu_text.isdecimal() and int(cpu_text) > 0,
            "positive_allocated_CPU_count_required")
    threads = min(RESOURCES["cpus"], int(cpu_text))
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    old = read_json(BASE / "outputs/pipeline_conformance/attempt-005/neutral_conformance_report.json")
    code = {}
    with zipfile.ZipFile(old["source_archive"]["path"]) as archive:
        for name in ("content_models", "content_train", "content_utils"):
            reference = old["source_members"][name]
            data = archive.read(reference["path"])
            require(hashlib.sha256(data).hexdigest() == reference["sha256"], "tagged_source_identity_mismatch")
            code[name] = data.decode()
    namespace = {}
    exec(compile(code["content_models"], "PINNED_V1_models.py", "exec"), namespace)
    torch.manual_seed(42)
    model = namespace["Converter"](9, 11)
    require([(m.in_features, m.out_features) for m in model.modules()
             if isinstance(m, torch.nn.Linear)] == [(9, 4), (4, 4), (4, 11)],
            "tagged_converter_width_mismatch")
    norms = [m for m in model.modules() if isinstance(m, torch.nn.InstanceNorm1d)]
    require(len(norms) == 2 and all(m.num_features == 1 and not m.affine
            and not m.track_running_stats and m.eps == 1e-5 for m in norms),
            "tagged_instance_normalization_mismatch")
    require([m.negative_slope for m in model.modules()
             if isinstance(m, torch.nn.LeakyReLU)] == [0.2, 0.2], "tagged_activation_mismatch")
    x = torch.arange(27, dtype=torch.float32).reshape(3, 9) / 7
    manual = x.reshape(3, 1, 9)
    for m in model.model:
        if isinstance(m, torch.nn.InstanceNorm1d):
            mean = manual.mean(-1, keepdim=True)
            manual = (manual - mean) / torch.sqrt(((manual - mean) ** 2).mean(-1, keepdim=True) + m.eps)
        else:
            manual = m(manual)
    actual = model(x)
    require(actual.shape == (3, 11) and torch.allclose(actual, manual.reshape(3, 11), atol=2e-6, rtol=2e-6),
            "tagged_converter_forward_conformance_failed")
    # Execute the exact qualified weight function without importing utilities
    # that read fitted neural decoders or use unrelated optional dependencies.
    tree = ast.parse(code["content_utils"])
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_layer_weights")
    ns = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "PINNED_V1_compute_layer_weights", "exec"), ns)
    reduction = tagged_content_loss_reduction(code["content_train"])
    blocks = {"conv": np.ones((2, 1, 3), dtype=np.float32),
              "fc": np.full((2, 1), 2, dtype=np.float32)}
    weights = ns["compute_layer_weights"](blocks, ["conv", "fc"])
    require(np.allclose([weights["conv"], weights["fc"]], [4 / 7, 3 / 7], atol=1e-6, rtol=0),
            "tagged_active_block_weight_conformance_failed")
    # Execute the pinned accumulation/reduction function with a synthetic
    # fixed decoder callback. This callback cannot open files or neural data.
    train_tree = ast.parse(code["content_train"])
    loss_node = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "chunk_process_batch")
    predictions = {"conv": torch.from_numpy(blocks["conv"] + 1).unsqueeze(1).requires_grad_(True),
                   "fc": torch.from_numpy(blocks["fc"] + 2).unsqueeze(1).requires_grad_(True)}
    class FixturePath:
        def build_model_path(self, layer):
            return layer
    ns.update({"torch": torch, "Variable": lambda t: t,
               "test_fastl2lir_div": lambda layer, value, index: predictions[layer]})
    exec(compile(ast.Module(body=[loss_node], type_ignores=[]), "PINNED_V1_chunk_process_batch", "exec"), ns)
    inputs = {k: torch.empty_like(torch.from_numpy(v)) for k, v in blocks.items()}
    loss = ns["chunk_process_batch"](lambda value: value, torch.zeros((2, 1)), blocks,
        inputs, ["conv", "fc"], torch.nn.MSELoss(reduction=reduction), FixturePath(), {"conv": 0, "fc": 0})
    require(torch.isfinite(loss).item() and abs(loss.item() - 48 / 7) < 2e-6,
            "tagged_loss_reduction_and_aggregation_failed")
    loss.backward()
    for layer, expected in (("conv", 8 / 7), ("fc", 12 / 7)):
        gradient = predictions[layer].grad
        require(gradient is not None and torch.allclose(
            gradient, torch.full_like(gradient, expected), atol=1e-6, rtol=0),
            "tagged_loss_gradient_conformance_failed")
    return {"passed": True, "fixture_only": True, "source_members": old["source_members"],
        "hidden_width": "source_input_width//2_for_both_hidden_layers",
        "normalization": {"kind": "InstanceNorm1d", "num_features": 1, "eps": 1e-5,
            "affine": False, "track_running_stats": False,
            "axes": "per_sample_across_hidden_positions_of_N_1_hidden_tensor_biased_variance"},
        "activation": "LeakyReLU_negative_slope_0.2",
        "loss": "float32_norms_of_complete_active_blocks_inverse_squared_then_normalize_sum_to_one_times_per_block_sum_squared_error_sum_over_layers",
        "loss_reduction": reduction,
        "loss_fixture": {"caller_constructor_and_argument_verified": True,
            "block_shapes": {k: list(v.shape) for k, v in blocks.items()},
            "block_element_counts": {k: int(v.size) for k, v in blocks.items()},
            "normalized_weights": {k: float(v) for k, v in weights.items()},
            "residual_per_element": {"conv": 1, "fc": 2},
            "expected_sum_loss": 48 / 7, "observed_loss": loss.item(),
            "counterfactual_mean_loss": 16 / 7,
            "expected_gradient_per_element": {"conv": 8 / 7, "fc": 12 / 7},
            "all_prediction_gradients_verified": True},
        "feature_normalization": "source_training_image_features_use_target_decoder_training_mean_and_ddof1_std_per_unit",
        "degenerate_active_norm": "stop_affected_fit_no_added_epsilon_or_block_exclusion",
        "runtime": {"numpy": np.__version__, "torch": torch.__version__},
        "torch_threads": threads,
        "real_images_or_neural_values_read": False, "OOD_access": False,
        "empirical_fitting_scoring": False}


def validate_feature_lock(lock, auth, public_report, alignment):
    validate_authorization(auth)
    validate_passed_report(public_report)
    require(lock.get("schema_version") == "ep21.protocol_lock.v1"
            and lock.get("episode_id") == "ep21" and lock.get("state") == "valid_prospective_lock",
            "valid_real_protocol_lock_required")
    require(lock.get("module_states", {}).get("laion_feature_preparation") == "runnable_locked",
            "feature_module_not_locked_runnable")
    require(lock.get("neural_execution_authorized") is False
            and lock.get("OOD_access_authorized") is False
            and lock.get("neural_or_image_values_used_to_resolve") is False,
            "lock_boundary_mismatch")
    require(lock.get("regular_image_binding") == alignment, "frozen_image_alignment_mismatch")
    require(lock.get("public_runtime_report_job") == public_report.get("job_id"),
            "public_report_provenance_mismatch")
    require(lock.get("converter_conformance", {}).get("passed") is True,
            "required_converter_conformance_missing")
    recipe = lock.get("feature_recipe", {})
    require(recipe.get("layers") == list(LAYERS) and recipe.get("shapes") == [list(s) for s in SHAPES]
            and recipe.get("dtype") == "float32" and recipe.get("checkpoint_members") == PUBLIC_MEMBERS
            and recipe.get("input_rule") == "RGB_direct_BICUBIC_224_BGR_CHW_float32_0_255_minus_qualified_channel_means",
            "frozen_feature_recipe_mismatch")
    return lock


def validate_loaded_runtime(public_report, converter):
    """Compare the two executed stages using module versions on both sides.

    Distribution metadata can retain a build suffix absent from __version__;
    it is recorded separately and is not comparable to a module-version field.
    """
    public_versions = public_report.get("runtime", {})
    feature_versions = converter.get("runtime", {})
    require(all(isinstance(public_versions.get(name), str)
                and public_versions[name]
                and public_versions[name] == feature_versions.get(name)
                for name in ("numpy", "torch")),
            "public_and_feature_runtime_mismatch")
    return dict(feature_versions)


def complete_protocol_lock(public_report, check, authorization_path=AUTH):
    validate_authorization(read_json(authorization_path))
    validate_passed_report(public_report)
    require(not LOCK.exists() and not CONFORMANCE.exists(), "existing_lock_or_attempt_requires_resume")
    check()
    _, groups, alignment, metadata = image_roles()
    decisions = read_yaml(BASE / "outputs/protocol_decisions_2026-09-30.yaml")
    protocol = read_yaml(BASE / "PROTOCOL_TABLE.yaml")
    bindings = read_yaml(BASE / "outputs/protocol_implementation_bindings_2026-09-30.yaml")
    mask = read_json(BASE / "outputs/steward_provisioning/independent_visual_masks/mask_report.json")
    neural = read_json(BASE / "outputs/steward_provisioning/regular_neural_structural_report.json")
    require(mask["publication"] == "complete" and neural["publication"] == "complete"
            and neural["contains_neural_values"] is False
            and neural["OOD_arrays_formed_retained_summarized"] is False
            and neural["model_fitting_scoring_run"] is False, "sealed_structural_reports_required")
    require(len(metadata["directed_pairs"]) == 20 and all(p["source_target_unique_overlap"] == 0
            for p in metadata["directed_pairs"]), "complete_disjoint_pair_table_required")
    for subject in neural["subjects"]:
        match = next(s for s in mask["subjects"] if s["participant"] == subject["participant"])
        require(subject["mask_index_sha256"] == match["voxel_indices_sha256"]
                and all(r["nonfinite_elements"] == 0 and r["repetitions_preserved"] is True
                        for r in subject["roles"].values()), "native_mask_and_finite_coverage_required")
    converter = converter_conformance()
    check()
    runtime = {"python": sys.version.split()[0], "interpreter": sys.executable,
               "prefix": sys.prefix, **{p: importlib.metadata.version(p)
               for p in ("numpy", "torch", "h5py", "Pillow")}}
    runtime["loaded_module_versions"] = validate_loaded_runtime(public_report, converter)
    # Resolve only formerly pending shared/A fields. Unqualified branches have
    # explicit locked dispositions, rather than convenient replacement assets.
    protocol["initial_state"] = "resolved_prospective_protocol"
    protocol["source_identity"]["software"]["source_environment_and_critical_library_versions"] = runtime
    laion = protocol["data_qualification"]["laion"]
    laion["whole_visual_cortex_mask_ids"] = "superseded_by_approved_independent_29_mask_union_native_C_order"
    laion["v1_v2_v3_v4_hvc_crosswalk"] = "subarea_reports_not_evaluable_unqualified_crosswalk"
    laion["remaining_neural_qualification"] = "sealed_regular_rows_finite_native_header_binding_complete_fixed_provider_exception_adopted"
    protocol["vgg19_feature_contract"]["exact_layer_aliases_shapes_flattening_and_normalization"] = bindings["vgg19_true_features"]
    protocol["vgg19_feature_contract"]["checkpoint"] = {
        "members": PUBLIC_MEMBERS, "strict_load": "passed", "public_report_job": public_report["job_id"]}
    protocol["roi_contract"]["native_beta_grid_binding"] = "completed_150_session_bindings_in_sealed_regular_neural_structural_report_no_resampling"
    protocol["response_preprocessing"]["content_converter"]["target_output_scale_definition"] = "fixed_target_decoder_subject_unique_training_standardized_response_space"
    protocol["feature_decoder_contract"]["planned_voxel_selection"]["ties_and_fewer_than_500_action"] = "ascending_stable_native_voxel_ID_ties_fewer_than_500_not_evaluable"
    c = protocol["content_converter_contract"]
    c["planned_architecture"]["hidden_width_rule"] = converter["hidden_width"]
    c["planned_architecture"]["normalization"] = converter["normalization"]
    c["content_loss"]["exact_weighting_layer_aggregation_and_numerical_stability"] = converter["loss"]
    c["content_loss"]["normalization_binding"] = converter["feature_normalization"]
    protocol["brain_loss_contract"]["regularization_value_selection_and_intercept"] = {
        "selection": decisions["delegated_LAION_adaptations"]["brain_loss"],
        "estimator": bindings["fastl2lir"], "numeric_alpha": "selected_later_from_tau_train_only_not_a_lock_input"}
    states = {m: "runnable_locked" for m in protocol["module_resolution"]["shared_modules"]
              + ["laion_feature_preparation", "laion_target_decoders", "laion_within_source_decoders",
                 "laion_content_converters", "laion_brain_loss_converters", "finding_a_feature_decoding"]}
    reasons = {"finding_b_reconstruction": "generator_archive_available_but_checkpoint_config_latent_and_DIST_runtime_binding_unqualified",
        "finding_c_identification": "reconstruction_unqualified_and_exact_tagged_bvlc_alexnet_pt_checkpoint_not_authenticated",
        "inter_site_decoders_and_converters": "external_native_data_and_handoffs_unprovisioned",
        "finding_d_inter_site_reconstruction": "external_native_data_and_handoffs_unprovisioned",
        "finding_e_inter_site_identification": "external_native_data_and_handoffs_unprovisioned",
        "artificial_image_module": "complete_371_image_OOD_pool_reserved_for_EP04",
        "comparability_A_C_E": "no_outcome_independent_margin_supplied_no_margin_invented",
        "visual_subareas": "unqualified_anatomical_crosswalk",
        "noise_ceiling_correction": "unqualified_estimator_no_threshold_invented"}
    states.update({m: "not_evaluable_locked" for m in reasons})
    lock = {"schema_version": "ep21.protocol_lock.v1", "episode_id": "ep21",
        "state": "valid_prospective_lock", "producer": "pipeline_conformance", "job_id": os.environ["SLURM_JOB_ID"],
        "neural_execution_authorized": False, "OOD_access_authorized": False,
        "neural_or_image_values_used_to_resolve": False, "scientific_results_observed": False,
        "module_states": states, "not_evaluable_reasons": reasons,
        "resolved_protocol_table": protocol, "adopted_decisions": decisions,
        "implementation_bindings": bindings, "regular_image_binding": alignment,
        "mask_amendment": read_yaml(BASE / "outputs/protocol_mask_amendment_2026-09-30.yaml"),
        "provider_measurement_amendment": read_yaml(BASE / "outputs/protocol_provider_beta_amendment_2026-09-30.yaml"),
        "native_neural_structural_report": "outputs/steward_provisioning/regular_neural_structural_report.json",
        "converter_conformance": converter, "runtime": runtime,
        "public_runtime_report_job": public_report["job_id"],
        "public_runtime_report": "outputs/pipeline_conformance/attempt-007/public_runtime_conformance_report.json",
        "reused_unchanged_checks": ["pipeline_conformance/attempt-005", "pipeline_conformance/attempt-006"],
        "feature_recipe": {"layers": list(LAYERS), "shapes": [list(s) for s in SHAPES], "dtype": "float32",
            "checkpoint_members": PUBLIC_MEMBERS,
            "input_rule": "RGB_direct_BICUBIC_224_BGR_CHW_float32_0_255_minus_qualified_channel_means",
            "HDF5_row_rule": "source_index_CSV_ordinal", "finite_mask_bitorder": "little",
            "TF32": False, "AMP": False, "historical_Caffe_parity": "unverified"},
        "directed_pairs": [{**p, "pair_id": "LAION:" + p["source"] + "->LAION:" + p["target"]}
                           for p in metadata["directed_pairs"]],
        "fixed_display": {"pair": "LAION:sub-01->LAION:sub-03",
            "image_ids": sorted(alignment["role_image_ids"]["tau_test"], key=lambda s: s.encode())[:4],
            "rule": decisions["reporting_and_inference"]["fixed_display"]},
        "evidence_limits": ["adapted_retrospective_five_participant_fixed_measurement_replication",
            "unverified_provider_tau_test_OOD_preprocessing_exclusions",
            "incomplete_independent_visual_union", "historical_Caffe_parity_unverified",
            "regular_test_evidence_correlated_with_EP04_lineage", "no_participant_population_inference"]}
    validate_feature_lock(lock, read_json(authorization_path), public_report, alignment)
    report = {"schema_version": "ep21.resolved_conformance_report.v1", "astra_analysis_id": "pipeline_conformance",
        "attempt": 8, "job_id": os.environ["SLURM_JOB_ID"], "outcome": "passed_required_shared_and_Finding_A_protocol_conformance",
        "converter_conformance": converter, "module_states": states, "not_evaluable_reasons": reasons,
        "public_runtime_report": lock["public_runtime_report"], "reused_unchanged_checks": lock["reused_unchanged_checks"],
        "actual_neural_fitting_scoring": False, "real_images_or_neural_values_read": False,
        "OOD_access": False, "protocol_lock": "outputs/protocol_lock.yaml"}
    publish_json(CONFORMANCE, report)
    publish_json(LOCK, lock)
    print(json.dumps({"stage": "pipeline_conformance", "status": "required_conformance_and_prospective_lock_complete",
                      "neural_execution_authorized": False}), flush=True)
    return lock


@dataclass(frozen=True)
class FeatureExportContext:
    source: Path
    rows: tuple
    lock_path: Path
    authorization_path: Path
    neural_capable: bool = False


def bind_feature_export_context(public_report, authorization_path=AUTH):
    auth = read_json(authorization_path)
    rows, _, alignment, _ = image_roles()
    validate_feature_lock(read_yaml(LOCK), auth, public_report, alignment)
    return FeatureExportContext(BASE / "inputs/laion_regular_feature_handoff", tuple(rows), LOCK, Path(authorization_path))
