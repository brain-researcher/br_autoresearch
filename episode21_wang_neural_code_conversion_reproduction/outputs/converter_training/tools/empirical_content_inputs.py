"""UNACTIVATED empirical content-input composition; no grant, CLI or launcher.

A future caller must supply both its separately admitted execution context and
the resolver that checks that context's exact operation, pair and resources.
This module implements no resolver or authority constructor. Neither a decoder
grant nor successful decoder assembly substitutes for converter access. The
legacy execution_adapter.bind_empirical_context remains closed and unchanged.

Input binding consumes the existing loss correction before any scientific
payload, then reuses the complete decoder reader, all-five-axis/canonical-twenty
initialization and selective source loader. Only the source's 4,712 normalized
unique-image rows are loaded; target response matrices are never requested.
The separate publishing operation delegates the unchanged 1,024-update driver.
Measurement-only admission cannot authorize that operation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import content_pair_driver as driver
import decoder_interface as interface
import schedule


CORRECTION_RELATIVE = "outputs/pipeline_conformance/converter_loss_correction_001/report.json"
CORRECTION_PATH = interface.assembly.BASE / CORRECTION_RELATIVE
CORRECTION_MAX_BYTES = 1_000_000
CORRECTED_LOSS = (
    "float32_norms_of_complete_active_blocks_inverse_squared_then_normalize_sum_to_one"
    "_times_per_block_sum_squared_error_sum_over_layers"
)
require = interface.require


@dataclass(frozen=True)
class LossCorrectionEvidence:
    """Observed implementation metadata, never an execution permission."""

    report_relative: str
    job_id: str
    loss: str
    numpy_version: str
    torch_version: str


def _read_loss_correction():
    path = CORRECTION_PATH
    require(path.resolve() == path.absolute(), "symlinked_loss_correction_refused")
    with path.open("rb") as stream:
        raw = stream.read(CORRECTION_MAX_BYTES + 1)
    require(len(raw) <= CORRECTION_MAX_BYTES, "loss_correction_exceeds_metadata_bound")
    record = json.loads(raw)
    require(isinstance(record, dict), "loss_correction_record_required")
    corrected = record.get("converter_conformance", {})
    require(isinstance(corrected, dict), "corrected_converter_conformance_required")
    fixture = corrected.get("loss_fixture", {})
    runtime = corrected.get("runtime", {})
    require(record.get("status") == "passed_source_sum_loss_correction"
            and record.get("job_id") == "46881096"
            and record.get("scope") == "public_source_and_small_analytic_fixture_only"
            and record.get("historical_report") == "outputs/pipeline_conformance/attempt-008/conformance_report.json"
            and record.get("historical_lock") == "outputs/protocol_lock.yaml"
            and record.get("historical_artifacts_preserved") is True
            and record.get("corrects_only") == ["converter_conformance.loss",
                "resolved_protocol_table.content_converter_contract.content_loss.exact_weighting_layer_aggregation_and_numerical_stability"]
            and corrected.get("passed") is True and corrected.get("fixture_only") is True
            and corrected.get("loss") == CORRECTED_LOSS and corrected.get("loss_reduction") == "sum"
            and isinstance(fixture, dict)
            and fixture.get("caller_constructor_and_argument_verified") is True
            and fixture.get("all_prediction_gradients_verified") is True,
            "existing_source_sum_loss_and_gradient_correction_required")
    require(all(record.get(flag) is False for flag in (
                "decoder_fits_and_VGG_features_affected", "empirical_fitting_scoring_authorized",
                "neural_or_image_values_read", "OOD_access"))
            and all(corrected.get(flag) is False for flag in (
                "real_images_or_neural_values_read", "OOD_access", "empirical_fitting_scoring")),
            "loss_correction_is_implementation_evidence_not_empirical_authority")
    require(isinstance(runtime, dict) and all(isinstance(runtime.get(key), str) and runtime[key]
            for key in ("numpy", "torch")), "observed_loss_correction_runtime_required")
    return LossCorrectionEvidence(CORRECTION_RELATIVE, record["job_id"], corrected["loss"],
                                  runtime["numpy"], runtime["torch"])


def _admit(context, resolver, *, operation, source, target, budget,
           execution_options=None, allocation_identity=None):
    require(context is not None and callable(resolver), "explicit_external_context_resolver_required")
    require((source, target) in schedule.PAIR_ORDER, "canonical_nonself_content_pair_required")
    # The resolver is supplied by the future authorized entrypoint. Returning a
    # decoder Boolean or another truthy token is deliberately not this API.
    result = resolver(context, operation=operation, source=source, target=target,
                      budget=budget, execution_options=execution_options)
    require(result is None, "external_resolver_must_raise_or_return_None")
    interface.assembly._require_reader_context(budget)
    identity = (budget.allocation["job_id"], budget.allocation["hostname"])
    require(allocation_identity is None or identity == allocation_identity,
            "bound_content_inputs_cannot_move_to_another_allocation")
    return identity


@dataclass(frozen=True)
class BoundContentInputs:
    """Caller-owned admitted inputs; this value does not grant later operations.

    Nested panel/native-axis objects have their existing immutability contract.
    A measurement runner may consume panel/pair only after its own operation
    admission; binding alone permits neither optimization nor publication.
    """

    execution_context: object
    require_admitted_context: Callable
    budget: object
    allocation_identity: tuple[str, str]
    loss_correction: LossCorrectionEvidence
    panel: driver.ContentPanelInitialization
    pair: interface.ContentPair

    @property
    def source(self):
        return self.pair.source

    @property
    def target(self):
        return self.pair.target_axis.participant


def bind_content_inputs(execution_context, *, require_admitted_context,
                        source, target, budget) -> BoundContentInputs:
    """Resolve access before any I/O; bind the fixed full-size empirical inputs.

    The external resolver owns protocol/access/resource admission, including
    all five normalization axes and one-at-a-time canonical model construction.
    Its budget implements the existing allocation/deadline/check interface;
    this function does not reuse or enlarge the decoder's 7GiB WorkBudget.
    """
    identity = _admit(execution_context, require_admitted_context,
        operation="bind_content_inputs", source=source, target=target, budget=budget)
    correction = _read_loss_correction()
    manifest = interface.assembly.load_complete_manifest(
        interface.assembly.BASE / interface.assembly.MANIFEST, budget=budget)
    panel = driver.prepare_content_panel(manifest, budget=budget)
    source_index, target_index = map(schedule.PARTICIPANTS.index, (source, target))
    source_ids = panel.training_image_ids[source_index]
    target_ids = panel.training_image_ids[target_index]
    interface.require_image_order(source_ids, source_ids)
    interface.require_image_order(target_ids, target_ids)
    require(not set(source_ids).intersection(target_ids),
            "source_converter_target_decoder_overlap_forbidden_before_source_read")
    # Recheck after preparation callbacks, before the one source response read.
    _admit(execution_context, require_admitted_context, operation="bind_content_inputs",
           source=source, target=target, budget=budget, allocation_identity=identity)
    training = interface.load_source_training(manifest, participant=source,
                                              expected_image_ids=source_ids, budget=budget)
    driver._same_axis(training.axis, panel.axes[source_index])
    interface.require_image_order(training.image_ids, source_ids)
    pair = interface.bind_content_pair(manifest, training, target=target, budget=budget)
    driver._same_axis(pair.target_axis, panel.axes[target_index])
    interface.require_image_order(pair.target_normalization.training_image_ids.tolist(), target_ids)
    _admit(execution_context, require_admitted_context, operation="bind_content_inputs",
           source=source, target=target, budget=budget, allocation_identity=identity)
    return BoundContentInputs(execution_context, require_admitted_context, budget, identity,
                              correction, panel, pair)


def train_and_publish_full_pair(bound: BoundContentInputs, *, device, block_units,
                                row_chunk_size, backend, final_path, maximum_bytes,
                                on_iteration=None):
    """Separately admit the full operation, then use the tested final-only driver.

    No step-count option, measurement escalation, retry or partial publication
    is introduced. Exceptions propagate, including an iteration callback's
    interruption. The future resolver must reject measurement-only contexts.
    """
    require(isinstance(bound, BoundContentInputs), "bound_content_inputs_required")
    options = MappingProxyType(dict(device=device, block_units=block_units,
        row_chunk_size=row_chunk_size, backend=backend, final_path=Path(final_path),
        maximum_bytes=maximum_bytes))
    _admit(bound.execution_context, bound.require_admitted_context,
        operation="train_and_publish_full_pair", source=bound.source.axis.participant,
        target=bound.target, budget=bound.budget, execution_options=options,
        allocation_identity=bound.allocation_identity)
    require(_read_loss_correction() == bound.loss_correction,
            "loss_correction_changed_after_content_input_binding")
    return driver.train_and_publish_content_pair(bound.panel, bound.source,
        target=bound.target, budget=bound.budget, on_iteration=on_iteration, **options)
