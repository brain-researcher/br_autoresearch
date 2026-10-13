"""UNACTIVATED content-pair orchestration over caller-admitted training roles.

This library joins the existing complete-decoder reader, canonical pinned
initialization, full-batch training and final-only publisher. It has no CLI,
empirical admission, source-response loader, retry, scoring or successor. The
closed execution_adapter remains unchanged. A future caller must separately
admit converter access, the explicit loss-correction evidence, the numerical
backend and finite compute/memory/storage resources before calling this code.

Preparation reads the five already frozen training-normalization records, not
their normalized_X matrices. It captures one seed42 sequence over ALL twenty
canonical pairs, retaining only RNG states and one shared 19-layer schedule.
Pair execution receives its full source SourceTraining from that caller. It
does not treat one independently scheduled pair as a complete panel result.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

import decoder_interface as interface
import final_content_artifact as final
import schedule
import train_content as training


@dataclass(frozen=True)
class ContentPanelInitialization:
    """Caller-owned fixed axes and compact canonical states; no trained models.

    As with existing PairInitialization objects, nested tensors must remain
    unchanged. This object is an implementation binding, never access authority.
    """

    manifest: interface.assembly.CompleteDecoderManifest
    axes: tuple[interface.NativeAxis, ...]
    training_image_ids: tuple[tuple[str, ...], ...]
    initializations: tuple[schedule.PairInitialization, ...]
    channel_schedule: schedule.ChannelSchedule
    converter_factory: schedule.ConverterFactory


@dataclass(frozen=True)
class PublishedContentPair:
    source: str
    target: str
    pair_index: int
    path: Path
    completion: training.CompletedContentFit


def prepare_content_panel(manifest, *, budget) -> ContentPanelInitialization:
    """Bind all five widths to the actual complete decoder's frozen native axes.

    Reuse the existing normalization admission instead of accepting arbitrary
    width mappings. No normalized source responses, features or coefficients
    are read here. Public pinned model source is loaded by the existing loader.
    One constructor is live at a time; allocation sizing must accommodate the
    largest model even when the eventual pair is scheduled independently.
    """
    interface.assembly._require_reader_context(budget)
    axes, image_ids = [], []
    for participant in schedule.PARTICIPANTS:
        budget.check()
        _, normalization = interface._state_context(manifest, participant, budget)
        axis = interface.native_axis(normalization)
        ids = tuple(normalization.training_image_ids.tolist())
        interface.require_image_order(ids, ids)
        axes.append(axis)
        image_ids.append(ids)
    factory = schedule.load_pinned_converter_factory()

    def checked_factory(input_width, output_width):
        budget.check()
        model = factory(input_width, output_width)
        budget.check()
        return model

    widths = {axis.participant: len(axis.eligible_native_ids) for axis in axes}
    initializations = schedule.capture_pair_initializations(widths, checked_factory)
    channel_schedule = schedule.make_channel_schedule()
    budget.check()
    return ContentPanelInitialization(manifest, tuple(axes), tuple(image_ids),
        initializations, channel_schedule, factory)


def _same_axis(actual, expected):
    interface.require(isinstance(actual, interface.NativeAxis)
                      and actual.participant == expected.participant,
                      "source_or_target_participant_differs_from_captured_axis")
    for field in ("full_native_ids", "eligible_native_ids", "eligible_full_columns", "mean", "std"):
        interface.require(np.array_equal(getattr(actual, field), getattr(expected, field)),
                          "source_or_target_axis_differs_from_captured_decoder_state")


def train_and_publish_content_pair(
    panel: ContentPanelInitialization,
    source: interface.SourceTraining,
    *,
    target: str,
    budget,
    device,
    block_units: int,
    row_chunk_size: int,
    backend: Literal["sparse", "dense"],
    final_path,
    maximum_bytes: int,
    on_iteration: Callable[[dict], None] | None = None,
    provider_factory: Callable | None = None,
) -> PublishedContentPair:
    """Initialize once, run all 1,024 full-4,712-row steps, publish once.

    Source order/normalization and target coordinates must match the records
    that determined ALL twenty initializations. The supplied matrix is copied
    to an explicit device without any new normalization, filtering or row split.
    A CPU copy of the final model is also needed by the existing artifact helper;
    the caller's memory bound must cover these copies and Adam/autograd state.

    Any deadline, numerical, callback or publication exception propagates. There
    is no final success result on failure and no intermediate checkpoint, restart
    or automatic retry. Existing final or partial files refuse before fitting;
    publication still uses the writer's exclusive operation to handle races.
    A separately admitted caller may select a provider factory; None preserves
    the ordinary interface provider. Its close method, if present, runs even
    on callback interruption, before any final-artifact preparation.
    """
    interface.assembly._require_reader_context(budget)
    require = interface.require
    require(isinstance(panel, ContentPanelInitialization)
            and len(panel.axes) == len(panel.training_image_ids) == len(schedule.PARTICIPANTS)
            and len(panel.initializations) == len(schedule.PAIR_ORDER),
            "complete_five_participant_twenty_pair_initialization_required")
    require(isinstance(source, interface.SourceTraining), "supplied_source_training_required")
    identity = (source.axis.participant, target)
    require(identity in schedule.PAIR_ORDER, "canonical_nonself_content_pair_required")
    require(type(block_units) is int and 1 <= block_units <= interface.assembly.MAX_BLOCK_UNITS,
            "explicit_bounded_decoder_block_required")
    require(type(row_chunk_size) is int and 1 <= row_chunk_size <= training.TRAINING_ROWS,
            "explicit_bounded_arithmetic_row_chunk_required")
    require(backend in ("sparse", "dense"), "explicit_qualified_numerical_backend_required")
    selected_device = torch.device(device)
    require(selected_device.type in ("cpu", "cuda"), "explicit_caller_allocated_CPU_or_CUDA_device_required")
    require(on_iteration is None or callable(on_iteration), "iteration_callback_must_be_callable")
    require(provider_factory is None or callable(provider_factory), "provider_factory_must_be_callable")
    path = final._file_context(final_path, budget, maximum_bytes)
    for existing in (path, path.with_name(path.name + ".partial")):
        if os.path.lexists(existing):
            raise FileExistsError(existing)

    source_index = schedule.PARTICIPANTS.index(identity[0])
    target_index = schedule.PARTICIPANTS.index(target)
    _same_axis(source.axis, panel.axes[source_index])
    interface.require_image_order(source.image_ids, panel.training_image_ids[source_index])
    require(source.feature_root == interface.assembly.BASE / "outputs/steward_provisioning"
            / "regular_vgg19_sealed" / identity[0], "fixed_source_unique_feature_partition_required")
    values = source.normalized_X
    require(isinstance(values, np.ndarray) and values.dtype == np.dtype("float32")
            and values.shape == (training.TRAINING_ROWS, len(source.axis.eligible_native_ids)),
            "all_4712_float32_eligible_source_rows_required")
    for first in range(0, training.TRAINING_ROWS, 64):
        budget.check()
        require(bool(np.isfinite(values[first:first + 64]).all()), "nonfinite_supplied_source_training")

    # Existing binding verifies complete catalog, target normalization and exact
    # source/target image disjointness; it never reads target neural responses.
    pair = interface.bind_content_pair(panel.manifest, source, target=target, budget=budget)
    _same_axis(pair.target_axis, panel.axes[target_index])
    interface.require_image_order(pair.target_normalization.training_image_ids.tolist(),
                                  panel.training_image_ids[target_index])
    pair_index = schedule.PAIR_ORDER.index(identity)
    initial = panel.initializations[pair_index]
    require(initial.pair_index == pair_index and (initial.source, initial.target) == identity
            and initial.input_width == len(source.axis.eligible_native_ids)
            and initial.output_width == len(pair.target_axis.eligible_native_ids),
            "canonical_initialization_must_match_bound_pair_axes")
    budget.check()
    model = schedule.initialize_pair(initial, panel.converter_factory)
    budget.check()
    model = model.to(device=selected_device)
    # Copy preserves the admitted NumPy matrix and avoids a writable Tensor view
    # of its frozen storage. The full row count persists in every optimizer step.
    source_tensor = torch.tensor(values, dtype=torch.float32, device=selected_device)
    factory = interface.content_block_provider if provider_factory is None else provider_factory
    provider = factory(pair, budget=budget, block_units=block_units)
    try:
        completion = training.fit_content_pair(model, source_tensor, panel.channel_schedule,
            provider, check=budget.check, max_block_units=block_units,
            row_chunk_size=row_chunk_size, backend=backend, on_iteration=on_iteration)
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    del source_tensor, provider
    budget.check()
    artifact = final.prepare_final_content_artifact(model, completion, pair, initial,
        panel.channel_schedule, check=budget.check)
    del model
    published = final.write_final_content_artifact(path, artifact, budget=budget,
                                                   maximum_bytes=maximum_bytes)
    return PublishedContentPair(identity[0], target, pair_index, published, completion)
