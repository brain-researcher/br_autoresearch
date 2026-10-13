"""Sixteen fixed original content pairs on the existing cached CUDA driver.

The caller separately admits each pair and finite allocation. This module adds
no reader, estimator, schedule, grant, retry or loop-length option. All already
submitted compositions and shared numerical implementations remain unchanged.
"""
from pathlib import Path
from types import MappingProxyType

import content_channel_cache as cache
import content_pair_driver as driver
import empirical_content_inputs as inputs


OPERATION = "train_and_publish_cached_cuda_remaining_pair"
# These are frozen subject-unique CONTENT eligible axes. In particular sub-01
# has12294 eligible coordinates; the brain comparator's full12295 axis is not
# interchangeable with this existing decoder/converter normalization.
SCOPES = MappingProxyType({
    "remaining-000": ("sub-01", "sub-03", 0, 12294, 12897),
    "remaining-001": ("sub-01", "sub-05", 1, 12294, 10891),
    "remaining-002": ("sub-01", "sub-06", 2, 12294, 12023),
    "remaining-003": ("sub-01", "sub-07", 3, 12294, 12057),
    "remaining-008": ("sub-05", "sub-01", 8, 10891, 12294),
    "remaining-009": ("sub-05", "sub-03", 9, 10891, 12897),
    "remaining-010": ("sub-05", "sub-06", 10, 10891, 12023),
    "remaining-011": ("sub-05", "sub-07", 11, 10891, 12057),
    "remaining-012": ("sub-06", "sub-01", 12, 12023, 12294),
    "remaining-013": ("sub-06", "sub-03", 13, 12023, 12897),
    "remaining-014": ("sub-06", "sub-05", 14, 12023, 10891),
    "remaining-015": ("sub-06", "sub-07", 15, 12023, 12057),
    "remaining-016": ("sub-07", "sub-01", 16, 12057, 12294),
    "remaining-017": ("sub-07", "sub-03", 17, 12057, 12897),
    "remaining-018": ("sub-07", "sub-05", 18, 12057, 10891),
    "remaining-019": ("sub-07", "sub-06", 19, 12057, 12023),
})
MAXIMUM_FINAL_BYTES = 1024**3
require = inputs.require


def train_and_publish_cached_cuda_remaining_pair(bound, *, stage, device,
        block_units, row_chunk_size, backend, final_path, maximum_bytes,
        on_iteration=None):
    """Bind one literal stage, then delegate the original complete fit once."""
    require(isinstance(stage, str) and stage in SCOPES, "fixed_CUDA_remaining_stage_required")
    source, target, pair_index, source_width, target_width = SCOPES[stage]
    require(isinstance(bound, inputs.BoundContentInputs), "bound_content_inputs_required")
    require(device == "cuda:0" and type(block_units) is int and block_units == 256
            and type(row_chunk_size) is int and row_chunk_size == 512 and backend == "sparse",
            "fixed_CUDA_remaining_numerical_options_required")
    require(type(maximum_bytes) is int and maximum_bytes == MAXIMUM_FINAL_BYTES,
            "fixed_CUDA_remaining_final_byte_cap_required")
    require(on_iteration is None or callable(on_iteration), "iteration_callback_must_be_callable")
    path = Path(final_path)
    require(path.is_absolute() and path.suffix == ".pt", "explicit_final_model_path_required")
    options = dict(device=device, block_units=block_units, row_chunk_size=row_chunk_size,
                   backend=backend, final_path=path, maximum_bytes=maximum_bytes)
    admitted = MappingProxyType({**options,
        "feature_read_mode": "bounded_channel_two_pass_cache",
        "maximum_channel_bytes": cache.MAX_CHANNEL_BYTES,
        "maximum_retained_bytes": cache.MAX_RETAINED_BYTES})
    inputs._admit(bound.execution_context, bound.require_admitted_context,
        operation=OPERATION, source=source, target=target, budget=bound.budget,
        execution_options=admitted, allocation_identity=bound.allocation_identity)
    require(bound.source.axis.participant == source and bound.target == target
            and bound.source.normalized_X.shape == (4712, source_width)
            and len(bound.source.axis.eligible_native_ids) == source_width
            and len(bound.pair.target_axis.eligible_native_ids) == target_width
            and len(bound.source.image_ids) == 4712
            and len(bound.panel.axes) == 5 and len(bound.panel.initializations) == 20,
            "fixed_full_CUDA_remaining_scope_required")
    channels = bound.panel.channel_schedule.channels_by_layer
    require(len(channels) == 19 and all(len(values) == 1024 for values in channels),
            "complete_shared_19_layer_1024_step_schedule_required")
    initial = bound.panel.initializations[pair_index]
    require(initial.pair_index == pair_index and (initial.source, initial.target) == (source, target)
            and initial.input_width == source_width and initial.hidden_width == source_width // 2
            and initial.output_width == target_width,
            "canonical_remaining_pair_initialization_required")
    require(inputs._read_loss_correction() == bound.loss_correction,
            "loss_correction_changed_after_content_input_binding")
    return driver.train_and_publish_content_pair(bound.panel, bound.source,
        target=target, budget=bound.budget, on_iteration=on_iteration,
        provider_factory=cache.make_cached_provider, **options)
