"""Fixed original content pairs 5–7 on the existing cached CUDA driver.

The caller separately admits each pair and finite allocation. This module adds
no reader, estimator, schedule, grant, retry or loop-length option. Pair 4's
running composition and all shared numerical implementations remain unchanged.
"""
from pathlib import Path
from types import MappingProxyType

import content_channel_cache as cache
import content_pair_driver as driver
import empirical_content_inputs as inputs


OPERATION = "train_and_publish_cached_cuda_roster_pair"
SCOPES = MappingProxyType({
    "roster-005": ("sub-03", "sub-05", 5, 10891),
    "roster-006": ("sub-03", "sub-06", 6, 12023),
    "roster-007": ("sub-03", "sub-07", 7, 12057),
})
MAXIMUM_FINAL_BYTES = 1024**3
require = inputs.require


def train_and_publish_cached_cuda_roster_pair(bound, *, stage, device,
        block_units, row_chunk_size, backend, final_path, maximum_bytes,
        on_iteration=None):
    """Bind one literal stage, then delegate the original complete fit once."""
    require(isinstance(stage, str) and stage in SCOPES, "fixed_CUDA_roster_stage_required")
    source, target, pair_index, target_width = SCOPES[stage]
    require(isinstance(bound, inputs.BoundContentInputs), "bound_content_inputs_required")
    require(device == "cuda:0" and type(block_units) is int and block_units == 256
            and type(row_chunk_size) is int and row_chunk_size == 512 and backend == "sparse",
            "fixed_CUDA_roster_numerical_options_required")
    require(type(maximum_bytes) is int and maximum_bytes == MAXIMUM_FINAL_BYTES,
            "fixed_CUDA_roster_final_byte_cap_required")
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
            and bound.source.normalized_X.shape == (4712, 12897)
            and len(bound.source.axis.eligible_native_ids) == 12897
            and len(bound.pair.target_axis.eligible_native_ids) == target_width
            and len(bound.source.image_ids) == 4712
            and len(bound.panel.axes) == 5 and len(bound.panel.initializations) == 20,
            "fixed_full_CUDA_roster_scope_required")
    channels = bound.panel.channel_schedule.channels_by_layer
    require(len(channels) == 19 and all(len(values) == 1024 for values in channels),
            "complete_shared_19_layer_1024_step_schedule_required")
    initial = bound.panel.initializations[pair_index]
    require(initial.pair_index == pair_index and (initial.source, initial.target) == (source, target)
            and initial.input_width == 12897 and initial.output_width == target_width,
            "canonical_roster_pair_initialization_required")
    require(inputs._read_loss_correction() == bound.loss_correction,
            "loss_correction_changed_after_content_input_binding")
    return driver.train_and_publish_content_pair(bound.panel, bound.source,
        target=target, budget=bound.budget, on_iteration=on_iteration,
        provider_factory=cache.make_cached_provider, **options)
