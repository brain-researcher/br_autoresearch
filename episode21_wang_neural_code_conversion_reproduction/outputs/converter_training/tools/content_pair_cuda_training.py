"""Dormant full cached-CUDA composition; no loader, grant, loop or launcher.

The caller must independently admit full production on the original canonical
pair. A four-update measurement cannot be promoted through this operation.
The existing driver owns all 1,024 updates, canonical schedule, cache cleanup
and exclusive final-only publication. This module changes no arithmetic and
offers no shorter fit, resume, retry, scoring or extra data role.
"""
from pathlib import Path
from types import MappingProxyType

import content_channel_cache as cache
import content_pair_driver as driver
import empirical_content_inputs as inputs


OPERATION = "train_and_publish_cached_cuda_full_pair"
SOURCE, TARGET, PAIR_INDEX = "sub-03", "sub-01", 4
CACHE_MODE = "bounded_channel_two_pass_cache"
MAXIMUM_FINAL_BYTES = 1024**3
require = inputs.require


def train_and_publish_cached_cuda_full_pair(bound, *, device, block_units,
        row_chunk_size, backend, final_path, maximum_bytes, on_iteration=None):
    """Separately admit the fixed full fit, then compose existing components.

    The external resolver receives the six ordinary execution options plus
    the three fixed cache declarations before any provider is selected. Actual
    allocation/device/precision/deadline authority remains with that resolver
    and its budget; this adapter does not construct or expand it.
    """
    require(isinstance(bound, inputs.BoundContentInputs), "bound_content_inputs_required")
    require(device == "cuda:0" and type(block_units) is int and block_units == 256
            and type(row_chunk_size) is int and row_chunk_size == 512 and backend == "sparse",
            "fixed_cached_CUDA_sparse_256_by512_execution_required")
    require(type(maximum_bytes) is int and maximum_bytes == MAXIMUM_FINAL_BYTES,
            "fixed_one_GiB_final_model_cap_required")
    require(on_iteration is None or callable(on_iteration), "iteration_callback_must_be_callable")
    path = Path(final_path)
    require(path.is_absolute() and path.suffix == ".pt", "absolute_final_pt_path_required")
    driver_options = dict(device=device, block_units=block_units, row_chunk_size=row_chunk_size,
                          backend=backend, final_path=path, maximum_bytes=maximum_bytes)
    admitted_options = MappingProxyType({**driver_options, "feature_read_mode": CACHE_MODE,
        "maximum_channel_bytes": cache.MAX_CHANNEL_BYTES,
        "maximum_retained_bytes": cache.MAX_RETAINED_BYTES})
    inputs._admit(bound.execution_context, bound.require_admitted_context,
        operation=OPERATION, source=SOURCE, target=TARGET, budget=bound.budget,
        execution_options=admitted_options, allocation_identity=bound.allocation_identity)
    require(bound.source.axis.participant == SOURCE and bound.target == TARGET
            and bound.source.normalized_X.shape == (4712, 12897)
            and len(bound.source.axis.eligible_native_ids) == 12897
            and len(bound.pair.target_axis.eligible_native_ids) == 12294
            and len(bound.panel.channel_schedule.channels_by_layer) == 19
            and len(bound.panel.initializations) == 20,
            "original_full_pair4_rows_axes_and_layers_required")
    initial = bound.panel.initializations[PAIR_INDEX]
    require(initial.pair_index == PAIR_INDEX and (initial.source, initial.target) == (SOURCE, TARGET)
            and initial.input_width == 12897 and initial.output_width == 12294,
            "original_pair4_canonical_initialization_required")
    require(inputs._read_loss_correction() == bound.loss_correction,
            "loss_correction_changed_after_content_input_binding")
    return driver.train_and_publish_content_pair(bound.panel, bound.source,
        target=TARGET, budget=bound.budget, on_iteration=on_iteration,
        provider_factory=cache.make_cached_provider, **driver_options)
