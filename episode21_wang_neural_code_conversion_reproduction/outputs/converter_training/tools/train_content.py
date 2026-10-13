"""Unactivated content-converter training loop over already admitted objects.

No file I/O, CLI, data admission, submission or publication occurs here. The
future empirical driver must supply the corrected loss binding, converter
authority, complete decoder collection, exact role/axis reader, source-faithful
initialization and qualified numerical backend. The current decoder grant does
not admit calling this loop on real responses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from content_loss import ActiveLayer, BlockProvider, content_loss_step
from decoder_interface import active_layer_units
from schedule import ChannelSchedule, ITERATIONS, make_channel_schedule


TRAINING_ROWS = 4712
LEARNING_RATE = 0.00005
DECAY_START = 512


@dataclass(frozen=True)
class CompletedContentFit:
    """Completion metadata only; the caller retains its final in-memory model."""

    iterations: int
    rows: int
    last_step_loss_before_update: float
    final_used_learning_rate: float
    backend: str
    bitwise_source_equivalence: bool = False


def tagged_lr_factor(iteration: int) -> float:
    """Pinned LambdaLR(1024, offset=0, decay_start=512).step expression."""
    if type(iteration) is not int or not 0 <= iteration <= ITERATIONS:
        raise ValueError("learning_rate_iteration_out_of_range")
    return 1.0 - max(0, iteration - DECAY_START) / (ITERATIONS - DECAY_START)


def fit_content_pair(
    model: torch.nn.Module,
    source_normalized: torch.Tensor,
    channel_schedule: ChannelSchedule,
    provider: BlockProvider,
    *,
    check: Callable[[], None],
    max_block_units: int,
    row_chunk_size: int,
    backend: Literal["sparse", "dense"],
    on_iteration: Callable[[dict], None] | None = None,
) -> CompletedContentFit:
    """Run all 1,024 full-batch steps; failures propagate without a final result.

    ``check`` is the future caller's existing allocation/deadline check, not
    authority to read data. The provider also checks that budget on bounded
    reads. ``on_iteration`` receives scalars only; no intermediate checkpoint
    or outcome-based stopping rule is implemented. Publication belongs to the
    admitted driver and may happen only after this function returns normally.
    An interrupted optimizer is not resumable by calling this function again.
    """
    if not callable(check):
        raise ValueError("allocation_check_required")
    check()
    if (not isinstance(channel_schedule, ChannelSchedule)
            or channel_schedule != make_channel_schedule()):
        raise ValueError("exact_shared_source_channel_schedule_required")
    if (not isinstance(source_normalized, torch.Tensor)
            or source_normalized.dtype != torch.float32
            or source_normalized.ndim != 2
            or source_normalized.shape[0] != TRAINING_ROWS
            or source_normalized.requires_grad):
        raise ValueError("all_4712_fixed_source_training_rows_required")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("initialized_pinned_converter_required")
    if on_iteration is not None and not callable(on_iteration):
        raise ValueError("iteration_callback_must_be_callable")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, tagged_lr_factor)
    final_loss = final_rate = None
    for iteration in range(ITERATIONS):
        check()
        active = tuple(ActiveLayer(layer, channel, active_layer_units(layer, channel))
                       for layer, channel in channel_schedule.blocks(iteration))
        used_rate = float(optimizer.param_groups[0]["lr"])
        result = content_loss_step(model, optimizer, source_normalized, active, provider,
            expected_rows=TRAINING_ROWS, max_block_units=max_block_units,
            row_chunk_size=row_chunk_size, backend=backend)
        final_loss, final_rate = float(result.loss.item()), used_rate
        # Do not keep a response-sized adjoint or old iteration graph alive.
        del result
        scheduler.step()  # Tagged order: optimizer step first, then LR update.
        if on_iteration is not None:
            on_iteration({"iteration": iteration, "completed_iterations": iteration + 1,
                          "training_loss": final_loss, "learning_rate_used": used_rate})
    check()
    return CompletedContentFit(ITERATIONS, TRAINING_ROWS, final_loss, final_rate, backend)
