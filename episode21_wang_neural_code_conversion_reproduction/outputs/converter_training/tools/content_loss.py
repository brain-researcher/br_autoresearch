"""UNACTIVATED, array-only full-batch content-loss numerical engine.

The caller owns admission, all 4,712 source rows, native-axis mapping, the
canonical layer/channel schedule, source model initialization and Adam/LR
schedule. Converted responses are ALREADY in target-decoder standardized
input coordinates. Provider targets must use the frozen TARGET decoder's
feature means/scales on the SOURCE participant's image features.

There is no file reader, empirical authority, model construction or scheduler.
A provider is repeatable and immutable for the duration of a call: its two
passes return the same complete active targets and frozen decoder. This simple
implementation rereads targets on both passes; full-scale I/O and runtime are
unqualified. No cached norms or bitwise source equivalence are claimed.

The source norm expression is retained: float32 norm, then inverse square,
then normalization across active layers. Block/row reductions, sparse mm and
the explicit adjoint change floating-point accumulation order relative to the
source's complete dense tensors. Focused forward/gradient equivalence remains
to be measured on compute before any empirical use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence

import torch


@dataclass(frozen=True)
class ActiveLayer:
    """One scheduled conv map (channel=int), or all FC units (channel=None)."""

    layer: str
    channel: int | None
    units: int


@dataclass(frozen=True)
class LossBlock:
    """Contiguous units of one active layer in its declared feature order.

    Columns/weights have shape [units, selected_voxels], bias [units], and
    standardized_features [ALL source images, units]. Columns refer to the
    eligible target output axis, not the full native axis. Production admission
    supplies 500 selected voxels; tiny numerical fixtures may use fewer.
    Tensor or NumPy array inputs are accepted; frozen values may not require
    gradients. The interface adapter owns the authentic native IDs/coordinates.
    """

    unit_start: int
    eligible_columns: object
    weights: object
    bias: object
    standardized_features: object


@dataclass(frozen=True)
class ContentLossResult:
    """Detached numerical result; never an empirical or conformance receipt."""

    loss: torch.Tensor
    response_adjoint: torch.Tensor
    active_layers: tuple[ActiveLayer, ...]
    norms: torch.Tensor
    layer_weights: torch.Tensor
    layer_squared_errors: torch.Tensor
    backend: str
    bitwise_source_equivalence: bool = False


BlockProvider = Callable[[ActiveLayer], Iterable[LossBlock]]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_int(value, name):
    _require(type(value) is int and value > 0, name + " must be a positive integer")


def _frozen_tensor(value, *, dtype, device, name):
    if isinstance(value, torch.Tensor):
        _require(not value.requires_grad, name + " must be frozen")
        result = value.detach().to(device=device)
    else:
        # Copy provider arrays: no writable Torch view of a read-only NumPy buffer.
        result = torch.tensor(value, device=device)
    _require(result.dtype == dtype, name + " has the wrong dtype")
    _require(bool(torch.isfinite(result).all()), name + " must be finite")
    return result


def _blocks(provider, active, *, rows, width, max_block_units, device):
    next_unit = 0
    for block in provider(active):
        _require(isinstance(block, LossBlock), "provider must yield LossBlock")
        _require(type(block.unit_start) is int and block.unit_start == next_unit,
                 "active unit coverage must be contiguous and ordered")
        columns = _frozen_tensor(block.eligible_columns, dtype=torch.int64,
                                 device=device, name="eligible_columns")
        weights = _frozen_tensor(block.weights, dtype=torch.float32,
                                 device=device, name="decoder weights")
        bias = _frozen_tensor(block.bias, dtype=torch.float32, device=device, name="decoder bias")
        targets = _frozen_tensor(block.standardized_features, dtype=torch.float32,
                                 device=device, name="standardized targets")
        _require(columns.ndim == 2 and 0 < columns.shape[0] <= max_block_units
                 and 0 < columns.shape[1] <= width, "invalid bounded selected-voxel block")
        units = columns.shape[0]
        _require(weights.shape == columns.shape and bias.shape == (units,)
                 and targets.shape == (rows, units), "decoder/target block shapes differ")
        _require(next_unit + units <= active.units, "active block exceeds declared units")
        _require(bool(((columns >= 0) & (columns < width)).all()),
                 "selected columns are outside the eligible target axis")
        sorted_columns = torch.sort(columns, dim=1).values
        _require(not bool((sorted_columns[:, 1:] == sorted_columns[:, :-1]).any()),
                 "a feature unit has duplicate selected voxel columns")
        next_unit += units
        yield columns, weights, bias, targets
    _require(next_unit == active.units, "active layer has missing units")


def _decoder_matrix(columns, weights, width, backend):
    units = columns.shape[0]
    unit_ids = torch.arange(units, device=columns.device).unsqueeze(1).expand_as(columns)
    if backend == "sparse":
        coordinates = torch.stack((columns.reshape(-1), unit_ids.reshape(-1)))
        return torch.sparse_coo_tensor(coordinates, weights.reshape(-1),
                                      (width, units), device=weights.device).coalesce()
    matrix = torch.zeros((width, units), dtype=torch.float32, device=weights.device)
    matrix[columns.reshape(-1), unit_ids.reshape(-1)] = weights.reshape(-1)
    return matrix


def content_loss_and_adjoint(
    converted_responses: torch.Tensor,
    active_layers: Sequence[ActiveLayer],
    provider: BlockProvider,
    *,
    expected_rows: int,
    max_block_units: int,
    row_chunk_size: int,
    backend: Literal["sparse", "dense"],
) -> ContentLossResult:
    """Compute complete weighted SSE and its gradient w.r.t. converted input.

    All source rows participate in both passes. Feature/row blocking changes
    storage and reduction order, not sample support or optimizer step count.
    Temporary decoder storage is sparse selected-voxel entries or an explicit
    bounded [target_width, block_units] dense block. Products use at most
    [row_chunk_size, target_width] adjoint workspace; never [N, U, 500].
    Sparse support is device/runtime dependent and has no silent fallback.
    """
    _positive_int(expected_rows, "expected_rows")
    _positive_int(max_block_units, "max_block_units")
    _positive_int(row_chunk_size, "row_chunk_size")
    _require(backend in ("sparse", "dense"), "select an explicit sparse or dense backend")
    _require(isinstance(converted_responses, torch.Tensor)
             and converted_responses.dtype == torch.float32
             and converted_responses.ndim == 2
             and converted_responses.shape[0] == expected_rows
             and converted_responses.shape[1] > 0, "complete float32 response matrix required")
    _require(bool(torch.isfinite(converted_responses.detach()).all()),
             "converted responses must be finite")
    layers = tuple(active_layers)
    _require(bool(layers) and all(isinstance(layer, ActiveLayer) for layer in layers),
             "explicit active layers required")
    _require(len({layer.layer for layer in layers}) == len(layers),
             "each scheduled layer occurs exactly once")
    for layer in layers:
        _require(isinstance(layer.layer, str) and bool(layer.layer), "layer name required")
        _positive_int(layer.units, "active units")
        _require(layer.channel is None or type(layer.channel) is int and layer.channel >= 0,
                 "channel must be a selected nonnegative index or None for FC")
    width, device = converted_responses.shape[1], converted_responses.device
    block_args = dict(rows=expected_rows, width=width, max_block_units=max_block_units, device=device)

    # Nothing in the fixed decoder, target features or layer weights is part of
    # autograd. Only the resulting full response adjoint reaches the converter.
    with torch.no_grad():
        norms = []
        for active in layers:
            squared_norm = torch.zeros((), dtype=torch.float32, device=device)
            for _, _, _, targets in _blocks(provider, active, **block_args):
                for start in range(0, expected_rows, row_chunk_size):
                    squared_norm.add_(targets[start:start + row_chunk_size].square().sum())
            norm = squared_norm.sqrt()  # Keep source float32 sqrt-then-square.
            _require(bool(torch.isfinite(norm)) and bool(norm > 0),
                     "complete active-layer norm is zero or nonfinite")
            norms.append(norm)
        norms = torch.stack(norms)
        inverse_squared = norms.square().reciprocal()
        _require(bool(torch.isfinite(inverse_squared).all()) and bool((inverse_squared > 0).all()),
                 "inverse squared norms are not finite positive float32 values")
        denominator = inverse_squared.sum()
        _require(bool(torch.isfinite(denominator)) and bool(denominator > 0),
                 "layer weight denominator is nonfinite")
        layer_weights = inverse_squared / denominator
        _require(bool((layer_weights > 0).all()), "normalized layer weight underflow")
        responses = converted_responses.detach()
        adjoint = torch.zeros_like(responses)
        loss = torch.zeros((), dtype=torch.float32, device=device)
        squared_errors = []
        for index, active in enumerate(layers):
            layer_sse = torch.zeros_like(loss)
            factor = 2 * layer_weights[index]
            for columns, weights, bias, targets in _blocks(provider, active, **block_args):
                decoder = _decoder_matrix(columns, weights, width, backend)
                for start in range(0, expected_rows, row_chunk_size):
                    stop = min(start + row_chunk_size, expected_rows)
                    rows = responses[start:stop]
                    if backend == "sparse":
                        predicted = torch.sparse.mm(decoder.transpose(0, 1), rows.T).T
                    else:
                        predicted = rows @ decoder
                    residual = predicted + bias - targets[start:stop]
                    layer_sse.add_(residual.square().sum())
                    if backend == "sparse":
                        contribution = torch.sparse.mm(decoder, (factor * residual).T).T
                    else:
                        contribution = (factor * residual) @ decoder.T
                    adjoint[start:stop].add_(contribution)
                del decoder
            squared_errors.append(layer_sse)
            loss.add_(layer_weights[index] * layer_sse)
        _require(bool(torch.isfinite(loss)) and bool(torch.isfinite(adjoint).all()),
                 "content loss or response adjoint is nonfinite")
        return ContentLossResult(loss, adjoint, layers, norms, layer_weights,
                                 torch.stack(squared_errors), backend)


def content_loss_step(
    converter: torch.nn.Module,
    optimizer: torch.optim.Adam,
    source_normalized: torch.Tensor,
    active_layers: Sequence[ActiveLayer],
    provider: BlockProvider,
    *,
    expected_rows: int,
    max_block_units: int,
    row_chunk_size: int,
    backend: Literal["sparse", "dense"],
) -> ContentLossResult:
    """One full-batch forward, one converter backward, one Adam step.

    Initialization, optimizer parameters, iteration number, LR scheduler and
    publication are the admitted driver's responsibility. No optimizer step
    occurs on an invalid loss/adjoint/parameter gradient. An optimizer failure
    is not transactional: the caller must abort, never retry this step blindly.
    """
    _require(isinstance(optimizer, torch.optim.Adam), "the selected optimizer is Adam")
    _require(isinstance(source_normalized, torch.Tensor)
             and source_normalized.dtype == torch.float32
             and source_normalized.ndim == 2 and source_normalized.shape[0] == expected_rows
             and not source_normalized.requires_grad
             and bool(torch.isfinite(source_normalized).all()),
             "full fixed standardized source batch required")
    parameters = tuple(parameter for parameter in converter.parameters() if parameter.requires_grad)
    optimized = tuple(parameter for group in optimizer.param_groups for parameter in group["params"])
    _require(bool(parameters) and len(optimized) == len(parameters)
             and {id(parameter) for parameter in optimized} == {id(parameter) for parameter in parameters},
             "Adam must own exactly this converter's trainable parameters")
    optimizer.zero_grad(set_to_none=True)
    converted = converter(source_normalized)
    _require(converted.requires_grad, "converter output must retain its parameter graph")
    result = content_loss_and_adjoint(converted, active_layers, provider,
        expected_rows=expected_rows, max_block_units=max_block_units,
        row_chunk_size=row_chunk_size, backend=backend)
    converted.backward(result.response_adjoint)
    _require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                 for parameter in parameters), "converter parameter gradients must be finite")
    optimizer.step()
    return result
