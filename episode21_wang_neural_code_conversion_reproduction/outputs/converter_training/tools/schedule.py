"""Pinned content-converter channel schedule and ordered CPU initialization.

Source: InterSiteNeuralCodeConversion V1.0.0, commit
edbff02edd6a08d3c6c829af45a957fd14bdc460, NCC_content_loss/NCC_train.py
(`prepare_indices` and `main`) and NCC_content_loss/models.py (`Converter`).
The source prepares one layer-major schedule before its ordered pair loop.
This module preserves that schedule and the one-seed initialization sequence
while allowing a later admitted runner to execute pairs in another job order.

Import and schedule construction perform no file I/O. The explicit source
loader reuses the existing conformance report's exact archive/member/SHA binding
for the pinned Converter class; no empirical data, checkpoint or authorization
is read. Its source input and target output widths must be the frozen eligible
native-voxel counts, not the lengths of full-native normalization vectors.
Construction is CPU/float32, as in the source before model.cuda(). GPU execution
and cross-version bitwise equivalence are not established by these helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import platform
import random
from typing import Any, Callable, Mapping


SEED = 42
ITERATIONS = 1024
PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
PAIR_ORDER = tuple(permutations(PARTICIPANTS, 2))
LAYER_ORDER = (
    "fc8", "fc7", "fc6",
    "conv5_4", "conv5_3", "conv5_2", "conv5_1",
    "conv4_4", "conv4_3", "conv4_2", "conv4_1",
    "conv3_4", "conv3_3", "conv3_2", "conv3_1",
    "conv2_2", "conv2_1", "conv1_2", "conv1_1",
)
_CHANNEL_COUNTS = {
    "conv1_1": 64, "conv1_2": 64,
    "conv2_1": 128, "conv2_2": 128,
    "conv3_1": 256, "conv3_2": 256, "conv3_3": 256, "conv3_4": 256,
    "conv4_1": 512, "conv4_2": 512, "conv4_3": 512, "conv4_4": 512,
    "conv5_1": 512, "conv5_2": 512, "conv5_3": 512, "conv5_4": 512,
}
ConverterFactory = Callable[[int, int], Any]


def load_pinned_converter_factory() -> ConverterFactory:
    """Load only the already-bound public models.py, without constructing a model.

    Reuse feature_protocol.converter_conformance's attempt-005 report and member
    hash. This is the same source-only route, not a new archive copy, receipt or
    empirical-admission mechanism. Missing source material is an error; there is
    no downloader, alternate implementation or path discovery fallback.
    """
    import hashlib
    import json
    from pathlib import Path
    import zipfile

    base = Path(__file__).resolve().parents[3]
    report_path = base / "outputs/pipeline_conformance/attempt-005/neutral_conformance_report.json"
    with report_path.open("rb") as source:
        report_bytes = source.read(1_000_001)
    if len(report_bytes) > 1_000_000:
        raise ValueError("source_report_exceeds_metadata_bound")
    report = json.loads(report_bytes)
    reference = report["source_members"]["content_models"]
    with zipfile.ZipFile(report["source_archive"]["path"]) as archive:
        with archive.open(reference["path"]) as source:
            data = source.read(65_537)
    if len(data) > 65_536:
        raise ValueError("pinned_model_source_exceeds_text_bound")
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("tagged_source_identity_mismatch")
    namespace = {}
    exec(compile(data.decode("utf-8"), "PINNED_V1_models.py", "exec"), namespace)
    return namespace["Converter"]


@dataclass(frozen=True)
class ChannelSchedule:
    """One shared schedule; None means every unit in that FC layer.

    The source uses a zero chunk index for FC layers, not FC unit zero. The
    engine-facing None makes that distinction explicit. Tuple storage permits
    reuse across pairs without consuming RNG or advancing a mutable cursor.
    """

    channels_by_layer: tuple[tuple[int | None, ...], ...]
    python_version: str

    def blocks(self, iteration: int) -> tuple[tuple[str, int | None], ...]:
        if type(iteration) is not int or not 0 <= iteration < ITERATIONS:
            raise ValueError("iteration_must_be_an_integer_in_0_through_1023")
        return tuple((layer, channels[iteration])
                     for layer, channels in zip(LAYER_ORDER, self.channels_by_layer))


def make_channel_schedule() -> ChannelSchedule:
    """Generate the tagged 1,024-step schedule, without changing global RNG.

    Each layer completes all its permutation cycles before generation moves
    to the next layer. After the first cycle, sample the *previous permutation*,
    not range(C) again and not a with-replacement channel draw. Exact replay is
    tied to the recorded Python runtime's random.sample implementation.
    """
    rng = random.Random(SEED)
    schedule = []
    for layer in LAYER_ORDER:
        if layer.startswith("fc"):
            schedule.append((None,) * ITERATIONS)
            continue
        count = _CHANNEL_COUNTS[layer]
        current = rng.sample(range(count), count)
        indices = []
        for iteration in range(ITERATIONS):
            if iteration > 0 and iteration % count == 0:
                current = rng.sample(current, count)
            indices.append(current[iteration % count])
        schedule.append(tuple(indices))
    return ChannelSchedule(tuple(schedule), platform.python_version())


@dataclass(frozen=True)
class PairInitialization:
    """CPU RNG replay state for one canonical pair, not fitted model weights.

    Tensor states are cloned on capture and use. They remain caller-owned
    objects and must be retained unchanged with the future pair artifact.
    Runtime equality is required on replay; it is not historical-runtime
    conformance or authentication of the supplied source factory.
    """

    pair_index: int
    source: str
    target: str
    input_width: int
    hidden_width: int
    output_width: int
    torch_version: str
    cpu_rng_before: Any
    cpu_rng_after: Any


def _eligible_widths(widths: Mapping[str, int]) -> dict[str, int]:
    if set(widths) != set(PARTICIPANTS):
        raise ValueError("exact_five_participant_eligible_widths_required")
    copied = {person: widths[person] for person in PARTICIPANTS}
    # The source applies InstanceNorm to the hidden spatial axis; width one
    # cannot support its training-time variance calculation.
    if any(type(width) is not int or width < 4 for width in copied.values()):
        raise ValueError("eligible_widths_must_be_integers_at_least_four")
    return copied


def _check_cpu_float32_model(model: Any, torch: Any) -> None:
    if not isinstance(model, torch.nn.Module):
        raise ValueError("pinned_Converter_module_factory_required")
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError("pinned_Converter_must_be_constructed_on_CPU_in_float32")


def capture_pair_initializations(
    eligible_widths: Mapping[str, int], converter_factory: ConverterFactory,
) -> tuple[PairInitialization, ...]:
    """Capture all 20 initializations in one canonical source-major pass.

    Factory contract: the authenticated pinned Converter(input_nc, output_nc),
    with hidden width input_nc//2, CPU parameters and unchanged InstanceNorm
    and LeakyReLU layers. No training, optimizer, data loading or other RNG
    consumer belongs inside the factory. The pinned model/training loop has
    no stochastic forward layer, so intervening pair training does not consume
    CPU torch random numbers. A future change to that fact invalidates replay.

    All predecessors are constructed even if an eventual pair job is skipped
    or already complete: scheduling cannot change a later pair's initialization.
    This is the canonical fresh 20-pair sequence, not the source script's
    filesystem-dependent skip behavior. Peak live model count is one; caller
    allocation sizing must still accommodate the largest eligible-width model.
    These functions mutate the process CPU RNG temporarily and must not run
    concurrently with another RNG consumer in the same process.
    """
    widths = _eligible_widths(eligible_widths)
    import torch

    if torch.get_default_dtype() != torch.float32:
        raise ValueError("source_float32_default_dtype_required")
    records = []
    # Seed CPU once without touching CUDA generators. Source constructors run
    # on CPU before any .cuda(), so this is their torch.manual_seed(42) stream.
    with torch.random.fork_rng(devices=[]):
        initial = torch.Generator(device="cpu").manual_seed(SEED).get_state()
        torch.set_rng_state(initial)
        for index, (source, target) in enumerate(PAIR_ORDER):
            before = torch.get_rng_state().clone()
            model = converter_factory(widths[source], widths[target])
            _check_cpu_float32_model(model, torch)
            after = torch.get_rng_state().clone()
            records.append(PairInitialization(
                index, source, target, widths[source], widths[source] // 2,
                widths[target], str(torch.__version__), before, after,
            ))
            del model
    return tuple(records)


def initialize_pair(record: PairInitialization, converter_factory: ConverterFactory) -> Any:
    """Construct one pinned pair model from its saved canonical CPU RNG state.

    Preserve the caller's CPU RNG on success and failure. A factory whose random
    consumption differs from capture is refused. Source identity/architecture
    qualification remains the caller's responsibility; RNG equality alone is
    not an architecture check. No model is moved to a GPU here.
    """
    if (type(record.pair_index) is not int or not 0 <= record.pair_index < len(PAIR_ORDER)
            or (record.source, record.target) != PAIR_ORDER[record.pair_index]):
        raise ValueError("canonical_pair_identity_required")
    if (type(record.input_width) is not int or record.input_width < 4
            or type(record.output_width) is not int or record.output_width < 4
            or record.hidden_width != record.input_width // 2):
        raise ValueError("canonical_eligible_width_binding_required")
    import torch

    if str(torch.__version__) != record.torch_version:
        raise ValueError("captured_torch_runtime_required")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("source_float32_default_dtype_required")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(record.cpu_rng_before.clone())
        model = converter_factory(record.input_width, record.output_width)
        _check_cpu_float32_model(model, torch)
        if not torch.equal(torch.get_rng_state(), record.cpu_rng_after):
            raise ValueError("factory_CPU_RNG_consumption_differs_from_canonical_capture")
    return model
