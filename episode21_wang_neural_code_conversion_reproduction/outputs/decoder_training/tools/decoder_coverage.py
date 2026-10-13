"""Exact decoder identities and compact coverage; no payload reads or execution.

The full target is all five participants and all 19 layers. This module plans
coverage only: it neither authorizes fitting nor claims that a planned row exists.
"""
from dataclasses import dataclass
import math

import train_retained_prefix as prefix

PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
PER_PARTICIPANT = 14_861_288
TOTAL = 74_306_440
RETAINED = 51_328
require = prefix.require


@dataclass(frozen=True, order=True)
class Span:
    participant: str
    layer: str
    channel: int
    start: int
    stop: int

    @property
    def shape(self):
        require(self.participant in PARTICIPANTS and self.layer in prefix.LAYERS,
                "locked_participant_and_layer_required")
        return prefix.SHAPES[prefix.LAYERS.index(self.layer)]

    def validate(self):
        shape = self.shape
        count, size = (shape[0], math.prod(shape[1:])) if len(shape) == 3 else (1, shape[0])
        require(all(type(x) is int for x in (self.channel, self.start, self.stop))
                and 0 <= self.channel < count and 0 <= self.start < self.stop <= size,
                "exact_channel_spatial_interval_required")
        return self

    @property
    def count(self):
        return self.stop - self.start

    @property
    def key(self):
        return self.participant, self.layer, self.channel

    def record(self):
        self.validate()
        return {"participant": self.participant, "layer": self.layer, "channel": self.channel,
                "spatial_Fortran_interval": [self.start, self.stop]}


def complete_spans(participants=PARTICIPANTS, layers=prefix.LAYERS):
    require(len(set(participants)) == len(participants) and len(set(layers)) == len(layers),
            "duplicate_scope_required_no")
    for participant in participants:
        for layer in layers:
            shape = prefix.SHAPES[prefix.LAYERS.index(layer)]
            channels, size = (shape[0], math.prod(shape[1:])) if len(shape) == 3 else (1, shape[0])
            for channel in range(channels):
                yield Span(participant, layer, channel, 0, size).validate()


def retained_spans():
    # Prefix spatial indices differ from a contiguous full-layer Fortran prefix.
    for layer in prefix.LAYERS:
        yield Span("sub-01", layer, 0, 0, 64)
    yield Span("sub-01", "conv1_1", 0, 64, 50176)


def subtract_retained(span):
    """A fixed subtraction of completed identities, independent of fitted values."""
    span.validate()
    intervals = [(span.start, span.stop)]
    for old in retained_spans():
        if old.key != span.key:
            continue
        remaining = []
        for start, stop in intervals:
            if old.stop <= start or old.start >= stop:
                remaining.append((start, stop))
            else:
                if start < old.start:
                    remaining.append((start, old.start))
                if old.stop < stop:
                    remaining.append((old.stop, stop))
        intervals = remaining
    return [Span(*span.key, start, stop) for start, stop in intervals]


def remaining_spans():
    for whole in complete_spans():
        yield from subtract_retained(whole)


def next_layer_spans():
    # Complete the first required full layer; never replay its completed channel 0.
    return tuple(Span("sub-01", "conv1_1", channel, 0, 50176) for channel in range(1, 64))


def pack_work(spans, max_units=1_003_520):
    """Deterministic whole-channel packing; never mix participant training states.

    Prefix exclusions can leave a genuine partial first channel. No extra target
    is introduced to fill a work item or a 64-target arithmetic block.
    """
    require(type(max_units) is int and max_units > 0, "positive_work_cap_required")
    current, size = [], 0
    for span in spans:
        span.validate()
        require(span.count <= max_units, "work_cap_cannot_split_channel")
        if current and (size + span.count > max_units or span.participant != current[0].participant):
            yield tuple(current)
            current, size = [], 0
        current.append(span)
        size += span.count
    if current:
        yield tuple(current)


def verify_complete_coverage(spans, *, participants=PARTICIPANTS, layers=prefix.LAYERS):
    """Reject duplicates, gaps and scope additions before asserting completeness."""
    expected = {span.key: span for span in complete_spans(participants, layers)}
    grouped = {}
    for span in spans:
        span.validate()
        require(span.key in expected, "coverage_outside_required_scope")
        grouped.setdefault(span.key, []).append(span)
    require(set(grouped) == set(expected), "missing_required_channels")
    total = 0
    for key, whole in expected.items():
        cursor = 0
        for span in sorted(grouped[key], key=lambda value: value.start):
            require(span.start == cursor, "coverage_gap_or_overlap")
            cursor = span.stop
            total += span.count
        require(cursor == whole.stop, "incomplete_channel")
    return total


def coordinates(shape, channel, spatial):
    """Tensor coordinate, source full-F output ID and packed-mask full-C ID."""
    if len(shape) == 1:
        require(channel == 0 and 0 <= spatial < shape[0], "FC_coordinate_outside_scope")
        return (spatial,), spatial, spatial
    channels, height, width = shape
    require(0 <= channel < channels and 0 <= spatial < height * width,
            "convolutional_coordinate_outside_scope")
    h, w = spatial % height, spatial // height
    return (channel, h, w), channel + channels * spatial, channel * height * width + h * width + w
