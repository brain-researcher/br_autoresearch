#!/usr/bin/env python3
"""Finite real-feature EP18 false-partition cache and deterministic search.

Prospective recipe, to be recorded by the coordinator before EEG: all six
families, all six fixed outer folds, seed 20261012, 8 members, 64 starts/member,
100000 legal proposals/start. This stage constructs image/context candidates;
it NEVER reports full false-bank qualification. Exact selected-design spectrum,
leverage and term-EDF rechecks remain mandatory and cannot regenerate this bank.

Per fold/family use the ten training blocks' population column scaling (zero
SD columns map to zero), followed by per-image L2 normalization. Zero norms
fail. Match held-out minimum cosine distance to the ten training group members,
mean training-pair cosine scatter, and held-out pair cosine scatter separately.

Context is fixed per image before searching. Use signed 20-ms bins over the
overlap horizon ±(800+FIR_left+FIR_right) source samples; samples are 1 ms. Bins
are anchored to multiples of 20, the last bin includes its right endpoint,
and no event outside the actual horizon or physical sequence enters. Each bin
contains neighbor count, target fraction, coarse-button fraction and mean
focal-to-neighbor cosine distance for every family. Empty-bin fractions and
means are zero, accompanied by the explicit zero neighbor count; self distance
is exactly zero. The button flag is located at its source image onset, never
at provider RT. Add presentation/308, six within-block-position indicators and
seconds to the authenticated physical start/end. For every scalar context
channel compare train/held-out group means and population within-group variance.
Also apply the .01 absolute mean/five-quantile rate bound to target/button
group means. These are prospective analyst definitions, not uniquely implied
by the prose policy. Physical span and FIR support choices MUST be supplied.

Initial groups are a random concept ordering with twelve distinct cyclic
shifts, mapped to each concept's image in each block. Each proposal draws a
block, first group, second-group starting position and one uniform variate;
cyclically choose the first legal distinct second group. Swap within the block.
Energy is the equal-descriptor mean of squared reference-standardized first
and second centered-moment discrepancies. For zero reference SD, the objective
uses the declared 1e-12 equality unit; acceptance still requires exact equality
within that bound. T(s)=.1*(.001)**(s/100000). Accept downhill/ties, otherwise
log(U)<-delta/T. Only two groups' statistics change; no EEG enters this process.

Check start 0 and each 1000 proposals. Recompute aggregate moments at each
checkpoint. A failing necessary mean bound skips sorting; otherwise check ALL
full empirical W1/five-quantile/mean bounds and diversity |ARI|<=.01. First
passing checkpoint wins. All 64 starts may be exhausted; a time/resource stop
preserves the same logical counters/RNG/assignment and is NOT underidentified.
Each continuation has a fresh segment directory and an explicit prior state.
Counters and proposal RNG do not reset. No automatic successor, retry or EEG
release occurs. A killed, incomplete write cannot serve as a checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np

from feature_extractors import DIMENSIONS, require_compute
from prepare_false_partition_inputs import load_release, read_json, require


RECIPE = "ep18_false_image_context_search_20261012"
SEED, GROUPS, BLOCKS, IMAGES = 20261012, 1854, 12, 22248
MEMBERS, STARTS, SWAPS, CHECKPOINT = 8, 64, 100000, 1000
QUANTILES = np.array([.05, .25, .50, .75, .95])
FOLDS = tuple((index, index + 6) for index in range(6))
FAMILIES = tuple(DIMENSIONS)
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


class BudgetStop(RuntimeError):
    pass


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def atomic_json(path, value, cap=8*1024**2):
    payload = encoded(value)
    require(len(payload) <= cap)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path, arrays, cap):
    require(sum(np.asarray(value).nbytes + 2048 for value in arrays.values()) <= cap)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    require(temporary.stat().st_size <= cap)
    os.replace(temporary, path)


@contextmanager
def budget(deadline, rss_bytes):
    """Stop only between atomic logical operations; outer Slurm bounds native work."""
    require(math.isfinite(deadline) and deadline > time.time() and rss_bytes > 0)
    stopped = [False]
    def stop(*_):
        stopped[0] = True
    def check():
        if stopped[0] or time.time() >= deadline:
            raise BudgetStop("time_or_signal")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > rss_bytes:
            # This deliberate soft budget is resumable. A genuine allocation
            # MemoryError remains a diagnosed failure, never an auto-retry.
            raise BudgetStop("rss_cap")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.))
    try:
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
        yield check
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def load_config(filename):
    config = read_json(filename)
    required = {"recipe_id", "inputs_dir", "participant_id", "cache_dir", "recorded_span_recipe",
                "filter_left_support_samples", "filter_right_support_samples", "pair_block_rows",
                "cache_byte_cap", "segment_byte_cap", "not_after_unix"}
    require(set(config) == required and config["recipe_id"] == RECIPE)
    for key in ("inputs_dir", "cache_dir"):
        require(Path(config[key]).is_absolute() and ".." not in Path(config[key]).parts)
    for key in ("filter_left_support_samples", "filter_right_support_samples"):
        require(type(config[key]) is int and config[key] >= 0)
    for key in ("pair_block_rows", "cache_byte_cap", "segment_byte_cap"):
        require(type(config[key]) is int and config[key] > 0)
    require(config["segment_byte_cap"] >= 64*1024**2 and math.isfinite(config["not_after_unix"]))
    config["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return config


def load_inputs(config):
    root = Path(config["inputs_dir"])
    report, settings = read_json(root / "report.json"), read_json(root / "settings.json")
    person = config["participant_id"]
    require(report["status"] == "false_partition_inputs_complete" and person in report["participants"]
            and not (root / "failure.json").exists())
    index = read_json(root / "image_index.json")
    require(len(index["main_opaque_ids"]) == IMAGES and set(index["families"]) == set(FAMILIES))
    with np.load(root / (person + ".npz"), allow_pickle=False) as source:
        events = {name: np.array(source[name], copy=True) for name in source.files}
    fields = ("concept", "exemplar", "block", "physical_sequence", "within_block_sequence",
              "presentation", "sample_one_based", "sample_zero_based", "target_flag", "response_flag")
    require(set(events) == set(fields) | {"concept_by_block_image_row"})
    require(all(events[name].shape == (IMAGES,) and events[name].dtype == np.int64 for name in fields))
    require(events["concept_by_block_image_row"].shape == (GROUPS, BLOCKS)
            and np.array_equal(np.sort(events["concept_by_block_image_row"].ravel()), np.arange(IMAGES)))
    release, people, event_paths, _ = load_release(settings["release_manifest"])
    require(person in people)
    return index, events, release, event_paths[person]


def physical_spans(config, events, release, event_path, check):
    from brainvision_sequence_reader import (RecordedSpanRecipe, bind_development_paths,
        build_recorded_sequence_spans, read_recording_metadata)
    paths = bind_development_paths(development_manifest=release["development_manifest"],
        bids_root=event_path.parents[2], participant_id=config["participant_id"])
    metadata = read_recording_metadata(paths, header_byte_limit=8*1024**2,
        marker_byte_limit=128*1024**2, budget_check=check)
    order = np.lexsort((events["presentation"], events["physical_sequence"]))
    source_rows = tuple({"isteststim": "0", "sample": str(events["sample_one_based"][index]),
        "sequencenumber": str(events["physical_sequence"][index]),
        "blocksequencenumber": str(events["block"][index]),
        "withinsequencenumber": str(events["within_block_sequence"][index]),
        "presentationnumber": str(events["presentation"][index])} for index in order)
    spans = build_recorded_sequence_spans(metadata, source_rows,
                                         recipe=RecordedSpanRecipe(**config["recorded_span_recipe"]))
    return {span.physical_sequence: span for span in spans}


def packed_offset(first, second):
    low, high = np.minimum(first, second), np.maximum(first, second)
    return low * (2*IMAGES-low-1)//2 + high-low-1


def distances(packed, first, second):
    first, second = np.broadcast_arrays(first, second)
    same = first == second
    offset = packed_offset(first, second)
    return np.where(same, 0., packed[np.where(same, 0, offset)])


def normalized_features(index, events, family, fold):
    source = np.load(index["families"][family]["path"], mmap_mode="r", allow_pickle=False)
    values = np.asarray(source[np.asarray(index["main_to_feature_row"])], dtype=np.float64)
    require(values.shape == (IMAGES, DIMENSIONS[family]) and np.isfinite(values).all())
    train = ~np.isin(events["block"], FOLDS[fold])
    training = values[train]
    mean, scale = training.mean(axis=0), training.std(axis=0, ddof=0)
    constant = np.all(training == training[0], axis=0)
    mean[constant], scale[constant] = training[0, constant], 0.
    del training, source
    active = scale != 0.
    values[:, active] = (values[:, active] - mean[active]) / scale[active]
    values[:, ~active] = 0.
    norms = np.linalg.norm(values, axis=1)
    require(np.isfinite(norms).all() and np.all(norms > 0))
    values /= norms[:, None]
    return values, {"training_images": int(train.sum()), "zero_sd_columns": np.flatnonzero(~active).tolist()}


def prepare_pair_cache(config, index, events, progress, check):
    root = Path(config["cache_dir"])
    count = IMAGES*(IMAGES-1)//2
    for fold in range(6):
        for family in FAMILIES:
            key = f"fold{fold}_{family}"
            final, partial = root/(key + ".npy"), root/(key + ".partial.npy")
            if key in progress["completed_pairs"]:
                require(final.stat().st_size == progress["completed_pairs"][key]["bytes"])
                continue
            check()
            values, scaling = normalized_features(index, events, family, fold)
            start = progress["active_pair_row"] if progress["active_pair"] == key else 0
            require(progress["active_pair"] in (None, key))
            if progress["active_pair"] is None:
                require(not partial.exists() and not final.exists())
                cache = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float64, shape=(count,))
                progress.update(active_pair=key, active_pair_row=0)
                atomic_json(root/"progress.json", progress)
            else:
                cache = np.load(partial, mmap_mode="r+", allow_pickle=False)
                require(cache.shape == (count,) and cache.dtype == np.float64)
            for first in range(start, IMAGES, config["pair_block_rows"]):
                check()
                stop = min(IMAGES, first + config["pair_block_rows"])
                block = 1. - values[first:stop] @ values.T
                require(np.isfinite(block).all())
                for row in range(first, stop):
                    offset = row*(2*IMAGES-row-1)//2
                    cache[offset:offset+IMAGES-row-1] = block[row-first, row+1:]
                cache.flush()
                progress["active_pair_row"] = stop
                atomic_json(root/"progress.json", progress)
            del cache, values
            os.replace(partial, final)
            progress["completed_pairs"][key] = dict(scaling, bytes=final.stat().st_size)
            progress.update(active_pair=None, active_pair_row=0)
            atomic_json(root/"progress.json", progress)


def context_schema(config):
    horizon = 800 + config["filter_left_support_samples"] + config["filter_right_support_samples"]
    bound = ((horizon+19)//20)*20
    edges = np.arange(-bound, bound+1, 20, dtype=np.int64)
    names = ["presentation_fraction"] + [f"within_block_{i}" for i in range(6)] + ["seconds_from_start", "seconds_to_end"]
    rate = [False]*len(names)
    for left, right in zip(edges[:-1], edges[1:]):
        suffix = f"{left}_{right}"
        names += ["neighbor_count_"+suffix, "target_fraction_"+suffix, "button_fraction_"+suffix]
        names += [family+"_neighbor_distance_"+suffix for family in FAMILIES]
        rate += [False, True, True] + [False]*len(FAMILIES)
    return horizon, edges, names, np.asarray(rate, dtype=bool)


def prepare_context_cache(config, events, spans, progress, check):
    root = Path(config["cache_dir"])
    horizon, edges, names, rate = context_schema(config)
    sequences = {sequence: np.flatnonzero(events["physical_sequence"] == sequence) for sequence in range(72)}
    for fold in range(6):
        key, final = str(fold), root/f"context_fold{fold}.npy"
        if key in progress["completed_contexts"]:
            require(final.stat().st_size == progress["completed_contexts"][key])
            continue
        check()
        require(not final.exists())  # An unrecorded final needs diagnosed reconciliation, never overwrite it.
        partial = root/f"context_fold{fold}.partial.npy"
        # A previously interrupted context file has no scientific completion;
        # resume overwrites only this run's explicitly tracked partial file.
        context = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float64, shape=(IMAGES, len(names)))
        context[:] = 0.
        pairs = [np.load(root/f"fold{fold}_{family}.npy", mmap_mode="r", allow_pickle=False) for family in FAMILIES]
        for sequence, images in sequences.items():
            check()
            samples = events["sample_zero_based"][images]
            span = spans[sequence]
            for image in images:
                sample = events["sample_zero_based"][image]
                context[image, 0] = events["presentation"][image]/308.
                context[image, 1+events["within_block_sequence"][image]] = 1.
                context[image, 7:9] = [(sample-span.start_sample)/1000., (span.stop_sample-1-sample)/1000.]
                selected = images[np.abs(samples-sample) <= horizon]
                delta = events["sample_zero_based"][selected]-sample
                bins = np.minimum((delta-edges[0])//20, len(edges)-2)
                for bin_index in np.unique(bins):
                    neighbors = selected[bins == bin_index]
                    first = 9 + int(bin_index)*9
                    context[image, first:first+3] = [len(neighbors), events["target_flag"][neighbors].mean(),
                                                  events["response_flag"][neighbors].mean()]
                    for family_index, packed in enumerate(pairs):
                        context[image, first+3+family_index] = distances(packed, image, neighbors).mean()
        context.flush(); del context, pairs
        os.replace(partial, final)
        progress["completed_contexts"][key] = final.stat().st_size
        atomic_json(root/"progress.json", progress)
    return names, rate


def prepare_cache(config, check):
    root = Path(config["cache_dir"])
    config_id = digest(config)
    index, events, release, event_path = load_inputs(config)
    horizon, edges, names, rate = context_schema(config)
    pair_bytes = 36*(IMAGES*(IMAGES-1)//2*8 + 256)
    context_bytes = 6*(IMAGES*len(names)*8 + 256)
    require(pair_bytes+context_bytes+16*1024**2 <= config["cache_byte_cap"])
    if not root.exists():
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        atomic_json(root/"settings.json", {"config": config, "config_id": config_id, "EEG_read": False})
        progress = {"config_id": config_id, "completed_pairs": {}, "completed_contexts": {},
                    "active_pair": None, "active_pair_row": 0}
        atomic_json(root/"progress.json", progress)
    else:
        require(read_json(root/"settings.json")["config_id"] == config_id and not (root/"manifest.json").exists())
        progress = read_json(root/"progress.json")
        require(progress["config_id"] == config_id)
    spans = physical_spans(config, events, release, event_path, check)
    prepare_pair_cache(config, index, events, progress, check)
    names, rate = prepare_context_cache(config, events, spans, progress, check)
    check()
    atomic_npz(root/"event_index.npz", events, 8*1024**2)
    atomic_json(root/"manifest.json", {"config_id": config_id, "recipe_id": RECIPE,
        "participant_id": config["participant_id"], "status": "complete_image_context_cache",
        "pairs": progress["completed_pairs"], "contexts": progress["completed_contexts"],
        "context_names": names, "context_rate_channels": rate.tolist(), "horizon_samples": horizon,
        "source_sample_rate_hz": 1000, "groups": GROUPS, "images": IMAGES,
        "main_opaque_ids": index["main_opaque_ids"], "EEG_read": False,
        "algorithm_seed": SEED, "source_code_sha256": config["code_sha256"]})


def search_array_bytes(context_channels):
    """Array admission before loading; excludes native workspace/interpreter RSS.

    All 36 packed pair arrays are resident to avoid random OAK faults per swap.
    Reference/current/sorted/standardized descriptor arrays and full-group
    context gathers are included with conservative simultaneous-copy factors.
    This is approximately a 70-GiB-or-larger search, not a 7-GiB worker.
    """
    pairs = 36 * IMAGES*(IMAGES-1)//2 * 8
    contexts = 6 * IMAGES * context_channels * 8
    descriptors = 144 + 24*context_channels
    metric_values = (144 + 24*context_channels) * GROUPS * 8
    return pairs + 2*contexts + 10*metric_values + 512*1024**2, descriptors


def load_search_cache(config, array_byte_cap, check):
    root = Path(config["cache_dir"])
    manifest = read_json(root/"manifest.json")
    require(manifest["status"] == "complete_image_context_cache" and manifest["config_id"] == digest(config)
            and manifest["source_code_sha256"] == config["code_sha256"] and manifest["EEG_read"] is False)
    channels = len(manifest["context_names"])
    estimated, descriptors = search_array_bytes(channels)
    require(estimated <= array_byte_cap)
    # Four checkpoint-sized allowances cover old/new checkpoint, final bank,
    # settings and operational records; no directory scan or silent deletion.
    estimated_checkpoint = 8*descriptors*5*8 + 8*IMAGES*4 + 8*descriptors*2 + 4*1024**2
    require(estimated_checkpoint <= config["segment_byte_cap"]//4)
    count = IMAGES*(IMAGES-1)//2
    pairs = np.empty((36, count), dtype=np.float64)
    for fold in range(6):
        for family_index, family in enumerate(FAMILIES):
            check()
            key = f"fold{fold}_{family}"
            path = root/(key + ".npy")
            require(path.stat().st_size == manifest["pairs"][key]["bytes"])
            source = np.load(path, mmap_mode="r", allow_pickle=False)
            require(source.shape == (count,) and source.dtype == np.float64)
            for start in range(0, count, 1024**2):
                check()
                values = source[start:start+1024**2]
                require(np.isfinite(values).all())
                pairs[fold*6+family_index, start:start+len(values)] = values
            del source
    contexts = []
    for fold in range(6):
        check()
        path = root/f"context_fold{fold}.npy"
        require(path.stat().st_size == manifest["contexts"][str(fold)])
        values = np.load(path, allow_pickle=False)
        require(values.shape == (IMAGES, channels) and values.dtype == np.float64 and np.isfinite(values).all())
        contexts.append(values)
    with np.load(root/"event_index.npz", allow_pickle=False) as source:
        events = {name: np.array(source[name], copy=True) for name in source.files}
    rate = np.asarray(manifest["context_rate_channels"], dtype=bool)
    require(rate.shape == (channels,))
    return manifest, pairs, contexts, events, rate, estimated


def group_metrics(assignment, pairs, contexts):
    """Exact metrics for any rows of a legal block-ordered assignment.

    One proposal invokes this only for its two affected rows. Canonical block
    reduction order avoids accumulating errors in the group metrics themselves.
    Distances for all 36 fold/family combinations are gathered in one operation.
    """
    rows, channels = len(assignment), contexts[0].shape[1]
    offsets = packed_offset(assignment[:, :, None], assignment[:, None, :])
    diagonal = np.eye(BLOCKS, dtype=bool)[None, :, :]
    matrix = pairs[:, np.where(diagonal, 0, offsets)]
    matrix[:, :, np.arange(BLOCKS), np.arange(BLOCKS)] = 0.
    nearest = np.empty((36, rows, 2), dtype=np.float64)
    scatter = np.empty((72, rows, 1), dtype=np.float64)
    context_metrics = np.empty((24*channels, rows, 1), dtype=np.float64)
    triangle = np.triu_indices(10, k=1)
    for fold, held_tuple in enumerate(FOLDS):
        held = np.asarray(held_tuple)
        train = np.asarray([block for block in range(BLOCKS) if block not in held_tuple])
        family = matrix[fold*6:(fold+1)*6]
        held_train = family[:, :, held[:, None], train[None, :]]
        nearest[fold*6:(fold+1)*6] = held_train.min(axis=-1)
        training = family[:, :, train[:, None], train[None, :]]
        scatter[fold*12:fold*12+6, :, 0] = training[:, :, triangle[0], triangle[1]].mean(axis=-1)
        scatter[fold*12+6:(fold+1)*12, :, 0] = family[:, :, held[0], held[1]]
        values = contexts[fold][assignment]
        training, heldout = values[:, train], values[:, held]
        first = fold*4*channels
        for index, array in enumerate((training.mean(axis=1), heldout.mean(axis=1),
                                      training.var(axis=1, ddof=0), heldout.var(axis=1, ddof=0))):
            context_metrics[first+index*channels:first+(index+1)*channels, :, 0] = array.T
    require(all(np.isfinite(value).all() for value in (nearest, scatter, context_metrics)))
    return nearest, scatter, context_metrics


class MetricBlock:
    """Fixed reference, exact current group metrics and incremental moments."""
    def __init__(self, reference, rate):
        self.shape = reference.shape
        flat = reference.reshape(reference.shape[0], -1)
        self.reference_sorted = np.sort(flat, axis=1)
        self.reference_mean = flat.mean(axis=1)
        self.reference_sd = flat.std(axis=1, ddof=1)
        constant = np.all(flat == flat[:, :1], axis=1)
        self.reference_mean[constant], self.reference_sd[constant] = flat[constant, 0], 0.
        self.units = np.where(self.reference_sd == 0., 1e-12, self.reference_sd)
        self.reference_quantiles = np.quantile(flat, QUANTILES, axis=1, method="linear").T
        centered = (flat-self.reference_mean[:, None])/self.units[:, None]
        self.reference_second = np.mean(centered*centered, axis=1)
        self.rate = np.asarray(rate, dtype=bool)
        require(self.rate.shape == (len(reference),))
        self.values = self.total = self.squares = None

    def bind(self, values):
        require(values.shape == self.shape and np.isfinite(values).all())
        self.values = values
        self.reanchor()

    def reanchor(self):
        centered = (self.values-self.reference_mean[:, None, None])/self.units[:, None, None]
        self.total = centered.sum(axis=(1, 2))
        self.squares = (centered*centered).sum(axis=(1, 2))

    def change(self, rows, values):
        before = (self.values[:, rows]-self.reference_mean[:, None, None])/self.units[:, None, None]
        after = (values-self.reference_mean[:, None, None])/self.units[:, None, None]
        return (self.total + (after-before).sum(axis=(1, 2)),
                self.squares + (after*after-before*before).sum(axis=(1, 2)))

    def energy(self, moments=None):
        total, squares = (self.total, self.squares) if moments is None else moments
        count = self.shape[1]*self.shape[2]
        mean, second = total/count, squares/count
        return float(np.sum(mean*mean + (second-self.reference_second)**2))

    def exact_acceptance(self):
        """All descriptors checked; moment objective never certifies acceptance.

        Columns: standardized mean, W1, largest quantile discrepancy, largest
        absolute target/response mean-or-quantile discrepancy, and maximum
        pointwise error in zero-SD descriptors (in 1e-12 units). For zero-SD
        rows the first three units are likewise 1e-12, not a fabricated SD.
        """
        candidate = np.sort(self.values.reshape(len(self.values), -1), axis=1)
        quantiles = np.quantile(candidate, QUANTILES, axis=1, method="linear").T
        # Identical raw-value reduction to the necessary-mean precheck avoids
        # a sort-dependent floating-point boundary disagreement.
        absolute_mean = np.abs(self.values.mean(axis=(1, 2))-self.reference_mean)
        absolute_quantile = np.abs(quantiles-self.reference_quantiles).max(axis=1)
        difference = np.abs(candidate-self.reference_sorted)
        diagnostics = np.column_stack((absolute_mean/self.units,
            difference.mean(axis=1)/self.units, absolute_quantile/self.units,
            np.where(self.rate, np.maximum(absolute_mean, absolute_quantile), 0.),
            np.where(self.reference_sd == 0., difference.max(axis=1)/1e-12, 0.)))
        regular = self.reference_sd != 0.
        passed = (np.all(diagnostics[regular, :3] <= .10)
                  and np.all(diagnostics[~regular, 4] <= 1.)
                  and np.all(diagnostics[:, 3] <= .01))
        return bool(passed), diagnostics

    def fails_necessary_mean(self):
        # Use raw means, not drifted moments. This is only a necessary test;
        # a passing mean always proceeds to exact full-distribution checks.
        difference = np.abs(self.values.mean(axis=(1, 2))-self.reference_mean)
        return bool(np.any(difference > np.where(self.reference_sd == 0., 1e-12, .1*self.reference_sd)))


def prepare_metric_blocks(true_groups, pairs, contexts, rate, check):
    check()
    reference = group_metrics(true_groups, pairs, contexts)
    context_rate = np.tile(np.concatenate((rate, rate, np.zeros(2*len(rate), dtype=bool))), 6)
    blocks = tuple(MetricBlock(value, flags) for value, flags in zip(reference,
        (np.zeros(36, dtype=bool), np.zeros(72, dtype=bool), context_rate)))
    return blocks


def valid_assignment(assignment, events):
    require(assignment.shape == (GROUPS, BLOCKS) and assignment.dtype.kind in "iu")
    require(np.array_equal(np.sort(assignment.ravel()), np.arange(IMAGES)))
    require(np.array_equal(events["block"][assignment], np.broadcast_to(np.arange(BLOCKS), assignment.shape)))
    concepts = np.sort(events["concept"][assignment], axis=1)
    require(np.all(np.diff(concepts, axis=1) > 0))


def initial_assignment(events, member, start):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, member, start])))
    order, shifts = rng.permutation(GROUPS), rng.choice(GROUPS, BLOCKS, replace=False)
    concepts = order[(np.arange(GROUPS)[:, None]+shifts[None, :]) % GROUPS]
    assignment = np.asarray(events["concept_by_block_image_row"][concepts, np.arange(BLOCKS)], dtype=np.int32)
    valid_assignment(assignment, events)
    return assignment, rng


def legal_proposal(assignment, concepts, rng):
    block, first, origin = int(rng.integers(BLOCKS)), int(rng.integers(GROUPS)), int(rng.integers(GROUPS))
    uniform = float(rng.random())  # Draw even when downhill; resume never changes RNG consumption.
    other = np.arange(BLOCKS) != block
    first_concept = concepts[assignment[first, block]]
    first_other = concepts[assignment[first, other]]
    for distance in range(GROUPS):
        second = (origin+distance) % GROUPS
        if (second != first and concepts[assignment[second, block]] not in first_other
                and first_concept not in concepts[assignment[second, other]]):
            rows = np.array([first, second])
            changed = assignment[rows].copy()
            changed[:, block] = changed[::-1, block]
            return rows, changed, uniform
    raise ValueError("No legal group partner; this is not an exhausted search budget")


def adjusted_rand(first, second):
    # Equal groups of size twelve: exact integer contingency pair count, then
    # the standard adjusted Rand normalization. Group labels are immaterial.
    a, b = np.empty(IMAGES, dtype=np.int64), np.empty(IMAGES, dtype=np.int64)
    a[first.ravel()] = np.repeat(np.arange(GROUPS), BLOCKS)
    b[second.ravel()] = np.repeat(np.arange(GROUPS), BLOCKS)
    counts = np.unique(a*GROUPS+b, return_counts=True)[1]
    observed = int(np.sum(counts*(counts-1)//2))
    marginal = GROUPS*BLOCKS*(BLOCKS-1)//2
    expected = marginal*marginal/(IMAGES*(IMAGES-1)/2)
    return float((observed-expected)/(marginal-expected))


def advance_assignment(assignment, concepts, pairs, contexts, blocks, rng, swaps):
    """One complete legal proposal; return whether accepted, never a partial commit."""
    rows, proposed, uniform = legal_proposal(assignment, concepts, rng)
    values = group_metrics(proposed, pairs, contexts)
    moments = [block.change(rows, value) for block, value in zip(blocks, values)]
    dimensions = sum(len(block.reference_mean) for block in blocks)
    old = sum(block.energy() for block in blocks)/dimensions
    new = sum(block.energy(moment) for block, moment in zip(blocks, moments))/dimensions
    require(math.isfinite(old) and math.isfinite(new))
    temperature = .1 * (.001 ** (swaps/SWAPS))
    accepted = new <= old or uniform == 0. or math.log(uniform) < -(new-old)/temperature
    if accepted:
        assignment[rows] = proposed
        for block, value, (total, squares) in zip(blocks, values, moments):
            block.values[:, rows] = value
            block.total, block.squares = total, squares
    return bool(accepted)


def random_capacity_bank():
    return np.stack([np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, 8+k, 0])))
                     .permutation(IMAGES).reshape(GROUPS, BLOCKS) for k in range(MEMBERS)]).astype(np.int32)


def checkpoint_payload(state, assignment, accepted, accepted_diagnostics, blocks, rng):
    metadata = dict(state, rng_state=None if rng is None else rng.bit_generator.state)
    arrays = {"metadata_utf8": np.frombuffer(encoded(metadata), dtype=np.uint8),
              "assignment": assignment,
              "accepted": np.asarray(accepted, dtype=np.int32).reshape(-1, GROUPS, BLOCKS),
              "accepted_diagnostics": np.asarray(accepted_diagnostics, dtype=np.float64)}
    if state["phase"] == "search":
        for index, block in enumerate(blocks):
            arrays[f"total_{index}"], arrays[f"squares_{index}"] = block.total, block.squares
    return arrays


def restore_checkpoint(path, config, events):
    require(path.is_absolute() and path.name == "checkpoint.npz"
            and path.stat().st_size <= config["segment_byte_cap"]//4
            and not path.with_name("checkpoint.npz.pending").exists())
    # A prior terminal bank or diagnosed program error is never auto-retried.
    report = read_json(path.parent/"report.json")
    require(report["status"] == "resource_incomplete" and report["checkpoint_available"] is True)
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    state = json.loads(arrays.pop("metadata_utf8").tobytes().decode())
    require(state["config_id"] == digest(config) and state["recipe_id"] == RECIPE
            and state["phase"] in ("need_start", "search")
            and 0 <= state["start"] < STARTS and 0 <= state["swaps"] <= SWAPS
            and 0 <= state["total_accepted_proposals"] <= state["total_legal_proposals"] <= MEMBERS*STARTS*SWAPS)
    assignment, accepted = arrays.pop("assignment"), list(arrays.pop("accepted"))
    diagnostics = list(arrays.pop("accepted_diagnostics"))
    require(len(accepted) == len(diagnostics) == state["member"] < MEMBERS)
    require(len(state["acceptance_positions"]) == len(accepted))
    channels = len(read_json(Path(config["cache_dir"])/"manifest.json")["context_names"])
    require(all(value.shape == (108+24*channels, 5) and np.isfinite(value).all() for value in diagnostics))
    for value in accepted:
        valid_assignment(value, events)
    if state["phase"] == "search":
        valid_assignment(assignment, events)
        require(-1 <= state["last_checked"] <= state["swaps"]
                and (state["last_checked"] == -1 or state["last_checked"] % CHECKPOINT == 0))
    return state, assignment, accepted, diagnostics, arrays


def run_search(config, output, resume, array_byte_cap, check, report):
    manifest, pairs, contexts, events, rate, estimated = load_search_cache(config, array_byte_cap, check)
    report["array_admission_bytes"] = estimated
    blocks = prepare_metric_blocks(events["concept_by_block_image_row"], pairs, contexts, rate, check)
    assignment = np.empty((0, BLOCKS), dtype=np.int32)
    accepted, accepted_diagnostics, rng = [], [], None
    state = {"config_id": digest(config), "recipe_id": RECIPE, "member": 0, "start": 0, "swaps": 0,
             "phase": "need_start", "last_checked": -1, "total_legal_proposals": 0,
             "total_accepted_proposals": 0, "exact_checks": 0, "acceptance_positions": []}
    if resume is not None:
        state, assignment, accepted, accepted_diagnostics, saved = restore_checkpoint(resume, config, events)
        if state["phase"] == "search":
            values = group_metrics(assignment, pairs, contexts)
            for index, (block, current) in enumerate(zip(blocks, values)):
                block.bind(current)
                for name in ("total", "squares"):
                    prior = saved.pop(f"{name}_{index}")
                    require(prior.shape == (len(block.reference_mean),) and np.isfinite(prior).all())
                    setattr(block, name, prior)
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = state.pop("rng_state")
        else:
            require(state.pop("rng_state") is None)
        require(not saved)

    def save():
        arrays = checkpoint_payload(state, assignment, accepted, accepted_diagnostics, blocks, rng)
        atomic_npz(output/"checkpoint.npz", arrays, config["segment_byte_cap"]//4)
        report.update(checkpoint_available=True, member=state["member"], start=state["start"],
            swaps=state["swaps"], total_legal_proposals=state["total_legal_proposals"],
            total_accepted_proposals=state["total_accepted_proposals"], exact_checks=state["exact_checks"])

    try:
        save()
        while state["phase"] in ("need_start", "search"):
            check()
            if state["phase"] == "need_start":
                assignment, rng = initial_assignment(events, state["member"], state["start"])
                for block, values in zip(blocks, group_metrics(assignment, pairs, contexts)):
                    block.bind(values)
                state.update(phase="search", swaps=0, last_checked=-1)
            if state["swaps"] % CHECKPOINT == 0 and state["last_checked"] != state["swaps"]:
                for block in blocks:
                    block.reanchor()
                state["last_checked"] = state["swaps"]
                if not any(block.fails_necessary_mean() for block in blocks):
                    state["exact_checks"] += 1
                    findings = [block.exact_acceptance() for block in blocks]
                    diversity = [adjusted_rand(assignment, previous) for previous in accepted]
                    if all(value[0] for value in findings) and all(abs(value) <= .01 for value in diversity):
                        valid_assignment(assignment, events)
                        accepted.append(assignment.copy())
                        accepted_diagnostics.append(np.concatenate([value[1] for value in findings], axis=0))
                        state["acceptance_positions"].append({"member": state["member"], "start": state["start"],
                            "swaps": state["swaps"], "pairwise_adjusted_rand": diversity})
                        state.update(member=state["member"]+1, start=0, swaps=0, last_checked=-1,
                                     phase="candidate_complete" if len(accepted) == MEMBERS else "need_start")
                        assignment, rng = np.empty((0, BLOCKS), dtype=np.int32), None
                        save()
                        continue
                save()
            if state["swaps"] == SWAPS:
                state.update(start=state["start"]+1, swaps=0, last_checked=-1,
                             phase="exhausted" if state["start"]+1 == STARTS else "need_start")
                assignment, rng = np.empty((0, BLOCKS), dtype=np.int32), None
                save()
                continue
            if advance_assignment(assignment, events["concept"], pairs, contexts, blocks, rng, state["swaps"]):
                state["total_accepted_proposals"] += 1
            state["swaps"] += 1
            state["total_legal_proposals"] += 1
        save()
    except BudgetStop as error:
        save()
        report.update(status="resource_incomplete", budget_stop_reason=str(error))
        return
    if state["phase"] == "exhausted":
        report.update(status="exhausted_unavailable", disposition="cannot_construct_all_eight_under_frozen_budget")
        return
    require(state["phase"] == "candidate_complete" and len(accepted) == MEMBERS)
    bank = np.stack(accepted)
    random_bank = random_capacity_bank()
    fingerprints = [hashlib.sha256(np.asarray(value, dtype="<i4").tobytes()).hexdigest() for value in bank]
    atomic_npz(output/"candidate_bank.npz", {"primary": bank, "random_capacity_diagnostic": random_bank,
        "acceptance_diagnostics": np.stack(accepted_diagnostics)}, config["segment_byte_cap"]//4)
    atomic_json(output/"candidate_bank.json", {"config_id": digest(config), "recipe_id": RECIPE,
        "participant_id": config["participant_id"], "main_opaque_ids": manifest["main_opaque_ids"],
        "primary_bank_fingerprints": fingerprints, "acceptance_positions": state["acceptance_positions"],
        "descriptor_order": ["nearest:fold,family", "scatter:fold,train_or_held,family",
                             "context:fold,trainmean_heldmean_trainvar_heldvar,channel"],
        "context_channels": manifest["context_names"], "diagnostic_columns":
            ["mean_scaled", "W1_scaled", "maximum_quantile_scaled", "maximum_absolute_rate", "zero_SD_maximum_error_units"],
        "full_false_bank_qualified": False, "selected_design_recheck_required": True,
        "regeneration_after_selected_design_recheck": False, "EEG_read": False}, cap=4*1024**2)
    report.update(status="complete_image_context_candidate_bank", primary_members=MEMBERS,
                  capacity_diagnostic_members=MEMBERS, full_false_bank_qualified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "search"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New operational segment directory")
    parser.add_argument("--resume", type=Path, help="Explicit prior resource-incomplete checkpoint for search")
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--rss-byte-cap", type=int, required=True)
    parser.add_argument("--array-byte-cap", type=int, required=True)
    args = parser.parse_args(argv)
    owned, report, started = False, {}, time.time()
    try:
        require_compute()
        require(all(os.environ.get(name) == "1" for name in THREAD_ENV))
        os.umask(0o077)
        config = load_config(args.config)
        require(args.output.is_absolute() and ".." not in args.output.parts
                and 0 < args.array_byte_cap < args.rss_byte_cap
                and args.deadline <= config["not_after_unix"]
                and (args.phase == "search" or args.resume is None))
        args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
        owned = True
        atomic_json(args.output/"settings.json", {"config": config, "phase": args.phase,
            "resume": None if args.resume is None else str(args.resume), "deadline": args.deadline,
            "rss_byte_cap": args.rss_byte_cap, "array_byte_cap": args.array_byte_cap,
            "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat()})
        report.update(status="started", phase=args.phase, EEG_read=False, full_false_bank_qualified=False,
                      checkpoint_available=False, participant_id=config["participant_id"])
        with budget(args.deadline, args.rss_byte_cap) as check:
            if args.phase == "prepare":
                # Largest feature-normalization/reduction buffers plus one
                # pair block; cache memmaps additionally count toward actual RSS.
                estimate = 8*(6*IMAGES*max(DIMENSIONS.values()) + config["pair_block_rows"]*IMAGES) + 512*1024**2
                require(estimate <= args.array_byte_cap)
                prepare_cache(config, check)
                report["status"] = "complete_image_context_cache"
            else:
                run_search(config, args.output, args.resume, args.array_byte_cap, check, report)
        return 0 if report["status"].startswith("complete_") else 75 if report["status"] == "resource_incomplete" else 2
    except BudgetStop as error:
        report.update(status="resource_incomplete", budget_stop_reason=str(error))
        return 75
    except BaseException as error:
        # No original paths, event contents, credentials or model captions in
        # exceptions/logs. Diagnosed errors never authorize an automatic retry.
        report.update(status="stopped_error", error_type=type(error).__name__)
        return 1
    finally:
        if owned:
            report.update(elapsed_seconds=time.time()-started,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                finished_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(args.output/"report.json", report, cap=1024**2)


if __name__ == "__main__":
    raise SystemExit(main())
