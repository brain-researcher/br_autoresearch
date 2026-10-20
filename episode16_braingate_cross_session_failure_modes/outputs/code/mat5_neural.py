"""Role-sealed selective reader for EP16 neural windows and cursor rows.

Extends the frozen field-selective Level-5 reader (mat5_structural.py, unchanged).
The participant role is checked twice before any neural or cursor value is decoded:
from the archive identity supplied by the caller, and from the in-file participant_id
read in the structural pass. Only the selected 30-bin window rows of the requested
neural fields and cursor rows at starts of whole, inside-block trials are kept.
"""
from __future__ import annotations

import struct

import numpy as np

from ep16_roles import RoleSealError, authorize_protected_read
from mat5_structural import (DTYPES, MISSING, SelectedRows, StructuralReader, element, exact,
                             field_vector, raw_subelement, read_structural)

WINDOW_BINS = 30


class SelectedWindowRows:
    """Rows of an n_time_bins x n_channels matrix to keep (sorted, unique)."""

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.int64)


class NeuralReader(StructuralReader):
    """StructuralReader plus row selection for n x p numeric matrices."""

    def matrix(self, stream, selection, path, top=False):
        _, flags_raw = raw_subelement(stream, self.endian)
        flags = struct.unpack(self.endian + "I", flags_raw[:4])[0]
        array_class = flags & 255
        _, shape_raw = raw_subelement(stream, self.endian)
        shape = np.frombuffer(shape_raw, dtype=self.endian + "i4").astype(int)
        _, name_raw = raw_subelement(stream, self.endian)
        name = name_raw.decode("utf-8")
        if top:
            path = name
            if name not in selection:
                self.skipped_paths.add(path)
                return name, MISSING
            selection = selection[name]
        if flags & 0x800:
            raise ValueError(f"Complex selected field is unsupported: {path}")
        size = int(np.prod(shape))
        if array_class == 2:
            if not isinstance(selection, dict):
                raise ValueError(f"Struct requires an explicit field whitelist: {path}")
            _, length_raw = raw_subelement(stream, self.endian)
            width = struct.unpack(self.endian + "i", length_raw)[0]
            _, names_raw = raw_subelement(stream, self.endian)
            if width <= 0 or len(names_raw) % width:
                raise ValueError("Invalid struct field names")
            fields = [names_raw[i:i + width].split(b"\0", 1)[0].decode("utf-8")
                      for i in range(0, len(names_raw), width)]
            objects = []
            for _ in range(size):
                obj = {}
                for field in fields:
                    child = element(stream, self.endian)
                    child_path = path + "." + field
                    if field not in selection:
                        self.skipped_paths.add(child_path)
                    else:
                        if child.kind != 14:
                            raise ValueError("Selected struct value is not a matrix")
                        _, obj[field] = self.matrix(child.data, selection[field], child_path)
                    child.finish()
                objects.append(obj)
            return name, objects[0] if size == 1 else objects
        if array_class == 1 or array_class == 4 or not isinstance(selection, (SelectedRows, SelectedWindowRows)):
            raise ValueError(f"Neural reader selects numeric rows only: {path}")
        item = element(stream, self.endian)
        if item.kind not in DTYPES or array_class not in range(6, 16):
            raise ValueError(f"Unsupported selected primitive encoding at {path}")
        dtype = np.dtype(self.endian + DTYPES[item.kind])
        if len(shape) != 2:
            raise ValueError(f"Selected matrix must be two-dimensional: {path}")
        n_rows, n_cols = int(shape[0]), int(shape[1])
        rows = np.asarray(selection.rows, dtype=np.int64)
        if isinstance(selection, SelectedRows) and n_cols != 2:
            raise ValueError("Row selection with SelectedRows is for n x 2 position matrices")
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= n_rows):
            raise ValueError(f"Selected rows outside matrix support at {path}")
        if item.data.remaining != n_rows * n_cols * dtype.itemsize:
            raise ValueError(f"Numeric dimensions differ from element length at {path}")
        unique, inverse = np.unique(rows, return_inverse=True)
        selected = np.empty((len(unique), n_cols), dtype=np.float64)
        for column in range(n_cols):  # column-major: read one column, keep selected rows
            values = np.frombuffer(exact(item.data, n_rows * dtype.itemsize), dtype=dtype)
            selected[:, column] = values[unique]
        item.finish()
        self.decoded_paths.add(path)
        return name, selected[inverse]


def window_rows(trials, blocks, bins=WINDOW_BINS):
    """Trials whose 30-bin window [start, start+30) lies inside their declared block."""
    starts = field_vector(trials, "trial_start_index").astype(np.int64)
    ends = field_vector(trials, "trial_end_index").astype(np.int64)
    trial_blocks = field_vector(trials, "block_id").astype(np.int64)
    bounds = {int(b): (int(s), int(e)) for b, s, e in zip(field_vector(blocks, "block_id"),
              field_vector(blocks, "sample_start_index"), field_vector(blocks, "sample_end_index"))}
    whole = np.array([bounds[int(b)][0] <= s < e <= bounds[int(b)][1]
                      for b, s, e in zip(trial_blocks, starts, ends)], dtype=bool)
    inside = np.array([bounds[int(b)][0] <= s and s + bins <= bounds[int(b)][1]
                       for b, s in zip(trial_blocks, starts)], dtype=bool)
    return starts, ends, trial_blocks, whole, inside


def read_session_windows(source, participant, features=("sbp", "tx_4_5"), token=None, half_windows=("sbp",)):
    """Decode structural fields, then (after two role checks) cursor at trial start and window means.

    Returns window means (trial x electrode, NaN unless all 30 bins finite and window inside block),
    odd/even half-window means for the listed features (leakage control), cursor and target at start.
    Cursor rows are NaN unless the trial is whole and its window is inside the declared block.
    """
    authorize_protected_read(participant, token)
    values, trace = read_structural(source)
    if values["participant_id"] != participant:
        raise RoleSealError("In-file participant differs from the authorized archive identity")
    authorize_protected_read(values["participant_id"], token)
    starts, ends, trial_blocks, whole, inside = window_rows(values["trials"], values["blocks"])
    keep = np.flatnonzero(inside)
    cursor_keep = np.flatnonzero(whole & inside)
    rows = np.unique((starts[keep, None] + np.arange(WINDOW_BINS)[None, :]).reshape(-1))
    reader = NeuralReader(source)
    selection = {"cursor_position": SelectedRows(starts[cursor_keep]),
                 "neural": {f: SelectedWindowRows(rows) for f in features}}
    got = reader.read(selection)
    if set(got) != {"cursor_position", "neural"} or set(got["neural"]) != set(features):
        raise ValueError("Selected neural/cursor variables are missing")
    position = {int(r): k for k, r in enumerate(rows)}
    cursor = np.full((len(starts), 2), np.nan)
    cursor[cursor_keep] = np.asarray(got["cursor_position"], dtype=np.float64)
    out = {"cursor_start": cursor}
    for feature in features:
        matrix = got["neural"][feature]
        means = np.full((len(starts), matrix.shape[1]), np.nan)
        halves = np.full((2, len(starts), matrix.shape[1]), np.nan) if feature in half_windows else None
        for trial in keep:
            index = [position[int(starts[trial] + b)] for b in range(WINDOW_BINS)]
            window = matrix[index]
            complete = np.isfinite(window).all(axis=0)
            means[trial, complete] = window[:, complete].mean(axis=0)
            if halves is not None:
                halves[0, trial, complete] = window[0::2, complete].mean(axis=0)
                halves[1, trial, complete] = window[1::2, complete].mean(axis=0)
        out[feature] = means
        if halves is not None:
            out[feature + "_odd_even"] = halves
    access = {"structural": trace, "protected_decoded_paths": sorted(reader.decoded_paths),
              "skipped_paths": sorted(reader.skipped_paths), "window_rows_decoded": int(len(rows)),
              "cursor_rows_decoded": int(len(cursor_keep))}
    return values, out, {"starts": starts, "ends": ends, "trial_blocks": trial_blocks, "whole": whole,
                         "window_inside_block": inside}, access
