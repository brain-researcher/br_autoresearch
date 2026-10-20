#!/usr/bin/env python3.11
"""
EP09 cortical Laplace field and soma frames.

Run inside a verified Slurm compute allocation:
    python3.11 -I ep09_cortical_laplace.py \
        CCF_PAYLOAD_DIR TOOLS_DIR ZIP CELL_MANIFEST_CSV OUT_DIR

Only initial SWC comments/blank lines and the first data row are read.
The first data row must be node 1, type 1, parent -1.
"""
from __future__ import annotations

import os

THREAD_ENV = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_DEFAULT_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    )
}
THREAD_ENV.update({"OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE"})
os.environ.update(THREAD_ENV)

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import re
import signal
import socket
import stat
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True

# Refuse executable use before importing numerical libraries.
if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise SystemExit("ERROR: SLURM_JOB_ID is required.")
    if not sys.flags.isolated:
        raise SystemExit("ERROR: invoke with python3.11 -I.")

import numpy as np
import scipy
from scipy import ndimage, sparse
from scipy.sparse.linalg import LinearOperator, cg


SHAPE = (528, 320, 456)
SPACING_UM = 25
MIDLINE_ML_UM = 5700
STRIDES = (SHAPE[1] * SHAPE[2], SHAPE[2], 1)
ISO_ID = 315
FIBRE_ID = 1009
VENTRICLE_ID = 73
RTOL = 1e-10
ATOL = 0.0
MAXITER = 50_000
GRADIENT_MIN_NORM = 1e-12
TANGENT_FALLBACK_NORM = 1e-6
CHUNK = 262_144
HASH_CHUNK = 8 * 1024 * 1024
MAX_SWC_LINE_BYTES = 65_536
MAX_INITIAL_SWC_LINES = 10_000

OTHER = 0
FIBRE = 1
DOMAIN_UNLAMINATED = 2
# Reviewer correction v2 (2026-10-17): layer-1 nodes of non-Isocortex cortex
# (e.g. ENTl1) mark the lateral continuation of the cortical sheet.
NONISO_L1 = 99
CTX_ID = 688
MIDLINE_INDEX = MIDLINE_ML_UM // SPACING_UM  # ML voxel index whose lower face is the midline
TOKENS = ("1", "2", "2/3", "3", "4", "5", "6", "6a", "6b")
LAYER_CODE = {token: i + 3 for i, token in enumerate(TOKENS)}
POPCOUNT = np.array([i.bit_count() for i in range(64)], dtype=np.uint8)
FACE_ORDER = ("AP-", "AP+", "DV-", "DV+", "ML-", "ML+")
AXES = ("ap", "dv", "ml")
VECTOR_PREFIXES = ("r", "t1", "t2", "r_c", "t1_c", "t2_c")
CSV_COLUMNS = (
    ["cell", "brain", "ep09_role", "soma_voxel_label", "hemisphere", "depth"]
    + [f"{prefix}_{axis}" for prefix in VECTOR_PREFIXES for axis in AXES]
    + ["missing_reason"]
)
MANIFEST_COLUMNS = {
    "cell", "brain", "line", "area", "layer",
    "pool_key", "pool_size", "ep09_role",
}
FIELD_NAME = "cortical_laplace_field.npz"
REPORT_NAME = "cortical_laplace_report.json"
FRAMES_NAME = "soma_cortical_frames.csv"
OUTPUT_NAMES = (FIELD_NAME, REPORT_NAME, FRAMES_NAME)
DECIMAL_RE = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z"
)

FROZEN = {
    "contract": "EP09 measurement contract v1.2",
    "atlas": {
        "annotation": "annotation_25_ccf_2017.nrrd",
        "graph": "structure_graph_1.json",
        "graph_id": 1,
        "shape": list(SHAPE),
        "array_axes": ["AP", "DV", "ML"],
        "spacing_um": SPACING_UM,
        "voxel_extent_um": "[25*i, 25*i+25) on each axis",
        "voxel_center_um": "(i+0.5)*25 on each axis",
        "midline_ml_um": MIDLINE_ML_UM,
        "isocortex_id": ISO_ID,
        "fibre_tract_id": FIBRE_ID,
        "ventricular_system_id": VENTRICLE_ID,
    },
    "domain": "Descendant-or-self of Isocortex 315; six-connectivity.",
    "laminar_classification": {
        "rule": (
            "Inside Isocortex: name contains 'layer', case-insensitively, "
            "or acronym equals parent acronym plus an accepted token. "
            "Read the token strictly as the suffix after the parent acronym; "
            "reject a laminar node with an unknown suffix."
        ),
        "tokens_in_display_order": list(TOKENS),
        "aggregate_tokens": "2/3 and 6 overlap finer tokens; no strict median-order test.",
    },
    "boundary_faces": {
        "interior": "Domain neighbour: standard graph-Laplacian coupling.",
        "pial": (
            "v2 reviewer correction: source token 1 and a neighbour that is "
            "background, Isocortex across the midline, or any other non-fibre "
            "structure except another region's layer-1 node: Dirichlet phi=0. "
            "v1 used only background neighbours, leaving the medial and orbital "
            "pial surfaces no-flux."
        ),
        "white_matter": (
            "Source token 6b and neighbour descendant-or-self of 1009: "
            "Dirichlet phi=1."
        ),
        "other": "Omitted, including every off-grid face: no flux.",
        "ghost_discretization": (
            "Boundary value is at neighbour centre. Each Dirichlet face "
            "adds 1 to diagonal and its prescribed value to RHS."
        ),
        "face_bit_order": list(FACE_ORDER),
        "l6a_fibre_faces": "Diagnostic only; never a Dirichlet boundary.",
    },
    "components": (
        "v2: six-connected components labelled separately on each side of the "
        "midline (no coupling across ML 5700 um). Solve each component only if "
        "it has both pial and white-matter faces. Other components retain NaN."
    ),
    "solver": {
        "matrix": "scipy.sparse.csr_matrix",
        "arithmetic": "float64",
        "method": "scipy.sparse.linalg.cg",
        "preconditioner": "Jacobi: inverse matrix diagonal",
        "rtol": RTOL,
        "atol": ATOL,
        "maxiter": MAXITER,
        "x0": "zeros",
        "final_residual": "||b-Ax||_2 / ||b||_2, explicitly recomputed",
        "success": "CG info=0, finite solution/residual, final residual <= rtol.",
        "clipping": False,
    },
    "gradient": {
        "arithmetic": "Unrounded float64 solution; evaluated on demand at soma voxels.",
        "both_neighbours": "(phi_plus-phi_minus)/(2*25 um)",
        "plus_only": "(phi_plus-phi_center)/(25 um)",
        "minus_only": "(phi_center-phi_minus)/(25 um)",
        "neither": "NaN",
        "eligible_neighbours": "Finite, solved domain voxels only; no boundary ghosts.",
        "radial": "-gradient / norm(gradient)",
        "missing": "Any nonfinite component or norm < 1e-12 per um.",
    },
    "sampling": {
        "coordinate_parse": "fractions.Fraction of each decimal coordinate token",
        "voxel": "floor(coordinate/25), independently on AP, DV, ML",
        "interpolation": False,
        "nearest_voxel_search": False,
        "invalid_sample": "Depth and all frames missing if outside, unsolved or invalid gradient.",
        "out_of_grid": "outside_isocortex; soma_voxel_label is blank.",
        "depth_squared": "Derived downstream as CSV depth squared; preserves missingness.",
    },
    "frame": {
        "e_ap": [1, 0, 0],
        "e_ml": [0, 0, 1],
        "t1": "Normalize e_AP-(e_AP dot r)*r; use projected e_ML if norm < 1e-6.",
        "t2": "r cross t1",
        "hemisphere": "Exact soma ML <5700: left; >5700: right; equality: ambiguous.",
        "right_canonical": (
            "M=diag(1,1,-1); r_c=M*r, t1_c=M*t1, t2_c=r_c cross t1_c."
        ),
        "left_canonical": "Unchanged raw frame.",
        "ambiguous_canonical": (
            "Retain valid depth/raw frame; canonical frame blank; "
            "missing_reason=ambiguous_hemisphere."
        ),
    },
    "outputs": {
        "npz": {
            "flat_indices": "uint32, ascending C-order flat indices of solved voxels",
            "phi": "float32, corresponding values; unsolved/outside voxels omitted",
            "shape": "int32[3]",
            "omission": "Reconstruct omitted voxels as NaN.",
        },
        "csv_numeric_precision": "17 significant digits from float64 calculations",
        "csv_missing": "Empty fields",
        "quantiles": "numpy.quantile, method='linear', over solved float64 values",
        "creation": "Exclusive creation; chmod 0444 after successful writes.",
    },
    "input_access": {
        "somata": "All manifest roles permitted; no role affects the field.",
        "swc": (
            "Skip initial blank/comment lines; parse only the first data row, "
            "requiring node=1, type=1, parent=-1. Never scan later data rows."
        ),
        "zip": (
            "Central-directory metadata and first soma rows only. "
            "No whole-ZIP hash or full-member CRC verification."
        ),
    },
}


class SolverFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Soma:
    voxel: tuple[int, int, int]
    hemisphere: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    s = path.stat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f"Not a regular file: {path}")
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def fingerprint(path: Path) -> dict:
    before = stat_signature(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(HASH_CHUNK):
            h.update(block)
    if stat_signature(path) != before:
        raise RuntimeError(f"Input changed while hashing: {path}")
    return {
        "path": str(path),
        "sha256": h.hexdigest(),
        "size_bytes": before[2],
        "device": before[0],
        "inode": before[1],
        "mtime_ns": before[3],
        "ctime_ns": before[4],
    }


def array_sha256(array: np.ndarray) -> str:
    """Hash contiguous, little-endian element bytes, without a header."""
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    h = hashlib.sha256()
    if array.nbytes:
        view = memoryview(array).cast("B")
        for start in range(0, len(view), HASH_CHUNK):
            h.update(view[start:start + HASH_CHUNK])
    return h.hexdigest()


def handle_sha256(handle) -> str:
    handle.flush()
    position = handle.tell()
    handle.seek(0)
    h = hashlib.sha256()
    while block := handle.read(HASH_CHUNK):
        h.update(block)
    handle.seek(position)
    return h.hexdigest()


class OutputSet:
    """Own only files created by this invocation; roll them back on failure."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.handles = {}
        self.identities = {}
        self.created_directory = False
        self.committed = False

    def _owns(self, name: str) -> bool:
        try:
            s = (self.directory / name).lstat()
        except FileNotFoundError:
            return False
        return (s.st_dev, s.st_ino) == self.identities[name]

    def _rollback(self) -> list[str]:
        errors = []
        for name, handle in self.handles.items():
            try:
                handle.close()
            except Exception as exc:
                errors.append(f"close {name}: {exc}")
        for name in self.identities:
            try:
                if self._owns(name):
                    (self.directory / name).unlink()
            except Exception as exc:
                errors.append(f"remove {name}: {exc}")
        if self.created_directory:
            try:
                self.directory.rmdir()
            except OSError:
                pass  # Never remove unrelated contents.
        return errors

    def __enter__(self):
        try:
            try:
                self.directory.mkdir(parents=True, exist_ok=False)
                self.created_directory = True
            except FileExistsError:
                if not self.directory.is_dir():
                    raise
            for name in OUTPUT_NAMES:
                flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(self.directory / name, flags, 0o600)
                try:
                    s = os.fstat(fd)
                    self.identities[name] = (s.st_dev, s.st_ino)
                    self.handles[name] = os.fdopen(fd, "w+b")
                except BaseException:
                    os.close(fd)
                    raise
            return self
        except BaseException as exc:
            errors = self._rollback()
            if errors:
                raise RuntimeError("Rollback errors: " + "; ".join(errors)) from exc
            raise

    def commit(self) -> None:
        for name, handle in self.handles.items():
            if not self._owns(name):
                raise RuntimeError(f"Reserved output was removed/replaced: {name}")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        for handle in self.handles.values():
            handle.close()
        self.committed = True

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self.committed:
            errors = self._rollback()
            if errors:
                raise RuntimeError("Rollback errors: " + "; ".join(errors)) from exc
        return False


def graph_class_table(nodes: dict) -> tuple[np.ndarray, np.ndarray]:
    normalized = {int(key): value for key, value in nodes.items()}
    if len(normalized) != len(nodes):
        raise ValueError("Duplicate graph IDs after integer normalization.")
    for required in (ISO_ID, FIBRE_ID, VENTRICLE_ID):
        if required not in normalized:
            raise ValueError(f"Graph lacks required structure ID {required}.")

    classes = {0: OTHER}
    for node_id, node in normalized.items():
        if not 0 < node_id <= np.iinfo(np.uint32).max:
            raise ValueError(f"Invalid graph ID: {node_id}")
        if int(node["id"]) != node_id:
            raise ValueError(f"Graph key/id mismatch: {node_id}")
        path = tuple(int(x) for x in str(node["structure_id_path"]).split("/") if x)
        if not path or path[-1] != node_id:
            raise ValueError(f"Invalid structure_id_path for {node_id}.")

        is_domain = ISO_ID in path
        is_fibre = FIBRE_ID in path
        if is_domain and is_fibre:
            raise ValueError(f"Structure {node_id} is both Isocortex and fibre tract.")

        code = FIBRE if is_fibre else OTHER
        if not is_domain and not is_fibre and CTX_ID in path:
            parent_id = node.get("parent_structure_id")
            parent = normalized.get(int(parent_id)) if parent_id is not None else None
            parent_acronym = str(parent["acronym"]) if parent is not None else ""
            acronym = str(node["acronym"])
            if (
                "layer" in str(node["name"]).lower()
                and parent_acronym
                and acronym.startswith(parent_acronym)
                and acronym[len(parent_acronym):] == "1"
            ):
                code = NONISO_L1
        if is_domain:
            code = DOMAIN_UNLAMINATED
            acronym = str(node["acronym"])
            parent_id = node.get("parent_structure_id")
            parent = normalized.get(int(parent_id)) if parent_id is not None else None
            parent_acronym = str(parent["acronym"]) if parent is not None else ""
            suffix = (
                acronym[len(parent_acronym):]
                if parent_acronym and acronym.startswith(parent_acronym)
                else None
            )
            laminar = "layer" in str(node["name"]).lower() or suffix in LAYER_CODE
            if laminar:
                if suffix not in LAYER_CODE:
                    raise ValueError(
                        f"Unknown laminar token: ID={node_id}, acronym={acronym!r}, "
                        f"parent={parent_acronym!r}, suffix={suffix!r}"
                    )
                code = LAYER_CODE[suffix]
        classes[node_id] = code

    ids = np.array(sorted(classes), dtype=np.uint32)
    codes = np.array([classes[int(i)] for i in ids], dtype=np.uint8)
    return ids, codes


def classify_annotation(annotation: np.ndarray, ids, table) -> np.ndarray:
    flat = annotation.ravel(order="C")
    result = np.empty(flat.size, dtype=np.uint8)
    for start in range(0, flat.size, CHUNK):
        block = flat[start:start + CHUNK]
        positions = np.searchsorted(ids, block)
        safe = np.minimum(positions, ids.size - 1)
        known = (positions < ids.size) & (ids[safe] == block)
        if not np.all(known):
            unknown = np.unique(block[~known])[:16].tolist()
            raise ValueError(f"Annotation IDs absent from graph, first examples: {unknown}")
        result[start:start + block.size] = table[safe]
    return result


def neighbours(flat_indices: np.ndarray):
    """Yield face ID, positions in input, and in-grid neighbour flat indices."""
    flat = np.asarray(flat_indices, dtype=np.int64)
    for axis, (width, stride) in enumerate(zip(SHAPE, STRIDES)):
        coordinate = (flat // stride) % width
        for side, sign in enumerate((-1, 1)):
            valid = coordinate > 0 if sign < 0 else coordinate < width - 1
            positions = np.flatnonzero(valid)
            yield 2 * axis + side, positions, flat[positions] + sign * stride


def boundary_counts(annotation_flat, codes, domain_flat, domain_codes):
    pial_mask = np.zeros(domain_flat.size, dtype=np.uint8)
    wm_mask = np.zeros(domain_flat.size, dtype=np.uint8)
    l6a_with_fibre = 0
    pial_category_counts = {
        "background": 0, "across_midline": 0, "other_structure": 0,
        "l1_to_noniso_l1_noflux": 0, "l1_to_fibre_noflux": 0,
    }

    for start in range(0, domain_flat.size, CHUNK):
        stop = min(start + CHUNK, domain_flat.size)
        local_codes = domain_codes[start:stop]
        pial = pial_mask[start:stop]
        wm = wm_mask[start:stop]
        l6a_touch = np.zeros(stop - start, dtype=bool)

        for face, positions, adjacent in neighbours(domain_flat[start:stop]):
            source_codes = local_codes[positions]
            adjacent_codes = codes[adjacent]
            is_fibre = adjacent_codes == FIBRE
            # Reviewer correction v2: the whole outer surface of L1 is pial,
            # including the medial surface where the hemispheres abut and
            # faces against non-cortical-sheet structures (e.g. ORB against
            # the olfactory bulb). Faces into another region's layer 1 (the
            # lateral continuation of the sheet) and into fibre tracts stay
            # no-flux.
            source_ml = domain_flat[start:stop][positions].astype(np.int64) % SHAPE[2]
            across = (source_ml < MIDLINE_INDEX) != (adjacent % SHAPE[2] < MIDLINE_INDEX)
            adjacent_domain = (adjacent_codes >= DOMAIN_UNLAMINATED) & (adjacent_codes < NONISO_L1)
            is_l1 = source_codes == LAYER_CODE["1"]
            pial_condition = is_l1 & (
                (adjacent_domain & across)
                | (~adjacent_domain & ~is_fibre & (adjacent_codes != NONISO_L1))
            )
            pial_category_counts["background"] += int(np.count_nonzero(pial_condition & (annotation_flat[adjacent] == 0)))
            pial_category_counts["across_midline"] += int(np.count_nonzero(pial_condition & adjacent_domain))
            pial_category_counts["other_structure"] += int(np.count_nonzero(pial_condition & ~adjacent_domain & (annotation_flat[adjacent] != 0)))
            pial_category_counts["l1_to_noniso_l1_noflux"] += int(np.count_nonzero(is_l1 & (adjacent_codes == NONISO_L1)))
            pial_category_counts["l1_to_fibre_noflux"] += int(np.count_nonzero(is_l1 & is_fibre))
            pial_hits = positions[pial_condition]
            wm_hits = positions[
                (source_codes == LAYER_CODE["6b"]) & is_fibre
            ]
            l6a_hits = positions[
                (source_codes == LAYER_CODE["6a"]) & is_fibre
            ]
            pial[pial_hits] |= np.uint8(1 << face)
            wm[wm_hits] |= np.uint8(1 << face)
            l6a_touch[l6a_hits] = True
        l6a_with_fibre += int(np.count_nonzero(l6a_touch))

    pial_counts = POPCOUNT[pial_mask]
    wm_counts = POPCOUNT[wm_mask]
    report = {
        "pial_faces": int(pial_counts.sum(dtype=np.int64)),
        "white_matter_faces": int(wm_counts.sum(dtype=np.int64)),
        "l6a_voxels_with_fibre_tract_face": l6a_with_fibre,
        "pial_face_categories_v2": pial_category_counts,
        "surface_pinning": {
            "domain_flat_indices_sha256": array_sha256(domain_flat),
            "domain_flat_indices_encoding": "ascending uint32 little-endian",
            "pial_face_bitmasks_sha256": array_sha256(pial_mask),
            "white_matter_face_bitmasks_sha256": array_sha256(wm_mask),
            "bitmask_encoding": (
                "One uint8 per domain voxel in ascending flat-index order; "
                "bits 0..5 are AP-, AP+, DV-, DV+, ML-, ML+."
            ),
        },
    }
    return pial_counts, wm_counts, report


def grouped_counts(component_ids, values, n_components):
    result = np.zeros(n_components + 1, dtype=np.int64)
    for start in range(0, component_ids.size, CHUNK):
        ids = component_ids[start:start + CHUNK]
        if values is None:
            local = np.bincount(ids, minlength=n_components + 1)
        else:
            # Integer face counts are exactly representable by float64.
            local = np.bincount(
                ids,
                weights=values[start:start + ids.size],
                minlength=n_components + 1,
            ).astype(np.int64)
        result += local
    return result


def build_matrix(flat_indices, row_map, pial_counts, wm_counts):
    n = flat_indices.size
    if n == 0 or 7 * n > np.iinfo(np.int32).max:
        raise SolverFailure("Unsupported matrix size.")

    degree = np.zeros(n, dtype=np.uint8)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        local_degree = degree[start:stop]
        for _, positions, adjacent in neighbours(flat_indices[start:stop]):
            mapped = row_map[adjacent]
            local_degree[positions[mapped >= 0]] += 1

    diagonal = (
        degree.astype(np.int16)
        + pial_counts.astype(np.int16)
        + wm_counts.astype(np.int16)
    )
    if np.any(diagonal <= 0) or np.any(diagonal > 6):
        raise SolverFailure("Invalid matrix diagonal.")

    indptr = np.empty(n + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(degree + np.uint8(1), dtype=np.int64, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    data = np.full(int(indptr[-1]), -1.0, dtype=np.float64)
    indices[indptr[:-1]] = np.arange(n, dtype=np.int32)
    data[indptr[:-1]] = diagonal
    cursor = indptr[:-1].copy()
    cursor += 1

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        for _, positions, adjacent in neighbours(flat_indices[start:stop]):
            mapped = row_map[adjacent]
            keep = mapped >= 0
            rows = start + positions[keep]
            indices[cursor[rows]] = mapped[keep]
            cursor[rows] += 1

    if not np.array_equal(cursor, indptr[1:]):
        raise SolverFailure("CSR construction count mismatch.")

    matrix = sparse.csr_matrix((data, indices, indptr), shape=(n, n), copy=False)
    matrix.sort_indices()
    return matrix, diagonal.astype(np.float64)


def solve_field(labels, component_ids, domain_flat, pial_counts, wm_counts, n_components):
    counts = grouped_counts(component_ids, None, n_components)
    pial_by_component = grouped_counts(component_ids, pial_counts, n_components)
    wm_by_component = grouped_counts(component_ids, wm_counts, n_components)
    eligible = (pial_by_component > 0) & (wm_by_component > 0)
    eligible[0] = False

    # Stable sorting preserves ascending flat indices within each component.
    ordering = np.argsort(component_ids, kind="stable").astype(np.int32)
    offsets = np.empty(n_components + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts[1:], out=offsets[1:])

    # Reuse the full-volume component-label allocation as a voxel-to-row map.
    row_map = labels.ravel(order="C")
    row_map.fill(-1)
    phi = np.full(domain_flat.size, np.nan, dtype=np.float64)
    details = []

    for component in range(1, n_components + 1):
        item = {
            "component": component,
            "voxels": int(counts[component]),
            "pial_faces": int(pial_by_component[component]),
            "white_matter_faces": int(wm_by_component[component]),
        }
        if not eligible[component]:
            item["status"] = "unsolved"
            item["reasons"] = []
            if pial_by_component[component] == 0:
                item["reasons"].append("no_pial_face")
            if wm_by_component[component] == 0:
                item["reasons"].append("no_white_matter_face")
            details.append(item)
            continue

        rows = ordering[offsets[component - 1]:offsets[component]]
        flat_indices = domain_flat[rows]
        n = rows.size
        print(f"Building component {component}: {n} voxels", flush=True)
        setup_start = time.perf_counter()
        row_map[flat_indices] = np.arange(n, dtype=np.int32)
        try:
            matrix, diagonal = build_matrix(
                flat_indices, row_map, pial_counts[rows], wm_counts[rows]
            )
        finally:
            row_map[flat_indices] = -1
        setup_seconds = time.perf_counter() - setup_start

        rhs = wm_counts[rows].astype(np.float64)
        rhs_norm = float(np.linalg.norm(rhs))
        if not math.isfinite(rhs_norm) or rhs_norm == 0.0:
            raise SolverFailure(f"Component {component}: invalid RHS norm.")

        inverse_diagonal = 1.0 / diagonal
        preconditioner = LinearOperator(
            (n, n),
            matvec=lambda vector, scale=inverse_diagonal: scale * vector,
            dtype=np.float64,
        )
        iterations = 0

        def callback(_):
            nonlocal iterations
            iterations += 1

        solver_start = time.perf_counter()
        solution, info = cg(
            matrix,
            rhs,
            x0=np.zeros(n, dtype=np.float64),
            rtol=RTOL,
            atol=ATOL,
            maxiter=MAXITER,
            M=preconditioner,
            callback=callback,
        )
        residual = rhs - matrix @ solution
        relative_residual = float(np.linalg.norm(residual) / rhs_norm)
        solve_seconds = time.perf_counter() - solver_start
        print(
            f"Component {component}: iterations={iterations}, info={info}, "
            f"relative_residual={relative_residual:.17g}",
            flush=True,
        )
        if (
            info != 0
            or not np.all(np.isfinite(solution))
            or not math.isfinite(relative_residual)
            or relative_residual > RTOL
        ):
            raise SolverFailure(
                f"Component {component} failed: info={info}, "
                f"iterations={iterations}, relative_residual={relative_residual:.17g}"
            )

        phi[rows] = solution
        item.update({
            "status": "solved",
            "solver": {
                "iterations": iterations,
                "cg_info": int(info),
                "relative_residual": relative_residual,
                "rhs_norm": rhs_norm,
                "nnz": int(matrix.nnz),
                "matrix_setup_seconds": setup_seconds,
                "solve_and_residual_seconds": solve_seconds,
            },
        })
        details.append(item)
        del (
            matrix, diagonal, rhs, inverse_diagonal, preconditioner,
            solution, residual, rows, flat_indices,
        )

    # Permanent map to compact, globally ascending domain indices.
    row_map[domain_flat] = np.arange(domain_flat.size, dtype=np.int32)
    solved_count = int(np.count_nonzero(eligible[1:]))
    report = {
        "total": n_components,
        "solved_components": solved_count,
        "unsolved_components": n_components - solved_count,
        "solved_voxels": int(counts[eligible].sum()),
        "unsolved_voxels": int(counts[~eligible].sum()),
        "details": details,
    }
    return phi, row_map, report


def field_diagnostics(phi, domain_codes):
    valid = np.isfinite(phi)
    values = phi[valid]
    field_min = float(values.min()) if values.size else None
    field_max = float(values.max()) if values.size else None
    outside_range = int(np.count_nonzero((values < 0.0) | (values > 1.0)))
    layers = {}

    for token in TOKENS:
        mask = domain_codes == LAYER_CODE[token]
        layer_values = phi[mask & valid]
        if layer_values.size:
            q05, median, q95 = np.quantile(
                layer_values, [0.05, 0.5, 0.95],
                method="linear", overwrite_input=True,
            )
            quantiles = {
                "p05": float(q05),
                "median": float(median),
                "p95": float(q95),
            }
        else:
            quantiles = {"p05": None, "median": None, "p95": None}
        layers[token] = {
            "domain_voxels": int(np.count_nonzero(mask)),
            "solved_voxels": int(layer_values.size),
            **quantiles,
        }

    report = {
        "domain_voxels": int(phi.size),
        "solved_voxels": int(values.size),
        "unsolved_voxels": int(phi.size - values.size),
        "field_min": field_min,
        "field_max": field_max,
        "voxels_outside_closed_unit_interval": outside_range,
        "expected_range": [0.0, 1.0],
        "unlaminated_domain_voxels": int(
            np.count_nonzero(domain_codes == DOMAIN_UNLAMINATED)
        ),
        "per_laminar_token": layers,
        "expected_laminar_pattern": "Increasing depth from layer 1 toward layer 6b.",
    }
    return valid, values, report


def read_manifest(path: Path) -> list[dict[str, str]]:
    records = []
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if (
            fields is None
            or len(fields) != len(set(fields))
            or set(fields) != MANIFEST_COLUMNS
        ):
            raise ValueError(
                f"Manifest must have exactly these columns: {sorted(MANIFEST_COLUMNS)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed manifest row at CSV line {line_number}.")
            for name in ("cell", "brain", "ep09_role"):
                if not row[name] or row[name] != row[name].strip():
                    raise ValueError(f"Invalid {name!r} at CSV line {line_number}.")
            if row["cell"] in seen:
                raise ValueError(f"Duplicate manifest cell: {row['cell']}")
            seen.add(row["cell"])
            records.append(row)
    if not records:
        raise ValueError("Manifest has no cells.")
    return records


def member_name(record: dict[str, str]) -> str:
    brain = record["brain"]
    cell = record["cell"]
    prefix = brain + "_"
    if not cell.startswith(prefix):
        raise ValueError(f"Cell {cell!r} does not start with manifest brain {brain!r}.")
    sample = cell[len(prefix):]
    for value in (brain, sample):
        if (
            not value or value in (".", "..")
            or any(character in value for character in ("/", "\\", "\x00"))
        ):
            raise ValueError(f"Unsafe or malformed SWC identifier: {cell!r}")
    return f"data_v3/{brain}/{sample}.swc"


def exact_decimal(token: bytes) -> Fraction:
    text = token.decode("ascii")
    if DECIMAL_RE.fullmatch(text) is None:
        raise ValueError(f"Not a finite decimal token: {text!r}")
    return Fraction(text)


def read_first_soma_row(handle, member: str) -> tuple[Soma, bytes]:
    for _ in range(MAX_INITIAL_SWC_LINES):
        raw = handle.readline(MAX_SWC_LINE_BYTES + 1)
        if not raw:
            raise ValueError(f"No first SWC data row: {member}")
        if len(raw) > MAX_SWC_LINE_BYTES:
            raise ValueError(f"SWC initial line exceeds size limit: {member}")
        stripped = raw.strip()
        if not stripped or stripped.startswith(b"#"):
            continue

        # Parse this row only. A mismatch is an error, never a search forward.
        fields = stripped.split()
        if len(fields) != 7:
            raise ValueError(f"First SWC data row must have seven fields: {member}")
        try:
            identifiers = (int(fields[0]), int(fields[1]), int(fields[6]))
            if identifiers != (1, 1, -1):
                raise ValueError("expected node=1, type=1, parent=-1")
            coordinates = tuple(exact_decimal(token) for token in fields[2:5])
            radius = exact_decimal(fields[5])
            if radius < 0:
                raise ValueError("negative soma radius")
        except (ValueError, UnicodeError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid first soma row in {member}: {exc}") from exc

        voxel = tuple(int(coordinate // SPACING_UM) for coordinate in coordinates)
        ml = coordinates[2]
        hemisphere = (
            "left" if ml < MIDLINE_ML_UM
            else "right" if ml > MIDLINE_ML_UM
            else "ambiguous"
        )
        return Soma(voxel=voxel, hemisphere=hemisphere), raw
    raise ValueError(f"Too many initial SWC comment/blank lines: {member}")


def read_somata(zip_path: Path, manifest):
    before = stat_signature(zip_path)
    somata = []
    digest = hashlib.sha256()
    with zipfile.ZipFile(zip_path, "r") as archive:
        entries = archive.infolist()
        occurrences = Counter(entry.filename for entry in entries)
        member_count = len(entries)
        for record in manifest:
            member = member_name(record)
            if occurrences[member] != 1:
                raise ValueError(
                    f"Expected exactly one ZIP member {member!r}; "
                    f"found {occurrences[member]}."
                )
            with archive.open(member, "r") as handle:
                soma, raw = read_first_soma_row(handle, member)
            somata.append(soma)
            for part in (record["cell"].encode("utf-8"), raw):
                digest.update(len(part).to_bytes(8, "little"))
                digest.update(part)

    if stat_signature(zip_path) != before:
        raise RuntimeError("ZIP changed while reading soma rows.")
    report = {
        "path": str(zip_path),
        "size_bytes": before[2],
        "member_count": member_count,
        "soma_rows_read": len(somata),
        "soma_rows_sha256": digest.hexdigest(),
        "soma_rows_hash_encoding": (
            "Manifest order; for each cell: uint64le byte length + UTF-8 cell, "
            "then uint64le byte length + raw first SWC data line, including "
            "its line terminator when present."
        ),
        "whole_zip_sha256": None,
    }
    return somata, report, before


def radial_at_soma(voxel, flat_index, center_index, phi, row_map):
    center = float(phi[center_index])
    gradient = np.empty(3, dtype=np.float64)

    for axis, stride in enumerate(STRIDES):
        minus = None
        plus = None
        # v2: never difference across the midline (hemispheres are decoupled).
        left_side = voxel[2] < MIDLINE_INDEX
        minus_ok = voxel[axis] > 0 and (
            axis != 2 or (voxel[2] - 1 < MIDLINE_INDEX) == left_side
        )
        plus_ok = voxel[axis] < SHAPE[axis] - 1 and (
            axis != 2 or (voxel[2] + 1 < MIDLINE_INDEX) == left_side
        )
        if minus_ok:
            index = int(row_map[flat_index - stride])
            if index >= 0 and math.isfinite(float(phi[index])):
                minus = float(phi[index])
        if plus_ok:
            index = int(row_map[flat_index + stride])
            if index >= 0 and math.isfinite(float(phi[index])):
                plus = float(phi[index])

        if minus is not None and plus is not None:
            gradient[axis] = (plus - minus) / (2.0 * SPACING_UM)
        elif plus is not None:
            gradient[axis] = (plus - center) / SPACING_UM
        elif minus is not None:
            gradient[axis] = (center - minus) / SPACING_UM
        else:
            gradient[axis] = np.nan

    if not np.all(np.isfinite(gradient)):
        return None
    norm = float(np.linalg.norm(gradient))
    if not math.isfinite(norm) or norm < GRADIENT_MIN_NORM:
        return None
    return -gradient / norm


def tangent_frame(radial):
    ap = np.array([1.0, 0.0, 0.0])
    t1 = ap - float(np.dot(ap, radial)) * radial
    norm = float(np.linalg.norm(t1))
    if norm < TANGENT_FALLBACK_NORM:
        ml = np.array([0.0, 0.0, 1.0])
        t1 = ml - float(np.dot(ml, radial)) * radial
        norm = float(np.linalg.norm(t1))
    t1 /= norm
    return t1, np.cross(radial, t1)


def sample_soma(record, soma, annotation_flat, phi, row_map):
    row = {column: "" for column in CSV_COLUMNS}
    row.update({
        "cell": record["cell"],
        "brain": record["brain"],
        "ep09_role": record["ep09_role"],
        "hemisphere": soma.hemisphere,
    })
    voxel = soma.voxel
    if not all(0 <= coordinate < width for coordinate, width in zip(voxel, SHAPE)):
        row["missing_reason"] = "outside_isocortex"
        return row, True

    flat_index = sum(coordinate * stride for coordinate, stride in zip(voxel, STRIDES))
    row["soma_voxel_label"] = str(int(annotation_flat[flat_index]))
    center_index = int(row_map[flat_index])
    if center_index < 0:
        row["missing_reason"] = "outside_isocortex"
        return row, False
    if not math.isfinite(float(phi[center_index])):
        row["missing_reason"] = "unsolved_component"
        return row, False

    radial = radial_at_soma(voxel, flat_index, center_index, phi, row_map)
    if radial is None:
        row["missing_reason"] = "invalid_gradient"
        return row, False

    t1, t2 = tangent_frame(radial)
    row["depth"] = format(float(phi[center_index]), ".17g")
    vectors = {"r": radial, "t1": t1, "t2": t2}
    if soma.hemisphere == "right":
        reflection = np.array([1.0, 1.0, -1.0])
        radial_c = reflection * radial
        t1_c = reflection * t1
        vectors.update({
            "r_c": radial_c,
            "t1_c": t1_c,
            "t2_c": np.cross(radial_c, t1_c),
        })
    elif soma.hemisphere == "left":
        vectors.update({"r_c": radial, "t1_c": t1, "t2_c": t2})
    else:
        row["missing_reason"] = "ambiguous_hemisphere"

    for prefix, vector in vectors.items():
        for axis, value in zip(AXES, vector):
            row[f"{prefix}_{axis}"] = format(float(value), ".17g")
    return row, False


def write_frames(binary_handle, manifest, somata, annotation, phi, row_map):
    by_role = {}
    out_of_grid = 0
    text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(
            text_handle, fieldnames=CSV_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for record, soma in zip(manifest, somata, strict=True):
            row, outside_grid = sample_soma(
                record, soma, annotation.ravel(order="C"), phi, row_map
            )
            writer.writerow(row)
            out_of_grid += int(outside_grid)
            role = record["ep09_role"]
            counts = by_role.setdefault(role, {
                "total": 0,
                "missing_depth": 0,
                "missing_radial_axis": 0,
                "missing_canonical_frame": 0,
                "missing_depth_or_any_frame": 0,
                "missing_reasons": {},
            })
            missing_depth = row["depth"] == ""
            missing_radial = row["r_ap"] == ""
            missing_canonical = row["r_c_ap"] == ""
            counts["total"] += 1
            counts["missing_depth"] += int(missing_depth)
            counts["missing_radial_axis"] += int(missing_radial)
            counts["missing_canonical_frame"] += int(missing_canonical)
            counts["missing_depth_or_any_frame"] += int(
                missing_depth or missing_radial or missing_canonical
            )
            reason = row["missing_reason"]
            if reason:
                reasons = counts["missing_reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
        text_handle.flush()
    finally:
        # Keep the exclusively created binary handle owned by OutputSet.
        text_handle.detach()
    return {"by_role": by_role, "out_of_grid_somata": out_of_grid}


def interrupt_handler(signum, _frame):
    raise InterruptedError(f"Received signal {signal.Signals(signum).name}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("CCF_PAYLOAD_DIR", type=Path)
    parser.add_argument("TOOLS_DIR", type=Path)
    parser.add_argument("ZIP", type=Path)
    parser.add_argument("CELL_MANIFEST_CSV", type=Path)
    parser.add_argument("OUT_DIR", type=Path)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID is required.")
    if not sys.flags.isolated:
        raise RuntimeError("Invoke with python3.11 -I.")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt_handler)

    started_utc = utc_now()
    started_clock = time.perf_counter()
    hostname = socket.gethostname()
    print(f"Hostname={hostname} SLURM_JOB_ID={job_id}", flush=True)

    payload = args.CCF_PAYLOAD_DIR.expanduser().resolve()
    tools_dir = args.TOOLS_DIR.expanduser().resolve()
    zip_path = args.ZIP.expanduser().resolve()
    manifest_path = args.CELL_MANIFEST_CSV.expanduser().resolve()
    out_dir = args.OUT_DIR.expanduser().resolve()
    paths = {
        "annotation": payload / "annotation_25_ccf_2017.nrrd",
        "graph": payload / "structure_graph_1.json",
        "manifest": manifest_path,
        "loader": (tools_dir / "derive_ccf_soma_layers.py").resolve(),
        "script": Path(__file__).resolve(),
    }
    sources = {name: fingerprint(path) for name, path in paths.items()}
    manifest = read_manifest(manifest_path)
    somata, zip_report, zip_signature = read_somata(zip_path, manifest)

    sys.path.insert(0, str(tools_dir))
    loader = importlib.import_module("derive_ccf_soma_layers")
    if Path(loader.__file__).resolve() != paths["loader"]:
        raise RuntimeError("Imported loader does not match the pinned loader file.")

    with OutputSet(out_dir) as outputs:
        header, annotation = loader.read_nrrd(paths["annotation"])
        if not isinstance(annotation, np.ndarray):
            raise ValueError("read_nrrd did not return a NumPy array.")
        if tuple(annotation.shape) != SHAPE:
            raise ValueError(f"Expected annotation shape {SHAPE}; got {annotation.shape}.")
        if annotation.dtype.kind != "u" or annotation.dtype.itemsize != 4:
            raise ValueError(f"Expected uint32 annotation; got {annotation.dtype}.")
        annotation = np.ascontiguousarray(annotation, dtype=np.uint32)
        if 7 * annotation.size > np.iinfo(np.int32).max:
            raise ValueError("Atlas exceeds the supported int32 CSR indexing bound.")

        nodes, _by_acr = loader.load_graph(paths["graph"])
        graph_ids, graph_codes = graph_class_table(nodes)
        codes = classify_annotation(annotation, graph_ids, graph_codes)
        domain = ((codes >= DOMAIN_UNLAMINATED) & (codes < NONISO_L1)).reshape(SHAPE)
        # Reviewer correction v2: hemispheres are decoupled at the midline, so
        # components are labelled separately on each side of it.
        structure = ndimage.generate_binary_structure(3, 1)
        left_labels, n_left = ndimage.label(domain[:, :, :MIDLINE_INDEX], structure=structure)
        right_labels, n_right = ndimage.label(domain[:, :, MIDLINE_INDEX:], structure=structure)
        right_labels[right_labels > 0] += n_left
        labels = np.ascontiguousarray(
            np.concatenate([left_labels, right_labels], axis=2), dtype=np.int32
        )
        del left_labels, right_labels
        n_components = int(n_left + n_right)
        domain_flat = np.flatnonzero(domain.ravel(order="C")).astype("<u4")
        component_ids = labels.ravel(order="C")[domain_flat]
        domain_codes = codes[domain_flat]
        del domain, nodes, _by_acr, graph_ids, graph_codes

        pial_counts, wm_counts, boundary_report = boundary_counts(
            annotation.ravel(order="C"), codes, domain_flat, domain_codes
        )
        del codes
        phi, row_map, component_report = solve_field(
            labels, component_ids, domain_flat, pial_counts, wm_counts, n_components
        )
        del labels, component_ids, pial_counts, wm_counts

        solved, solved_values, field_report = field_diagnostics(phi, domain_codes)
        solved_indices = domain_flat[solved]
        float64_solution_sha256 = array_sha256(solved_values)

        np.savez_compressed(
            outputs.handles[FIELD_NAME],
            flat_indices=solved_indices.astype("<u4", copy=False),
            phi=solved_values.astype("<f4"),
            shape=np.array(SHAPE, dtype="<i4"),
        )
        field_npz_sha256 = handle_sha256(outputs.handles[FIELD_NAME])
        del solved, solved_values, solved_indices, domain_codes

        soma_report = write_frames(
            outputs.handles[FRAMES_NAME],
            manifest, somata, annotation, phi, row_map,
        )
        frames_sha256 = handle_sha256(outputs.handles[FRAMES_NAME])

        # Refuse to publish results if any pinned source changed during the run.
        for name, path in paths.items():
            if fingerprint(path) != sources[name]:
                raise RuntimeError(f"Pinned input changed during execution: {name}")
        if stat_signature(zip_path) != zip_signature:
            raise RuntimeError("ZIP metadata changed during execution.")

        warnings = []
        if field_report["solved_voxels"] == 0:
            warnings.append("No component has both prescribed boundary types.")
        if field_report["voxels_outside_closed_unit_interval"]:
            warnings.append("Unclipped solved field contains values outside [0,1].")
        animals = len({record["brain"] for record in manifest})
        if len(manifest) != 1703 or animals != 43:
            warnings.append(
                f"Manifest has {len(manifest)} cells and {animals} animals; "
                "the supplied cohort description states 1703 and 43."
            )

        print(
            f"Domain voxels={field_report['domain_voxels']}; "
            f"pial faces={boundary_report['pial_faces']}; "
            f"white-matter faces={boundary_report['white_matter_faces']}; "
            f"L6a voxels touching fibre="
            f"{boundary_report['l6a_voxels_with_fibre_tract_face']}",
            flush=True,
        )
        print(
            f"Components: solved={component_report['solved_components']} "
            f"({component_report['solved_voxels']} voxels), "
            f"unsolved={component_report['unsolved_components']} "
            f"({component_report['unsolved_voxels']} voxels)",
            flush=True,
        )
        print(
            f"Field range: min={field_report['field_min']}, "
            f"max={field_report['field_max']}",
            flush=True,
        )
        for token, item in field_report["per_laminar_token"].items():
            print(
                f"Layer {token}: solved_voxels={item['solved_voxels']}, "
                f"p05={item['p05']}, median={item['median']}, p95={item['p95']}",
                flush=True,
            )
        for role, item in soma_report["by_role"].items():
            print(
                f"Role {role}: total={item['total']}, "
                f"missing_depth={item['missing_depth']}, "
                f"missing_radial_axis={item['missing_radial_axis']}, "
                f"missing_canonical_frame={item['missing_canonical_frame']}, "
                f"missing_depth_or_any_frame={item['missing_depth_or_any_frame']}",
                flush=True,
            )
        for warning in warnings:
            print(f"Diagnostic: {warning}", flush=True)

        report = {
            "schema": "ep09_cortical_laplace_report_v1",
            "provenance": {
                "SLURM_JOB_ID": job_id,
                "SLURM_STEP_ID": os.environ.get("SLURM_STEP_ID"),
                "SLURMD_NODENAME": os.environ.get("SLURMD_NODENAME"),
                "hostname": hostname,
                "started_at_utc": started_utc,
                "finished_computation_at_utc": utc_now(),
                "elapsed_seconds_before_report_write": time.perf_counter() - started_clock,
                "versions": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "python_isolated": bool(sys.flags.isolated),
                "thread_environment": THREAD_ENV,
                "inputs": sources,
                "zip": zip_report,
                "manifest_cells": len(manifest),
                "manifest_animals": animals,
                "nrrd_header": {
                    key: str(header[key])
                    for key in (
                        "type", "dimension", "sizes", "encoding",
                        "space", "space directions", "space origin",
                    )
                    if key in header
                },
            },
            "frozen_conventions": FROZEN,
            "solver_settings": FROZEN["solver"],
            "diagnostics": {
                **field_report,
                **boundary_report,
                "components": component_report,
                "somata": soma_report,
                "warnings": warnings,
            },
            "outputs": {
                FIELD_NAME: {
                    "sha256": field_npz_sha256,
                    "solved_phi_float64_sha256": float64_solution_sha256,
                    "float64_hash_encoding": (
                        "Unrounded float64 little-endian values, in ascending "
                        "solved flat-index order; raw element bytes without header."
                    ),
                    "saved_phi_dtype": "float32",
                    "sampling_phi_dtype": "float64",
                },
                FRAMES_NAME: {
                    "sha256": frames_sha256,
                    "rows": len(manifest),
                    "columns": CSV_COLUMNS,
                },
            },
        }
        outputs.handles[REPORT_NAME].write(
            (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
        )
        outputs.commit()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("ERROR: interrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
