"""One adopted availability axis; original anatomy support stays unchanged.

This validates the recorded complete intersection, not authority to construct
one. No beta values, fitting, missingness repair or mask selection occur here.
"""
import io
from pathlib import Path

import numpy as np

from pilot_task_a_mapping import validate_support


FIELDS = {"keep", "original_voxel_ids", "selected_voxel_ids", "completed_maps"}


def load_selection(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError("availability-axis record exceeds its bound")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != FIELDS:
            raise ValueError("unexpected availability-axis fields")
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def select_support(original, forward_header, selection=None):
    """Return a checked support, indices and IDs; default is the original208.

    A supplied selection must bind every original ID, exact retained ordering
    and completed9024-map coverage. An empty intersection remains recorded by
    its builder but cannot enter fitting. This does not substitute a count
    threshold for downstream geometry and numerical certificates.
    """
    indices = validate_support(original, forward_header)
    original_ids = np.ravel_multi_index(indices.T, tuple(forward_header["shape"][:3]), order="C")
    support = {name: np.array(value, copy=True) for name, value in original.items()}
    if selection is None:
        return support, indices, tuple(map(int, original_ids))
    if set(selection) != FIELDS:
        raise ValueError("unexpected availability-axis fields")
    keep = np.asarray(selection["keep"])
    supplied_ids = np.asarray(selection["original_voxel_ids"])
    retained = np.asarray(selection["selected_voxel_ids"])
    completed = np.asarray(selection["completed_maps"])
    if (keep.dtype != np.dtype(bool) or keep.shape != (208,) or not keep.any()
            or supplied_ids.dtype.kind not in "iu" or supplied_ids.shape != (208,)
            or not np.array_equal(supplied_ids, original_ids)
            or retained.dtype.kind not in "iu" or retained.ndim != 1
            or not np.array_equal(retained, original_ids[keep])
            or completed.shape != () or completed.dtype.kind not in "iu"
            or int(completed) != 9024):
        raise ValueError("incomplete, empty or mismatched common availability axis")
    support["common_mask"][:] = False
    original_positions = np.flatnonzero(np.asarray(original["common_mask"]).ravel(order="C"))
    support["common_mask"].flat[original_positions[keep]] = True
    support["source_transform_voxel_ijk"] = indices[keep].copy()
    support["world_xyz_mm"] = support["world_xyz_mm"][keep].copy()
    selected = validate_support(support, forward_header, expected_points=int(keep.sum()))
    return support, selected, tuple(map(int, retained))
