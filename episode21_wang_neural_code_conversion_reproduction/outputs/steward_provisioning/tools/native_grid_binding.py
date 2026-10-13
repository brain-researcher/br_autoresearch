"""Compare declared native grids at NIfTI-1 header numerical precision."""

import itertools
import numpy as np


def compare_native_grids(beta, mask):
    if beta["shape"] != mask["shape"] or beta["spatial_units"] != mask["units"][0]:
        return {"consistent": False, "reason": "shape_or_unit_tag_mismatch"}
    a = np.asarray(beta["affine"], dtype=np.float64)
    b = np.asarray(mask["affine"], dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4) or not np.all(np.isfinite([a, b])):
        return {"consistent": False, "reason": "invalid_affine"}
    if not np.array_equal(a[3], [0, 0, 0, 1]) or not np.array_equal(b[3], a[3]):
        return {"consistent": False, "reason": "invalid_homogeneous_row"}
    corners = np.array(list(itertools.product(*[(0, n - 1) for n in beta["shape"]])))
    homogeneous = np.column_stack([corners, np.ones(8)])
    # NIfTI-1 coordinate parameters are float32. Use a standard forward-error
    # allowance for two four-term homogeneous transforms, with no outcome-
    # based spatial tolerance and no resampling. Check corners because the
    # affine difference reaches its coordinate-wise extrema there.
    unit_roundoff = np.finfo(np.float32).eps / 2
    gamma8 = 8 * unit_roundoff / (1 - 8 * unit_roundoff)
    difference = homogeneous @ (a - b).T
    precision_bound = gamma8 * (np.abs(homogeneous) @ (np.abs(a) + np.abs(b)).T)
    try:
        relative = (np.linalg.inv(a) @ (homogeneous @ b.T).T).T - homogeneous
        reverse = (np.linalg.inv(b) @ (homogeneous @ a.T).T).T - homogeneous
    except np.linalg.LinAlgError:
        return {"consistent": False, "reason": "singular_affine"}
    max_displacement = max(np.max(np.abs(relative[:, :3])), np.max(np.abs(reverse[:, :3])))
    consistent = bool(np.all(np.abs(difference[:, :3]) <= precision_bound[:, :3])
                      and max_displacement < 0.5)
    return {"consistent": consistent,
            "criterion": "NIfTI1_float32_gamma8_forward_error_at_full_grid_corners",
            "float32_unit_roundoff": float(unit_roundoff), "gamma8": float(gamma8),
            "max_corner_coordinate_difference": float(np.max(np.abs(difference[:, :3]))),
            "max_corner_voxel_difference": float(np.max(np.abs(relative[:, :3]))),
            "max_reverse_corner_voxel_difference": float(np.max(np.abs(reverse[:, :3]))),
            "nearest_native_index_guard": "strictly_less_than_half_voxel_both_directions",
            "max_precision_bound_coordinate_units": float(np.max(precision_bound[:, :3])),
            "mask_or_beta_geometry_changed": False}
