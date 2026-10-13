"""Approved fixed EP04 low-level transform; no file, HDF5 or network access.

The 2026-10-05 approval selects this unchanged transform and the separate
role-selecting steward reader. No trained detector, new foundation model or
image display is used; OOD remains closed.
"""
from io import BytesIO

import numpy as np

GRIDS = (5, 10, 20)
DIMENSION = 6 * sum(grid**2 for grid in GRIDS)  # 3150 fixed coordinates
CONTROL_ID = 'ep04_rgb_block_moments_5_10_20_v1'


def displayed_rgb(encoded):
    """Decode one already-authorized image byte string, never a source path.

Provider documents stored 1000x1000 RGB JPEG / RGBA PNG images. Composite
transparency onto byte RGB128, using nearest-integer alpha compositing. No
EXIF rotation, color-profile reinterpretation, crop, resampling or augmentation.
The future OOD custodian would apply this same renderer; no OOD read is selected.
"""
    from PIL import Image
    with Image.open(BytesIO(encoded)) as image:
        if image.size != (1000, 1000) or image.format not in ('JPEG', 'PNG'):
            raise ValueError('Source image differs from the frozen provider format')
        if image.mode not in ('RGB', 'RGBA'):
            raise ValueError('Unsupported image mode; do not choose a conversion after scores')
        pixels = np.asarray(image)
        if pixels.dtype != np.uint8:
            raise ValueError('Expected byte-valued provider image')
        if image.mode == 'RGB':
            return pixels.copy()
        alpha = pixels[:, :, 3:].astype(np.uint32)
        numerator = pixels[:, :, :3].astype(np.uint32)*alpha + 128*(255-alpha)
        return ((numerator+127)//255).astype(np.uint8)


def descriptor(rgb):
    """Fixed RGB block means and population SDs; equal scales and statistics.

Values are byte RGB/255, not calibrated linear light. Each grid partitions the
stored square into equal nonoverlapping blocks. Every feature keeps fixed
physical intensity units; no across-image/neural fitting or L2 normalization.
"""
    values = np.asarray(rgb)
    if values.shape != (1000, 1000, 3) or values.dtype != np.uint8:
        raise ValueError('Expected the frozen displayed RGB byte array')
    values = values.astype(np.float64)/255.
    features = []
    for grid in GRIDS:
        width = 1000//grid
        blocks = values.reshape(grid, width, grid, width, 3)
        means = blocks.mean(axis=(1, 3))
        deviations = blocks.std(axis=(1, 3), ddof=0)
        features.append(np.concatenate((means.ravel(), deviations.ravel()))
                        / np.sqrt(6*grid*grid*len(GRIDS)))
    result = np.concatenate(features)
    if result.shape != (DIMENSION,) or not np.isfinite(result).all():
        raise ValueError('Invalid proposed low-level feature vector')
    return result


def low_level_rdm(features):
    """Euclidean geometry, computed in float64 before fixed partial-Spearman RSA."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != DIMENSION or not np.isfinite(x).all():
        raise ValueError('Invalid low-level feature matrix')
    squared_norm = np.einsum('ij,ij->i', x, x)
    squared = squared_norm[:, None]+squared_norm[None, :]-2*(x@x.T)
    result = np.sqrt(np.maximum(squared, 0.))
    np.fill_diagonal(result, 0.)
    return result


if __name__ == '__main__':
    raise SystemExit('Pure transform only; use the separately selected prepare_low_level_proposal.py reader')
