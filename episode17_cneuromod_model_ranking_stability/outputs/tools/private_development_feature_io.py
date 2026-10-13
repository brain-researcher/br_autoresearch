"""Trusted full-development image delivery and complete feature persistence.

No execution or reader release is selected by importing this module. Only the
fixed source metadata and exact own-archive members may supply the 2423 rows.
The caller owns archive authentication, private credential delivery, resource
admission and the separately released development view. Audit/calibration
pixels and all neural responses are outside this interface.
"""
from collections import Counter
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
import warnings

import h5py
import numpy as np
from PIL import Image

import development_features as delivery
import inspect_stimulus_archive as archive
import nested_prediction as nested
import private_stimulus_feature_pilot as pilot

FORMAT = "ep17-complete-development-features-v1"
FEATURE_BYTES = 148869120
INDEX_BYTES = 32 * 1024 * 1024


class FeatureIOError(ValueError):
    """Fixed code only; no private source exceptions are public evidence."""


def require(condition, code):
    if not condition:
        raise FeatureIOError(code)


def bind_development_metadata(report, concepts, images, provider):
    require(report.get("status") == "completed" and report.get("mode") == "full"
            and report.get("metadata_assignments_complete") is True,
            "completed_frozen_assignment_required")
    require(report.get("preserved_metadata") == {
        "concepts": 720, "images": 4320, "presentations": 49980, "runs": 833}
        and report.get("role_counts") == {
        "development": 480, "support_calibration": 120, "sealed_audit": 120},
        "frozen_assignment_scope_changed")
    require(len(concepts) == 720 and len(images) == 4320
            and Counter(row["role"] for row in concepts) == report["role_counts"]
            and sum(row["eligible_image"] == "True" for row in images) == 3634,
            "frozen_assignment_counts_changed")
    # This existing exact join checks every image/concept/provider relationship,
    # including preserved ineligible rows. Its pilot subset is not our roster.
    archive.select_development_images(concepts, images, provider)
    manifest = tuple(dict(row) for row in images
                     if row["role"] == "development" and row["eligible_image"] == "True")
    # Preserve source CSV order and original fold/block strings without sorting.
    design = nested.DevelopmentDesign(
        tuple(row["provider_image_path"] for row in manifest),
        tuple(row["concept_name"] for row in manifest),
        tuple(row["role"] for row in manifest),
        tuple(row["outer_fold"] for row in manifest),
        tuple(row["development_block"] for row in manifest))
    delivery._bind_manifest(design, manifest)
    return design, manifest


def load_development_metadata():
    report = json.loads(archive.bounded_bytes(archive.ROLES / "assignment_report.json", 256 * 1024))
    return bind_development_metadata(report,
        archive.read_csv(archive.ROLES / "concept_assignments.csv"),
        archive.read_csv(archive.ROLES / "image_assignments.csv"), archive.read_csv(archive.PROVIDER))


def member_rows(design):
    return [{"feature_row_zero_based": index, "provider_image_path": path,
             "member_match_count": ""} for index, path in enumerate(design.image_ids)]


@contextmanager
def exact_image_loader(source_zip, allowed_paths, image_id, password, check):
    """Open one previously authenticated exact member, before eager decoding.

    Archive identity/encryption/directory checks belong to the caller's single
    admitted open object. This reader never searches, extracts or substitutes.
    """
    require(image_id in allowed_paths, "image_outside_development_roster")
    check("before_member_read")
    entry = source_zip.getinfo(image_id)
    require(entry.filename == entry.orig_filename == image_id and not entry.is_dir()
            and not stat.S_ISLNK(entry.external_attr >> 16)
            and 0 < entry.compress_size <= pilot.MEMBER_CAP
            and 0 < entry.file_size <= pilot.MEMBER_CAP, "image_member_bounds_or_identity")
    with source_zip.open(entry, "r", pwd=password) as member:
        encoded = member.read(pilot.MEMBER_CAP + 1)
    require(len(encoded) == entry.file_size, "image_member_length_changed")
    check("after_member_read")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = Image.open(io.BytesIO(encoded))
        try:
            require(image.format == "JPEG" and getattr(image, "n_frames", 1) == 1
                    and min(image.size) > 0 and image.width * image.height <= pilot.PIXEL_CAP,
                    "image_format_or_pixel_cap")
            yield image
        finally:
            # Pillow's context exit closes the file handle but can retain the
            # decoded core. Explicit close also releases pixels on caller error.
            image.close()
    check("after_image_close")


def _index_payload(design, manifest):
    delivery._bind_manifest(design, manifest)
    payload = json.dumps(list(manifest), ensure_ascii=False, sort_keys=True,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    require(len(payload) <= INDEX_BYTES, "feature_index_cap")
    return payload


def _private_parent(path):
    info = path.parent.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) == 0o700, "owned_private_parent_required")


def _panel_headers(panels, design):
    require(tuple(panels) == nested.MODEL_IDS, "all_four_models_required")
    for model in nested.MODEL_IDS:
        panel = panels[model]
        require(isinstance(panel, nested.FeaturePanel) and panel.rows == design
                and tuple(panel.layers) == nested.LAYERS, "complete_bound_feature_panel_required")
        for layer in nested.LAYERS:
            value = panel.layers[layer]
            require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float32")
                    and value.shape == (nested.DEVELOPMENT_IMAGES, nested.FEATURE_WIDTHS[layer]),
                    "feature_shape_or_dtype_changed")


def save_complete_features(path, design, manifest, panels, *, byte_cap, check):
    """Write one private HDF5; preserve partials and never overwrite an attempt.

    Publication uses an exclusive hard link after complete close/fsync. No
    resource callback after publication can mislabel a committed file as failed.
    A missing operational closeout still requires coordinator reconciliation.
    """
    path = Path(path)
    require(path.name == "features.h5" and type(byte_cap) is int and byte_cap > FEATURE_BYTES,
            "explicit_complete_feature_storage_required")
    _private_parent(path)
    require(not path.exists() and not path.is_symlink(), "feature_attempt_already_exists")
    payload = _index_payload(design, manifest)
    require(FEATURE_BYTES + len(payload) < byte_cap, "feature_storage_allowance_too_small")
    _panel_headers(panels, design)
    check("before_feature_write")
    partial = path.with_name(path.name + ".partial")
    descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w+b") as raw:
        with h5py.File(raw, "w") as target:
            target.attrs["format"] = FORMAT
            target.attrs["complete"] = 0
            target.create_dataset("index_json", data=np.frombuffer(payload, dtype=np.uint8))
            for model in nested.MODEL_IDS:
                group = target.create_group(model)
                for layer in nested.LAYERS:
                    check("before_feature_dataset_write")
                    value = panels[model].layers[layer]
                    require(bool(np.isfinite(value).all()), "nonfinite_feature_dataset")
                    group.create_dataset(layer, data=value, dtype="float32")
                    target.flush()
                    require(os.fstat(raw.fileno()).st_size <= byte_cap, "feature_storage_cap")
                    check("after_feature_dataset_write")
            target.attrs["complete"] = 1
            target.flush()
        raw.flush()
        os.fsync(raw.fileno())
        size = os.fstat(raw.fileno()).st_size
        require(size <= byte_cap, "feature_storage_cap")
    check("before_feature_publication")
    os.link(partial, path)
    cleanup = True
    try:
        partial.unlink()
    except OSError:
        cleanup = False
    return {"bytes": size, "partial_cleanup_complete": cleanup}


def _ordinary_hdf_object(container, name, kind):
    require(isinstance(container.get(name, getlink=True), h5py.HardLink), "ordinary_feature_object_required")
    value = container[name]
    require(isinstance(value, kind), "feature_object_kind_changed")
    return value


def load_complete_features(path, design, manifest, *, byte_cap, check):
    """Return immutable complete panels only for an explicitly admitted reader.

    Matching caller metadata does not grant access or authenticate its origin.
    Never call this during source-only work or use it to inspect closed features.
    """
    path = Path(path)
    require(path.name == "features.h5" and type(byte_cap) is int and byte_cap > FEATURE_BYTES,
            "complete_feature_file_required")
    _private_parent(path)
    info = pilot._ordinary_private_file(path)
    require(0 < info.st_size <= byte_cap, "feature_file_size_cap")
    expected = _index_payload(design, manifest)
    check("before_feature_open")
    with h5py.File(path, "r") as source:
        require(source.attrs.get("format") == FORMAT and source.attrs.get("complete") == 1
                and set(source) == {"index_json", *nested.MODEL_IDS}, "complete_feature_container_required")
        index = _ordinary_hdf_object(source, "index_json", h5py.Dataset)
        require(index.shape == (len(expected),) and index.dtype == np.dtype("uint8")
                and not index.is_virtual and index.external is None, "feature_index_header_changed")
        datasets = {}
        for model in nested.MODEL_IDS:
            group = _ordinary_hdf_object(source, model, h5py.Group)
            require(set(group) == set(nested.LAYERS), "all_four_layers_required")
            for layer in nested.LAYERS:
                value = _ordinary_hdf_object(group, layer, h5py.Dataset)
                require(value.shape == (nested.DEVELOPMENT_IMAGES, nested.FEATURE_WIDTHS[layer])
                        and value.dtype == np.dtype("float32") and not value.is_virtual
                        and value.external is None and value.compression is None,
                        "feature_dataset_header_changed")
                datasets[model, layer] = value
        # All sixteen headers pass before any feature values are loaded.
        require(index[()].tobytes() == expected, "feature_index_binding_changed")
        panels = {}
        for model in nested.MODEL_IDS:
            layers = {}
            for layer in nested.LAYERS:
                check("before_feature_dataset_read")
                values = datasets[model, layer][()]
                require(bool(np.isfinite(values).all()), "nonfinite_feature_dataset")
                layers[layer] = np.frombuffer(values.tobytes(order="C"), dtype=np.float32).reshape(values.shape)
                del values
            panels[model] = nested.FeaturePanel(design, MappingProxyType(layers))
    check("after_feature_load")
    return MappingProxyType(panels)
