"""Bind existing, explicitly released development artifacts to the full fitter.

This is a separate full-response bridge, never an upgrade of cost-only panels.
The coordinator must select full development access and resources beforehand.
Only the 12 existing development HDF5 paths, their projected metadata, eight
calibration-derived A_all axes and one named complete feature bank are used.
There is no discovery, calibration/audit response access, fitting or release.

The existing response adapter reads complete physical 512-column stripes in
development copies, then retains the frozen eligible repeats/A_all columns.
These reads can include other development cells; they never open another role.
response_byte_cap covers the final means, not process peak RSS or workspaces.
The caller must separately admit all arrays, snapshots and fitting workspaces.
"""
from contextlib import ExitStack
from dataclasses import dataclass
import re
from types import MappingProxyType

import h5py

import released_development_inputs as existing


@dataclass(frozen=True)
class CompleteDevelopmentInputs:
    design: object
    features: object
    responses: object
    frozen_columns: object
    operational_dimensions: object
    complete_development_response_release: bool = True


def load_complete_development_inputs(*, feature_attempt, feature_byte_cap, metadata_byte_cap,
                                    response_byte_cap, resource_check):
    """Load the complete unchanged design; explicit procedural release is external."""
    require, event = existing.require, existing.InputEvent
    try:
        require(callable(resource_check), "external_resource_callback_required")
        check = resource_check
        require(isinstance(feature_attempt, str) and re.fullmatch(r"full[0-9]{3,6}", feature_attempt) is not None
                and all(type(cap) is int and cap > 0 for cap in
                        (feature_byte_cap, metadata_byte_cap, response_byte_cap)), "explicit_full_input_scope_required")
        check(event("begin_complete_development_loading"))
        report = existing._json(existing.FEATURE_PUBLIC / feature_attempt / "feature_report.json", metadata_byte_cap, check)
        require(report.get("analysis_id") == "stimulus_features" and report.get("attempt") == feature_attempt
                and report.get("status") == "completed_private_staging"
                and report.get("full_development_features_complete") is True
                and report.get("development_images") == 2423 and report.get("development_concepts") == 480
                and report.get("model_ids") == list(existing.nested.MODEL_IDS)
                and report.get("layers") == list(existing.nested.LAYERS)
                and report.get("neural_values_read") is False
                and report.get("audit_pixels_or_features_opened") is False,
                "completed_full_development_feature_attempt_required")
        design, manifest = existing.feature_io.load_development_metadata()
        entries = existing._copy_entries(existing._json(existing.COPY_PUBLIC / "materialization_report.json",
                                                        metadata_byte_cap, check))
        row_maps = existing._row_maps(entries, metadata_byte_cap, check)
        axes = existing._frozen_columns(metadata_byte_cap, check)
        means_bytes = 3 * existing.nested.DEVELOPMENT_IMAGES * sum(len(axis) for axis in axes.values()) * 8
        check(event("plan_complete_response_means", byte_count=means_bytes))
        require(0 < means_bytes <= response_byte_cap, "complete_response_byte_cap_exceeded")
        feature_path = existing.FEATURE_PRIVATE / feature_attempt / "development/features.h5"
        require(existing.feature_io.pilot._ordinary_private_file(feature_path).st_size == report.get("feature_file_bytes"),
                "feature_report_file_size_changed")
        # Header/ordinary-file checks for all 12 copies precede response values.
        with ExitStack() as stack:
            handles = {}
            for person in existing.endpoint.SUBJECTS:
                for stage in existing.endpoint.STAGES:
                    path = existing.COPY_PRIVATE / "development" / person / f"{stage}.h5"
                    entry = entries[person, stage]
                    check(event("open_complete_development_header", participant=person, stage=stage))
                    require(existing._ordinary_input_file(path, existing.COPY_PRIVATE) == entry["array_file_bytes"],
                            "development_copy_file_size_changed")
                    handles[person, stage] = stack.enter_context(h5py.File(path, "r", rdcc_nbytes=existing.HDF_CACHE_BYTES))
                    existing.columns.output_dataset(handles[person, stage], entry["shape"], entry["dtype_str"])
            features = existing.feature_io.load_complete_features(feature_path, design, manifest,
                                                                  byte_cap=feature_byte_cap, check=check)
            panels = existing.responses.build_development_responses(design, manifest, row_maps, handles, axes,
                                                                     resource_check=check)
        dimensions = dict(development_images=2423, development_concepts=480, models=4, layers_per_model=4,
            participant_ROI_groups=8, beta_stages=3, response_mean_bytes=means_bytes,
            calibrated_group_widths={f"{person}_{roi}": len(axes[person, roi]) for person, roi in existing.endpoint.GROUPS},
            open_response_HDF_cache_bytes=12 * existing.HDF_CACHE_BYTES,
            physical_stripes_can_include_other_development_cells=True,
            raw_calibration_or_audit_response_access=False)
        check(event("complete_development_loading", when="after"))
        return CompleteDevelopmentInputs(design, features, panels, axes, MappingProxyType(dimensions))
    except existing.DevelopmentInputError:
        raise
    except Exception:
        raise existing.DevelopmentInputError("complete_development_input_loading_failed") from None
