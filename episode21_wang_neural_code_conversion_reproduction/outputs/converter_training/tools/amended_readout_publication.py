"""Dormant policy labeling around the existing one-inode readout publisher.

Keep genuine fitted-object identities and the original scientific codecs.
Only HDF/manifest/report policy labels and the local final publication view
differ. No grant, reader-closure authority, retry or resource choice is made.
"""
from __future__ import annotations

import os
from pathlib import Path

import amended_full_models as models
import finding_a_staged_collection as staged


MODEL_POLICY = models.MODEL_POLICY
original = staged.original
require = original._require


class AmendedStagedCollection(staged.StagedFindingACollection):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        try:
            self._check()
            self._h5.attrs["model_policy"] = MODEL_POLICY
            self._check()
        except BaseException as error:
            self._abort(error)
            raise


def open_decoded_point(stage_h5, *, budget, maximum_bytes, require_admitted):
    reader = staged.open_decoded_point(stage_h5, budget=budget,
        maximum_bytes=maximum_bytes, require_admitted=require_admitted)
    try:
        require_admitted()
        require(reader._state._h5.attrs.get("model_policy") == MODEL_POLICY,
                "explicit_adapted_stage_model_policy_required")
        return reader
    except BaseException:
        reader.close()
        raise


class _PublicationView:
    """Local delegation only: never replace live module or state methods."""
    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        return getattr(self._state, name)

    def _sidecar(self, path, record, used):
        require(path in (self._state.manifest_path, self._state.report_path),
                "only_final_manifest_and_report_policy_labels_required")
        return self._state._sidecar(path, {**record, "model_policy": MODEL_POLICY}, used)


def finalize_decoded_point(stage_h5, final_h5, primary, incoming, dyadic, *,
                           budget, maximum_bytes, require_admitted, require_readers_closed):
    """Retain the original sole-writer sequence; label both final sidecars.

    This is the existing finalizer's short I/O composition with the policy-aware
    opener and local publication view. All point/result validation, arrays,
    statistics, file caps and exclusive sidecar publication remain delegated.
    """
    require(callable(require_readers_closed), "external_reader_closure_required")
    with open_decoded_point(stage_h5, budget=budget, maximum_bytes=maximum_bytes,
                           require_admitted=require_admitted) as reader:
        state = reader._state
        final = Path(final_h5)
        require(final.is_absolute() and final.suffix == ".h5"
            and final != state.path and final.parent.is_dir(), "new_absolute_final_destination_required")
        destinations = (final, final.with_name(final.name + ".partial"),
            final.with_suffix(".decoded_feature_manifest.json"), final.with_suffix(".finding_a_report.json"))
        for path in destinations:
            if os.path.lexists(path) or os.path.lexists(str(path) + ".partial"):
                raise FileExistsError(path)
        require(staged.codec.panel_to_record(primary.point, check=state._check)
            == staged.codec.panel_to_record(reader.point, check=state._check),
            "final_primary_point_must_equal_published_stage")
        state._bind_results(primary, incoming, dyadic)
        require_readers_closed()
        state._check()
        marker_record = staged._encoded(dict(stage_hdf=str(state.path), final_hdf=str(final),
            status="finalization_started_no_retry", model_policy=MODEL_POLICY))
        require(state.path.stat().st_size + state._stage_metadata.stat().st_size
            + len(marker_record) <= maximum_bytes, "collection_exceeds_caller_aggregate_byte_cap")
        with state._stage_marker.open("xb") as stream:
            stream.write(marker_record)
            stream.flush()
            os.fsync(stream.fileno())
        staged._sync_directory(state._stage_marker.parent)
        reader.close()
        original._require_file_limit(maximum_bytes)
        state.require_admitted()
        stage_path = state.path
        state.path, state.partial = final, destinations[1]
        state.manifest_path, state.report_path = destinations[2:]
        state._replay_only, state._finalizing = False, True
        os.link(stage_path, state.partial)
        state._closed = False
        state._h5 = None
        try:
            state._check()
            state._h5 = staged.h5py.File(state.partial, "r+", driver="sec2")
            require(not bool(state._h5.attrs["complete"])
                and bool(state._h5.attrs["decoded_point_complete"])
                and "resampling" not in state._h5
                and state._h5.attrs.get("model_policy") == MODEL_POLICY,
                "unfinalized_adapted_stage_required_for_sole_writer")
            return original.FindingACollection.publish(_PublicationView(state), primary, incoming, dyadic)
        except BaseException as error:
            state._abort(error)
            raise
