"""Unactivated lifecycle for a future, separately admitted amended decode.

No scientific imports, grant, CLI, retry, or historical closure certification.
The future caller supplies its existing AmendedStagedCollection to
observed_collection_class(), then uses ``with lifecycle.run():`` around the
collection context and unchanged decode/point operation. The subclass adds
work-stop checks and observes close; it inherits all numerical/publication and
855-slot requirements. Do not substitute it into the currently running process.

The caller must bind both clocks to its actual allocation and put the outer
TERM/KILL timeout at hard_stop_epoch, AFTER the explicitly selected cleanup
reserve. Work stops at work_stop_epoch. SIGTERM handling, when requested, only
requests cooperative stopping at the next check; it cannot interrupt a stalled
native call safely or make SIGKILL/native file failure recoverable. Missing or
partial close records are not affirmative evidence. Terminal job state, frozen
inputs and ordinary recovery validation remain separate caller responsibilities.
Fresh admission, run root, dependent-job bindings and finite resource selection
must be integrated separately. The future composition's shared check/admission
callback must call check_work before its unchanged admission/budget checks,
including input/model preparation before constructing the writer and subsequent
decode/point work. Wrapping only the writer is insufficient. This is not a runner.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import time


class DecodeWorkStopped(TimeoutError):
    """Scientific work must unwind, leaving the reserved cleanup interval."""


class DecodeLifecycle:
    def __init__(self, *, job_id, collection_path, record_path,
                 work_stop_epoch, hard_stop_epoch, clock=time.time):
        if not (isinstance(job_id, str) and job_id.isdigit()
                and os.environ.get("SLURM_JOB_ID") == job_id):
            raise ValueError("actual_current_job_identity_required")
        self.collection_path, self.record_path = Path(collection_path), Path(record_path)
        for path in (self.collection_path, self.record_path):
            if not path.is_absolute() or path.resolve() != path or not path.parent.is_dir():
                raise ValueError("existing_absolute_nonsymlink_parent_required")
        self.partial_path = Path(str(self.collection_path) + ".partial")
        if self.collection_path.suffix != ".h5" or self.record_path in (self.collection_path, self.partial_path):
            raise ValueError("distinct_collection_and_operational_record_required")
        for path in (self.collection_path, self.partial_path, self.record_path,
                     Path(str(self.record_path) + ".partial")):
            if os.path.lexists(path):
                raise FileExistsError(path)
        if not (all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (work_stop_epoch, hard_stop_epoch))
                and work_stop_epoch < hard_stop_epoch):
            raise ValueError("explicit_positive_cleanup_reserve_required")
        self.job_id, self.clock = job_id, clock
        self.work_stop_epoch, self.hard_stop_epoch = work_stop_epoch, hard_stop_epoch
        self.writer = None
        self.close_state, self.close_error, self.file_identity = "not_attempted", None, None
        self.close_completed_at = None
        self._term_requested = self._entered = self._running = self.record_published = False
        self.check_work()

    def check_work(self):
        if os.environ.get("SLURM_JOB_ID") != self.job_id:
            raise RuntimeError("decode_allocation_identity_changed")
        if self._term_requested or self.clock() >= self.work_stop_epoch:
            raise DecodeWorkStopped("cooperative_decode_work_stop")

    def _check_hard_stop(self):
        if os.environ.get("SLURM_JOB_ID") != self.job_id:
            raise RuntimeError("decode_allocation_identity_changed")
        if self.clock() >= self.hard_stop_epoch:
            raise TimeoutError("decode_cleanup_hard_stop")

    def _bind(self, writer, path):
        if not self._running or self.writer is not None or Path(path) != self.collection_path:
            raise ValueError("one_fresh_collection_inside_lifecycle_required")
        self.check_work()
        self.writer = writer

    def _request_term(self, unused_signal, unused_frame):
        self._term_requested = True

    def _publish(self, operation_error):
        self._check_hard_stop()
        record = dict(kind="ep21_future_decode_closure", job_id=self.job_id,
            collection_path=str(self.collection_path), partial_path=str(self.partial_path),
            work_stop_epoch=self.work_stop_epoch, hard_stop_epoch=self.hard_stop_epoch,
            cleanup_reserve_seconds=self.hard_stop_epoch - self.work_stop_epoch,
            term_requested=self._term_requested, close_state=self.close_state,
            writer_closed=self.close_state == "succeeded",
            hdf_close_completed=self.close_state == "succeeded",
            fsync_completed=self.close_state == "succeeded", file_identity=self.file_identity,
            close_completed_at_epoch=self.close_completed_at,
            cleanup_error_type=self.close_error,
            operation_error_type=None if operation_error is None else type(operation_error).__name__,
            historical_certification=False)
        encoded = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(encoded) > 8192:
            raise ValueError("bounded_closure_record_required")
        temporary = Path(str(self.record_path) + ".partial")
        linked = False
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._check_hard_stop()
            os.link(temporary, self.record_path)  # Exclusive publication; never overwrite.
            linked = True
            directory = os.open(self.record_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self._check_hard_stop()
            temporary.unlink()
        except BaseException:
            if linked:
                self.record_path.unlink()  # Keep the unaccepted .partial for diagnosis.
            raise
        self.record_published = True

    @contextmanager
    def run(self, *, handle_sigterm=False):
        if self._entered:
            raise RuntimeError("lifecycle_is_single_use")
        self.check_work()
        self._entered = self._running = True
        previous = None
        operation_error = None
        try:
            if handle_sigterm:
                previous = signal.signal(signal.SIGTERM, self._request_term)
            try:
                yield self
            except BaseException as error:
                operation_error = error
                raise
            finally:
                cleanup_error = None
                try:
                    if self.writer is not None:
                        self.writer.close()
                except BaseException as error:
                    cleanup_error = error
                try:
                    self._publish(operation_error or cleanup_error)
                except BaseException as error:
                    if operation_error is None:
                        raise
                    operation_error.add_note("closure record not published: " + type(error).__name__)
                if cleanup_error is not None:
                    if operation_error is None:
                        raise cleanup_error
                    operation_error.add_note("writer cleanup failed: " + type(cleanup_error).__name__)
        finally:
            self._running = False
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)


def observed_collection_class(base):
    """Wrap the supplied original sec2-HDF collection; change no scientific method.

    The future caller supplies AmendedStagedCollection explicitly. Its _check,
    close, _h5 and partial interfaces are required. An extra fsync of a duplicate
    of the actual native descriptor proves synchronization independently of a
    base's idempotent close flag. Failure never becomes success on a later call.
    """
    class ObservedCollection(base):
        def __init__(self, path, *, lifecycle, **kwargs):
            # Base constructors may call self._check() and self._abort().
            self._decode_lifecycle = lifecycle
            lifecycle._bind(self, path)
            super().__init__(path, **kwargs)

        def _check(self):
            self._decode_lifecycle.check_work()
            return super()._check()

        def close(self):
            lifecycle = self._decode_lifecycle
            handle = getattr(self, "_h5", None)
            if lifecycle.close_state != "not_attempted" or handle is None or not handle.id.valid:
                # A second close may still release resources, but cannot certify
                # a prior failed/unknown close or a pre-existing closed handle.
                return super().close()
            lifecycle.close_state = "in_progress"
            descriptor = None
            try:
                if Path(self.partial) != lifecycle.partial_path:
                    raise ValueError("observed_writer_path_changed")
                descriptor = os.dup(handle.id.get_vfd_handle())
                identity = os.fstat(descriptor)
                on_disk = lifecycle.partial_path.stat()
                if (identity.st_dev, identity.st_ino) != (on_disk.st_dev, on_disk.st_ino):
                    raise ValueError("observed_writer_inode_mismatch")
                lifecycle.file_identity = dict(device=identity.st_dev, inode=identity.st_ino)
                super().close()
                if handle.id.valid:
                    raise RuntimeError("native_hdf_close_did_not_complete")
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
                lifecycle.close_completed_at = lifecycle.clock()
                lifecycle.close_state = "succeeded"
            except BaseException as error:
                lifecycle.close_state, lifecycle.close_error = "failed", type(error).__name__
                raise
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    return ObservedCollection
