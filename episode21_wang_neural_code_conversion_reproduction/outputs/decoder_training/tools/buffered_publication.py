"""Opt-in bounded publication grouping; no fitting, HDF5 or source admission.

Wrap one fresh ChannelStore for one current channel. The numerical worker still
solves at most 64 rows at a time. Only publication groups up to four such calls;
the original store retains its data-before-marker fsyncs and durable cursor.

The integration MUST use an outer finally around fit_span:

    publication = BufferedPublication(store)
    try:
        fit_span(..., publication, ...)
    finally:
        publication.flush()

Time that final flush as publication work. Use ONLY store.committed_end and
store.uncertain_publication for terminal coverage. Pending rows are not durable.
There is no destructor, implicit retry, resumed channel or background work.
"""
from __future__ import annotations

import numbers
import signal

NUMERICAL_BLOCK_ROWS = 64
MAX_BUFFERED_BLOCKS = 4
MAX_BUFFERED_ROWS = NUMERICAL_BLOCK_ROWS * MAX_BUFFERED_BLOCKS


class PublicationBufferError(RuntimeError):
    pass


def _require(condition, reason):
    if not condition:
        raise PublicationBufferError(reason)


def _copy_row(row):
    """Copy only the two vectors; never retain a prepared block/base array.

    Scalars from fit_span are immutable Python/NumPy real scalars. Array dtype
    and numerical values are preserved: no cast, normalization or estimator
    operation is performed here. Finite/unique-ID validation stays in the store.
    The dtype-size bound excludes object arrays or arbitrarily wide payloads.
    """
    _require(len(row) == 6, "six_original_row_fields_required")
    ids, weights, *scalars = row
    _require(ids.shape == weights.shape == (500,), "exact_500_element_row_vectors_required")
    _require(ids.dtype.kind in "iu" and 0 < ids.dtype.itemsize <= 8
             and weights.dtype.kind == "f" and 0 < weights.dtype.itemsize <= 8,
             "bounded_integer_ids_and_real_floating_weights_required")
    _require(all(isinstance(value, numbers.Real) for value in scalars), "immutable_real_row_scalars_required")
    owned_ids, owned_weights = ids.copy(order="C"), weights.copy(order="C")
    owned_ids.setflags(write=False)
    owned_weights.setflags(write=False)
    return (owned_ids, owned_weights, *scalars)


class BufferedPublication:
    """One-channel adapter with at most 256 owned rows and no replay route.

    Owned vector storage is at most 2,048,000 bytes at the allowed maximum
    dtypes (actual int64/float32 rows: 1,536,000 bytes). Original ChannelStore
    stacks one vector field at a time, at most another 1,024,000 bytes. Including
    the caller's current 64-row vectors and small tuples/lists is conservatively
    below 4 MiB of publication-specific memory for the production numeric types.
    This does not count the worker's existing numerical arrays or HDF5 buffers.

    The adapter preserves a failure object after ANY delegated BaseException.
    Later flush/commit raises that object without another store call, even if
    the store advanced its durable cursor before delivering a deferred signal.
    """
    def __init__(self, store):
        _require(store.committed_end == store.span.start and store.uncertain_publication is None,
                 "fresh_certain_channel_store_required")
        self.store = store
        self._span = store.span
        self._published_end = store.committed_end
        self._accepted_end = self._published_end
        self._rows = []
        self._blocks = 0
        self._short_tail = False
        self._failure = None

    @property
    def pending_rows(self):
        """Rows still held in memory; never a committed-coverage assertion."""
        return len(self._rows)

    @property
    def failed(self):
        return self._failure is not None

    def _check_active(self):
        if self._failure is not None:
            raise self._failure
        _require(self.store.span == self._span, "one_unchanged_current_channel_required")
        _require(self.store.committed_end == self._published_end
                 and self.store.uncertain_publication is None,
                 "underlying_store_changed_outside_publication_buffer")

    def commit(self, begin, rows):
        self._check_active()
        _require(type(begin) is int and begin == self._accepted_end,
                 "disjoint_contiguous_ascending_buffered_rows_required")
        _require(not self._short_tail and 0 < len(rows) <= NUMERICAL_BLOCK_ROWS
                 and begin + len(rows) <= self._span.stop, "valid_next_numerical_block_or_final_partial_required")
        _require(self._blocks < MAX_BUFFERED_BLOCKS
                 and len(self._rows) + len(rows) <= MAX_BUFFERED_ROWS, "bounded_publication_buffer_required")
        # A delivered deadline after this short critical section must leave all
        # accepted rows available to the caller's outer-finally flush. Restore
        # the signal mask BEFORE calling the original store: it owns its own
        # durability critical section and deferred-signal exception semantics.
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
        try:
            owned = [_copy_row(row) for row in rows]
            self._rows.extend(owned)
            self._accepted_end += len(owned)
            self._blocks += 1
            self._short_tail = len(owned) < NUMERICAL_BLOCK_ROWS
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        if self._blocks == MAX_BUFFERED_BLOCKS or self._accepted_end == self._span.stop:
            self.flush()

    def flush(self):
        self._check_active()
        if not self._rows:
            return
        begin, end = self._published_end, self._accepted_end
        try:
            self.store.commit(begin, self._rows)
            # Never invent a cursor, including for unusual successful returns.
            _require(self.store.committed_end == end and self.store.uncertain_publication is None,
                     "delegated_store_did_not_certify_exact_publication")
            self._published_end = self.store.committed_end
            self._rows = []
            self._blocks = 0
        except BaseException as error:
            self._failure = error
            raise
