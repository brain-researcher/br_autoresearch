"""Six focused publication tests; execute only in coordinated Slurm.

Fixtures create their own small HDF5 fragments. No empirical inputs, model
fitting, source arrays, existing fragments or scheduler calls are involved.
"""
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import buffered_publication as buffered
import train_decoder_work_item as worker

layer = worker.layer


class FakeStore:
    def __init__(self, span):
        self.span = span
        self.committed_end = span.start
        self.uncertain_publication = None
        self.calls = []

    def commit(self, begin, rows):
        self.calls.append((begin, list(rows)))
        self.committed_end = begin + len(rows)


class BufferedPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worker.prefix.load_libraries()
        cls.np, cls.h5py = worker.prefix.np, worker.prefix.h5py

    def span(self, start=0, stop=300):
        return worker.contract.coverage.Span("sub-01", "conv1_1", 3, start, stop)

    def rows(self, begin, count):
        np = self.np
        return [(np.arange(500, dtype=np.int64) + index * 1000,
                 np.full(500, index / 8., dtype=np.float32), np.float32(index / 16.),
                 np.float32(index / 32.), np.float32(1. + index / 64.), index / 1000.)
                for index in range(begin, begin + count)]

    def test_exact_numeric_payload_and_ids_with_four_block_group_and_tail(self):
        np = self.np
        span = self.span(17, 287)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.h5py.File(root / "plain.h5", "x") as plain_file, \
                    self.h5py.File(root / "buffered.h5", "x") as buffered_file:
                plain = layer.ChannelStore(plain_file, root / "plain.h5", span, mock.Mock())
                store = layer.ChannelStore(buffered_file, root / "buffered.h5", span, mock.Mock())
                publication = buffered.BufferedPublication(store)
                with mock.patch.object(store, "commit", wraps=store.commit) as delegated:
                    for index, begin in enumerate(range(span.start, span.stop, 64)):
                        rows = self.rows(begin, min(64, span.stop - begin))
                        plain.commit(begin, rows)
                        publication.commit(begin, rows)
                        if index < 3:
                            self.assertEqual(store.committed_end, span.start)
                            self.assertEqual(publication.pending_rows, (index + 1) * 64)
                        # Inputs are no longer owned by the adapter. Mutating them
                        # must not alter delayed writes or keep their bases alive.
                        for row in rows:
                            row[0][:] = -99
                            row[1][:] = -99
                    publication.flush()
                self.assertEqual([(call.args[0], len(call.args[1])) for call in delegated.call_args_list],
                                 [(17, 256), (273, 14)])
                self.assertEqual(store.committed_end, span.stop)
                self.assertEqual(publication.pending_rows, 0)
                self.assertIsNone(store.uncertain_publication)
                self.assertEqual(set(plain.file), set(store.file))
                for name in plain.file:
                    np.testing.assert_array_equal(plain.file[name][:], store.file[name][:], err_msg=name)

    def test_original_fit_span_finally_flushes_partial_rows_on_deadline_or_solver_error(self):
        np = self.np
        span, native = self.span(0, 193), np.arange(501)
        bank = np.zeros((5, 193), dtype=np.float32)
        for error_kind in ("deadline", "solver", "next_prepare"):
            with self.subTest(error_kind=error_kind):
                store = FakeStore(span)
                publication = buffered.BufferedPublication(store)
                checks, solved, prepares = 0, 0, 0
                def check():
                    nonlocal checks
                    checks += 1
                    if error_kind == "deadline" and checks == 73:
                        raise worker.Stop("work_item_deadline_reached")
                def prepare(x, augmented, gram, targets, corrmat):
                    nonlocal prepares
                    prepares += 1
                    if error_kind == "next_prepare" and prepares == 3:
                        raise RuntimeError("prepare failed")
                    count = targets.shape[1]
                    return {"branch": "source_greater_than_500_with_native_ID_ties",
                        "right": np.zeros((count, 502)), "correlations": np.zeros((count, 501)),
                        "means": np.zeros((1, count), dtype=np.float32),
                        "scales": np.ones((1, count), dtype=np.float32)}
                def solve(*args):
                    nonlocal solved
                    if error_kind == "solver" and solved == 70:
                        raise RuntimeError("solver failed")
                    solved += 1
                    return np.arange(500), np.full(500, solved, dtype=np.float32), np.float32(solved)
                timing = {key: 0. for key in ("normalization_correlation_RHS_and_exact500_solve_seconds",
                                             "select_solve_seconds", "publication_seconds")}
                with mock.patch.object(worker.prefix, "prepare_layer_math", side_effect=prepare), \
                        mock.patch.object(worker.prefix, "solve_selected", side_effect=solve), \
                        mock.patch.object(buffered, "BufferedPublication", return_value=publication), \
                        self.assertRaises(worker.Stop if error_kind == "deadline" else RuntimeError):
                    worker.fit_span_with_publication_policy(bank, span, None, None, None, native, None,
                        store, mock.Mock(check=check), timing, publication_blocks=4)
                expected = 128 if error_kind == "next_prepare" else 70
                self.assertEqual(store.committed_end, expected)
                self.assertEqual(len(store.calls), 1)
                self.assertEqual(len(store.calls[0][1]), expected)
                self.assertEqual(publication.pending_rows, 0)
                self.assertFalse(publication.failed)
                for index, row in enumerate(store.calls[0][1]):
                    np.testing.assert_array_equal(row[1], np.full(500, index + 1, dtype=np.float32))

    def test_worker_default_policy_is_exact_direct_path_and_other_counts_refuse(self):
        arguments = [object() for _ in range(9)] + [{"publication_seconds": 0.}]
        result = object()
        with mock.patch.object(worker, "fit_span", return_value=result) as original, \
                mock.patch.object(buffered, "BufferedPublication") as forbidden:
            self.assertIs(worker.fit_span_with_publication_policy(*arguments), result)
            original.assert_called_once_with(*arguments)
            original.reset_mock()
            self.assertIs(worker.fit_span_with_publication_policy(*arguments, publication_blocks=1), result)
            original.assert_called_once_with(*arguments)
            original.reset_mock()
            for count in (0, 2, 5, True):
                with self.subTest(count=count), self.assertRaisesRegex(worker.Stop, "supported_publication"):
                    worker.fit_span_with_publication_policy(*arguments, publication_blocks=count)
            original.assert_not_called()
            forbidden.assert_not_called()
        self.assertEqual(arguments[-1]["publication_seconds"], 0.)

    def test_noncontiguous_duplicate_oversized_and_cross_channel_refusal(self):
        store = FakeStore(self.span())
        publication = buffered.BufferedPublication(store)
        publication.commit(0, self.rows(0, 64))
        for begin, count in ((0, 64), (65, 64), (64, 65), (64, 0)):
            with self.subTest(begin=begin, count=count), self.assertRaises(buffered.PublicationBufferError):
                publication.commit(begin, self.rows(begin, count))
        self.assertEqual(publication.pending_rows, 64)
        self.assertEqual(store.calls, [])
        publication.commit(64, self.rows(64, 2))
        with self.assertRaises(buffered.PublicationBufferError):
            publication.commit(66, self.rows(66, 1))
        publication.flush()
        self.assertEqual(store.committed_end, 66)
        original = store.span
        store.span = self.span(0, 301)
        with self.assertRaisesRegex(buffered.PublicationBufferError, "unchanged_current_channel"):
            publication.flush()
        store.span = original
        with self.assertRaisesRegex(buffered.PublicationBufferError, "fresh_certain"):
            buffered.BufferedPublication(store)

    def test_delegated_data_or_marker_failure_is_latched_and_never_retried(self):
        for uncertain in (False, True):
            with self.subTest(uncertain=uncertain):
                store = FakeStore(self.span())
                publication = buffered.BufferedPublication(store)
                error = OSError("marker failure" if uncertain else "data failure")
                calls = []
                def failure(begin, rows):
                    calls.append((begin, len(rows)))
                    if uncertain:
                        store.uncertain_publication = {"spatial_Fortran_interval": [begin, begin + len(rows)],
                                                       "phase": "completion_marker_fsync"}
                    raise error
                with mock.patch.object(store, "commit", side_effect=failure):
                    for begin in (0, 64, 128):
                        publication.commit(begin, self.rows(begin, 64))
                    with self.assertRaises(OSError) as initial:
                        publication.commit(192, self.rows(192, 64))
                    self.assertIs(initial.exception, error)
                    for operation in (publication.flush, lambda: publication.commit(256, self.rows(256, 1))):
                        with self.assertRaises(OSError) as repeated:
                            operation()
                        self.assertIs(repeated.exception, error)
                self.assertEqual(calls, [(0, 256)])
                self.assertTrue(publication.failed)
                self.assertEqual(store.committed_end, 0)
                self.assertEqual(publication.pending_rows, 256)
                self.assertEqual(store.uncertain_publication is not None, uncertain)

    def test_deferred_signal_after_durable_cursor_advance_cannot_trigger_replay(self):
        span = self.span(0, 128)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "channels.h5"
            with self.h5py.File(path, "x") as container:
                store = layer.ChannelStore(container, path, span, mock.Mock())
                publication = buffered.BufferedPublication(store)
                publication.commit(0, self.rows(0, 64))
                real_sync, calls = worker.prefix.fsync_h5, []
                def sync(handle):
                    real_sync(handle)
                    calls.append(True)
                    if len(calls) == 2:
                        # Original ChannelStore masks this signal until AFTER
                        # marker durability and committed_end have advanced.
                        signal.raise_signal(signal.SIGALRM)
                def stop(signum, frame):
                    raise worker.Stop("work_item_deadline_reached")
                previous = signal.signal(signal.SIGALRM, stop)
                try:
                    with mock.patch.object(worker.prefix, "fsync_h5", side_effect=sync):
                        with self.assertRaisesRegex(worker.Stop, "work_item_deadline_reached"):
                            publication.flush()
                        with self.assertRaisesRegex(worker.Stop, "work_item_deadline_reached"):
                            publication.flush()
                finally:
                    signal.signal(signal.SIGALRM, previous)
                self.assertEqual(len(calls), 2)
                self.assertTrue(publication.failed)
                self.assertEqual(store.committed_end, 64)
                self.assertIsNone(store.uncertain_publication)
                self.np.testing.assert_array_equal(store.file["state"][:64], self.np.full(64, 2))
                self.np.testing.assert_array_equal(store.file["state"][64:], self.np.zeros(64))


if __name__ == "__main__":
    unittest.main()
