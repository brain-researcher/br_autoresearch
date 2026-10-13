"""Deterministic dispatch fixtures: no real threads, subprocesses or source reads."""
import threading
import unittest
from unittest.mock import patch

import run_elastic_net_panel as panel


class FakeFuture:
    def __init__(self, owner, function, arguments):
        self.owner = owner
        self.function = function
        self.arguments = arguments
        self.finished = False
        self.cancelled = False
        self.value = None
        self.error = None

    def execute(self):
        if self.finished:
            return
        try:
            self.value = self.function(*self.arguments)
        except BaseException as error:
            self.error = error
        self.finished = True
        self.owner.outstanding -= 1

    def result(self):
        if not self.finished:
            raise AssertionError("Result requested before deterministic completion")
        if self.error is not None:
            raise self.error
        return self.value

    def cancel(self):
        if self.finished:
            return False
        self.cancelled = True
        self.finished = True
        self.owner.outstanding -= 1
        return True


class FakeExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.futures = []
        self.outstanding = 0
        self.maximum_outstanding = 0
        self.submission_snapshots = []
        self.joined = False

    def __enter__(self):
        return self

    def submit(self, function, *arguments):
        self.submission_snapshots.append(tuple(f.finished for f in self.futures))
        future = FakeFuture(self, function, arguments)
        self.futures.append(future)
        self.outstanding += 1
        self.maximum_outstanding = max(self.maximum_outstanding, self.outstanding)
        return future

    def __exit__(self, *_):
        for future in self.futures:
            future.execute()
        self.joined = True


class DispatchFixtures(unittest.TestCase):
    def setUp(self):
        self.executors = []
        self.reverse = True

        def executor_factory(max_workers):
            executor = FakeExecutor(max_workers)
            self.executors.append(executor)
            return executor

        def deterministic_wait(futures, return_when):
            self.assertIs(return_when, panel.FIRST_COMPLETED)
            # Complete one future at a time, deliberately reversing session
            # order by default; every other future remains unsettled.
            future = list(futures)[-1 if self.reverse else 0]
            future.execute()
            return {future}, set(futures)-{future}

        self.enterContext(patch.object(panel, "ThreadPoolExecutor", executor_factory))
        self.enterContext(patch.object(panel, "wait", deterministic_wait))

    def run_panel(self, sessions, run_one, **kwargs):
        settings = dict(started=0, max_seconds=1260,
                        stop_event=threading.Event(), clock=lambda: 0)
        settings.update(kwargs)
        return panel.run_sessions(sessions, 4, run_one, **settings)

    def assert_settled(self):
        executor = self.executors[-1]
        self.assertTrue(executor.joined)
        self.assertEqual(executor.outstanding, 0)
        self.assertTrue(all(f.finished for f in executor.futures))
        return executor

    def test_allocation_and_worker_validation(self):
        panel.validate_workers(4, 4)
        panel.validate_workers(1, 4)
        for workers, cpus in ((4, 1), (0, 4), (5, 8), (True, 4), (4, 4.0), (1, 0)):
            with self.subTest(workers=workers, cpus=cpus), self.assertRaises(ValueError):
                panel.validate_workers(workers, cpus)

    def test_fifteen_sessions_four_waves_and_original_output_order(self):
        sessions = [f"session_{i}" for i in range(15)]
        callbacks = []
        result = self.run_panel(sessions, lambda session: {"session": session},
            on_complete=lambda session, item: callbacks.append(session))
        self.assertEqual(result, [{"session": session} for session in sessions])
        self.assertEqual(callbacks, sessions[0:4][::-1]+sessions[4:8][::-1]
                         +sessions[8:12][::-1]+sessions[12:15][::-1])
        executor = self.assert_settled()
        self.assertEqual(executor.max_workers, 4)
        self.assertEqual(executor.maximum_outstanding, 4)
        self.assertEqual(len(executor.futures), 15)
        # No new wave starts until *every* prior wave future has settled.
        for start in (4, 8, 12):
            self.assertTrue(all(executor.submission_snapshots[start]))

    def test_worker_failure_stops_next_wave_and_preserves_original_error(self):
        self.reverse = False
        event = threading.Event()
        invoked = []

        def run_one(session):
            invoked.append(session)
            raise RuntimeError("original child failure")

        with self.assertRaisesRegex(RuntimeError, "original child failure"):
            self.run_panel(list(range(15)), run_one, stop_event=event)
        executor = self.assert_settled()
        self.assertEqual(len(executor.futures), 4)
        self.assertEqual(invoked, [0])
        self.assertTrue(event.is_set())

    def test_success_before_failure_is_recorded_but_no_replacements_start(self):
        completed = []
        invoked = []

        def run_one(session):
            invoked.append(session)
            if session == 2:
                raise ValueError("second completed child failed validation")
            return session

        with self.assertRaisesRegex(ValueError, "failed validation"):
            self.run_panel(list(range(15)), run_one,
                on_complete=lambda session, item: completed.append(item))
        executor = self.assert_settled()
        self.assertEqual(invoked, [3, 2])
        self.assertEqual(completed, [3])
        self.assertEqual(len(executor.futures), 4)

    def test_reserve_is_checked_before_wave_and_again_before_child(self):
        invoked = []
        with self.assertRaisesRegex(TimeoutError, "300s session reserve"):
            self.run_panel(list(range(15)), invoked.append,
                           max_seconds=600, clock=lambda: 301)
        self.assertEqual(len(self.assert_settled().futures), 0)
        self.assertEqual(invoked, [])

        # Wave selection still has its reserve, but time elapses before the
        # worker actually runs. No source-facing callback may then start.
        moments = iter((300, 301))
        with self.assertRaisesRegex(TimeoutError, "300s session reserve"):
            self.run_panel(list(range(15)), invoked.append,
                           max_seconds=600, clock=lambda: next(moments))
        self.assertEqual(len(self.assert_settled().futures), 4)
        self.assertEqual(invoked, [])

        # Equality is allowed by the prespecified complete 300-second reserve.
        self.assertEqual(self.run_panel(["only"], lambda session: session,
            max_seconds=600, clock=lambda: 300), ["only"])
        self.assert_settled()

    def test_preexisting_stop_and_callback_failure_settle_before_raising(self):
        event = threading.Event()
        event.set()
        invoked = []
        with self.assertRaises(TimeoutError):
            self.run_panel(list(range(15)), invoked.append, stop_event=event)
        self.assertEqual(len(self.assert_settled().futures), 0)
        self.assertEqual(invoked, [])

        def failed_callback(session, item):
            raise OSError("cannot record completed child")

        with self.assertRaisesRegex(OSError, "cannot record completed child"):
            self.run_panel(list(range(15)), invoked.append, on_complete=failed_callback)
        self.assertEqual(len(self.assert_settled().futures), 4)
        self.assertEqual(invoked, [3])


if __name__ == "__main__":
    unittest.main()
