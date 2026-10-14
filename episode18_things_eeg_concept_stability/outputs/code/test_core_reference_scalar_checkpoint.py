"""Two actual-solver interruption/continuation checks, compute allocation only.

These tiny numerical fixtures reuse the original overlapping-event operator.
They are implementation checks, not a simulation campaign or empirical result.
"""
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import continuous_fold_runner as folds
from continuous_sequence_model import JointSequenceDesign
import core_reference_scalar_checkpoint as checkpoint
from test_continuous_sequence_model import fixture


def inputs():
    part, _, _ = fixture()
    design = JointSequenceDesign([part])
    targets = np.column_stack([np.arange(8, dtype=float)**2/(column+3)
                              + np.cos(np.arange(8)*(column+1)/17.) for column in range(61)])
    weights = np.array([1., 2., 0., 1., 3., 1., 1., 2.])
    penalties = np.array([0., 0., .2, .3, .4, .5, .6, .7])
    limits = folds.SolverLimits(atol=1e-8, btol=1e-8, conlim=1e12, maxiter=10000)
    return design, targets, weights, penalties, limits


class Interrupted(RuntimeError):
    pass


def fail_solver(*args, **kwargs):
    raise RuntimeError('fixture_solver_failure')


class ScalarCheckpointTests(unittest.TestCase):
    def test_interrupted_parallel_resume_equals_unchanged_solver_and_reuses_only_completed(self):
        values = inputs()
        original = folds._fit(*values, lambda:None)
        prior_children = {item.pid for item in mp.active_children()}
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary)/'image_penalty_00.npz'
            binding = {'settings_id':'fixture', 'participant_id':'fixture',
                       'split':[0,1], 'model':'image', 'penalty_index':0}
            events = []
            def interrupt(event):
                events.append(event)
                if event['event'] == 'scalar_completed' and event['completed_scalars'] == 7:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                checkpoint.fit_checkpointed(*values, lambda:None, checkpoint=filename,
                    binding=binding, workers=2, record_byte_cap=1048576, progress=interrupt)
            with np.load(filename, allow_pickle=False) as saved:
                committed = saved['completed'].copy()
                self.assertEqual(int(committed.sum()), 7)
                self.assertTrue(np.all(saved['stop_codes'][committed] >= 0))
            self.assertEqual({item.pid for item in mp.active_children()}, prior_children)
            # An interrupted, uncommitted temporary never replaces authority.
            orphan = filename.with_name(filename.name+'.pending-interrupted')
            orphan.write_bytes(b'incomplete archive; do not reuse or delete')
            resumed_events = []
            resumed = checkpoint.fit_checkpointed(*values, lambda:None, checkpoint=filename,
                binding=binding, workers=2, record_byte_cap=1048576, progress=resumed_events.append)
            completed_columns = [event['column'] for event in resumed_events if event['event'] == 'scalar_completed']
            self.assertEqual(set(completed_columns), set(np.flatnonzero(~committed).tolist()))
            self.assertEqual(len(completed_columns), 54)
            self.assertTrue(orphan.exists())
            self.assertEqual(resumed_events[0]['reused_scalars'], 7)
            np.testing.assert_array_equal(resumed.coefficients, original.coefficients)
            self.assertEqual(resumed.stop_codes, original.stop_codes)
            self.assertEqual(resumed.iterations, original.iterations)
            self.assertFalse(resumed.coefficients.flags.writeable)
            with patch.object(checkpoint, 'fit_joint_ridge', side_effect=AssertionError('completed solve reran')):
                reused = checkpoint.fit_checkpointed(*values, lambda:None, checkpoint=filename,
                    binding=binding, workers=1, record_byte_cap=1048576, progress=lambda event:None)
            np.testing.assert_array_equal(reused.coefficients, original.coefficients)
            self.assertFalse(any('coefficients' in event or 'loss' in event or 'score' in event for event in events+resumed_events))
            self.assertEqual({item.pid for item in mp.active_children()}, prior_children)

    def test_mismatched_invalid_checkpoint_and_worker_failure_refuse_without_fallback(self):
        values = inputs()
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary)/'image_penalty_00.npz'
            binding = {'settings_id':'fixture', 'participant_id':'fixture', 'penalty_index':0}
            options = dict(checkpoint=filename, binding=binding, workers=1,
                           record_byte_cap=1048576, progress=lambda event:None)
            checkpoint.fit_checkpointed(*values, lambda:None, **options)
            original_bytes = filename.read_bytes()
            with self.assertRaisesRegex(ValueError, 'binding_or_solver'):
                checkpoint.fit_checkpointed(*values, lambda:None,
                    **{**options, 'binding':{**binding, 'penalty_index':1}})
            with self.assertRaisesRegex(ValueError, 'binding_or_solver'):
                checkpoint.fit_checkpointed(*values[:4], replace(values[4], atol=1e-7), lambda:None, **options)
            with self.assertRaisesRegex(ValueError, 'penalty_mismatch'):
                checkpoint.fit_checkpointed(*values[:3], values[3]+.1, values[4], lambda:None, **options)
            self.assertEqual(filename.read_bytes(), original_bytes)
            with np.load(filename, allow_pickle=False) as archive:
                saved = {name:archive[name].copy() for name in archive.files}
            for name, index, bad in [('coefficients',(0,0),np.nan), ('stop_codes',0,6), ('iterations',0,-1)]:
                changed = {key:array.copy() for key,array in saved.items()}
                changed[name][index] = bad
                with filename.open('wb') as stream:
                    np.savez(stream, **changed)
                with self.assertRaises(ValueError):
                    checkpoint.fit_checkpointed(*values, lambda:None, **options)
            filename.write_bytes(original_bytes)
            with self.assertRaisesRegex(ValueError, '61D'):
                checkpoint.fit_checkpointed(values[0], values[1][:,:60], *values[2:], lambda:None, **options)
            failed = Path(temporary)/'failed_penalty.npz'
            prior_children = {item.pid for item in mp.active_children()}
            with patch.object(checkpoint, 'fit_joint_ridge', fail_solver):
                with self.assertRaisesRegex(ValueError, 'scalar_worker_failed_no_retry'):
                    checkpoint.fit_checkpointed(*values, lambda:None,
                        **{**options, 'checkpoint':failed, 'workers':2})
            self.assertFalse(failed.exists())
            self.assertEqual({item.pid for item in mp.active_children()}, prior_children)
            # No real runaway child: emulate unsuccessful TERM/KILL/reaping.
            # The parent must retain its tracked PIDs and cannot emit the
            # workers_finished event or advance after this cleanup exception.
            surviving, connection = Mock(), Mock()
            surviving.is_alive.return_value = True
            with self.assertRaisesRegex(ValueError, 'cleanup_incomplete_stop_job'):
                checkpoint._finish_workers([surviving], [connection])
            connection.close.assert_called_once_with()
            surviving.terminate.assert_called_once_with()
            surviving.kill.assert_called_once_with()
            self.assertEqual(surviving.join.call_count, 2)


if __name__ == '__main__':
    unittest.main()
