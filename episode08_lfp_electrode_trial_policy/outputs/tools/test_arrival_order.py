"""New order-routing/canonicalization and endpoint-reference checks only."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from arrival_order_reader import OrderTrajectory, order_queues
from development_arrival_order import endpoint_replay
from prepare_development_trial_roles import DEVELOPMENT
from trial_allocation import POLICIES


def role():
    pools = [list(range(d*12, (d+1)*12)) for d in range(8)]
    return {"filename": DEVELOPMENT[0], "fitting_pool_by_direction": pools,
            "nested_training_sets": {"96": {"training_trials": sum(pools, [])}}}


class ArrivalOrderTest(unittest.TestCase):
    def test_only_postinitial_rows_are_reordered_and_seed_is_reproducible(self):
        r = role()
        for rep in range(3):
            queues = order_queues(r, rep)
            self.assertEqual(queues, order_queues(r, rep))
            for old, new in zip(r["fitting_pool_by_direction"], queues):
                self.assertEqual(tuple(old[:4]), new[:4])
                self.assertEqual(set(old[4:12]), set(new[4:]))
                self.assertEqual(len(set(new)), 12)
        with self.assertRaises(ValueError):
            order_queues(r, 3)

    def test_chosen_queue_read_and_original_canonical_final_arrays(self):
        r, calls = role(), []
        def read(ids, channels):
            calls.append((tuple(ids), tuple(channels)))
            return np.array([[t, -t, 2*t] for t in ids], dtype=float), np.array([[t, t*t] for t in ids], dtype=float)
        source = SimpleNamespace(role=r, read=read)
        initial = [t for q in r["fitting_pool_by_direction"] for t in q[:4]]
        x, y = read(initial, [9])
        calls.clear()
        final = []
        for rep in range(3):
            queues = order_queues(r, rep)
            history = OrderTrajectory(source, [9], x, y, queues)
            self.assertEqual(history.acquire(2), queues[2][4])
            self.assertEqual(calls[-1], ((queues[2][4],), (9,)))
            xx, yy, labels, ids = history.observed()
            self.assertEqual(ids, sorted(initial+[queues[2][4]]))
            for d in reversed(range(8)):
                while history.counts[d] < 12:
                    history.acquire(d)
            final.append(history.observed())
        for arrays in final:
            self.assertEqual(arrays[3], list(range(96)))
            for a, b in zip(arrays[:3], final[0][:3]):
                np.testing.assert_array_equal(a, b)
        invalid = [list(q) for q in order_queues(r, 0)]
        invalid[0][0], invalid[0][4] = invalid[0][4], invalid[0][0]
        n = len(calls)
        with self.assertRaises(ValueError):
            OrderTrajectory(source, [9], x, y, invalid)
        self.assertEqual(len(calls), n)

    def test_endpoint_replay_covers_each_order_seed_and_frozen_parameters(self):
        reference = {"status": "completed_two_trial_allocators_with_balanced_reference", "sessions": []}
        current = []
        for name in DEVELOPMENT:
            panels = {p: {"fitted_models": [{"training_trials": n, "electrodes": 8, "seed": seed,
                "rank": 2, "alpha": .01, "Q_common": .2+n/1000,
                "Q_wrong_common": .1, "pairing_increment": .1+n/1000}
                for n in (32, 48, 96) for seed in range(5)]} for p in POLICIES}
            reference["sessions"].append({"filename": name, "panels": copy.deepcopy(panels)})
            current.append({"filename": name, "orders": [{"panels": copy.deepcopy(panels)} for _ in range(3)]})
        self.assertEqual(endpoint_replay(current, reference), 0.)
        changed = copy.deepcopy(current)
        changed[0]["orders"][2]["panels"]["mean_error"]["fitted_models"][-1]["rank"] = 4
        with self.assertRaisesRegex(ArithmeticError, "rank/alpha"):
            endpoint_replay(changed, reference)
        changed = copy.deepcopy(current)
        changed[0]["orders"][2]["panels"]["mean_error"]["fitted_models"][-1]["Q_common"] += .001
        with self.assertRaisesRegex(ArithmeticError, "endpoint replay"):
            endpoint_replay(changed, reference)


if __name__ == "__main__":
    unittest.main()
