"""Focused proxy algebra, chosen-row boundary and reference replay fixtures."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from arrival_order_reader import OrderTrajectory
from development_mapping_variance import ORDERS, POLICY, attach_references
from development_design import ELECTRODE_COUNTS
from direction_training import residual_statistics
from mapping_variance import MappingDesign
from prepare_development_trial_roles import DEVELOPMENT
from trial_allocation import CHECKPOINTS, POLICIES


class MappingVarianceTest(unittest.TestCase):
    def test_updates_and_direction_expectations_match_independent_trace_difference(self):
        rng = np.random.default_rng(7301)
        x, y, labels = rng.normal(size=(32, 9)), rng.normal(size=(32, 4)), np.repeat(np.arange(8), 4)
        stats = residual_statistics(x, y, labels)
        state = MappingDesign(x, labels, stats, .03)
        z0, z, acquired_labels = state.z0.copy(), state.z0.copy(), labels.copy()
        m = z0.T @ z0 / 32
        for step in range(25):
            a = .96*np.eye(9)+z.T@z
            inv = np.linalg.inv(a)
            gains = np.array([np.trace(m@inv)-np.trace(m@np.linalg.inv(a+np.outer(row,row))) for row in z])
            expected = np.array([gains[acquired_labels == d].mean() for d in range(8)])
            counts = np.bincount(acquired_labels, minlength=8)
            chosen, values = state.choose(counts)
            np.testing.assert_allclose(values["utility"], expected, rtol=2e-10, atol=1e-12)
            self.assertAlmostEqual(values["design_index_before"], np.trace(m@inv), places=12)
            self.assertEqual(chosen, min(np.flatnonzero(counts<12), key=lambda d:(-expected[d],counts[d],d)))
            new = rng.normal(size=9)
            state.observe(new, chosen)
            z = np.vstack((z, (new-stats.direction_input_mean[chosen]-stats.input_mean)/stats.input_sd))
            acquired_labels = np.append(acquired_labels, chosen)
        np.testing.assert_allclose(state.inverse, np.linalg.inv(.96*np.eye(9)+z.T@z), rtol=2e-12, atol=2e-13)

    def test_zero_design_ties_balance_and_fixed_coordinates_are_copied(self):
        labels = np.repeat(np.arange(8), 4)
        x, y = np.zeros((32, 3)), np.arange(64).reshape(32,2).astype(float)
        stats = residual_statistics(x, y, labels)
        state = MappingDesign(x, labels, stats, .1)
        x[:] = 999
        stats.direction_input_mean[:] = 777
        stats.input_sd[:] = 10
        counts = np.full(8, 4, dtype=int)
        for total in range(64):
            chosen, values = state.choose(counts)
            self.assertEqual(chosen, total%8)
            self.assertEqual(values["utility"], [0.]*8)
            state.observe(np.zeros(3), chosen)
            counts[chosen] += 1
        with self.assertRaisesRegex(ValueError, "exhausted"):
            state.choose(counts)
        np.testing.assert_array_equal(state.scale, np.ones(3))

    def test_only_chosen_next_trial_reaches_proxy_after_decision(self):
        pools = [list(range(d*12,(d+1)*12)) for d in range(8)]
        role = {"fitting_pool_by_direction": pools}
        calls = []
        def read(ids, channels):
            calls.append((tuple(ids),tuple(channels)))
            return np.array([[t,t*t,-t] for t in ids],dtype=float), np.array([[t,2*t] for t in ids],dtype=float)
        initial = [t for pool in pools for t in pool[:4]]
        x,y = read(initial,[3]); calls.clear()
        labels = np.repeat(np.arange(8),4)
        history = OrderTrajectory(SimpleNamespace(role=role,read=read),[3],x,y,tuple(tuple(p) for p in pools))
        proxy = MappingDesign(x,labels,residual_statistics(x,y,labels),.1)
        chosen,before = proxy.choose(history.counts.copy())
        self.assertEqual(calls,[])
        trial = history.acquire(chosen)
        self.assertEqual(calls,[((pools[chosen][4],),(3,))])
        self.assertEqual(len(proxy.labels),32)
        proxy.observe(history.x[trial].copy(),chosen)
        self.assertEqual(len(proxy.labels),33)
        self.assertEqual(before["counts_before"],[4]*8)

    def test_reference_alignment_rejects_channel_capacity_and_endpoint_changes(self):
        sessions, original, arrival = reference_fixture()
        self.assertEqual(attach_references(sessions, original, arrival),0.)
        sessions, original, arrival = reference_fixture()
        sessions[0]["orders"][1]["trajectories"][0]["channels"] = [999]
        with self.assertRaisesRegex(ArithmeticError,"electrode"):
            attach_references(sessions,original,arrival)
        sessions, original, arrival = reference_fixture()
        sessions[0]["orders"][2]["panels"][POLICY]["fitted_models"][0]["rank"] = 2
        with self.assertRaisesRegex(ArithmeticError,"rank/alpha"):
            attach_references(sessions,original,arrival)
        sessions, original, arrival = reference_fixture()
        sessions[0]["orders"][3]["panels"][POLICY]["fitted_models"][0]["Q_common"] += 1e-4
        with self.assertRaisesRegex(ArithmeticError,"replay"):
            attach_references(sessions,original,arrival)


def reference_fixture():
    queues = [list(range(d*12,(d+1)*12)) for d in range(8)]
    rows, cells, histories = [], [], []
    for n in CHECKPOINTS:
        for e in ELECTRODE_COUNTS:
            scores = {"Q_common":.2,"Q_wrong_common":.1,"pairing_increment":.1,
                      "native_target_SSE":10.,"initial_SD_standardized_SSE":8.,"initial_mean_null_SSE":10.}
            cells.append({"training_trials":n,"electrodes":e,**scores})
            for seed in ([0] if e==96 else range(5)):
                rows.append({"training_trials":n,"electrodes":e,"seed":seed,"rank":1,"alpha":.1,**scores})
                if n==32:
                    histories.append({"electrodes":e,"seed":seed,"channels":list(range(e)),
                        "frozen_numeric_rank":1,"frozen_alpha":.1,"policy":"balanced","choices":[]})
    panel = {"cells":cells,"fitted_models":rows}
    original = {"status":"completed_two_trial_allocators_with_balanced_reference","held_session_accessed":False,"sessions":[]}
    arrival = {"status":"completed_three_order_robustness_comparison","held_session_accessed":False,"sessions":[]}
    sessions = []
    for name in DEVELOPMENT:
        animal = "Chewie" if name.startswith("Chewie") else "Mihili"
        prior = {"filename":name,"animal":animal,"panels":{p:copy.deepcopy(panel) for p in POLICIES},"trajectories":copy.deepcopy(histories)}
        original["sessions"].append(prior)
        arrival["sessions"].append({"filename":name,"animal":animal,"orders":[{"repetition":r,"queues":copy.deepcopy(queues),"panels":copy.deepcopy(prior["panels"]),"trajectories":copy.deepcopy(histories)} for r in range(3)]})
        sessions.append({"filename":name,"animal":animal,"orders":[{"order":r,"queues":copy.deepcopy(queues),"panels":{POLICY:copy.deepcopy(panel)},"trajectories":copy.deepcopy(histories)} for r in ORDERS]})
    return sessions,original,arrival


if __name__ == "__main__":
    unittest.main()
