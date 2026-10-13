"""Analytic implementation checks; fixture constants are not NeuroCam estimates."""

import math
import unittest

from switched_line import SourceLinePole
from r1_noise import R1CurrentPort, R1NoiseLimits, R1NoiseReduction


class SourceLinePoleTests(unittest.TestCase):
    def advance(self, pole, *, duration=0.25, drive=1.0, gain=1.0, tau=0.5, charge=0.0):
        return pole.advance(
            duration_seconds=duration,
            effective_input_volts=drive,
            gain=gain,
            tau_seconds=tau,
            switch_charge_coulomb=charge,
        )

    def test_known_step_endpoint_and_aperture_integral(self):
        pole = SourceLinePole(initial_volts=0.0, capacitance_farad=2.0)
        result = self.advance(pole, duration=1.0, drive=3.0, gain=2.0, tau=0.5)
        self.assertAlmostEqual(result.end_volts, 6 * (1 - math.exp(-2)))
        self.assertAlmostEqual(result.mean_volts, 6 * (1 - 0.5 * (1 - math.exp(-2))))
        self.assertEqual(pole.volts, result.end_volts)

    def test_partition_invariance_and_weighted_aperture_mean(self):
        whole = SourceLinePole(initial_volts=-2.0, capacitance_farad=1.0)
        split = SourceLinePole(initial_volts=-2.0, capacitance_farad=1.0)
        expected = self.advance(whole, duration=0.8, drive=4.0, tau=0.3)
        first = self.advance(split, duration=0.2, drive=4.0, tau=0.3)
        second = self.advance(split, duration=0.6, drive=4.0, tau=0.3)
        self.assertAlmostEqual(second.end_volts, expected.end_volts)
        self.assertAlmostEqual((0.2 * first.mean_volts + 0.6 * second.mean_volts) / 0.8,
                               expected.mean_volts)

    def test_row_change_retains_state_and_discarded_settling_history(self):
        pole = SourceLinePole(initial_volts=0.0, capacitance_farad=1.0)
        first = self.advance(pole, duration=0.5, drive=2.0, tau=0.5)
        second = self.advance(pole, duration=0.5, drive=0.0, tau=0.25)
        self.assertEqual(second.post_switch_volts, first.end_volts)
        self.assertAlmostEqual(second.end_volts, first.end_volts * math.exp(-2))
        self.assertGreater(second.end_volts, 0.0)

    def test_charge_conservation_with_fixed_capacitance(self):
        pole = SourceLinePole(initial_volts=1.0, capacitance_farad=2.0)
        result = self.advance(pole, duration=0.25, drive=0.0, tau=0.5, charge=-6.0)
        self.assertEqual(2.0 * (result.post_switch_volts - 1.0), -6.0)
        self.assertAlmostEqual(result.end_volts, -2.0 * math.exp(-0.5))

    def test_steady_state(self):
        pole = SourceLinePole(initial_volts=6.0, capacitance_farad=1.0)
        result = self.advance(pole, drive=3.0, gain=2.0)
        self.assertEqual(result.post_switch_volts, 6.0)
        self.assertEqual(result.mean_volts, 6.0)
        self.assertEqual(result.end_volts, 6.0)

    def test_extreme_time_ratios_have_finite_limits(self):
        tiny_aperture = SourceLinePole(initial_volts=0.0, capacitance_farad=1.0)
        tiny = self.advance(tiny_aperture, duration=1e-12, tau=1.0, drive=1.0)
        self.assertAlmostEqual(tiny.mean_volts / 5e-13, 1.0, places=11)
        short = SourceLinePole(initial_volts=2.0, capacitance_farad=1.0)
        result = self.advance(short, duration=1e-300, tau=1e300, drive=5.0)
        self.assertEqual(result.end_volts, 2.0)
        self.assertEqual(result.mean_volts, 2.0)
        long = SourceLinePole(initial_volts=2.0, capacitance_farad=1.0)
        result = self.advance(long, duration=1e300, tau=1e-300, drive=5.0)
        self.assertEqual(result.end_volts, 5.0)
        self.assertEqual(result.mean_volts, 5.0)

    def test_invalid_inputs_do_not_mutate_state(self):
        pole = SourceLinePole(initial_volts=2.0, capacitance_farad=1.0)
        for kwargs in ({"duration": 0.0}, {"duration": -1.0}, {"tau": 0.0},
                       {"drive": math.nan}, {"gain": math.inf}, {"charge": math.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.advance(pole, **kwargs)
            self.assertEqual(pole.volts, 2.0)
        for value in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(capacitance=value), self.assertRaises(ValueError):
                SourceLinePole(initial_volts=0.0, capacitance_farad=value)

    def noise_reduction(self, *, state="source:0", reference="ref:0", ports=None,
                        max_ports=3, max_bytes=72):
        if ports is None:
            ports = tuple(R1CurrentPort(f"i:{index}", coefficient,
                                       "explicit_conditional_current", f"component:{index}")
                          for index, coefficient in enumerate((1.0, -2.0, 0.0)))
        return R1NoiseReduction(state, reference, "analytic_fixture_reduction",
                                ports, R1NoiseLimits(max_ports, max_bytes))

    def noisy_pole(self):
        return SourceLinePole(initial_volts=1.0, capacitance_farad=2.0,
                              state_id="source:0", voltage_reference_id="ref:0")

    def noisy_advance(self, pole, sample, *, duration=0.4, charge=1.0, tau=0.5):
        return pole.advance(duration_seconds=duration, effective_input_volts=3.0,
                            gain=1.0, tau_seconds=tau, switch_charge_coulomb=charge,
                            noise_sample=sample)

    def test_held_current_exact_state_and_signed_original_port_work(self):
        sample = self.noise_reduction().sample_from_values(
            (0.2, -0.1, 0.7), port_voltage_offsets_volts=(-5.0, -0.2, 2.0))
        result = self.noisy_advance(self.noisy_pole(), sample)
        # Independent affine solution: Ieq=.4 A, target=3+.5*.4/2=3.1 V.
        post, target, duration, tau = 1.5, 3.1, 0.4, 0.5
        expected_end = target + (post - target) * math.exp(-duration / tau)
        expected_mean = target + (post - target) * tau / duration * (1 - math.exp(-duration / tau))
        self.assertAlmostEqual(sample.equivalent_current_amp, 0.4)
        self.assertEqual(result.post_switch_volts, post)
        self.assertAlmostEqual(result.end_volts, expected_end)
        self.assertAlmostEqual(result.mean_volts, expected_mean)
        expected_work = (
            duration * 0.2 * (expected_mean - 5.0),
            duration * -0.1 * (-2.0 * expected_mean - 0.2),
            duration * 0.7 * 2.0,
        )
        work = result.noise_work
        self.assertEqual(tuple(key for key, _ in work.port_work_joule), ("i:0", "i:1", "i:2"))
        for (_, actual), expected in zip(work.port_work_joule, expected_work):
            self.assertAlmostEqual(actual, expected)
        self.assertLess(work.port_work_joule[0][1], 0.0)
        self.assertGreater(work.port_work_joule[2][1], 0.0)  # b=0 does not erase this port.
        self.assertAlmostEqual(work.noise_current_source_interval_work_joule, sum(expected_work))
        self.assertAlmostEqual(work.equivalent_nodal_noise_interval_work_joule,
                               duration * 0.4 * expected_mean)
        self.assertAlmostEqual(work.known_port_voltage_offset_work_joule,
                               duration * (0.2 * -5.0 + -0.1 * -0.2 + 0.7 * 2.0))
        self.assertAlmostEqual(work.noise_current_source_interval_work_joule,
                               work.equivalent_nodal_noise_interval_work_joule
                               + work.known_port_voltage_offset_work_joule)

    def test_same_owned_draw_preserves_partition_state_aperture_and_work(self):
        currents, offsets = [0.2, -0.1, 0.7], [-5.0, -0.2, 2.0]
        sample = self.noise_reduction().sample_from_values(
            currents, port_voltage_offsets_volts=offsets)
        original = sample.numeric_values
        currents[:] = [100.0] * 3
        offsets[:] = [100.0] * 3
        self.assertEqual(sample.numeric_values, original)
        self.assertEqual(sample.reduction.retained_numeric_bytes, 72)
        expected = self.noisy_advance(self.noisy_pole(), sample, duration=0.8)
        split = self.noisy_pole()
        first = self.noisy_advance(split, sample, duration=0.2)
        second = self.noisy_advance(split, sample, duration=0.6, charge=0.0)
        self.assertAlmostEqual(second.end_volts, expected.end_volts)
        self.assertAlmostEqual((0.2 * first.mean_volts + 0.6 * second.mean_volts) / 0.8,
                               expected.mean_volts)
        for (_, total), (_, left), (_, right) in zip(
                expected.noise_work.port_work_joule, first.noise_work.port_work_joule,
                second.noise_work.port_work_joule):
            self.assertAlmostEqual(left + right, total)
        for name in ("noise_current_source_interval_work_joule",
                     "equivalent_nodal_noise_interval_work_joule",
                     "known_port_voltage_offset_work_joule"):
            self.assertAlmostEqual(getattr(first.noise_work, name) + getattr(second.noise_work, name),
                                   getattr(expected.noise_work, name))

    def test_noise_refusals_are_bounded_atomic_and_preserve_legacy_none(self):
        reduction = self.noise_reduction()
        sample = reduction.sample_from_values((0.2, -0.1, 0.7),
                                             port_voltage_offsets_volts=(-5.0, -0.2, 2.0))
        for state, reference in (("wrong", "ref:0"), ("source:0", "wrong"), (None, None)):
            pole = SourceLinePole(initial_volts=1.0, capacitance_farad=2.0,
                                  state_id=state, voltage_reference_id=reference)
            with self.subTest(state=state, reference=reference), self.assertRaises(ValueError):
                self.noisy_advance(pole, sample)
            self.assertEqual(pole.volts, 1.0)
        for bad in (math.nan, math.inf, 1 + 2j, True):
            with self.subTest(amplitude=bad), self.assertRaises(ValueError):
                reduction.sample_from_values((bad, 0.0, 0.0),
                                             port_voltage_offsets_volts=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            reduction.sample_from_values((0.0, 0.0), port_voltage_offsets_volts=(0.0,) * 3)
        with self.assertRaises(ValueError):
            self.noise_reduction(max_bytes=71)
        with self.assertRaises(ValueError):
            self.noise_reduction(ports=(R1CurrentPort("a", 1, "conditional", "same"),
                                        R1CurrentPort("b", 1, "conditional", "same")))
        consumed = []
        def unbounded_ports():
            while True:
                consumed.append(1)
                yield R1CurrentPort("a", 1, "conditional", "one")
        with self.assertRaises(ValueError):
            self.noise_reduction(ports=unbounded_ports(), max_ports=2)
        self.assertEqual(len(consumed), 3)  # One excess item, never unbounded tuple().
        # Projected-zero port still has physical work. Its overflow must not
        # commit even the otherwise valid boundary charge/state evolution.
        zero_port = self.noise_reduction(ports=(R1CurrentPort("zero", 0, "conditional", "zero"),))
        overflow = zero_port.sample_from_values((1e308,), port_voltage_offsets_volts=(1.0,))
        pole = self.noisy_pole()
        with self.assertRaises(ValueError):
            self.noisy_advance(pole, overflow, duration=2.0, charge=4.0)
        self.assertEqual(pole.volts, 1.0)
        # A target-shift overflow also refuses atomically, without cancellation
        # through an infinite intermediate or a falsely finite final state.
        one_port = self.noise_reduction(ports=(R1CurrentPort("one", 1, "conditional", "one"),))
        overflow_shift = one_port.sample_from_values((1e308,), port_voltage_offsets_volts=(0.0,))
        with self.assertRaises(ValueError):
            self.noisy_advance(pole, overflow_shift, tau=4.0)
        self.assertEqual(pole.volts, 1.0)
        default = SourceLinePole(initial_volts=2.0, capacitance_farad=1.0)
        explicit_none = SourceLinePole(initial_volts=2.0, capacitance_farad=1.0)
        left = self.advance(default, duration=1e-300, tau=1e300, drive=5.0)
        right = explicit_none.advance(duration_seconds=1e-300, effective_input_volts=5.0,
                                      gain=1.0, tau_seconds=1e300,
                                      switch_charge_coulomb=0.0, noise_sample=None)
        self.assertEqual(left, right)
        self.assertIsNone(right.noise_work)


if __name__ == "__main__":
    unittest.main()
