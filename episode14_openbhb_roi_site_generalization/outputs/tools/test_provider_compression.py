"""Three focused checks for the new grouping, train-only fit and paired intervals."""
import time
import unittest
import numpy as np
import run_provider_compression as run


class ProviderCompressionTest(unittest.TestCase):
    def test_provider_partition_and_control_strata(self):
        names, groups = run.admitted_metadata()
        for family, variants in groups.items():
            for variant, indices in variants.items():
                self.assertEqual(sorted(indices.ravel().tolist()), list(range(476)))
                for group in indices:
                    if family == 'anatomical':
                        strata = [(names[i][:2], next(f for f in run.PROVIDER_FAMILIES if names[i].endswith('_'+f))) for i in group]
                        self.assertEqual(set(strata), {(h, f) for h in ('lh', 'rh') for f in run.PROVIDER_FAMILIES})
                    else:
                        locations = [names[i].split('_', 1)[0] for i in group]
                        self.assertEqual(len(set(locations)), 68)
        with self.assertRaises(ValueError):
            run.provider_groups(names[:-1], ['invalid'], run.PROVIDER_FAMILIES)

    def test_held_values_do_not_fit_scaling_or_pca(self):
        _, groups = run.admitted_metadata()
        rng = np.random.RandomState(7)
        fit, held = rng.normal(size=(170, 476)), rng.normal(size=(3, 476))
        deadline = time.monotonic()+120
        original = {a: z for a, z, _, _ in run.representation_pairs(fit, held, groups, deadline)}
        altered = {a: z for a, z, _, _ in run.representation_pairs(fit, held*100+50, groups, deadline)}
        self.assertEqual(set(original), set(run.arm_menu()))
        for arm in original:
            np.testing.assert_array_equal(original[arm], altered[arm])
        # A full orthonormal block basis preserves the common-metric ridge fit.
        z = run.base.Scale().fit(fit).transform(fit)
        basis = np.zeros((476, 476))
        column = 0
        for indices in groups['anatomical']['provider']:
            block, _ = run.leading_basis(z[:, indices], len(indices))
            basis[np.ix_(indices, np.arange(column, column+len(indices)))] = block
            column += len(indices)
        y = rng.normal(size=len(z))
        raw, _ = run.base.ridge_grid(z, y, z[:3], np.array([10.]))
        rotated, _ = run.base.ridge_grid(z@basis, y, z[:3]@basis, np.array([10.]))
        np.testing.assert_allclose(raw, rotated, atol=1e-10, rtol=1e-10)

    def test_shared_hierarchical_draws_preserve_paired_constant_difference(self):
        arms = list(run.arm_menu())
        values = {arm: {str(i): {'domain': str(i//2), 'age': 20.+i, 'loss': float(i+1)+number/10.}
                        for i in range(6)} for number, arm in enumerate(arms)}
        draws = run.joint_bootstrap(values, time.monotonic()+120)
        for number, arm in enumerate(arms):
            np.testing.assert_allclose(draws[arm]-draws['raw'], number/10., atol=1e-12)
        values[arms[-1]]['0']['domain'] = 'wrong'
        with self.assertRaises(ValueError):
            run.joint_bootstrap(values, time.monotonic()+120)


if __name__ == '__main__':
    run.base.compute_only()
    unittest.main()
