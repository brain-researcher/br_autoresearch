"""Identity/order/payload boundary using generated feature rows only."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from prepare_development_embeddings import copy_selected, identity_join


class GuardedRows:
    def __init__(self, data, allowed):
        self.data, self.allowed, self.read_rows = data, allowed, []

    def __getitem__(self, selection):
        rows = selection[0] if isinstance(selection, tuple) else selection
        if not set(rows).issubset(self.allowed) or not np.all(np.diff(rows) > 0):
            raise AssertionError('Nondevelopment or unordered value read')
        self.read_rows.extend(rows.tolist())
        return self.data[selection]


class DevelopmentEmbeddingTest(unittest.TestCase):
    def test_join_restores_exact_frozen_order_and_never_reads_excluded_values(self):
        source_ids = ['closed_a', 'dev_b', 'closed_b', 'dev_c', 'dev_a']
        mapping = identity_join(source_ids, ['dev_a', 'dev_b', 'dev_c'])
        np.testing.assert_array_equal(mapping, [4, 1, 3])
        values = np.asarray([[99, 99], [1, -2], [88, 88], [3, -4], [5, -6]], dtype=np.float16)
        features = GuardedRows(values, {1, 3, 4})
        valid = GuardedRows(np.asarray([False, True, False, True, True]), {1, 3, 4})
        out = np.empty((3, 2), dtype=np.float16)
        progress = {'completed_rows': 0, 'logical_selected_feature_bytes': 0}
        copy_selected(features, valid, mapping, out, progress, chunk_rows=2)
        self.assertEqual(out.tobytes(), np.asarray([[5, -6], [1, -2], [3, -4]], dtype=np.float16).tobytes())
        self.assertEqual(progress, {'completed_rows': 3, 'logical_selected_feature_bytes': 12})
        self.assertEqual(features.read_rows, [1, 3, 4])
        self.assertEqual(valid.read_rows, [1, 3, 4])
        for source in (source_ids[:-1], source_ids+['dev_a']):
            with self.assertRaises(ValueError):
                identity_join(source, ['dev_a', 'dev_b', 'dev_c'])


if __name__ == '__main__':
    unittest.main()
