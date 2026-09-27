from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import numpy as np

    from ep12_executor.contract_roles import RANK_RULE
    from ep12_executor.pre_null_materializer import (
        PreNullMaterializerError,
        _component_rank,
    )

    SCIENTIFIC_STACK_AVAILABLE = True
except ImportError:
    SCIENTIFIC_STACK_AVAILABLE = False


@unittest.skipUnless(SCIENTIFIC_STACK_AVAILABLE, "needs NumPy/SciPy/PyArrow")
class PreNullMaterializerTests(unittest.TestCase):
    def _fitted(self, *, representation: str, representation_rank: int):
        model = SimpleNamespace(
            prior_means=np.zeros((3, 5)),
            diagnostics={
                "representation": representation,
                "representation_rank": representation_rank,
            },
        )
        return SimpleNamespace(M=model)

    def test_rank_uses_the_fitted_representation_not_covariance_fallbacks(self) -> None:
        rank = _component_rank(
            rule=RANK_RULE,
            fitted=self._fitted(
                representation="low_rank_count_model", representation_rank=2
            ),
            source_neuron_count=20,
            configuration={"representation": "low_rank_count_model", "rank": 2},
        )
        self.assertEqual(rank, 2)

    def test_rank_diagnostic_must_match_the_frozen_rule(self) -> None:
        with self.assertRaises(PreNullMaterializerError):
            _component_rank(
                rule=RANK_RULE,
                fitted=self._fitted(
                    representation="low_rank_count_model", representation_rank=4
                ),
                source_neuron_count=20,
                configuration={"representation": "low_rank_count_model", "rank": 2},
            )


if __name__ == "__main__":
    unittest.main()
