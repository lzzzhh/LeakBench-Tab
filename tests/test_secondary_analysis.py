import numpy as np

from scripts.analyze_secondary_v2 import hierarchical_bootstrap


def test_secondary_bootstrap_preserves_trailing_paired_axes():
    matrix = np.arange(3 * 2 * 4 * 5, dtype=float).reshape(3, 2, 4, 5)
    result = hierarchical_bootstrap(matrix, repetitions=17, seed=13)
    assert result.shape == (17, 4, 5)
    assert np.isfinite(result).all()
