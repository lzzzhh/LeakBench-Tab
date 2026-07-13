import numpy as np

from experiments.leakbench.run_diagnostic_suite import (
    METHODS,
    compute_blind_scores,
    evaluate_localization,
)


def test_diagnostic_suite_is_blind_and_localizes_direct_copy():
    rng = np.random.RandomState(7)
    y = rng.binomial(1, 0.5, size=500)
    X = rng.normal(size=(500, 5))
    X = np.column_stack([X, y])
    train = np.arange(350)
    validation = np.arange(350, 500)
    scores = compute_blind_scores(X[train], y[train], X[validation], y[validation], seed=13)
    assert set(scores) == set(METHODS)
    mask = np.array([False] * 5 + [True])
    for method, values in scores.items():
        assert values.shape == (6,), method
        assert np.isfinite(values).all(), method
        ap, normalized_ap, top5 = evaluate_localization(values, mask)
        assert ap > 0.5, method
        assert normalized_ap > 0.4, method
        assert top5 == 1.0, method


def test_localization_evaluation_rejects_missing_oracle_positive():
    try:
        evaluate_localization(np.array([0.1, 0.2]), np.array([False, False]))
    except ValueError as error:
        assert "positive feature" in str(error)
    else:
        raise AssertionError("empty oracle mask was accepted")
