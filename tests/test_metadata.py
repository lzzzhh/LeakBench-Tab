import numpy as np

from src.leakbench.diagnostics import (
    OperationalFeatureMetadata,
    OperationalMetadata,
    compute_operational_diagnostics,
)


def test_all_operational_axes_are_computed_without_oracle_labels():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(float)
    ids = [f"fid-{i}" for i in range(4)]
    metadata = OperationalMetadata(
        features={
            ids[0]: OperationalFeatureMetadata(stable_id=ids[0]),
            ids[1]: OperationalFeatureMetadata(stable_id=ids[1], available_at_prediction=False),
            ids[2]: OperationalFeatureMetadata(stable_id=ids[2], group_id="bundle"),
            ids[3]: OperationalFeatureMetadata(stable_id=ids[3], group_id="bundle"),
        },
        graph_edges=((ids[2], ids[3], 1.0),),
    )
    scores = compute_operational_diagnostics(X, y, ids, metadata)
    for values in (
        scores.predictive_impact,
        scores.availability_risk,
        scores.structural_risk,
        scores.environment_instability,
        scores.composite,
    ):
        assert values.shape == (4,)
        assert np.isfinite(values).all()
    assert scores.availability_risk[1] > scores.availability_risk[0]
    assert scores.structural_risk[2] == scores.structural_risk[3]
