"""benchmark_v2/information_regimes/controllers.py — O0/O1/O2 controllers."""
from __future__ import annotations
import numpy as np
from benchmark_v2.core.models import DetectorInput, InformationRegime, FeatureAvailability, FeatureSpec

class O0Controller:
    @staticmethod
    def build(task):
        n = len([f for f in task.feature_specs if f.role.value == "predictor"])
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        anon_ids = tuple(f"a{i:03d}" for i in range(n))
        X_shuffled = task.X[:, perm]
        id_map = {anon_ids[i]: task.feature_specs[perm[i]].feature_id for i in range(n)}
        return DetectorInput(regime=InformationRegime.O0, feature_ids=anon_ids,
            feature_names=tuple("" for _ in range(n)), feature_dtypes=tuple("float64" for _ in range(n)),
            train_X=X_shuffled.copy(), train_y=task.y.copy()), id_map

class O1Controller:
    @staticmethod
    def build(task, naming="natural"):
        n = len([f for f in task.feature_specs if f.role.value == "predictor"])
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        anon_ids = tuple(f"a{i:03d}" for i in range(n))
        X_shuffled = task.X[:, perm]
        id_map = {anon_ids[i]: task.feature_specs[perm[i]].feature_id for i in range(n)}
        names = tuple(task.feature_specs[perm[i]].name_pool.get(naming, f"var_{i:03d}") for i in range(n))
        return DetectorInput(regime=InformationRegime.O1, feature_ids=anon_ids,
            feature_names=names, feature_dtypes=tuple("float64" for _ in range(n)),
            train_X=X_shuffled.copy(), train_y=task.y.copy()), id_map

class O2Controller:
    @staticmethod
    def build(task, naming="natural"):
        di, id_map = O1Controller.build(task, naming)
        n = len(di.feature_ids)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        avails = []
        for i in range(n):
            orig = task.availability[perm[i]]
            avails.append(FeatureAvailability(feature_id=di.feature_ids[i],
                available_at_prediction=orig.available_at_prediction,
                source_table=orig.source_table))
        return DetectorInput(regime=InformationRegime.O2, feature_ids=di.feature_ids,
            feature_names=di.feature_names, feature_dtypes=di.feature_dtypes,
            train_X=di.train_X, train_y=di.train_y, feature_availability=tuple(avails)), id_map
