"""Construction-valid contamination mechanisms for LeakBench-Tab.

The injector keeps prediction-time invalidity (ground truth) separate from the
statistical strength of a generated feature.  Structured mechanisms receive and
return the time/entity/source metadata needed to audit their semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MechanismID(str, Enum):
    DIRECT_COPY = "M01_direct_copy"
    NOISY_PROXY = "M02_noisy_proxy"
    NONLINEAR = "M03_nonlinear"
    POST_OUTCOME = "M04_post_outcome"
    TEMPORAL_LEAK = "M05_temporal_leak"
    REDUNDANT_CLUSTER = "M06_redundant_cluster"
    SPARSE_SUBGROUP = "M07_sparse_subgroup"
    ENTITY_LEAK = "M08_entity_leak"
    SOURCE_LEAK = "M09_source_leak"
    MIXED = "M10_mixed"
    GRAPH_MEDIATED = "M11_graph_mediated"


@dataclass
class MechanismConfig:
    mechanism: MechanismID
    n_leakage_features: int = 1
    strength: float = 1.0
    noise_std: float = 0.05
    redundancy: int = 1
    coverage: float = 1.0
    nonlinearity: float = 0.0
    time_offset: float = 0.0
    source_overlap: float = 0.0
    subgroup_prevalence: float = 0.5
    seed: int = 42
    n_entities: int = 50
    n_sources: int = 8
    min_group_count: int = 2
    prior_strength: float = 5.0


@dataclass
class InjectedTask:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    leakage_mask: np.ndarray
    legitimate_mask: np.ndarray
    mechanism_labels: list[str]
    mechanism_params: list[dict]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    n_original: int
    n_injected: int
    timestamps: np.ndarray
    entity_ids: np.ndarray
    source_ids: np.ndarray
    feature_availability: dict[str, bool]
    groups: dict[str, list[str]] = field(default_factory=dict)
    graph_edges: list[tuple[str, str, float]] = field(default_factory=list)
    sample_metadata: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class _InjectedBlock:
    X: np.ndarray
    names: list[str]
    labels: list[str]
    params: list[dict]
    available_at_prediction: list[bool]
    groups: dict[str, list[str]] = field(default_factory=dict)
    graph_edges: list[tuple[str, str, float]] = field(default_factory=list)
    sample_metadata: dict[str, np.ndarray] = field(default_factory=dict)


class LeakBenchInjector:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def inject(
        self,
        X,
        y,
        configs,
        feature_names=None,
        split_type="auto",
        timestamps=None,
        entity_ids=None,
        source_ids=None,
    ):
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D and have the same number of rows as y")
        n_samples, n_orig = X.shape
        if feature_names is None:
            feature_names = [f"clean_{i}" for i in range(n_orig)]
        if len(feature_names) != n_orig:
            raise ValueError("feature_names must match the original feature count")

        timestamps = np.arange(n_samples, dtype=float) if timestamps is None else np.asarray(timestamps)
        if timestamps.shape != (n_samples,):
            raise ValueError("timestamps must have exactly one value per row")
        input_entities = None if entity_ids is None else np.asarray(entity_ids)
        input_sources = None if source_ids is None else np.asarray(source_ids)
        for name, values in (("entity_ids", input_entities), ("source_ids", input_sources)):
            if values is not None and values.shape != (n_samples,):
                raise ValueError(f"{name} must have exactly one value per row")

        blocks = [self._inject(cfg, X, y, timestamps, input_entities, input_sources) for cfg in configs]
        X_leak = np.column_stack([block.X for block in blocks]) if blocks else np.empty((n_samples, 0))
        X_full = np.column_stack([X, X_leak]).astype(np.float32)
        injected_names = [name for block in blocks for name in block.names]
        injected_labels = [label for block in blocks for label in block.labels]
        injected_params = [params for block in blocks for params in block.params]
        all_names = list(feature_names) + injected_names
        all_labels = [""] * n_orig + injected_labels
        all_params = [{}] * n_orig + injected_params

        leak_mask = np.zeros(X_full.shape[1], dtype=bool)
        for offset, label in enumerate(injected_labels):
            leak_mask[n_orig + offset] = label != "legitimate"

        availability = {name: True for name in feature_names}
        groups: dict[str, list[str]] = {}
        graph_edges: list[tuple[str, str, float]] = []
        sample_metadata: dict[str, np.ndarray] = {}
        final_entities = input_entities
        final_sources = input_sources
        for block in blocks:
            availability.update(dict(zip(block.names, block.available_at_prediction)))
            groups.update(block.groups)
            graph_edges.extend(block.graph_edges)
            sample_metadata.update(block.sample_metadata)
            if "entity_ids" in block.sample_metadata:
                final_entities = block.sample_metadata["entity_ids"]
            if "source_ids" in block.sample_metadata:
                final_sources = block.sample_metadata["source_ids"]

        requires_time = any(
            cfg.mechanism in {MechanismID.POST_OUTCOME, MechanismID.TEMPORAL_LEAK, MechanismID.ENTITY_LEAK}
            for cfg in configs
        )
        if split_type == "auto":
            split_type = "time" if requires_time else "random"
        if split_type == "time":
            order = np.argsort(timestamps, kind="stable")
        elif split_type == "random":
            order = self.rng.permutation(n_samples)
        else:
            raise ValueError("split_type must be one of: auto, random, time")
        train_end, val_end = int(n_samples * 0.6), int(n_samples * 0.8)

        return InjectedTask(
            X=X_full,
            y=y,
            feature_names=all_names,
            leakage_mask=leak_mask,
            legitimate_mask=~leak_mask,
            mechanism_labels=all_labels,
            mechanism_params=all_params,
            train_idx=order[:train_end],
            val_idx=order[train_end:val_end],
            test_idx=order[val_end:],
            n_original=n_orig,
            n_injected=X_leak.shape[1],
            timestamps=timestamps,
            entity_ids=np.full(n_samples, -1) if final_entities is None else np.asarray(final_entities),
            source_ids=np.full(n_samples, -1) if final_sources is None else np.asarray(final_sources),
            feature_availability=availability,
            groups=groups,
            graph_edges=graph_edges,
            sample_metadata=sample_metadata,
        )

    def _inject(self, cfg, X, y, timestamps, entity_ids, source_ids):
        methods = {
            MechanismID.DIRECT_COPY: self._m01,
            MechanismID.NOISY_PROXY: self._m02,
            MechanismID.NONLINEAR: self._m03,
            MechanismID.POST_OUTCOME: self._m04,
            MechanismID.TEMPORAL_LEAK: self._m05,
            MechanismID.REDUNDANT_CLUSTER: self._m06,
            MechanismID.SPARSE_SUBGROUP: self._m07,
            MechanismID.ENTITY_LEAK: self._m08,
            MechanismID.SOURCE_LEAK: self._m09,
            MechanismID.MIXED: self._m10,
            MechanismID.GRAPH_MEDIATED: self._m11,
        }
        return methods[cfg.mechanism](cfg, X, y, timestamps, entity_ids, source_ids)

    @staticmethod
    def _loo_prior(y):
        if len(y) <= 1:
            return np.full(len(y), 0.5, dtype=float)
        return (float(np.sum(y)) - y) / (len(y) - 1)

    @staticmethod
    def _block(X, names, labels, params, available=None, groups=None, graph_edges=None, metadata=None):
        if available is None:
            available = [False] * len(names)
        return _InjectedBlock(
            X=np.asarray(X, dtype=np.float32),
            names=names,
            labels=labels,
            params=params,
            available_at_prediction=available,
            groups={} if groups is None else groups,
            graph_edges=[] if graph_edges is None else graph_edges,
            sample_metadata={} if metadata is None else metadata,
        )

    def _future_window(self, y, timestamps, skip, width):
        order = np.argsort(timestamps, kind="stable")
        ordered_time = timestamps[order]
        ordered_y = y[order]
        cumulative = np.concatenate([[0.0], np.cumsum(ordered_y, dtype=float)])
        prior = self._loo_prior(y)
        result = prior.copy()
        counts = np.zeros(len(y), dtype=int)
        for row in range(len(y)):
            first_strict_future = np.searchsorted(ordered_time, timestamps[row], side="right")
            start = first_strict_future + max(0, skip - 1)
            end = min(len(y), start + width)
            if start < end:
                result[row] = (cumulative[end] - cumulative[start]) / (end - start)
                counts[row] = end - start
        return result, counts

    def _m01(self, c, X, y, timestamps, entity_ids, source_ids):
        strength = float(np.clip(c.strength, 0.0, 1.0))
        flips = self.rng.rand(len(y)) < 0.5 * (1.0 - strength)
        signal = np.where(flips, 1.0 - y, y).reshape(-1, 1)
        return self._block(signal, ["contam_M01_0"], ["M01"], [{"strength": strength, "flip_rate": 0.5 * (1.0 - strength)}])

    def _m02(self, c, X, y, timestamps, entity_ids, source_ids):
        prior = self._loo_prior(y)
        signal = c.strength * (y - prior) + self.rng.normal(0.0, c.noise_std, len(y))
        return self._block(signal.reshape(-1, 1), ["contam_M02_0"], ["M02"], [{"strength": c.strength, "noise_std": c.noise_std}])

    def _m03(self, c, X, y, timestamps, entity_ids, source_ids):
        threshold = float(np.median(X[:, 0]))
        side = np.where(X[:, 0] >= threshold, 1.0, -1.0)
        signal = c.strength * (2.0 * y - 1.0) * side + self.rng.normal(0.0, c.noise_std, len(y))
        return self._block(
            signal.reshape(-1, 1),
            ["contam_M03_0"],
            ["M03"],
            [{"strength": c.strength, "interaction_with": "clean_0", "threshold": threshold}],
        )

    def _m04(self, c, X, y, timestamps, entity_ids, source_ids):
        width = max(2, int(len(y) * 0.05))
        future, counts = self._future_window(y, timestamps, skip=1, width=width)
        prior = self._loo_prior(y)
        signal = prior + c.strength * (future - prior) + self.rng.normal(0.0, c.noise_std, len(y))
        return self._block(
            signal.reshape(-1, 1),
            ["contam_M04_0"],
            ["M04"],
            [{"strength": c.strength, "window": width, "excludes_current": True, "strictly_future": True}],
            metadata={"m04_future_count": counts},
        )

    def _m05(self, c, X, y, timestamps, entity_ids, source_ids):
        skip = max(1, int(len(y) * c.time_offset)) if c.time_offset > 0 else max(1, int(len(y) * 0.05))
        width = max(2, skip)
        future, counts = self._future_window(y, timestamps, skip=skip, width=width)
        prior = self._loo_prior(y)
        signal = prior + c.strength * (future - prior) + self.rng.normal(0.0, c.noise_std, len(y))
        return self._block(
            signal.reshape(-1, 1),
            ["contam_M05_0"],
            ["M05"],
            [{"strength": c.strength, "look_ahead_rows": skip, "excludes_current": True, "strictly_future": True}],
            metadata={"m05_future_count": counts},
        )

    def _m06(self, c, X, y, timestamps, entity_ids, source_ids):
        prior = self._loo_prior(y)
        count = max(1, int(c.redundancy))
        columns = []
        names = []
        for index in range(count):
            noise = c.noise_std * (1.0 + index / count)
            columns.append(c.strength * (y - prior) + self.rng.normal(0.0, noise, len(y)))
            names.append(f"contam_M06_r{index}")
        params = [{"strength": c.strength, "redundancy_index": index, "redundancy": count} for index in range(count)]
        return self._block(np.column_stack(columns), names, ["M06"] * count, params, groups={"M06_cluster": names})

    def _m07(self, c, X, y, timestamps, entity_ids, source_ids):
        in_subgroup = self.rng.rand(len(y)) < c.subgroup_prevalence
        prior = self._loo_prior(y)
        noise = self.rng.normal(0.0, c.noise_std, len(y))
        signal = np.where(in_subgroup, c.strength * (y - prior) + noise, 0.0)
        return self._block(
            signal.reshape(-1, 1),
            ["contam_M07_0"],
            ["M07"],
            [{"strength": c.strength, "coverage": float(in_subgroup.mean())}],
            metadata={"m07_covered": in_subgroup},
        )

    def _m08(self, c, X, y, timestamps, entity_ids, source_ids):
        n = len(y)
        if entity_ids is None:
            n_entities = min(max(2, int(c.n_entities)), max(2, n // 20))
            entity_ids = np.arange(n) % n_entities
            self.rng.shuffle(entity_ids)
        else:
            entity_ids = np.asarray(entity_ids)
            n_entities = len(np.unique(entity_ids))

        prior = self._loo_prior(y)
        future_rate = prior.copy()
        future_count = np.zeros(n, dtype=int)
        prior_weight = max(0.0, float(c.prior_strength))
        for entity in np.unique(entity_ids):
            rows = np.flatnonzero(entity_ids == entity)
            ordered = rows[np.argsort(timestamps[rows], kind="stable")]
            ordered_time = timestamps[ordered]
            ordered_y = y[ordered]
            cumulative = np.concatenate([[0.0], np.cumsum(ordered_y, dtype=float)])
            for row in ordered:
                start = np.searchsorted(ordered_time, timestamps[row], side="right")
                count = len(ordered) - start
                future_count[row] = count
                if count:
                    total = cumulative[-1] - cumulative[start]
                    future_rate[row] = (total + prior_weight * prior[row]) / (count + prior_weight)

        strength = float(np.clip(c.strength, 0.0, 1.0))
        signal = prior + strength * (future_rate - prior) + self.rng.normal(0.0, c.noise_std, n)
        params = [{
            "strength": strength,
            "n_entities": int(n_entities),
            "aggregation": "strict_future_entity_shrinkage_mean",
            "prior_weight": prior_weight,
            "excludes_current": True,
            "strictly_future": True,
        }]
        return self._block(
            signal.reshape(-1, 1),
            ["contam_M08_future_entity_rate"],
            ["M08"],
            params,
            metadata={"entity_ids": entity_ids, "m08_future_count": future_count, "m08_future_rate": future_rate},
        )

    def _m09(self, c, X, y, timestamps, entity_ids, source_ids):
        n_sources = max(4, int(c.n_sources))
        strength = float(np.clip(c.strength, 0.0, 1.0))
        separation = 3.0 * strength
        delta = np.linspace(-1.0, 1.0, n_sources)
        positive_prob = np.exp(separation * delta)
        positive_prob /= positive_prob.sum()
        negative_prob = np.exp(-separation * delta)
        negative_prob /= negative_prob.sum()

        generated = np.empty(len(y), dtype=int)
        for label, probabilities in ((0.0, negative_prob), (1.0, positive_prob)):
            rows = np.flatnonzero(y == label)
            required = n_sources * c.min_group_count
            if len(rows) < required:
                raise ValueError("M09 invalid cell: insufficient samples per source and class")
            assignments = np.repeat(np.arange(n_sources), c.min_group_count)
            remaining = len(rows) - required
            if remaining:
                assignments = np.concatenate(
                    [assignments, self.rng.choice(n_sources, size=remaining, p=probabilities)]
                )
            self.rng.shuffle(assignments)
            generated[rows] = assignments

        label_permutation = self.rng.permutation(n_sources)
        generated = label_permutation[generated]
        one_hot = np.eye(n_sources, dtype=np.float32)[generated]
        names = [f"contam_M09_source_{index}" for index in range(n_sources)]
        midpoint = 0.5 * (positive_prob + negative_prob)
        js = 0.5 * np.sum(positive_prob * np.log((positive_prob + 1e-12) / (midpoint + 1e-12)))
        js += 0.5 * np.sum(negative_prob * np.log((negative_prob + 1e-12) / (midpoint + 1e-12)))
        common = {
            "strength": strength,
            "n_sources": n_sources,
            "encoding": "one_hot",
            "outcome_dependent_assignment": True,
            "uses_target_rate_encoding": False,
            "label_permutation": label_permutation.tolist(),
            "js_divergence": float(js),
        }
        return self._block(
            one_hot,
            names,
            ["M09"] * n_sources,
            [common.copy() for _ in names],
            groups={"M09_source_one_hot": names},
            metadata={"source_ids": generated},
        )

    def _m10(self, c, X, y, timestamps, entity_ids, source_ids):
        legitimate = X[:, 0].astype(np.float32, copy=True)
        prior = self._loo_prior(y)
        contamination = c.strength * (y - prior) + self.rng.normal(0.0, c.noise_std, len(y))
        names = ["mixed_legitimate_clean_0", "contam_M10_target_proxy"]
        return self._block(
            np.column_stack([legitimate, contamination]),
            names,
            ["legitimate", "M10"],
            [
                {"type": "legitimate", "source_feature": "clean_0"},
                {"type": "contamination", "strength": c.strength, "noise_std": c.noise_std},
            ],
            available=[True, False],
            groups={"M10_mixed_group": names},
        )

    def _m11(self, c, X, y, timestamps, entity_ids, source_ids):
        count = max(2, int(c.n_leakage_features))
        prior = self._loo_prior(y)
        projections = self.rng.normal(size=count)
        block = (y - prior).reshape(-1, 1) * projections.reshape(1, -1) * c.strength
        block += self.rng.normal(0.0, c.noise_std, block.shape)
        names = [f"contam_M11_{index}" for index in range(count)]
        edges = [
            (names[left], names[right], 1.0)
            for left in range(count)
            for right in range(left + 1, count)
        ]
        return self._block(
            block,
            names,
            ["M11"] * count,
            [{"strength": c.strength, "n_components": count} for _ in names],
            groups={"M11_component": names},
            graph_edges=edges,
        )
