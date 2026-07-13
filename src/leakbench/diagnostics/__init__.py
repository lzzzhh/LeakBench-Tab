"""Operational I/A/S/E diagnostics with oracle labels kept out of scoring."""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class OperationalFeatureMetadata:
    """Metadata observable by a deployment-time schema/lineage audit."""

    stable_id: str
    available_at_prediction: bool = True
    lifecycle: str = "prediction_time"
    group_id: Optional[str] = None
    outcome_descendant: bool = False
    post_event_table: bool = False


@dataclass(frozen=True)
class OperationalMetadata:
    """Blind metadata used by diagnostics and governance.

    Keys and graph endpoints are stable feature IDs. Display names and
    contamination labels are deliberately absent.
    """

    features: Dict[str, OperationalFeatureMetadata]
    graph_edges: Tuple[Tuple[str, str, float], ...] = field(default_factory=tuple)

    def validate(self, feature_ids: Sequence[str]) -> None:
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("feature_ids must be unique")
        missing = set(feature_ids) - set(self.features)
        if missing:
            raise ValueError("operational metadata missing stable IDs: " + ", ".join(sorted(missing)))
        active_ids = set(feature_ids)
        unknown = {
            endpoint
            for src, dst, _ in self.graph_edges
            for endpoint in (src, dst)
            if endpoint not in active_ids
        }
        if unknown:
            raise ValueError("graph references unknown stable IDs: " + ", ".join(sorted(unknown)))

    def groups(self, feature_ids: Sequence[str]) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for feature_id in feature_ids:
            group_id = self.features[feature_id].group_id
            if group_id is not None:
                result.setdefault(group_id, []).append(feature_id)
        return result


@dataclass(frozen=True)
class OracleMetadata:
    """Contamination identity used only to evaluate a completed blind run."""

    leakage_by_feature_id: Dict[str, bool]

    def leakage_mask(self, feature_ids: Sequence[str]) -> np.ndarray:
        missing = set(feature_ids) - set(self.leakage_by_feature_id)
        if missing:
            raise ValueError("oracle metadata missing stable IDs: " + ", ".join(sorted(missing)))
        return np.asarray([self.leakage_by_feature_id[fid] for fid in feature_ids], dtype=bool)


@dataclass
class DiagnosticScores:
    feature_ids: List[str]
    predictive_impact: np.ndarray
    availability_risk: np.ndarray
    structural_risk: np.ndarray
    environment_instability: np.ndarray
    composite: np.ndarray

    def detection_auprc(self, oracle: OracleMetadata) -> float:
        truth = oracle.leakage_mask(self.feature_ids).astype(int)
        if truth.sum() == 0:
            return 0.0
        return float(average_precision_score(truth, self.composite))

    def top_k_recall(self, oracle: OracleMetadata, k: int = 5) -> float:
        truth = oracle.leakage_mask(self.feature_ids)
        if truth.sum() == 0:
            return 1.0
        order = np.argsort(self.composite, kind="stable")[::-1][:k]
        return float(truth[order].sum() / truth.sum())

    def legitimate_fpr(self, oracle: OracleMetadata, k: int = 5) -> float:
        truth = oracle.leakage_mask(self.feature_ids)
        order = np.argsort(self.composite, kind="stable")[::-1][:k]
        n_legitimate = int((~truth).sum())
        if n_legitimate == 0:
            return 0.0
        return float((~truth[order]).sum() / n_legitimate)

    def summary(self, oracle: OracleMetadata) -> Dict[str, float]:
        return {
            "detection_auprc": self.detection_auprc(oracle),
            "top5_recall": self.top_k_recall(oracle, 5),
            "legitimate_fpr_top5": self.legitimate_fpr(oracle, 5),
        }

    def values_by_id(self, field_name: str = "composite") -> Dict[str, float]:
        values = getattr(self, field_name)
        return {feature_id: float(values[i]) for i, feature_id in enumerate(self.feature_ids)}


def _seed_for_feature(feature_id: str) -> int:
    digest = hashlib.sha256(feature_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _normalise(values: np.ndarray, constant: float = 0.5) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    if len(values) == 0 or values.max() <= values.min():
        return np.full(len(values), constant, dtype=float)
    return (values - values.min()) / (values.max() - values.min())


def _single_feature_mi(values: np.ndarray, target: np.ndarray, feature_id: str) -> float:
    return float(
        mutual_info_regression(
            values.reshape(-1, 1),
            target,
            random_state=_seed_for_feature(feature_id),
        )[0]
    )


def compute_operational_diagnostics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_ids: Sequence[str],
    operational_metadata: OperationalMetadata,
    environment_ids: Optional[np.ndarray] = None,
) -> DiagnosticScores:
    """Score features using only blind statistical and operational inputs.

    Each feature is scored independently with a seed derived from its stable ID,
    so column permutation and display-name anonymisation cannot change its score.
    """

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    feature_ids = list(feature_ids)
    if X_train.ndim != 2 or X_train.shape[1] != len(feature_ids):
        raise ValueError("X_train columns must match feature_ids")
    if X_train.shape[0] != len(y_train):
        raise ValueError("X_train rows must match y_train")
    operational_metadata.validate(feature_ids)

    raw_impact = np.asarray(
        [_single_feature_mi(X_train[:, j], y_train, fid) for j, fid in enumerate(feature_ids)]
    )
    impact = _normalise(raw_impact)

    lifecycle_risk = {
        "prediction_time": 0.0,
        "pre_outcome": 0.0,
        "post_outcome": 1.0,
        "future_window": 1.0,
        "repayment": 0.8,
        "closed": 0.8,
        "subgroup_specific": 0.4,
        "entity_specific": 0.3,
        "source_specific": 0.3,
        "graph_component": 0.2,
    }
    availability = np.zeros(len(feature_ids), dtype=float)
    for j, feature_id in enumerate(feature_ids):
        meta = operational_metadata.features[feature_id]
        signals = [lifecycle_risk.get(meta.lifecycle, 0.5)]
        if not meta.available_at_prediction:
            signals.append(1.0)
        if meta.outcome_descendant:
            signals.append(1.0)
        if meta.post_event_table:
            signals.append(0.9)
        availability[j] = max(signals)

    weighted_degree = {feature_id: 0.0 for feature_id in feature_ids}
    for src, dst, weight in operational_metadata.graph_edges:
        weighted_degree[src] += abs(float(weight))
        weighted_degree[dst] += abs(float(weight))
    graph_risk = _normalise(np.asarray([weighted_degree[fid] for fid in feature_ids]), constant=0.0)

    groups = operational_metadata.groups(feature_ids)
    max_group_size = max([len(members) for members in groups.values()] or [1])
    group_risk = np.asarray(
        [
            (len(groups[operational_metadata.features[fid].group_id]) - 1) / max(1, max_group_size - 1)
            if operational_metadata.features[fid].group_id in groups
            else 0.0
            for fid in feature_ids
        ],
        dtype=float,
    )
    structural = np.maximum(graph_risk, group_risk)

    instability = np.zeros(len(feature_ids), dtype=float)
    if environment_ids is not None:
        environment_ids = np.asarray(environment_ids)
        if len(environment_ids) != X_train.shape[0]:
            raise ValueError("environment_ids rows must match X_train")
        environments = np.unique(environment_ids)
        if len(environments) >= 2:
            for j, feature_id in enumerate(feature_ids):
                environment_mi = []
                for environment in environments:
                    rows = environment_ids == environment
                    if rows.sum() > 10:
                        environment_mi.append(
                            _single_feature_mi(X_train[rows, j], y_train[rows], feature_id)
                        )
                if len(environment_mi) >= 2:
                    instability[j] = float(np.std(environment_mi))
            instability = _normalise(instability, constant=0.0)

    composite = 0.30 * impact + 0.30 * availability + 0.25 * structural + 0.15 * instability
    return DiagnosticScores(
        feature_ids=feature_ids,
        predictive_impact=impact,
        availability_risk=availability,
        structural_risk=structural,
        environment_instability=instability,
        composite=composite,
    )


# Backward-compatible public name with the corrected, blind signature.
compute_full_diagnostics = compute_operational_diagnostics
