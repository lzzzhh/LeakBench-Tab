"""Governance strategies over stable feature IDs and blind metadata."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.leakbench.diagnostics import OperationalMetadata, OracleMetadata


class GovernanceStrategy(str, Enum):
    NO_REMOVAL = "G1_no_removal"
    ORACLE_REMOVE_ALL = "G2_oracle_remove_all"
    FIXED_FIELD_BUDGET = "G3_fixed_field_budget"
    FIXED_GROUP_BUDGET = "G4_fixed_group_budget"
    SCORE_THRESHOLD = "G5_score_threshold"
    GRAPH_CUT = "G6_graph_cut"
    LIFECYCLE_REMOVAL = "G7_lifecycle_removal"
    CAPACITY_AWARE = "G8_capacity_aware"


class GovernanceStatus(str, Enum):
    APPLIED = "APPLIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class GovernanceResult:
    strategy: GovernanceStrategy
    feature_mask: np.ndarray
    status: GovernanceStatus = GovernanceStatus.APPLIED
    reason: str = ""
    quarantined_features: List[str] = field(default_factory=list)
    kept_features: List[str] = field(default_factory=list)
    n_quarantined: int = 0
    n_kept: int = 0
    review_units: int = 0
    strict_auc: float = 0.5
    oracle_gap: float = 0.0
    legitimate_retention: float = float("nan")
    false_quarantine_rate: float = float("nan")
    leakage_recall: float = float("nan")


def _budget_count(budget: float, n_units: int) -> int:
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must be between 0 and 1")
    if budget == 0.0 or n_units == 0:
        return 0
    return max(1, int(np.ceil(budget * n_units)))


def _connected_components(
    feature_ids: Sequence[str], graph_edges: Sequence[Tuple[str, str, float]]
) -> List[List[str]]:
    adjacency = {feature_id: set() for feature_id in feature_ids}
    graph_nodes = set()
    for src, dst, _ in graph_edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)
        graph_nodes.update((src, dst))

    components: List[List[str]] = []
    unseen = set(graph_nodes)
    order = {feature_id: i for i, feature_id in enumerate(feature_ids)}
    while unseen:
        start = min(unseen, key=order.get)
        stack = [start]
        component = []
        unseen.remove(start)
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = adjacency[node] & unseen
            unseen.difference_update(neighbours)
            stack.extend(sorted(neighbours, key=order.get, reverse=True))
        components.append(sorted(component, key=order.get))
    return components


def _result(
    strategy: GovernanceStrategy,
    feature_ids: Sequence[str],
    mask: np.ndarray,
    oracle_metadata: Optional[OracleMetadata],
    status: GovernanceStatus = GovernanceStatus.APPLIED,
    reason: str = "",
    review_units: int = 0,
) -> GovernanceResult:
    quarantined = [feature_ids[i] for i in np.where(mask < 0.5)[0]]
    kept = [feature_ids[i] for i in np.where(mask > 0.5)[0]]
    legitimate_retention = float("nan")
    false_quarantine_rate = float("nan")
    leakage_recall = float("nan")
    if oracle_metadata is not None and status == GovernanceStatus.APPLIED:
        truth = oracle_metadata.leakage_mask(feature_ids)
        quarantined_mask = mask < 0.5
        legitimate_retention = float((~truth & ~quarantined_mask).sum() / max(1, (~truth).sum()))
        false_quarantine_rate = float((~truth & quarantined_mask).sum() / max(1, quarantined_mask.sum()))
        leakage_recall = float((truth & quarantined_mask).sum() / max(1, truth.sum()))

    return GovernanceResult(
        strategy=strategy,
        feature_mask=mask,
        status=status,
        reason=reason,
        quarantined_features=quarantined,
        kept_features=kept,
        n_quarantined=len(quarantined),
        n_kept=len(kept),
        review_units=review_units,
        legitimate_retention=legitimate_retention,
        false_quarantine_rate=false_quarantine_rate,
        leakage_recall=leakage_recall,
    )


def apply_strategy(
    strategy: GovernanceStrategy,
    feature_ids: Sequence[str],
    diagnostic_scores: Sequence[float],
    operational_metadata: OperationalMetadata,
    oracle_metadata: Optional[OracleMetadata] = None,
    budget: float = 0.10,
    threshold: float = 0.5,
) -> GovernanceResult:
    """Apply a policy using blind metadata; oracle labels only score its result."""

    strategy = GovernanceStrategy(strategy)
    feature_ids = list(feature_ids)
    scores = np.asarray(diagnostic_scores, dtype=float)
    if len(scores) != len(feature_ids):
        raise ValueError("diagnostic_scores must match feature_ids")
    operational_metadata.validate(feature_ids)
    n_features = len(feature_ids)
    mask = np.ones(n_features, dtype=float)

    if strategy == GovernanceStrategy.NO_REMOVAL:
        return _result(strategy, feature_ids, mask, oracle_metadata)

    if strategy == GovernanceStrategy.ORACLE_REMOVE_ALL:
        if oracle_metadata is None:
            return _result(
                strategy,
                feature_ids,
                mask,
                oracle_metadata,
                GovernanceStatus.NOT_APPLICABLE,
                "oracle metadata was not supplied",
            )
        mask = (~oracle_metadata.leakage_mask(feature_ids)).astype(float)
        return _result(strategy, feature_ids, mask, oracle_metadata, review_units=int((mask < 0.5).sum()))

    if strategy in (GovernanceStrategy.FIXED_FIELD_BUDGET, GovernanceStrategy.CAPACITY_AWARE):
        n_remove = _budget_count(budget, n_features)
        order = np.argsort(scores, kind="stable")[::-1][:n_remove]
        mask[order] = 0.0
        return _result(strategy, feature_ids, mask, oracle_metadata, review_units=n_remove)

    if strategy == GovernanceStrategy.FIXED_GROUP_BUDGET:
        groups = operational_metadata.groups(feature_ids)
        if not groups:
            return _result(
                strategy,
                feature_ids,
                mask,
                oracle_metadata,
                GovernanceStatus.NOT_APPLICABLE,
                "no operational group metadata",
            )
        score_by_id = dict(zip(feature_ids, scores))
        ranked_groups = sorted(
            groups,
            key=lambda group_id: (max(score_by_id[fid] for fid in groups[group_id]), group_id),
            reverse=True,
        )
        n_groups = _budget_count(budget, len(ranked_groups))
        selected = set(ranked_groups[:n_groups])
        index_by_id = {feature_id: i for i, feature_id in enumerate(feature_ids)}
        for group_id in selected:
            for feature_id in groups[group_id]:
                mask[index_by_id[feature_id]] = 0.0
        return _result(strategy, feature_ids, mask, oracle_metadata, review_units=n_groups)

    if strategy == GovernanceStrategy.SCORE_THRESHOLD:
        mask = (scores <= threshold).astype(float)
        return _result(
            strategy,
            feature_ids,
            mask,
            oracle_metadata,
            review_units=int((mask < 0.5).sum()),
        )

    if strategy == GovernanceStrategy.GRAPH_CUT:
        if not operational_metadata.graph_edges:
            return _result(
                strategy,
                feature_ids,
                mask,
                oracle_metadata,
                GovernanceStatus.NOT_APPLICABLE,
                "no operational feature graph",
            )
        components = _connected_components(feature_ids, operational_metadata.graph_edges)
        score_by_id = dict(zip(feature_ids, scores))
        components.sort(
            key=lambda component: max(score_by_id[fid] for fid in component), reverse=True
        )
        n_components = _budget_count(budget, len(components))
        selected_ids = {fid for component in components[:n_components] for fid in component}
        for j, feature_id in enumerate(feature_ids):
            if feature_id in selected_ids:
                mask[j] = 0.0
        return _result(strategy, feature_ids, mask, oracle_metadata, review_units=n_components)

    if strategy == GovernanceStrategy.LIFECYCLE_REMOVAL:
        risky_lifecycles = {"post_outcome", "future_window", "repayment", "closed"}
        for j, feature_id in enumerate(feature_ids):
            meta = operational_metadata.features[feature_id]
            if (
                meta.lifecycle in risky_lifecycles
                or not meta.available_at_prediction
                or meta.outcome_descendant
                or meta.post_event_table
            ):
                mask[j] = 0.0
        return _result(
            strategy,
            feature_ids,
            mask,
            oracle_metadata,
            review_units=int((mask < 0.5).sum()),
        )

    raise ValueError("unsupported governance strategy: " + strategy.value)
