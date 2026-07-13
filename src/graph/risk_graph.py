"""risk_graph.py — V4-Graph: Three-class feature risk decomposition."""
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.feature_selection import mutual_info_regression


@dataclass
class FieldRiskProfile:
    feature_name: str; predictive_impact: float = 0.5
    availability_risk: float = 0.5; structural_risk: float = 0.5
    composite_risk: float = 0.5


def compute_predictive_impact(X_train, y_train, feature_names):
    mi = mutual_info_regression(X_train, y_train, random_state=42)
    mi = np.nan_to_num(mi, nan=0.0)
    if mi.max() > mi.min(): mi = (mi - mi.min()) / (mi.max() - mi.min())
    else: mi = np.full_like(mi, 0.5)
    return {n: float(np.clip(mi[j], 0.0, 1.0)) for j, n in enumerate(feature_names)}


def compute_availability_risk(feature_names, outcome_descendants=None, post_event_tables=None):
    scores = {}
    for name in feature_names:
        s, w = 0.0, 0.0
        if outcome_descendants and name in outcome_descendants: s += 0.9; w += 1.0
        if post_event_tables and name in post_event_tables: s += 0.7; w += 1.0
        scores[name] = float(np.clip(s / max(0.01, w), 0.0, 1.0)) if w > 0 else 0.5
    return scores


def compute_structural_risk(feature_names, graph_edges=None, known_leakage=None):
    scores = {n: 0.3 for n in feature_names}
    if graph_edges and known_leakage:
        adj = defaultdict(list)
        for src, tgt, w in graph_edges:
            adj[src].append((tgt, w)); adj[tgt].append((src, w))
        from collections import deque
        leak_set = known_leakage & set(feature_names)
        dist = {}
        queue = deque([(s, 0) for s in leak_set])
        for s, d in queue:
            if s not in dist:
                dist[s] = d
                for nb, _ in adj.get(s, []): queue.append((nb, d+1))
        for name in feature_names:
            if name in dist: scores[name] = {0:0.9, 1:0.6, 2:0.3}.get(dist[name], 0.1)
    return scores


def compute_composite_risk(feature_names, impact, availability, structural, alpha=0.4, beta=0.4, gamma=0.2):
    return {name: float(np.clip(alpha*availability.get(name,0.5) + beta*structural.get(name,0.5) + gamma*impact.get(name,0.5)*availability.get(name,0.5), 0.0, 1.0)) for name in feature_names}
