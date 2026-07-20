"""T0-B Semantic Registry — instantiates PolicyGroupView and SemanticEvaluationLabels per key.

Reads policy_group_registry_v2.json and bundle manifest to construct
oracle-isolated groups for any (dataset_index, mechanism, strength, seed) key.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.t0_b.policy_views import PolicyGroupView, SemanticEvaluationLabels

_ROOT = Path(__file__).resolve().parents[2]


def load_registry():
    with open(_ROOT / "configs/edbt_t0_b/policy_group_registry_v2.json") as f:
        return json.load(f)


def load_eval_labels_config():
    with open(_ROOT / "configs/edbt_t0_b/semantic_evaluation_labels_v2.json") as f:
        return json.load(f)


def build_groups_for_key(
    dataset_index: int,
    mechanism: str,
    strength: str,
    training_seed: int,
    n_original: int,
    n_injected: int,
) -> tuple[list[PolicyGroupView], str]:
    """Build oracle-isolated PolicyGroupView list for one key.

    Returns:
        (groups, mapping_hash) where mapping_hash is SHA256 of the deterministic mapping.
    """
    registry = load_registry()
    mech_groups = registry["mechanisms"].get(mechanism, {})
    injected = mech_groups.get("injected_groups", [])

    groups: list[PolicyGroupView] = []

    # Original columns: each is a singleton group
    for i in range(n_original):
        gid = f"g_orig_{i:03d}"
        groups.append(PolicyGroupView(
            opaque_group_id=gid,
            member_encoded_indices=(i,),
            group_size=1,
        ))

    # Injected groups
    total_cols = n_original + n_injected
    injected_start = n_original

    for ig in injected:
        gid = ig["opaque_group_id"]
        member_count = ig.get("member_count", 1)

        if isinstance(member_count, int):
            count = member_count
        else:
            count = n_injected  # dynamic: member_count is a string like "dynamic: ..."

        # M10 special case: two singleton groups at different positions
        if mechanism == "M10" and len(injected) == 2:
            if "g_inj_001" in gid:
                start, end = n_original, n_original
            else:
                start, end = n_original + 1, n_original + 1
        else:
            start = injected_start
            end = injected_start + count - 1
            injected_start = end + 1  # advance for next group

        member_indices = tuple(range(start, end + 1))
        groups.append(PolicyGroupView(
            opaque_group_id=gid,
            member_encoded_indices=member_indices,
            group_size=len(member_indices),
        ))

    # Verify: every encoded column belongs to exactly one group
    covered = set()
    for g in groups:
        for idx in g.member_encoded_indices:
            if idx in covered:
                raise ValueError(f"Column {idx} assigned to multiple groups in key ({dataset_index}, {mechanism}, {strength}, {training_seed})")
            covered.add(idx)

    expected = set(range(total_cols))
    missing = expected - covered
    if missing:
        raise ValueError(f"Columns {missing} not assigned to any group in key ({dataset_index}, {mechanism}, {strength}, {training_seed})")

    # Compute mapping hash
    mapping_repr = json.dumps([
        {"gid": g.opaque_group_id, "members": list(g.member_encoded_indices)}
        for g in sorted(groups, key=lambda g: g.opaque_group_id)
    ], sort_keys=True)
    mapping_hash = hashlib.sha256(mapping_repr.encode()).hexdigest()

    return groups, mapping_hash


def build_evaluation_labels(
    dataset_index: int,
    mechanism: str,
    strength: str,
    training_seed: int,
    leak_mask: np.ndarray,
    groups: list[PolicyGroupView],
) -> SemanticEvaluationLabels:
    """Build evaluation-only labels AFTER selection is complete."""
    eval_config = load_eval_labels_config()
    mech_labels = eval_config["mechanisms"].get(mechanism, {})

    contaminated_ids = frozenset(mech_labels.get("contaminated_group_ids", []))
    legitimate_ids = frozenset(mech_labels.get("legitimate_group_ids", []))

    # If "all g_orig_*" pattern, expand
    orig_ids = {g.opaque_group_id for g in groups if g.opaque_group_id.startswith("g_orig_")}

    return SemanticEvaluationLabels(
        leak_mask=tuple(bool(m) for m in leak_mask),
        contaminated_group_ids=contaminated_ids,
        legitimate_group_ids=orig_ids | {i for i in legitimate_ids if i.startswith("g_inj_")},
    )
