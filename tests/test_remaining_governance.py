"""Focused tests for the prospective remaining-governance protocols."""
from __future__ import annotations

import numpy as np

from scripts.analyze_remaining_governance import paired
from scripts.run_remaining_governance import m09_groups, random_selection, selection_hash


def test_m09_semantic_groups_collapse_only_one_hot_block():
    groups = m09_groups(12, 20)
    assert len(groups) == 13
    assert all(group.tolist() == [index] for index, group in enumerate(groups[:12]))
    assert groups[-1].tolist() == list(range(12, 20))


def test_random_group_selection_is_deterministic_and_unique():
    first = random_selection(13, 3, 2026071700, 4, 42)
    second = random_selection(13, 3, 2026071700, 4, 42)
    assert np.array_equal(first, second)
    assert len(set(first)) == 3


def test_selection_hash_binds_unit_type():
    assert selection_hash([1, 3], "semantic_group") != selection_hash([1, 3], "encoded_column")
    assert selection_hash([3, 1], "semantic_group") == selection_hash([1, 3], "semantic_group")


def test_paired_averages_random_seeds_within_key():
    rows = [
        {"task": "A", "training_seed": 13, "policy": "P3_blind_mi", "strict_distance_reduction": 0.5},
        {"task": "A", "training_seed": 13, "policy": "P2_random", "strict_distance_reduction": 0.1},
        {"task": "A", "training_seed": 13, "policy": "P2_random", "strict_distance_reduction": 0.3},
    ]
    import pandas as pd
    result = paired(pd.DataFrame(rows), ["task", "training_seed"])
    assert result.iloc[0].paired == 0.3

