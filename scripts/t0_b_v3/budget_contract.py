#!/usr/bin/env python3
"""T0-B V3 Budget Contract — integer half-up rounding."""
from __future__ import annotations


def compute_k(n_units: int, budget_basis_points: int) -> int:
    if n_units <= 0:
        raise ValueError(f"n_units must be positive, got {n_units}")
    if budget_basis_points <= 0:
        raise ValueError(f"budget_basis_points must be positive, got {budget_basis_points}")
    k = (n_units * budget_basis_points + 5000) // 10000
    k = max(1, min(k, n_units))
    return int(k)
