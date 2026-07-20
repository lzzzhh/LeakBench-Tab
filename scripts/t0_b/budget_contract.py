"""T0-B Budget Contract — deterministic half-up integer rounding.

No Python round(). Uses integer arithmetic only.
"""
from __future__ import annotations


def compute_k(n_units: int, budget_basis_points: int) -> int:
    """Compute k = max(1, floor((n_units * bp + 5000) / 10000)) capped at n_units.

    Args:
        n_units: total number of selectable units (groups or columns).
        budget_basis_points: budget in basis points (500=5%, 1000=10%, 2000=20%).

    Returns:
        k: number of units to remove.

    Raises:
        ValueError: if n_units <= 0 or budget_basis_points <= 0.
    """
    if n_units <= 0:
        raise ValueError(f"n_units must be positive, got {n_units}")
    if budget_basis_points <= 0:
        raise ValueError(f"budget_basis_points must be positive, got {budget_basis_points}")

    # k = floor((n * bp + 5000) / 10000) — deterministic half-up
    k = (n_units * budget_basis_points + 5000) // 10000
    k = max(1, k)
    k = min(k, n_units)
    return int(k)
