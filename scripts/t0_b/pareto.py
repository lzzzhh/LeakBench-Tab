"""T0-B Pareto Analysis — strict/weak dominance on frozen dimensions."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    policy_id: str
    full_group_recall: float
    directional_repair: float
    legit_retention: float
    overcorrection: float
    introduced_distortion: float
    runtime: float

    def to_vector(self) -> tuple[float, ...]:
        """Convert to ordered vector for dominance comparison.
        Maximize: [0]=full_group_recall, [1]=directional_repair, [2]=legit_retention
        Minimize: [3]=overcorrection, [4]=introduced_distortion, [5]=runtime
        """
        return (
            self.full_group_recall,
            self.directional_repair,
            self.legit_retention,
            -self.overcorrection,          # negate for "minimize → maximize"
            -self.introduced_distortion,
            -self.runtime,
        )


def weakly_dominates(a: ParetoPoint, b: ParetoPoint, tolerance: float = 1e-12) -> bool:
    """A weakly dominates B if A >= B on all maximize dimensions and A <= B on all minimize dimensions."""
    va = a.to_vector()
    vb = b.to_vector()
    return all(va[i] >= vb[i] - tolerance for i in range(len(va)))


def strictly_dominates(a: ParetoPoint, b: ParetoPoint, tolerance: float = 1e-12) -> bool:
    """A strictly dominates B if weakly dominates AND better on at least one dimension."""
    if not weakly_dominates(a, b, tolerance):
        return False
    va = a.to_vector()
    vb = b.to_vector()
    return any(va[i] > vb[i] + tolerance for i in range(len(va)))


def pareto_frontier(points: list[ParetoPoint], tolerance: float = 1e-12) -> list[str]:
    """Return Pareto-optimal policy IDs."""
    optimal: list[str] = []
    for i, pi in enumerate(points):
        dominated = False
        for j, pj in enumerate(points):
            if i == j:
                continue
            if strictly_dominates(pj, pi, tolerance):
                dominated = True
                break
        if not dominated:
            optimal.append(pi.policy_id)
    return optimal
