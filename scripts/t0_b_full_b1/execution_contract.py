"""T0-B Full-B1 Execution Contract — workload balancing, shard assignment."""
from __future__ import annotations
import hashlib


def workload_estimate(key: dict) -> float:
    """Estimate computational workload for a key based on metadata only."""
    n = key["n_original"] + key["n_injected"]
    return float(n) * 1.0


def balanced_shard_assignment(keys: list[dict], shard_count: int = 64) -> list[dict]:
    """Capacity-constrained workload-balanced shard assignment.

    60 shards × 86 keys, 4 shards × 85 keys (5500/64).
    """
    n_keys = len(keys)
    base = n_keys // shard_count
    remainder = n_keys % shard_count
    capacities = [base + 1] * remainder + [base] * (shard_count - remainder)

    # Sort by workload descending, then canonical_key_id ascending
    sorted_keys = sorted(keys, key=lambda k: (-workload_estimate(k), k["canonical_key_id"]))

    shard_loads = [0.0] * shard_count
    shard_counts = [0] * shard_count
    assignments = []

    for k in sorted_keys:
        wl = workload_estimate(k)
        # Find eligible shard: under capacity, lowest current load, smallest shard_id
        best = None
        best_load = float("inf")
        for sid in range(shard_count):
            if shard_counts[sid] < capacities[sid]:
                if shard_loads[sid] < best_load:
                    best_load = shard_loads[sid]
                    best = sid
        if best is None:
            raise RuntimeError(f"No shard available for key {k['canonical_key_id']}")
        shard_loads[best] += wl
        shard_counts[best] += 1
        assignments.append({"canonical_key_id": k["canonical_key_id"], "shard_id": best})

    return assignments
