#!/usr/bin/env python3
"""T0-B V3 Seed Contract — SHA256 cryptographic P2 seed derivation."""
from __future__ import annotations
import hashlib

SEED_CONTRACT_ID = b"t0b_random_matched_v3"


def derive_p2_seed(
    governance_seed: int, dataset_index: int, mechanism: str,
    strength: str, training_seed: int,
    cost_contract_id: str, budget_basis_points: int,
) -> int:
    payload = (
        SEED_CONTRACT_ID
        + b"\0" + str(governance_seed).encode()
        + b"\0" + str(dataset_index).encode()
        + b"\0" + mechanism.encode()
        + b"\0" + strength.encode()
        + b"\0" + str(training_seed).encode()
        + b"\0" + cost_contract_id.encode()
        + b"\0" + str(budget_basis_points).encode()
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)
