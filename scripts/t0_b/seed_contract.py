"""T0-B Seed Contract — cryptographic P2 seed derivation.

V2 uses SHA256-based derivation with explicit namespace separation.
No Python hash(), no float serialization, no unordered JSON.
"""
from __future__ import annotations
import hashlib

# Sentinel to verify correct contract is used
SEED_CONTRACT_ID = b"t0b_random_matched_v2"


def derive_p2_seed(
    governance_seed: int,
    dataset_index: int,
    mechanism: str,
    strength: str,
    training_seed: int,
    cost_contract_id: str,
    budget_basis_points: int,
) -> int:
    """Derive deterministic P2 seed from all namespace dimensions.

    Payload is UTF-8 encoded, null-byte separated, in a fixed order.
    """
    payload = (
        SEED_CONTRACT_ID
        + b"\0"
        + str(governance_seed).encode("utf-8")
        + b"\0"
        + str(dataset_index).encode("utf-8")
        + b"\0"
        + mechanism.encode("utf-8")
        + b"\0"
        + strength.encode("utf-8")
        + b"\0"
        + str(training_seed).encode("utf-8")
        + b"\0"
        + cost_contract_id.encode("utf-8")
        + b"\0"
        + str(budget_basis_points).encode("utf-8")
    )
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    return seed
