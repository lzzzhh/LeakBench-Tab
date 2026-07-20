"""T0-B V3 Selection Hash Contract.

Separate namespaces for encoded-column and semantic-group selections.
All hashes include: contract version, canonical key, policy, cost contract, budget.
"""
from __future__ import annotations
import hashlib, struct
import numpy as np

_ENCODED_PREFIX = b"t0b_encoded_selection_v3\0"
_SEMANTIC_PREFIX = b"t0b_semantic_selection_v3\0"


def _canonical_key_bytes(
    dataset_index: int, mechanism: str, strength: str, training_seed: int,
    bundle_key: str, bundle_sha256: str,
) -> bytes:
    return (
        str(dataset_index).encode()
        + b"\0" + mechanism.encode()
        + b"\0" + strength.encode()
        + b"\0" + str(training_seed).encode()
        + b"\0" + bundle_key.encode()
        + b"\0" + bundle_sha256.encode()
    )


def _policy_context(policy_id: str, cost_contract_id: str, budget_basis_points: int) -> bytes:
    return (
        policy_id.encode()
        + b"\0" + cost_contract_id.encode()
        + b"\0" + str(budget_basis_points).encode()
    )


def hash_encoded_selection(
    dataset_index: int, mechanism: str, strength: str, training_seed: int,
    bundle_key: str, bundle_sha256: str,
    policy_id: str, cost_contract_id: str, budget_basis_points: int,
    encoded_indices: np.ndarray,
) -> str:
    """SHA256 hash of encoded-column selection.

    encoded_indices: sorted int64 indices of removed columns.
    """
    payload = (
        _ENCODED_PREFIX
        + _canonical_key_bytes(dataset_index, mechanism, strength, training_seed, bundle_key, bundle_sha256)
        + b"\0" + _policy_context(policy_id, cost_contract_id, budget_basis_points)
    )
    sorted_idx = np.sort(encoded_indices).astype(np.int64)
    payload += sorted_idx.tobytes()
    return hashlib.sha256(payload).hexdigest()


def hash_semantic_selection(
    dataset_index: int, mechanism: str, strength: str, training_seed: int,
    bundle_key: str, bundle_sha256: str,
    policy_id: str, cost_contract_id: str, budget_basis_points: int,
    group_ids: list[str],
) -> str:
    """SHA256 hash of semantic-group selection.

    group_ids: sorted neutral opaque group IDs of removed groups.
    """
    payload = (
        _SEMANTIC_PREFIX
        + _canonical_key_bytes(dataset_index, mechanism, strength, training_seed, bundle_key, bundle_sha256)
        + b"\0" + _policy_context(policy_id, cost_contract_id, budget_basis_points)
    )
    sorted_gids = sorted(group_ids)
    for gid in sorted_gids:
        payload += gid.encode() + b"\0"
    return hashlib.sha256(payload).hexdigest()
