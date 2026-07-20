"""T0-B Full-B1 Run-Key Contract — shared key formatting for plan/runner/validator.

All plan builders, runners, and validators MUST use these functions.
No ad-hoc string concatenation for run-ID lookup keys.
"""
from __future__ import annotations


def baseline_lookup_key(baseline_type: str) -> str:
    """Return lookup key for baseline rows. Values: 'strict', 'full'."""
    assert baseline_type in ("strict", "full"), f"Invalid baseline_type: {baseline_type}"
    return baseline_type


def governed_lookup_key(
    policy: str,
    contract: str,
    budget_bp: int,
    governance_seed_index: int,
) -> str:
    """Return lookup key for governed rows.

    P2: P2_{contract}_{budget_bp}_{gov_seed_index}  (index 0-19)
    P3-P6: {policy}_{contract}_{budget_bp}_-1
    """
    if policy == "P2":
        assert 0 <= governance_seed_index <= 19, f"P2 governance_seed_index must be 0-19, got {governance_seed_index}"
    else:
        assert governance_seed_index == -1, f"{policy} governance_seed_index must be -1, got {governance_seed_index}"
    return f"{policy}_{contract}_{budget_bp}_{governance_seed_index}"


def build_run_id_lookup(run_rows: list[dict]) -> dict[str, dict[str, str]]:
    """Build canonical_key_id → lookup_key → run_id mapping from run plan rows.

    Returns: {canonical_key_id: {lookup_key: run_id}}
    """
    lookup: dict[str, dict[str, str]] = {}
    for r in run_rows:
        cid = r["canonical_key_id"]
        if cid not in lookup:
            lookup[cid] = {}
        if r["run_type"] == "baseline":
            lk = baseline_lookup_key(r["baseline_type"])
        else:
            lk = governed_lookup_key(
                r["policy"], r["contract"], r["budget_bp"],
                r["governance_seed_index"]
            )
        if lk in lookup[cid]:
            raise ValueError(f"Duplicate lookup key {lk} for canonical_key_id {cid}")
        lookup[cid][lk] = r["run_id"]
    return lookup


def expected_lookup_keys_for_key() -> set[str]:
    """Return the complete set of expected lookup keys for one canonical key."""
    keys = {"strict", "full"}
    for ct in ("semantic_group", "encoded_column"):
        for bp in (500, 1000, 2000):
            for gi in range(20):
                keys.add(governed_lookup_key("P2", ct, bp, gi))
            for pid in ("P3", "P4", "P5", "P6"):
                keys.add(governed_lookup_key(pid, ct, bp, -1))
    assert len(keys) == 2 + 144  # 2 baseline + 144 governed
    return keys


def validate_lookup_complete(lookup: dict[str, str]) -> list[str]:
    """Validate that lookup has all expected keys. Returns list of errors."""
    errors = []
    expected = expected_lookup_keys_for_key()
    actual = set(lookup.keys())
    missing = expected - actual
    extra = actual - expected
    if missing:
        errors.append(f"Missing lookup keys: {sorted(missing)[:10]}...")
    if extra:
        errors.append(f"Extra lookup keys: {sorted(extra)[:10]}...")
    return errors
