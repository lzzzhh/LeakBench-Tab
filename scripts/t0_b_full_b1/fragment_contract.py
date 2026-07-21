"""T0-B Fragment Contract — manifest, receipt, validation dataclasses."""
from __future__ import annotations
from dataclasses import dataclass, field
import gzip, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd
from numbers import Integral


# ====================================================================
# Strict scalar validation
# ====================================================================

def require_integral_scalar(value, field_name: str) -> int:
    """Require strict Python/numpy integer, rejecting bool, float, str, None, NaN."""
    if isinstance(value, (bool, np.bool_)):
        raise SelectionContractError(f"{field_name}: bool not allowed (got {value})")
    if not isinstance(value, Integral):
        raise SelectionContractError(f"{field_name}: expected integral, got {type(value).__name__} (value={value})")
    try:
        v = int(value)
    except (ValueError, TypeError):
        raise SelectionContractError(f"{field_name}: cannot convert to int: {value}")
    return v


def require_positive_integral(value, field_name: str) -> int:
    v = require_integral_scalar(value, field_name)
    if v < 0:
        raise SelectionContractError(f"{field_name}: must be non-negative, got {v}")
    return v


def require_nonempty_str(value, field_name: str) -> str:
    """Require non-empty string, rejecting None, NaN, and non-str types."""
    if not isinstance(value, str):
        raise SelectionContractError(f"{field_name}: expected str, got {type(value).__name__}")
    if not value.strip():
        raise SelectionContractError(f"{field_name}: empty or whitespace-only")
    return value.strip()

@dataclass
class ProductionGuard:
    real_bundle_loads: int = 0
    real_model_calls: int = 0
    real_selector_calls: int = 0

    def snapshot(self) -> dict:
        return {"real_bundle_loads": self.real_bundle_loads,
                "real_model_calls": self.real_model_calls,
                "real_selector_calls": self.real_selector_calls}

    def delta(self, before: "ProductionGuard") -> dict:
        return {"real_bundle_loads": self.real_bundle_loads - before.real_bundle_loads,
                "real_model_calls": self.real_model_calls - before.real_model_calls,
                "real_selector_calls": self.real_selector_calls - before.real_selector_calls}


@dataclass
class SyntheticCallCounter:
    lr_calls: int = 0
    p3_calls: int = 0
    p4_calls: int = 0
    p5_calls: int = 0
    p6_calls: int = 0

    def snapshot(self) -> dict:
        return {"lr_calls": self.lr_calls, "p3_calls": self.p3_calls,
                "p4_calls": self.p4_calls, "p5_calls": self.p5_calls, "p6_calls": self.p6_calls}

    def delta(self, before: "SyntheticCallCounter") -> dict:
        return {"lr_calls": self.lr_calls - before.lr_calls,
                "p3_calls": self.p3_calls - before.p3_calls,
                "p4_calls": self.p4_calls - before.p4_calls,
                "p5_calls": self.p5_calls - before.p5_calls,
                "p6_calls": self.p6_calls - before.p6_calls}


# ====================================================================
# Validation result
# ====================================================================

@dataclass
class CompletedKeyValidation:
    is_complete: bool
    errors: list[str] = field(default_factory=list)
    baseline_rows: int = 0
    governed_rows: int = 0
    selection_rows: int = 0
    failure_rows: int = 0
    duplicate_run_ids: list[str] = field(default_factory=list)
    missing_run_ids: list[str] = field(default_factory=list)
    extra_run_ids: list[str] = field(default_factory=list)
    null_run_id_count: int = 0
    receipt_valid: bool = False
    fragment_manifest_valid: bool = False
    fragment_sha_valid: bool = False
    run_id_closure_valid: bool = False
    selection_closure_valid: bool = False
    selection_payload_valid: bool = False
    realized_cost_valid: bool = False
    semantic_atomicity_valid: bool = False
    mapping_valid: bool = False
    m09_atomicity_valid: bool = False
    manifest_selection_digest_valid: bool = False
    planned_run_ids_sha256: str | None = None
    produced_run_ids_sha256: str | None = None


# ====================================================================
# Fragment manifest
# ====================================================================

def _row_sha256(row: dict) -> str:
    """Canonical JSON SHA256 of a single row."""
    text = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((text + "\n").encode()).hexdigest()


def _ids_sha256(ids: list[str]) -> str:
    """SHA256 of sorted ID list, one per line, trailing newline."""
    content = "\n".join(sorted(ids)) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def _sorted_counts_sha256(values: list[str]) -> str:
    """SHA256 of sorted values (preserving duplicates), one per line."""
    content = "\n".join(sorted(values)) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def build_fragment_manifest(
    cid: str,
    key_plan_row: dict,
    planned_run_ids: list[str],
    produced_run_ids: list[str],
    baseline_path: Path,
    governed_path: Path,
    selection_path: Path,
    failure_path: Path,
    plan_manifest_sha256: str,
) -> dict:
    """Build deterministic fragment manifest."""
    baseline_rows = len(pd.read_csv(pd.io.common.BytesIO(gzip.decompress(baseline_path.read_bytes()))))
    governed_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(governed_path.read_bytes())),
                              dtype={"selection_hash": str})
    selection_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(selection_path.read_bytes())),
                               dtype={"selection_hash": str, "removed_encoded_indices": str, "removed_group_ids": str})
    failure_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(failure_path.read_bytes())))

    governed_rows = len(governed_df)
    selection_rows = len(selection_df)
    failure_rows = len(failure_df)

    # Selection hash multiset (preserves duplicates)
    sel_hashes = sorted(selection_df["selection_hash"].tolist())
    sel_multiset_sha = _sorted_counts_sha256(sel_hashes)

    # Selection payload digest
    sel_payloads = []
    for _, r in selection_df.sort_values("selection_hash").iterrows():
        sel_payloads.append(_canonical_selection_json(r))
    sel_payload_digest_sha = _sorted_counts_sha256(sel_payloads)

    return {
        "schema_version": 1,
        "canonical_key_id": cid,
        "scientific_freeze_sha": "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845",
        "execution_contract_version": "v1",
        "plan_manifest_sha256": plan_manifest_sha256,
        "key_plan_row_sha256": _row_sha256(key_plan_row),
        "planned_run_ids_sha256": _ids_sha256(planned_run_ids),
        "produced_run_ids_sha256": _ids_sha256(produced_run_ids),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "governed_sha256": hashlib.sha256(governed_path.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "failure_sha256": hashlib.sha256(failure_path.read_bytes()).hexdigest(),
        "baseline_rows": baseline_rows,
        "governed_rows": governed_rows,
        "selection_rows": selection_rows,
        "failure_rows": failure_rows,
        "selection_hash_multiset_sha256": sel_multiset_sha,
        "selection_payload_digest_sha256": sel_payload_digest_sha,
    }


class SelectionContractError(ValueError):
    """Raised when selection array fields are invalid."""
    pass


# ====================================================================
# Array field parsing
# ====================================================================

def parse_int_array_json(value) -> list[int]:
    """Parse a JSON-encoded integer array. Raw CSV input: MUST be a JSON string."""
    if not isinstance(value, str):
        raise SelectionContractError(f"removed_encoded_indices: expected JSON string, got {type(value).__name__}")
    try:
        arr = json.loads(value)
    except json.JSONDecodeError as e:
        raise SelectionContractError(f"JSON parse error: {e}")
    if not isinstance(arr, list):
        raise SelectionContractError(f"expected JSON list, got {type(arr).__name__}")
    result = []
    for i, item in enumerate(arr):
        if isinstance(item, (bool, np.bool_)):
            raise SelectionContractError(f"index {i}: bool not allowed")
        if not isinstance(item, Integral):
            raise SelectionContractError(f"index {i}: expected int, got {type(item).__name__}")
        v = int(item)
        if v < 0: raise SelectionContractError(f"index {i}: negative index {v}")
        result.append(v)
    if len(result) != len(set(result)):
        raise SelectionContractError(f"duplicate encoded indices")
    return sorted(result)


def parse_group_id_array_json(value) -> list[str]:
    """Parse a JSON-encoded string array for group IDs. Raw CSV input: MUST be a JSON string."""
    if not isinstance(value, str):
        raise SelectionContractError(f"removed_group_ids: expected JSON string, got {type(value).__name__}")
    try:
        arr = json.loads(value)
    except json.JSONDecodeError as e:
        raise SelectionContractError(f"JSON parse error: {e}")
    if not isinstance(arr, list):
        raise SelectionContractError(f"expected JSON list, got {type(arr).__name__}")
    result = []
    for i, item in enumerate(arr):
        if not isinstance(item, str) or item.strip() == "":
            raise SelectionContractError(f"index {i}: expected non-empty str, got {type(item).__name__}")
        result.append(item.strip())
    if len(result) != len(set(result)):
        raise SelectionContractError(f"duplicate group IDs")
    return sorted(result)


# ====================================================================
# Canonical selection payload
# ====================================================================

def _canonical_selection_json(row: dict) -> str:
    """Canonical JSON for a selection row (shared by builder and validator).
    Uses strict parsing via normalize_selection_payload, then canonical serialization."""
    payload = normalize_selection_payload(row)
    return canonical_selection_payload_json(payload)


def canonical_selection_payload_json(payload: dict) -> str:
    """Canonical JSON for an already-normalized selection payload."""
    return json.dumps({
        "selection_hash": payload["selection_hash"],
        "policy": payload["policy"],
        "contract": payload["contract"],
        "budget_bp": payload["budget_bp"],
        "removed_encoded_indices": payload["removed_encoded_indices"],
        "removed_group_ids": payload["removed_group_ids"],
        "realized_encoded_cost": payload["realized_encoded_cost"],
    }, sort_keys=True, separators=(",", ":"))


def normalize_selection_payload(row: dict) -> dict:
    """Parse and normalize a selection row into a validated dict."""
    removed = parse_int_array_json(row["removed_encoded_indices"])
    groups = parse_group_id_array_json(row["removed_group_ids"])

    h = require_nonempty_str(row["selection_hash"], "selection_hash")
    policy = require_nonempty_str(row["policy"], "policy")
    if policy not in ("P2", "P3", "P4", "P5", "P6"):
        raise SelectionContractError(f"invalid policy: {policy}")
    contract = require_nonempty_str(row["contract"], "contract")
    if contract not in ("semantic_group", "encoded_column"):
        raise SelectionContractError(f"invalid contract: {contract}")
    bp = require_integral_scalar(row["budget_bp"], "budget_bp")
    if bp not in (500, 1000, 2000):
        raise SelectionContractError(f"invalid budget_bp: {bp}")
    cost = require_positive_integral(row["realized_encoded_cost"], "realized_encoded_cost")
    return {
        "selection_hash": h, "policy": policy, "contract": contract, "budget_bp": bp,
        "removed_encoded_indices": removed, "removed_group_ids": groups, "realized_encoded_cost": cost,
    }


# ====================================================================
# Selection validation functions
# ====================================================================

def validate_selection_multiset_closure(governed_df, normalized_payloads: list[dict]) -> list[str]:
    """Verify governed.selection_hash multiset == selection multiset."""
    errors = []
    from collections import Counter
    gov_counter = Counter(str(h) for h in governed_df["selection_hash"].dropna())
    sel_counter = Counter(p["selection_hash"] for p in normalized_payloads)
    if gov_counter != sel_counter:
        missing = {k: v for k, v in sel_counter.items() if gov_counter.get(k, 0) < v}
        extra = {k: v for k, v in gov_counter.items() if sel_counter.get(k, 0) < v}
        if missing:
            errors.append(f"selection multiset: {sum(missing.values())} missing occurrences")
        if extra:
            errors.append(f"selection multiset: {sum(extra.values())} extra occurrences")
    return errors


def validate_selection_payload_consistency(normalized_payloads: list[dict]) -> list[str]:
    """Same selection_hash must have identical payload."""
    errors = []
    by_hash = {}
    for pi, p in enumerate(normalized_payloads):
        h = p["selection_hash"]
        if h not in by_hash:
            by_hash[h] = (pi, p)
        else:
            prev_i, prev = by_hash[h]
            for key in ["policy", "contract", "budget_bp", "removed_encoded_indices", "removed_group_ids", "realized_encoded_cost"]:
                if prev[key] != p[key]:
                    errors.append(f"selection_hash {h[:16]}: {key} differs between row {prev_i} and {pi}")
                    break
    return errors


def validate_selection_realized_cost(normalized_payloads: list[dict]) -> list[str]:
    """realized_encoded_cost must equal len(removed_encoded_indices)."""
    errors = []
    for pi, p in enumerate(normalized_payloads):
        expected = len(p["removed_encoded_indices"])
        actual = p["realized_encoded_cost"]
        if actual != expected:
            errors.append(f"selection row {pi}: cost={actual}, expected={expected}")
    return errors


def validate_governed_realized_cost(governed_df, payload_by_hash: dict) -> list[str]:
    """Governed realized_cost must match selection payload cost."""
    errors = []
    for gi, row in governed_df.iterrows():
        try:
            h = require_nonempty_str(row["selection_hash"], f"governed row {gi} selection_hash")
            gcost = require_positive_integral(row["realized_cost"], f"governed row {gi} realized_cost")
        except SelectionContractError as e:
            errors.append(str(e))
            continue
        if h in payload_by_hash:
            expected = payload_by_hash[h]["realized_encoded_cost"]
            if gcost != expected:
                errors.append(f"governed row {gi}: cost={gcost}, selection cost={expected}")
    return errors


# ====================================================================
# Semantic validation
# ====================================================================

def validate_semantic_group_atomicity(payload: dict, group_members: dict[str, set[int]]) -> list[str]:
    """For semantic_group contract, removed columns must exactly equal group union."""
    errors = []
    if payload["contract"] != "semantic_group":
        return errors
    gids = payload["removed_group_ids"]
    for gid in gids:
        if gid not in group_members:
            errors.append(f"unknown group: {gid}")
            return errors
    if errors:
        return errors
    expected_indices = set()
    for gid in gids:
        expected_indices.update(group_members[gid])
    actual_indices = set(payload["removed_encoded_indices"])
    missing = expected_indices - actual_indices
    extra = actual_indices - expected_indices
    if missing:
        errors.append(f"semantic-group: missing encoded indices: {sorted(missing)[:10]}")
    if extra:
        errors.append(f"semantic-group: extra encoded indices: {sorted(extra)[:10]}")
    if missing or extra:
        errors.append("semantic-group: partial group removal detected")
    return errors


def validate_manifest_selection_digest(manifest: dict, sel_hashes: list[str], sel_payloads: list[dict]) -> list[str]:
    """Verify manifest selection hash multiset and payload digest match actual data."""
    errors = []
    actual_multiset = _sorted_counts_sha256(sorted(sel_hashes))
    if actual_multiset != manifest.get("selection_hash_multiset_sha256", ""):
        errors.append("manifest selection_hash_multiset_sha256 mismatch")
    canonical = [canonical_selection_payload_json(p) for p in sel_payloads]
    actual_digest = _sorted_counts_sha256(sorted(canonical))
    if actual_digest != manifest.get("selection_payload_digest_sha256", ""):
        errors.append("manifest selection_payload_digest_sha256 mismatch")
    return errors


# ====================================================================
# Policy mapping validation
# ====================================================================

@dataclass
class ValidatedPolicyMapping:
    group_members: dict[str, frozenset[int]]
    all_encoded_indices: frozenset[int]
    n_total_columns: int


def validate_policy_mapping(policy_mapping: dict, key_plan_row: dict) -> ValidatedPolicyMapping:
    """Validate policy mapping structure."""
    if not isinstance(policy_mapping, dict):
        raise SelectionContractError("policy_mapping must be dict")
    groups = policy_mapping.get("groups")
    if not isinstance(groups, list) or len(groups) == 0:
        raise SelectionContractError("policy_mapping.groups must be non-empty list")

    n_total = key_plan_row.get("n_total_columns") or key_plan_row.get("n_original", 0) + key_plan_row.get("n_injected", 0)
    if n_total <= 0:
        raise SelectionContractError(f"cannot determine n_total_columns from key_plan_row")

    group_members = {}; all_indices = set(); seen_gids = set()
    for gi, g in enumerate(groups):
        if not isinstance(g, dict):
            raise SelectionContractError(f"group {gi}: must be dict")
        gid = require_nonempty_str(g.get("opaque_group_id", ""), f"group {gi} opaque_group_id")
        if gid in seen_gids:
            raise SelectionContractError(f"duplicate group ID: {gid}")
        seen_gids.add(gid)
        members = g.get("member_encoded_indices")
        if not isinstance(members, list):
            raise SelectionContractError(f"group {gid}: member_encoded_indices must be list")
        validated = []
        for mi, m in enumerate(members):
            idx = require_integral_scalar(m, f"group {gid} member {mi}")
            if idx < 0 or idx >= n_total:
                raise SelectionContractError(f"group {gid}: index {idx} out of bounds [0, {n_total})")
            validated.append(idx)
        if len(validated) != len(set(validated)):
            raise SelectionContractError(f"group {gid}: duplicate member indices")
        gs = require_integral_scalar(g.get("group_size", 0), f"group {gid} group_size")
        if gs != len(validated):
            raise SelectionContractError(f"group {gid}: group_size={gs} != len(members)={len(validated)}")
        group_members[gid] = frozenset(validated)
        overlap = all_indices & frozenset(validated)
        if overlap:
            raise SelectionContractError(f"group {gid}: encoded indices {sorted(overlap)} already claimed by other groups")
        all_indices.update(validated)
    return ValidatedPolicyMapping(group_members=group_members, all_encoded_indices=frozenset(all_indices), n_total_columns=n_total)


# ====================================================================
# Semantic mapping validation
# ====================================================================

@dataclass(frozen=True)
class ValidatedSemanticMapping:
    leak_group_ids: tuple[str, ...]
    leak_encoded_indices: frozenset[int]


def validate_semantic_mapping(
    semantic_mapping,
    key_plan_row: dict,
    policy_mapping: ValidatedPolicyMapping,
) -> ValidatedSemanticMapping:
    """Validate semantic mapping structure. For M09, verifies leak union is exactly 8 columns."""
    if not isinstance(semantic_mapping, dict):
        raise SelectionContractError("semantic mapping: must be dict")
    leak_gids_raw = semantic_mapping.get("leak_group_ids")
    mech = key_plan_row.get("mechanism", "")

    if leak_gids_raw is None:
        if mech == "M09":
            raise SelectionContractError("M09: semantic mapping must have leak_group_ids")
        return ValidatedSemanticMapping(leak_group_ids=(), leak_encoded_indices=frozenset())

    if not isinstance(leak_gids_raw, list):
        raise SelectionContractError(f"semantic mapping leak_group_ids: expected list, got {type(leak_gids_raw).__name__}")

    # Validate uniqueness and type
    seen = set()
    validated_gids = []
    for i, gid in enumerate(leak_gids_raw):
        gid_str = require_nonempty_str(gid, f"leak_group_ids[{i}]")
        if gid_str in seen:
            raise SelectionContractError(f"semantic mapping: duplicate leak_group_id '{gid_str}'")
        seen.add(gid_str)
        validated_gids.append(gid_str)

    # Verify all groups exist in policy mapping
    leak_indices = set()
    for gid in validated_gids:
        if gid not in policy_mapping.group_members:
            raise SelectionContractError(f"semantic mapping: leak_group_id '{gid}' not in policy mapping")
        leak_indices.update(policy_mapping.group_members[gid])

    # M09: union must be exactly 8
    if mech == "M09":
        if len(leak_indices) != 8:
            raise SelectionContractError(f"M09 leak union size={len(leak_indices)}, expected 8")
        if not validated_gids:
            raise SelectionContractError("M09: leak_group_ids must be non-empty")

    return ValidatedSemanticMapping(leak_group_ids=tuple(validated_gids), leak_encoded_indices=frozenset(leak_indices))


# ====================================================================
# Encoded-column contract validation
# ====================================================================

def validate_encoded_column_contract(payload: dict, mapping: ValidatedPolicyMapping) -> list[str]:
    """For encoded_column contract: all indices in range, unknown groups rejected."""
    errors = []
    if payload["contract"] != "encoded_column":
        return errors
    for idx in payload["removed_encoded_indices"]:
        if idx < 0 or idx >= mapping.n_total_columns:
            errors.append(f"encoded-column: index {idx} out of bounds [0, {mapping.n_total_columns})")
    for gid in payload["removed_group_ids"]:
        if gid not in mapping.group_members:
            errors.append(f"encoded-column: unknown group '{gid}'")
    return errors


# ====================================================================
# M09 validation (uses explicit semantic mapping)
# ====================================================================

def validate_m09_eight_columns(payload: dict, validated_semantic: ValidatedSemanticMapping, key_plan_row: dict) -> list[str]:
    """M09 semantic-group contract: if leak group is selected, all 8 columns must be removed."""
    errors = []
    if key_plan_row.get("mechanism") != "M09":
        return errors
    leak_indices = validated_semantic.leak_encoded_indices
    if len(leak_indices) != 8:
        errors.append(f"M09 leak union size={len(leak_indices)}, expected 8")
    if payload["contract"] == "semantic_group":
        selected = set(payload["removed_group_ids"])
        if selected & set(validated_semantic.leak_group_ids):
            removed = set(payload["removed_encoded_indices"])
            if not leak_indices.issubset(removed):
                missing = leak_indices - removed
                errors.append(f"M09: leak group selected but {len(missing)} columns missing")
    return errors


# ====================================================================
# Main validator
# ====================================================================

def validate_completed_key(
    key_plan_row: dict,
    planned_run_ids: list[str],
    fragment_dir: Path,
    plan_manifest_sha256: str,
    policy_mapping: dict,
    semantic_mapping: dict,
) -> CompletedKeyValidation:
    """Validate a completed key fragment directory."""
    errors = []
    cid = key_plan_row.get("canonical_key_id", "unknown")
    result = CompletedKeyValidation(is_complete=False, errors=errors)

    # ── Receipt check ──
    receipt_path = fragment_dir / "completion_receipt.json"
    if not receipt_path.exists():
        errors.append("completion receipt missing")
        return result
    try:
        receipt = json.loads(receipt_path.read_text())
    except json.JSONDecodeError:
        errors.append("completion receipt corrupt")
        return result
    if receipt.get("status") != "complete":
        errors.append(f"receipt status: {receipt.get('status')}")
        return result
    if receipt.get("canonical_key_id") != cid:
        errors.append("receipt cid mismatch")
        return result
    result.receipt_valid = True

    # ── Fragment manifest check ──
    manifest_path = fragment_dir / "fragment_manifest.json"
    if not manifest_path.exists():
        errors.append("fragment manifest missing")
        return result
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        errors.append("fragment manifest corrupt")
        return result
    if manifest.get("canonical_key_id") != cid:
        errors.append("manifest cid mismatch")
        return result
    if receipt.get("fragment_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        errors.append("receipt manifest SHA mismatch")
        return result
    result.fragment_manifest_valid = True

    # ── File SHA checks ──
    for name, key in [("baseline", "baseline_rows"), ("governed", "governed_rows"), ("selection", "selection_rows"), ("failure", "failure_rows")]:
        fp = fragment_dir / f"{name}.csv.gz"
        if not fp.exists():
            errors.append(f"{name} fragment missing")
            continue
        actual_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        if actual_sha != manifest.get(f"{name}_sha256", ""):
            errors.append(f"{name} SHA mismatch")
    if errors:
        return result
    result.fragment_sha_valid = True

    # ── Row counts ──
    bl_path = fragment_dir / "baseline.csv.gz"
    gl_path = fragment_dir / "governed.csv.gz"
    sl_path = fragment_dir / "selection.csv.gz"
    fl_path = fragment_dir / "failure.csv.gz"
    bl_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(bl_path.read_bytes())))
    gl_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(gl_path.read_bytes())),
                         dtype={"selection_hash": str})
    sl_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(sl_path.read_bytes())),
                         dtype={"selection_hash": str, "removed_encoded_indices": str, "removed_group_ids": str})
    fl_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(fl_path.read_bytes())))

    result.baseline_rows = len(bl_df)
    result.governed_rows = len(gl_df)
    result.selection_rows = len(sl_df)
    result.failure_rows = len(fl_df)

    if result.baseline_rows != 2:
        errors.append(f"baseline rows: {result.baseline_rows}")
    if result.governed_rows != 144:
        errors.append(f"governed rows: {result.governed_rows}")
    if result.failure_rows != 0:
        errors.append(f"failure rows: {result.failure_rows}")
    if errors:
        return result

    # ── Run ID checks ──
    produced_ids = list(bl_df["run_id"]) + list(gl_df["run_id"])
    null_count = sum(1 for rid in produced_ids if not isinstance(rid, str) or rid == "")
    result.null_run_id_count = null_count
    if null_count > 0:
        errors.append(f"null run IDs: {null_count}")

    # Detect duplicates BEFORE set comparison
    seen = {}; dups = []
    for rid in produced_ids:
        if rid in seen:
            if seen[rid] == 1: dups.append(rid)
            seen[rid] += 1
        else:
            seen[rid] = 1
    result.duplicate_run_ids = dups
    if dups:
        errors.append(f"duplicate run IDs: {len(dups)}")
        return result

    produced_set = set(produced_ids)
    planned_set = set(planned_run_ids)
    result.missing_run_ids = sorted(planned_set - produced_set)
    result.extra_run_ids = sorted(produced_set - planned_set)
    if result.missing_run_ids:
        errors.append(f"missing run IDs: {len(result.missing_run_ids)}")
    if result.extra_run_ids:
        errors.append(f"extra run IDs: {len(result.extra_run_ids)}")
    if result.missing_run_ids or result.extra_run_ids:
        return result
    result.run_id_closure_valid = True

    # ── Selection semantic validation ──
    try:
        normalized_payloads = [normalize_selection_payload(row) for _, row in sl_df.iterrows()]
    except SelectionContractError as e:
        errors.append(f"selection payload parse error: {e}")
        return result

    # Multiset closure (with payload normalization)
    multiset_errors = validate_selection_multiset_closure(gl_df, normalized_payloads)
    if multiset_errors:
        errors.extend(multiset_errors); return result
    result.selection_closure_valid = True

    # Payload consistency (same hash, same payload)
    payload_errors = validate_selection_payload_consistency(normalized_payloads)
    if payload_errors:
        errors.extend(payload_errors); return result
    result.selection_payload_valid = True

    # Selection realized-cost closure
    sc_errors = validate_selection_realized_cost(normalized_payloads)
    if sc_errors:
        errors.extend(sc_errors); return result

    # Governed realized-cost closure
    payload_by_hash = {p["selection_hash"]: p for p in normalized_payloads}
    gc_errors = validate_governed_realized_cost(gl_df, payload_by_hash)
    if gc_errors:
        errors.extend(gc_errors); return result
    result.realized_cost_valid = True

    # ── Policy mapping + semantic validation ──
    try:
        validated_map = validate_policy_mapping(policy_mapping, key_plan_row)
    except SelectionContractError as e:
        errors.append(f"policy mapping invalid: {e}")
        result.is_complete = False; return result
    result.mapping_valid = True

    # Encoded-column contract validation
    for p in normalized_payloads:
        ec_errors = validate_encoded_column_contract(p, validated_map)
        if ec_errors:
            errors.extend(ec_errors)
    if errors:
        result.is_complete = False; return result

    # Semantic-group atomicity
    for p in normalized_payloads:
        sem_errors = validate_semantic_group_atomicity(p, {gid: set(members) for gid, members in validated_map.group_members.items()})
        if sem_errors:
            errors.extend(sem_errors)
    if errors:
        result.is_complete = False; return result

    # M09 validation (uses validated semantic mapping)
    try:
        validated_semantic = validate_semantic_mapping(semantic_mapping, key_plan_row, validated_map)
    except SelectionContractError as e:
        errors.append(f"semantic mapping invalid: {e}")
        result.is_complete = False; return result
    for p in normalized_payloads:
        try:
            m09_errors = validate_m09_eight_columns(p, validated_semantic, key_plan_row)
        except SelectionContractError as e:
            errors.append(f"semantic mapping invalid: {e}")
            result.is_complete = False; return result
        if m09_errors:
            errors.extend(m09_errors)
    if errors:
        result.is_complete = False; return result
    # Set semantic flags — M09 passes for non-M09 keys, validated for M09 keys
    result.semantic_atomicity_valid = True
    result.m09_atomicity_valid = True

    # Manifest selection digest closure
    sel_hashes = [p["selection_hash"] for p in normalized_payloads]
    manifest_errors = validate_manifest_selection_digest(manifest, sel_hashes, normalized_payloads)
    if manifest_errors:
        errors.extend(manifest_errors)
        result.is_complete = False; return result
    result.manifest_selection_digest_valid = True

    result.is_complete = (
        result.receipt_valid and result.fragment_manifest_valid and result.fragment_sha_valid
        and result.run_id_closure_valid and result.selection_closure_valid
        and result.selection_payload_valid and result.realized_cost_valid
        and result.mapping_valid and result.semantic_atomicity_valid
        and result.m09_atomicity_valid and result.manifest_selection_digest_valid
    )
    return result
