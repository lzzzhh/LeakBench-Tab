"""T0-B Fragment Contract — manifest, receipt, validation dataclasses."""
from __future__ import annotations
from dataclasses import dataclass, field
import gzip, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd


# ====================================================================
# Counters
# ====================================================================

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
    governed_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(governed_path.read_bytes())))
    selection_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(selection_path.read_bytes())))
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
        sel_payloads.append(json.dumps({
            "selection_hash": r["selection_hash"],
            "policy": r["policy"],
            "contract": r["contract"],
            "budget_bp": int(r["budget_bp"]),
            "removed_encoded_indices": sorted(json.loads(r["removed_encoded_indices"])),
            "removed_group_ids": sorted(json.loads(r["removed_group_ids"])),
            "realized_encoded_cost": int(r["realized_encoded_cost"]),
        }, sort_keys=True, separators=(",", ":")))
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


def validate_completed_key(
    key_plan_row: dict,
    planned_run_ids: list[str],
    fragment_dir: Path,
    plan_manifest_sha256: str,
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
    gl_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(gl_path.read_bytes())))
    sl_df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(sl_path.read_bytes())))
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

    # ── Selection multiset closure ──
    from collections import Counter
    gov_counter = Counter(gl_df["selection_hash"].dropna())
    sel_counter = Counter(sl_df["selection_hash"].dropna())
    if gov_counter != sel_counter:
        errors.append("selection multiset mismatch")
        return result
    result.selection_closure_valid = True

    result.is_complete = len(errors) == 0
    return result
