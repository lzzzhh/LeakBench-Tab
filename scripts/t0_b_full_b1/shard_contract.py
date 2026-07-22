"""T0-B Shard Contract — deterministic manifest, digest helpers, artifact validator."""
from __future__ import annotations
from dataclasses import dataclass, field
import csv, gzip, hashlib, io, json
from pathlib import Path
import pandas as pd

from scripts.t0_b_full_b1.fragment_contract import (
    SCIENTIFIC_FREEZE_SHA, EXECUTION_CONTRACT_VERSION,
    key_plan_row_sha256,
)

SHARD_MANIFEST_SCHEMA_VERSION = 1

# Forbidden dynamic fields in deterministic manifest
_FORBIDDEN_DYNAMIC_FIELDS = frozenset({
    "new_keys", "complete_keys", "recomputed", "skipped", "quarantined",
    "invalid", "counter", "timestamp", "completed_utc", "resume", "repair",
    "wall_clock", "pid", "hostname",
})


@dataclass
class ShardArtifactValidation:
    is_valid: bool
    errors: list[str] = field(default_factory=list)

    manifest_schema_valid: bool = False
    provenance_valid: bool = False
    artifact_sha_valid: bool = False
    planned_scope_valid: bool = False
    fragment_aggregate_valid: bool = False
    fragment_sources_valid: bool = False
    row_counts_valid: bool = False
    key_universe_valid: bool = False
    run_id_universe_valid: bool = False
    selection_multiset_valid: bool = False
    fragment_manifest_set_valid: bool = False

    baseline_rows: int = 0
    governed_rows: int = 0
    selection_rows: int = 0
    failure_rows: int = 0


# ═══════════════════════════════════════════════════════════════════
# Canonical digest helpers
# ═══════════════════════════════════════════════════════════════════

def sorted_lines_sha256(values: list[str]) -> str:
    content = "\n".join(sorted(values)) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def canonical_key_ids_sha256(canonical_key_ids: list[str]) -> str:
    """Deterministic SHA of sorted, deduplicated key IDs."""
    if any(not k or k == "" for k in canonical_key_ids):
        raise ValueError("null or empty canonical key ID")
    if len(set(canonical_key_ids)) != len(canonical_key_ids):
        raise ValueError("duplicate canonical key IDs")
    return sorted_lines_sha256(canonical_key_ids)


def produced_run_ids_sha256(baseline_df: pd.DataFrame, governed_df: pd.DataFrame) -> str:
    """Deterministic SHA of produced run IDs (baseline + governed)."""
    bl_ids = baseline_df["run_id"].tolist()
    gl_ids = governed_df["run_id"].tolist()
    all_ids = bl_ids + gl_ids
    if any(not isinstance(rid, str) or rid == "" for rid in all_ids):
        raise ValueError("null or empty produced run ID")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate produced run IDs")
    return sorted_lines_sha256(all_ids)


def selection_hash_multiset_sha256(selection_df: pd.DataFrame) -> str:
    """Deterministic SHA of selection hashes, preserving multiplicity."""
    return sorted_lines_sha256(selection_df["selection_hash"].tolist())


def fragment_manifest_set_sha256(key_fragment_dirs: list[Path]) -> str:
    """Deterministic SHA of (cid, fragment_manifest_sha) pairs."""
    entries = []
    for fdir in sorted(key_fragment_dirs, key=lambda p: p.name):
        cid = fdir.name
        mp = fdir / "fragment_manifest.json"
        if not mp.exists():
            raise ValueError(f"fragment_manifest.json missing for {cid}")
        msha = hashlib.sha256(mp.read_bytes()).hexdigest()
        entries.append(f"{cid}\t{msha}")
    return sorted_lines_sha256(entries)


# ═══════════════════════════════════════════════════════════════════
# Strict fragment CSV parser + source validation
# ═══════════════════════════════════════════════════════════════════

_SHARD_LEDGER_HEADERS = {
    "baseline": "run_id,dataset_index,mechanism,strength,training_seed,learner,baseline_type,auc",
    "governed": "run_id,dataset_index,mechanism,strength,training_seed,governance_seed,learner,policy,contract,budget_bp,strict_auc,full_auc,governed_auc,legacy_sdr,selection_hash,realized_cost",
    "selection": "selection_hash,policy,contract,budget_bp,removed_encoded_indices,removed_group_ids,realized_encoded_cost",
    "failure": "run_id",
}

_EXPECTED_FIELD_COUNTS = {name: len(h.split(",")) for name, h in _SHARD_LEDGER_HEADERS.items()}


def _parse_fragment_csv_strict(
    fragment_dir: Path, cid: str, name: str, header: str,
) -> list[str]:
    """Parse a fragment CSV with physical-line validation. Returns original raw data rows.

    Rejects: gzip corruption, UTF-8 errors, exact header mismatch, blank physical
    lines, wrong field counts, CSV quoting errors, embedded newlines in fields,
    multiline logical records spanning physical lines.
    Never silently normalizes, requotes, or drops rows.
    """
    raw = _check_common_fragment_file(fragment_dir, cid, name)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"active key {cid}: {name}.csv.gz UTF-8 decode error: {exc}") from exc

    if "\r" in text:
        raise ValueError(f"active key {cid}: {name}.csv.gz contains CR (carriage return)")

    # Must end with exactly one trailing newline
    if not text.endswith("\n"):
        raise ValueError(f"active key {cid}: {name}.csv.gz missing trailing newline")

    # Split into physical lines (remove trailing empty from final \n)
    physical_lines = text[:-1].split("\n")

    if not physical_lines:
        raise ValueError(f"active key {cid}: {name}.csv.gz is empty")

    # Exact raw header comparison
    if physical_lines[0] != header:
        raise ValueError(
            f"active key {cid}: {name}.csv.gz exact header mismatch, "
            f"expected [{header[:80]}], got [{physical_lines[0][:80]}]"
        )

    expected_count = _EXPECTED_FIELD_COUNTS[name]
    data_rows = []

    for i, physical_line in enumerate(physical_lines[1:], start=1):
        # Blank physical line
        if physical_line == "":
            raise ValueError(
                f"active key {cid}: {name}.csv.gz blank data record at row {i}"
            )

        # Parse this single physical line with csv.reader
        try:
            parsed = list(csv.reader([physical_line], strict=True))
        except csv.Error as exc:
            raise ValueError(f"active key {cid}: {name}.csv.gz CSV parse error: {exc}") from exc

        # Must produce exactly one record
        if len(parsed) != 1:
            raise ValueError(
                f"active key {cid}: {name}.csv.gz embedded newline or multiline "
                f"CSV record is forbidden at row {i}"
            )

        row = parsed[0]
        if len(row) != expected_count:
            raise ValueError(
                f"active key {cid}: {name}.csv.gz row {i} has {len(row)} fields; "
                f"expected {expected_count}"
            )

        # Reject embedded newline/carriage return in any field
        for fi, field in enumerate(row):
            if "\n" in field or "\r" in field:
                raise ValueError(
                    f"active key {cid}: {name}.csv.gz embedded newline in field {fi} at row {i}"
                )

        # Return the original physical data row
        data_rows.append(physical_line)

    return data_rows


def _check_common_fragment_file(
    fragment_dir: Path, cid: str, name: str,
) -> bytes:
    """Read a single fragment gzip file. Raises ValueError on any failure."""
    fp = fragment_dir / f"{name}.csv.gz"
    if not fp.exists():
        raise ValueError(f"active key {cid}: {name}.csv.gz missing")
    if not fp.is_file():
        raise ValueError(f"active key {cid}: {name}.csv.gz is not a regular file")
    try:
        raw_bytes = fp.read_bytes()
        decompressed = gzip.decompress(raw_bytes)
    except (gzip.BadGzipFile, EOFError) as exc:
        raise ValueError(f"active key {cid}: {name}.csv.gz gzip corrupt: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"active key {cid}: {name}.csv.gz read error: {exc}") from exc
    return decompressed


def _compute_ids_sha256(ids: list[str]) -> str:
    """Deterministic SHA-256 of sorted ID list (same as fragment contract _ids_sha256)."""
    content = "\n".join(sorted(ids)) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def collect_validated_active_fragment_rows(
    *,
    output_dir: Path,
    shard_key_rows: list[dict],
    shard_run_rows: list[dict],
    plan_manifest_sha256: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Validate all fragment source bindings and return validated data rows.

    For each planned active key, validates:
    - fragment_manifest.json existence, parse, provenance
    - fragment file SHA against manifest declarations
    - strict CSV parse of all four fragment files
    - row count binding
    - planned/produced run-ID digests

    Returns (collected_rows, errors) where collected_rows maps ledger name to
    list of validated raw data-row strings. If errors is non-empty, collected_rows
    is empty and the caller must NOT proceed with aggregation.
    """
    out = Path(output_dir)
    errors = []
    collected = {name: [] for name in _SHARD_LEDGER_HEADERS}

    cids = []
    if not shard_key_rows:
        errors.append("no planned keys for shard")
        return collected, errors
    if not shard_run_rows:
        errors.append("no planned runs for shard")
        return collected, errors
    for kp in shard_key_rows:
        cid = kp.get("canonical_key_id")
        if not cid or not isinstance(cid, str) or cid.strip() == "":
            errors.append(f"invalid canonical_key_id: {cid!r}")
            continue
        cids.append(cid)
    if len(set(cids)) != len(cids):
        errors.append("duplicate canonical_key_ids in shard_key_rows")
    if errors:
        return collected, errors

    # Key/run membership closure
    run_key_ids = set()
    for r in shard_run_rows:
        rcid = r.get("canonical_key_id")
        if not rcid or not isinstance(rcid, str) or rcid.strip() == "":
            errors.append(f"run plan row has invalid canonical_key_id: {rcid!r}")
        run_key_ids.add(rcid)
    extra_run_keys = run_key_ids - set(cids)
    if extra_run_keys:
        errors.append(f"run plan contains keys outside shard key universe: {sorted(extra_run_keys)}")
    for cid in cids:
        if cid not in run_key_ids:
            errors.append(f"planned key {cid} has no run rows")
    if errors:
        return collected, errors

    fragment_dir = out / "key_fragments"

    for kp, cid in zip(shard_key_rows, cids):
        kdir = fragment_dir / cid
        if not kdir.exists() or not kdir.is_dir():
            errors.append(f"active key {cid}: key_fragments directory missing")
            continue

        # ── Fragment manifest validation ──
        mp = kdir / "fragment_manifest.json"
        if not mp.exists():
            errors.append(f"active key {cid}: fragment_manifest.json missing")
            continue
        if not mp.is_file():
            errors.append(f"active key {cid}: fragment_manifest.json is not a regular file")
            continue
        try:
            fm = json.loads(mp.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"active key {cid}: fragment_manifest.json corrupt JSON: {exc}")
            continue
        if not isinstance(fm, dict):
            errors.append(f"active key {cid}: fragment_manifest.json not a JSON object")
            continue

        # Provenance
        if fm.get("schema_version") != 1:
            errors.append(f"active key {cid}: fragment manifest schema_version mismatch")
        if fm.get("canonical_key_id") != cid:
            errors.append(f"active key {cid}: fragment manifest canonical_key_id mismatch")
        if fm.get("scientific_freeze_sha") != SCIENTIFIC_FREEZE_SHA:
            errors.append(f"active key {cid}: fragment manifest scientific_freeze_sha mismatch")
        if fm.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
            errors.append(f"active key {cid}: fragment manifest execution_contract_version mismatch")
        if fm.get("plan_manifest_sha256") != plan_manifest_sha256:
            errors.append(f"active key {cid}: fragment manifest plan_manifest_sha256 mismatch")
        if fm.get("key_plan_row_sha256") != key_plan_row_sha256(kp):
            errors.append(f"active key {cid}: fragment manifest key_plan_row_sha256 mismatch")

        # Planned run-ID digest
        planned_for_key = sorted(r["run_id"] for r in shard_run_rows if r.get("canonical_key_id") == cid)
        if fm.get("planned_run_ids_sha256") != _compute_ids_sha256(planned_for_key):
            errors.append(f"active key {cid}: fragment manifest planned_run_ids_sha256 mismatch")

        if errors:
            continue  # skip CSV parse if provenance failed

        # ── Fragment file SHA + strict CSV + row counts ──
        fragment_names = ["baseline", "governed", "selection", "failure"]
        fragment_rows = {}
        fragment_row_counts = {}

        for name in fragment_names:
            fp = kdir / f"{name}.csv.gz"
            if not fp.exists():
                errors.append(f"active key {cid}: {name}.csv.gz missing")
                continue
            actual_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
            manifest_sha = fm.get(f"{name}_sha256")
            if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
                errors.append(f"active key {cid}: fragment manifest {name}_sha256 invalid format")
                continue
            if actual_sha != manifest_sha:
                errors.append(f"active key {cid}: {name}.csv.gz sha256 mismatch against fragment manifest")
                continue

            # Strict CSV parse
            try:
                rows = _parse_fragment_csv_strict(kdir, cid, name, _SHARD_LEDGER_HEADERS[name])
            except ValueError as exc:
                errors.append(str(exc))
                continue
            fragment_rows[name] = rows
            fragment_row_counts[name] = len(rows)

        if errors:
            continue

        # Row count binding
        for name in fragment_names:
            manifest_rows = fm.get(f"{name}_rows")
            if not isinstance(manifest_rows, int) or isinstance(manifest_rows, bool):
                errors.append(f"active key {cid}: fragment manifest {name}_rows invalid type")
            elif manifest_rows != fragment_row_counts.get(name, -1):
                errors.append(
                    f"active key {cid}: {name} row count mismatch: "
                    f"manifest={manifest_rows}, actual={fragment_row_counts[name]}"
                )

        # Produced run-ID digest
        produced_ids = fragment_rows.get("baseline", []) + fragment_rows.get("governed", [])
        produced_run_ids = [r.split(",")[0] for r in produced_ids]
        if any(not rid or rid == "" for rid in produced_run_ids):
            errors.append(f"active key {cid}: null/empty produced run ID")
        elif len(set(produced_run_ids)) != len(produced_run_ids):
            errors.append(f"active key {cid}: duplicate produced run IDs")
        else:
            actual_produced_sha = _compute_ids_sha256(produced_run_ids)
            manifest_produced = fm.get("produced_run_ids_sha256")
            if manifest_produced != actual_produced_sha:
                errors.append(f"active key {cid}: fragment manifest produced_run_ids_sha256 mismatch")

        if errors:
            continue

        # Collect validated rows
        for name in fragment_names:
            collected[name].extend(fragment_rows[name])

    return collected, errors


# ═══════════════════════════════════════════════════════════════════
# Canonical active-fragment aggregate → deterministic shard ledger bytes
# ═══════════════════════════════════════════════════════════════════

def build_canonical_shard_ledger_bytes(
    *,
    output_dir: Path,
    shard_key_rows: list[dict],
    shard_run_rows: list[dict],
    plan_manifest_sha256: str,
) -> dict[str, bytes]:
    """Build deterministic shard ledger bytes from validated active fragment rows.

    Validates all fragment source bindings (manifest provenance, file SHA,
    strict CSV parse, row counts, run-ID digests), then aggregates validated
    rows into canonical sorted ledger bytes.

    Never reads baseline_ledger.csv.gz, governed_ledger.csv.gz, etc.
    Only reads from key_fragments/<cid>/<name>.csv.gz.
    """
    collected, errors = collect_validated_active_fragment_rows(
        output_dir=output_dir,
        shard_key_rows=shard_key_rows,
        shard_run_rows=shard_run_rows,
        plan_manifest_sha256=plan_manifest_sha256,
    )
    if errors:
        raise ValueError("\n".join(errors))

    result = {}
    for name, header in _SHARD_LEDGER_HEADERS.items():
        rows = sorted(collected[name])
        # Data rows are already properly quoted CSV lines (from csv.writer in _parse_fragment_csv_strict)
        content = header + "\n" + (("\n".join(rows) + "\n") if rows else "")
        result[name] = gzip.compress(content.encode("utf-8"), mtime=0)

    return result


# ═══════════════════════════════════════════════════════════════════
# Build shard manifest
# ═══════════════════════════════════════════════════════════════════

def build_shard_manifest(
    *,
    mode: str,
    shard_id: int,
    plan_manifest: dict,
    plan_manifest_sha256: str,
    shard_key_rows: list[dict],
    shard_run_rows: list[dict],
    output_dir: Path,
) -> dict:
    """Build deterministic shard manifest from on-disk ledger files."""
    out = Path(output_dir)

    # Read actual ledger bytes + DataFrames
    ledger_specs = {
        "baseline": ("baseline_ledger.csv.gz", {"run_id": str, "baseline_type": str}),
        "governed": ("governed_ledger.csv.gz", {"selection_hash": str, "run_id": str}),
        "selection": ("selection_ledger.csv.gz", {"selection_hash": str}),
        "failure": ("failure_ledger.csv.gz", None),
    }

    dfs = {}
    shas = {}
    for name, (fname, dtype) in ledger_specs.items():
        fp = out / fname
        raw = fp.read_bytes()
        shas[name] = hashlib.sha256(raw).hexdigest()
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        dfs[name] = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(raw)), **kwargs)

    bl_df, gl_df, sl_df, fl_df = dfs["baseline"], dfs["governed"], dfs["selection"], dfs["failure"]

    baseline_rows = len(bl_df)
    governed_rows = len(gl_df)
    selection_rows = len(sl_df)
    failure_rows = len(fl_df)

    # Key universe
    cids = sorted(k["canonical_key_id"] for k in shard_key_rows)

    # Run universe
    planned_run_ids = sorted(r["run_id"] for r in shard_run_rows)
    produced_run_ids = produced_run_ids_sha256(bl_df, gl_df)

    # Selection multiset
    sel_multiset = selection_hash_multiset_sha256(sl_df)

    # Fragment manifest set
    key_dirs = [out / "key_fragments" / cid for cid in cids]
    frag_set = fragment_manifest_set_sha256(key_dirs)

    # Plan declared tool seal
    tool_seal = plan_manifest.get("tool_seal_sha", plan_manifest.get("plan_tool_seal_sha", ""))

    return {
        "schema_version": SHARD_MANIFEST_SCHEMA_VERSION,
        "mode": mode,
        "shard_id": int(shard_id),
        "scientific_freeze_sha": SCIENTIFIC_FREEZE_SHA,
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "plan_manifest_sha256": plan_manifest_sha256,
        "plan_declared_tool_seal_sha": tool_seal,

        "key_count": len(cids),
        "baseline_rows": baseline_rows,
        "governed_rows": governed_rows,
        "selection_rows": selection_rows,
        "failure_rows": failure_rows,
        "downstream_rows": baseline_rows + governed_rows,

        "baseline_sha256": shas["baseline"],
        "governed_sha256": shas["governed"],
        "selection_sha256": shas["selection"],
        "failure_sha256": shas["failure"],

        "completed_key_ids_sha256": canonical_key_ids_sha256(cids),
        "planned_run_ids_sha256": sorted_lines_sha256(planned_run_ids),
        "produced_run_ids_sha256": produced_run_ids,
        "selection_hash_multiset_sha256": sel_multiset,
        "fragment_manifest_set_sha256": frag_set,
    }


# ═══════════════════════════════════════════════════════════════════
# Validate shard artifacts
# ═══════════════════════════════════════════════════════════════════

def _read_ledger_csv(fp: Path, name: str, dtype: dict | None = None) -> pd.DataFrame:
    raw = fp.read_bytes()
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return pd.read_csv(pd.io.common.BytesIO(gzip.decompress(raw)), **kwargs)


def _sha256_file(fp: Path) -> str:
    return hashlib.sha256(fp.read_bytes()).hexdigest()


def validate_shard_artifacts(
    *,
    output_dir: Path,
    plan_manifest: dict,
    plan_manifest_sha256: str,
    shard_key_rows: list[dict],
    shard_run_rows: list[dict],
) -> ShardArtifactValidation:
    out = Path(output_dir)
    errors = []
    result = ShardArtifactValidation(is_valid=False, errors=errors)

    # ── Scope: non-empty valid input ──
    if not shard_key_rows:
        errors.append("no planned keys for shard")
        return result
    if not shard_run_rows:
        errors.append("no planned runs for shard")
        return result
    cids = []
    for kp in shard_key_rows:
        cid = kp.get("canonical_key_id")
        if not cid or not isinstance(cid, str) or cid.strip() == "":
            errors.append(f"invalid canonical_key_id in shard_key_rows: {cid!r}")
            continue
        cids.append(cid)
    if len(set(cids)) != len(cids):
        errors.append("duplicate canonical_key_ids in shard_key_rows")
        return result
    for r in shard_run_rows:
        rcid = r.get("canonical_key_id")
        if not rcid or not isinstance(rcid, str) or rcid.strip() == "":
            errors.append(f"run plan row has invalid canonical_key_id: {rcid!r}")
    for c in cids:
        if not any(r.get("canonical_key_id") == c for r in shard_run_rows):
            errors.append(f"planned key {c} has no run rows")
    if errors:
        return result
    result.planned_scope_valid = True

    # ── A. Manifest existence + parse ──
    mp = out / "shard_manifest.json"
    if not mp.exists():
        errors.append("shard_manifest.json missing")
        return result
    try:
        sm = json.loads(mp.read_text())
    except json.JSONDecodeError:
        errors.append("shard manifest corrupt JSON")
        return result
    if not isinstance(sm, dict):
        errors.append("shard manifest not a JSON object")
        return result

    # ── No dynamic fields ──
    for field in _FORBIDDEN_DYNAMIC_FIELDS:
        if field in sm:
            errors.append(f"shard manifest contains dynamic field: {field}")

    # ── Schema + provenance ──
    if sm.get("schema_version") != SHARD_MANIFEST_SCHEMA_VERSION:
        errors.append(f"shard manifest schema_version: expected {SHARD_MANIFEST_SCHEMA_VERSION}")
    if sm.get("mode") not in ("synthetic", "production"):
        errors.append(f"shard manifest invalid mode: {sm.get('mode')}")
    if not isinstance(sm.get("shard_id"), int) or sm["shard_id"] < 0:
        errors.append(f"shard manifest invalid shard_id: {sm.get('shard_id')}")
    if sm.get("scientific_freeze_sha") != SCIENTIFIC_FREEZE_SHA:
        errors.append("shard manifest scientific_freeze_sha mismatch")
    if sm.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
        errors.append("shard manifest execution_contract_version mismatch")
    if sm.get("plan_manifest_sha256") != plan_manifest_sha256:
        errors.append("shard manifest plan_manifest_sha256 mismatch")
    tool_seal = plan_manifest.get("tool_seal_sha", plan_manifest.get("plan_tool_seal_sha", ""))
    if sm.get("plan_declared_tool_seal_sha") != tool_seal:
        errors.append("shard manifest plan_declared_tool_seal_sha mismatch")
    if errors:
        return result
    result.manifest_schema_valid = True
    result.provenance_valid = True

    # ── B. Artifact bytes SHA ──
    artifact_shas = {}
    for name in ["baseline", "governed", "selection", "failure"]:
        fp = out / f"{name}_ledger.csv.gz"
        if not fp.exists():
            errors.append(f"{name}_ledger.csv.gz missing")
            continue
        artifact_shas[name] = _sha256_file(fp)
        manifest_sha = sm.get(f"{name}_sha256")
        if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
            errors.append(f"shard manifest {name}_sha256 invalid format")
        elif manifest_sha != artifact_shas[name]:
            errors.append(f"shard manifest {name}_sha256 mismatch")
    if errors:
        return result
    result.artifact_sha_valid = True

    # ── C. Read CSVs + row counts ──
    dfs = {}
    for name in ["baseline", "governed", "selection", "failure"]:
        fp = out / f"{name}_ledger.csv.gz"
        dfs[name] = _read_ledger_csv(fp, name,
            dtype={"selection_hash": str, "run_id": str} if name in ("governed", "selection", "baseline") else None)
    bl_df, gl_df, sl_df, fl_df = dfs["baseline"], dfs["governed"], dfs["selection"], dfs["failure"]

    result.baseline_rows = len(bl_df)
    result.governed_rows = len(gl_df)
    result.selection_rows = len(sl_df)
    result.failure_rows = len(fl_df)

    for name, actual, key in [("baseline", result.baseline_rows, "baseline_rows"),
                                ("governed", result.governed_rows, "governed_rows"),
                                ("selection", result.selection_rows, "selection_rows"),
                                ("failure", result.failure_rows, "failure_rows")]:
        if sm.get(key) != actual:
            errors.append(f"shard manifest {key} mismatch: expected {actual}, got {sm.get(key)}")
    if sm.get("downstream_rows") != result.baseline_rows + result.governed_rows:
        errors.append("shard manifest downstream_rows mismatch")
    if errors:
        return result
    result.row_counts_valid = True

    # ── D. Key universe ──
    cids = sorted(k["canonical_key_id"] for k in shard_key_rows)
    if sm.get("key_count") != len(cids):
        errors.append("shard manifest key_count mismatch")
    if sm.get("completed_key_ids_sha256") != canonical_key_ids_sha256(cids):
        errors.append("shard manifest completed_key_ids_sha256 mismatch")
    # Verify active key directories
    frag_dir = out / "key_fragments"
    if frag_dir.exists():
        actual_dirs = {d.name for d in frag_dir.iterdir() if d.is_dir() and d.name not in ("quarantine",)}
        expected_dirs = set(cids)
        missing = expected_dirs - actual_dirs
        extra = actual_dirs - expected_dirs
        if missing:
            errors.append(f"missing active key directories: {sorted(missing)}")
        if extra:
            errors.append(f"extra active key directories: {sorted(extra)}")
    if errors:
        return result
    result.key_universe_valid = True

    # ── E. Run universe ──
    all_produced = bl_df["run_id"].tolist() + gl_df["run_id"].tolist()
    if any(not isinstance(r, str) or r == "" for r in all_produced):
        errors.append("null or empty produced run ID in ledgers")
    if len(set(all_produced)) != len(all_produced):
        errors.append("duplicate produced run IDs in ledgers")
    planned_ids = sorted(r["run_id"] for r in shard_run_rows)
    if sm.get("planned_run_ids_sha256") != sorted_lines_sha256(planned_ids):
        errors.append("shard manifest planned_run_ids_sha256 mismatch")
    actual_produced_sha = produced_run_ids_sha256(bl_df, gl_df)
    if sm.get("produced_run_ids_sha256") != actual_produced_sha:
        errors.append("shard manifest produced_run_ids_sha256 mismatch")
    planned_set = set(planned_ids)
    produced_set = set(all_produced)
    if planned_set - produced_set:
        errors.append(f"missing run IDs: {len(planned_set - produced_set)}")
    if produced_set - planned_set:
        errors.append(f"extra run IDs: {len(produced_set - planned_set)}")
    if errors:
        return result
    result.run_id_universe_valid = True

    # ── F. Selection multiset ──
    actual_multiset = selection_hash_multiset_sha256(sl_df)
    if sm.get("selection_hash_multiset_sha256") != actual_multiset:
        errors.append("shard manifest selection_hash_multiset_sha256 mismatch")
    if errors:
        return result
    result.selection_multiset_valid = True

    # ── G. Fragment manifest set ──
    key_dirs = [out / "key_fragments" / cid for cid in cids]
    try:
        actual_frag_set = fragment_manifest_set_sha256(key_dirs)
    except ValueError as e:
        errors.append(str(e))
        return result
    if sm.get("fragment_manifest_set_sha256") != actual_frag_set:
        errors.append("shard manifest fragment_manifest_set_sha256 mismatch")
    if errors:
        return result
    result.fragment_manifest_set_valid = True

    # ── H. Active-fragment source binding ──
    collected, source_errors = collect_validated_active_fragment_rows(
        output_dir=output_dir,
        shard_key_rows=shard_key_rows,
        shard_run_rows=shard_run_rows,
        plan_manifest_sha256=plan_manifest_sha256,
    )
    if source_errors:
        errors.extend(source_errors)
        return result
    result.fragment_sources_valid = True

    # ── I. Active-fragment aggregate byte-level closure ──
    try:
        expected_bytes = build_canonical_shard_ledger_bytes(
            output_dir=output_dir,
            shard_key_rows=shard_key_rows,
            shard_run_rows=shard_run_rows,
            plan_manifest_sha256=plan_manifest_sha256,
        )
    except ValueError as exc:
        errors.append(str(exc))
        return result

    for name in ["baseline", "governed", "selection", "failure"]:
        disk_path = out / f"{name}_ledger.csv.gz"
        if not disk_path.exists():
            continue
        disk_bytes = disk_path.read_bytes()
        expected = expected_bytes[name]
        if disk_bytes != expected:
            disk_sha = hashlib.sha256(disk_bytes).hexdigest()
            exp_sha = hashlib.sha256(expected).hexdigest()
            errors.append(
                f"{name} shard ledger does not match canonical active-fragment "
                f"aggregate (expected_sha256={exp_sha}, actual_sha256={disk_sha})"
            )
    if errors:
        return result
    result.fragment_aggregate_valid = True

    result.is_valid = (
        result.planned_scope_valid
        and result.manifest_schema_valid and result.provenance_valid
        and result.artifact_sha_valid and result.fragment_sources_valid
        and result.fragment_aggregate_valid
        and result.row_counts_valid
        and result.key_universe_valid and result.run_id_universe_valid
        and result.selection_multiset_valid and result.fragment_manifest_set_valid
    )
    return result
