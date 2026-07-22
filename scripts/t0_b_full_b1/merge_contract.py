"""T0-B R10c Merge Contract — plan validation, scope closure, shard-set admission."""
from __future__ import annotations
from dataclasses import dataclass, field
import gzip, hashlib, json, re, zlib
from pathlib import Path

from scripts.t0_b_full_b1.fragment_contract import (
    SCIENTIFIC_FREEZE_SHA, EXECUTION_CONTRACT_VERSION,
)
from scripts.t0_b_full_b1.shard_contract import (
    validate_shard_artifacts, ShardArtifactValidation,
)

_SHARD_DIR_RE = re.compile(r"^shard_(0|[1-9][0-9]*)$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class ShardSetAdmissionResult:
    """Structured result of strict shard-set admission."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)

    plan_valid: bool = False
    scope_valid: bool = False
    directory_universe_valid: bool = False
    all_shards_valid: bool = False
    global_counts_valid: bool = False

    planned_shard_ids: list[int] = field(default_factory=list)
    validated_shard_ids: list[int] = field(default_factory=list)

    canonical_keys: int = 0
    baseline_rows: int = 0
    governed_rows: int = 0
    selection_rows: int = 0
    failure_rows: int = 0
    downstream_rows: int = 0


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_strict_integer(value, field_name: str, parent: str) -> int:
    """Require a strict integer (not bool, not float, not str)."""
    if isinstance(value, bool):
        raise ValueError(f"{parent} {field_name} must be a non-negative integer, got bool ({value})")
    if not isinstance(value, int):
        raise ValueError(f"{parent} {field_name} must be a non-negative integer, got {type(value).__name__} ({value})")
    return value


def _require_hex64(value, field_name: str, parent: str) -> str:
    if not isinstance(value, str) or not _HEX64_RE.match(value):
        raise ValueError(f"{parent} {field_name} must be a 64-char hex string, got {value!r}")
    return value


def _load_jsonl_gzip_strict(path: Path, label: str) -> list[dict]:
    """Strict gzip/UTF-8/JSONL loader. Never silently normalizes."""
    if not path.exists():
        raise ValueError(f"{label} file missing: {path}")
    if not path.is_file():
        raise ValueError(f"{label} path is not a regular file: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} read error: {exc}") from exc

    try:
        decompressed = gzip.decompress(raw)
    except (gzip.BadGzipFile, EOFError) as exc:
        raise ValueError(f"{label} gzip decode error: {exc}") from exc

    try:
        text = decompressed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} UTF-8 decode error: {exc}") from exc

    if not text:
        raise ValueError(f"{label} is empty")

    if not text.endswith("\n"):
        raise ValueError(f"{label} missing trailing newline")

    # Remove trailing newline, split
    content = text[:-1]
    lines = content.split("\n")

    if not lines:
        raise ValueError(f"{label} contains no records")

    rows = []
    for i, line in enumerate(lines, start=1):
        if line == "":
            raise ValueError(f"{label} contains blank JSONL record at line {i}")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} JSON parse error at line {i}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{label} row {i} is not a JSON object")
        rows.append(obj)

    return rows


def validate_plan_schema(plan_manifest: dict, expected_mode: str | None) -> list[str]:
    """Strict plan manifest schema validation. Returns errors."""
    errors = []

    if not isinstance(plan_manifest, dict):
        errors.append("plan manifest not a JSON object")
        return errors

    # Mode
    mode = plan_manifest.get("mode", "synthetic")
    if mode not in ("synthetic", "production"):
        errors.append(f"plan manifest invalid mode: {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        errors.append(f"synthetic flag incompatible with production plan (mode={mode!r})")

    # Fixed provenance fields
    if plan_manifest.get("scientific_freeze_sha") != SCIENTIFIC_FREEZE_SHA:
        errors.append("plan manifest scientific_freeze_sha mismatch")
    if plan_manifest.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
        errors.append("plan manifest execution_contract_version mismatch")

    # SHA fields (must be 64-char hex)
    for sha_field in ["key_plan_sha256", "run_plan_sha256"]:
        try:
            _require_hex64(plan_manifest.get(sha_field), sha_field, "plan manifest")
        except ValueError as exc:
            errors.append(str(exc))

    # Required count fields
    count_fields = [
        "shard_count", "canonical_keys",
        "baseline_rows", "governed_rows", "selection_rows",
        "failure_rows", "downstream_rows",
    ]
    for field in count_fields:
        if field not in plan_manifest:
            errors.append(f"plan manifest missing required field: {field}")
            continue
        try:
            val = _require_strict_integer(plan_manifest[field], field, "plan manifest")
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if field == "shard_count" and val < 1:
            errors.append(f"plan manifest shard_count must be > 0, got {val}")
        if field == "canonical_keys" and val < 1:
            errors.append(f"plan manifest canonical_keys must be > 0, got {val}")
        if field == "failure_rows" and val != 0:
            errors.append(f"plan manifest failure_rows must equal 0, got {val}")
        if field == "baseline_rows" and val < 0:
            errors.append(f"plan manifest baseline_rows must be non-negative, got {val}")
        if field == "governed_rows" and val < 0:
            errors.append(f"plan manifest governed_rows must be non-negative, got {val}")
        if field == "selection_rows" and val < 0:
            errors.append(f"plan manifest selection_rows must be non-negative, got {val}")
        if field == "downstream_rows" and val < 0:
            errors.append(f"plan manifest downstream_rows must be non-negative, got {val}")

    # Downstream invariant
    if all(f in plan_manifest for f in ["downstream_rows", "baseline_rows", "governed_rows"]):
        try:
            dr = _require_strict_integer(plan_manifest["downstream_rows"], "downstream_rows", "plan manifest")
            br = _require_strict_integer(plan_manifest["baseline_rows"], "baseline_rows", "plan manifest")
            gr = _require_strict_integer(plan_manifest["governed_rows"], "governed_rows", "plan manifest")
            if dr != br + gr:
                errors.append(
                    f"plan manifest downstream_rows mismatch: "
                    f"declared={dr}, baseline_plus_governed={br + gr}"
                )
        except ValueError:
            pass  # already reported above

    return errors


def validate_plan(
    plan_manifest: dict,
    plan_dir: Path,
) -> tuple[list[str], list[dict], list[dict]]:
    """Validate plan manifest and load key/run plans. Returns (errors, keys, runs)."""
    errors = []
    keys = []
    runs = []

    # Plan SHA validation happens first
    kp_path = plan_dir / "full_b1_key_plan.jsonl.gz"
    rp_path = plan_dir / "full_b1_run_plan.jsonl.gz"

    if not kp_path.exists() or not kp_path.is_file():
        errors.append("key plan file missing")
    else:
        declared_kp_sha = plan_manifest.get("key_plan_sha256")
        if declared_kp_sha is not None:
            actual_kp_sha = _sha256_file(kp_path)
            if actual_kp_sha != declared_kp_sha:
                errors.append("key plan SHA mismatch")
        try:
            keys = _load_jsonl_gzip_strict(kp_path, "key plan")
        except ValueError as exc:
            errors.append(str(exc))

    if not rp_path.exists() or not rp_path.is_file():
        errors.append("run plan file missing")
    else:
        declared_rp_sha = plan_manifest.get("run_plan_sha256")
        if declared_rp_sha is not None:
            actual_rp_sha = _sha256_file(rp_path)
            if actual_rp_sha != declared_rp_sha:
                errors.append("run plan SHA mismatch")
        try:
            runs = _load_jsonl_gzip_strict(rp_path, "run plan")
        except ValueError as exc:
            errors.append(str(exc))

    return errors, keys, runs


def validate_global_scope(
    plan_manifest: dict,
    key_rows: list[dict],
    run_rows: list[dict],
) -> list[str]:
    """Validate global key/run universe closure. Returns errors."""
    errors = []

    if not key_rows:
        errors.append("key plan is empty")
        return errors
    if not run_rows:
        errors.append("run plan is empty")
        return errors

    # Canonical key IDs + strict shard_id typing
    cid_set = set()
    for i, kp in enumerate(key_rows, start=1):
        cid = kp.get("canonical_key_id")
        if not cid or not isinstance(cid, str) or cid.strip() == "":
            errors.append(f"key plan row {i}: invalid canonical_key_id: {cid!r}")
        elif cid in cid_set:
            errors.append(f"duplicate canonical_key_id in key plan: {cid}")
        else:
            cid_set.add(cid)
        sid = kp.get("shard_id")
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 0:
            errors.append(f"key plan row {i}: shard_id must be a non-negative integer, got {type(sid).__name__} ({sid!r})")

    # Run IDs + strict shard_id typing
    rid_set = set()
    for i, rp in enumerate(run_rows, start=1):
        rid = rp.get("run_id")
        if not rid or not isinstance(rid, str) or rid.strip() == "":
            errors.append(f"run plan row {i}: invalid run_id: {rid!r}")
        elif rid in rid_set:
            errors.append(f"duplicate run_id in run plan: {rid}")
        else:
            rid_set.add(rid)
        sid = rp.get("shard_id")
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 0:
            errors.append(f"run plan row {i} ({rid}): shard_id must be a non-negative integer, got {type(sid).__name__} ({sid!r})")

    if errors:
        return errors

    # Key-shard assignment
    key_shard = {}
    for kp in key_rows:
        cid = kp["canonical_key_id"]
        sid = kp["shard_id"]
        key_shard[cid] = sid

    # Run→key membership + shard consistency
    run_key_shard = {}
    for rp in run_rows:
        rid = rp["run_id"]
        cid = rp.get("canonical_key_id", "")
        sid = rp["shard_id"]
        if cid not in cid_set:
            errors.append(f"run {rid}: canonical_key_id {cid} not in key plan")
        run_key_shard.setdefault(cid, set()).add(rid)
        expected_sid = key_shard.get(cid)
        if expected_sid is not None and sid != expected_sid:
            errors.append(
                f"run {rid} assigned to shard {sid} but key {cid} belongs to shard {expected_sid}"
            )

    # Keys without runs
    for cid in cid_set:
        if cid not in run_key_shard:
            errors.append(f"planned key {cid} has no run rows")

    # Derived shard universe: key vs run
    key_shard_ids = sorted({v for v in key_shard.values()})
    run_shard_ids = sorted({rp["shard_id"] for rp in run_rows if isinstance(rp.get("shard_id"), int) and not isinstance(rp.get("shard_id"), bool)})
    if key_shard_ids != run_shard_ids:
        errors.append(
            f"key/run plan shard universes differ: key={key_shard_ids}, run={run_shard_ids}"
        )

    # Manifest shard_count check
    try:
        manifest_count = _require_strict_integer(plan_manifest.get("shard_count"), "shard_count", "plan manifest")
    except ValueError:
        manifest_count = 0
    if isinstance(manifest_count, int) and not isinstance(manifest_count, bool) and manifest_count > 0:
        if len(key_shard_ids) != manifest_count:
            errors.append(
                f"plan shard_count mismatch: manifest={manifest_count}, derived={len(key_shard_ids)}"
            )

    return errors


def validate_shard_directory_universe(
    shard_root: Path,
    planned_shard_ids: list[int],
) -> list[str]:
    """Validate shard directory names and existence. Returns errors."""
    errors = []

    if not shard_root.exists():
        errors.append(f"shard-root does not exist: {shard_root}")
        return errors
    if not shard_root.is_dir():
        errors.append(f"shard-root is not a directory: {shard_root}")
        return errors
    if shard_root.is_symlink():
        errors.append(f"shard-root is a symlink: {shard_root}")
        return errors

    planned_set = set(planned_shard_ids)
    found = set()

    for entry in sorted(shard_root.iterdir()):
        name = entry.name
        if _SHARD_DIR_RE.match(name):
            sid = int(name.split("_")[1])
            if not entry.is_dir():
                errors.append(f"shard entry is not a directory: {name}")
            elif entry.is_symlink():
                errors.append(f"shard directory is a symlink: {name}")
            else:
                found.add(sid)
        elif name.startswith("shard_"):
            errors.append(f"non-canonical shard entry: {name}")

    missing = planned_set - found
    extra = found - planned_set

    for sid in sorted(missing):
        errors.append(f"missing planned shard directory: shard_{sid}")
    for sid in sorted(extra):
        errors.append(f"extra unplanned shard directory: shard_{sid}")

    return errors


def validate_admitted_shard_manifest_schema(
    shard_manifest: dict,
    planned_shard_id: int,
    planned_key_count: int,
    plan_mode: str,
    plan_sha: str,
) -> list[str]:
    """Strict R10c shard-manifest admission schema. Returns per-field errors."""
    errors = []
    sid = planned_shard_id  # for error messages

    # Required fields
    required_fields = [
        "schema_version", "mode", "shard_id", "key_count",
        "baseline_rows", "governed_rows", "selection_rows",
        "failure_rows", "downstream_rows",
        "scientific_freeze_sha", "execution_contract_version",
        "plan_manifest_sha256",
    ]
    for field in required_fields:
        if field not in shard_manifest:
            errors.append(f"shard {sid}: shard manifest missing required field: {field}")
    if errors:
        return errors

    # shard_id: strict integer, no bool, match planned
    manifest_sid = shard_manifest["shard_id"]
    if isinstance(manifest_sid, bool) or not isinstance(manifest_sid, int):
        errors.append(
            f"shard {sid}: shard manifest shard_id must be a non-negative integer, "
            f"got {type(manifest_sid).__name__} ({manifest_sid!r})"
        )
    elif manifest_sid < 0:
        errors.append(f"shard {sid}: shard manifest shard_id must be non-negative, got {manifest_sid}")
    elif manifest_sid != sid:
        errors.append(
            f"shard identity mismatch: directory={sid}, manifest={manifest_sid}, planned={sid}"
        )

    # mode
    if shard_manifest["mode"] != plan_mode:
        errors.append(
            f"shard {sid}: mode mismatch: manifest={shard_manifest['mode']}, plan={plan_mode}"
        )

    # Provenance
    if shard_manifest.get("scientific_freeze_sha") != SCIENTIFIC_FREEZE_SHA:
        errors.append(f"shard {sid}: shard manifest scientific_freeze_sha mismatch")
    if shard_manifest.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
        errors.append(f"shard {sid}: shard manifest execution_contract_version mismatch")
    if shard_manifest.get("plan_manifest_sha256") != plan_sha:
        errors.append(f"shard {sid}: shard manifest plan_manifest_sha256 mismatch")
    # schema_version: strict integer 1
    sv = shard_manifest.get("schema_version")
    if isinstance(sv, bool) or not isinstance(sv, int):
        errors.append(
            f"shard {sid}: shard manifest schema_version must be integer 1, "
            f"got {type(sv).__name__} ({sv!r})"
        )
    elif sv != 1:
        errors.append(
            f"shard {sid}: shard manifest schema_version must equal 1, got {sv}"
        )

    # Count fields: strict integer
    count_fields = [
        ("key_count", True, planned_key_count),  # (field, positive, expected)
        ("baseline_rows", False, None),
        ("governed_rows", False, None),
        ("selection_rows", False, None),
        ("failure_rows", False, 0),
        ("downstream_rows", False, None),
    ]
    count_values = {}
    for field, positive, expected in count_fields:
        val = shard_manifest[field]
        if isinstance(val, bool) or not isinstance(val, int):
            errors.append(
                f"shard {sid}: shard manifest {field} must be a non-negative integer, "
                f"got {type(val).__name__} ({val!r})"
            )
            continue
        if val < 0:
            errors.append(f"shard {sid}: shard manifest {field} must be non-negative, got {val}")
            continue
        if positive and val <= 0:
            errors.append(f"shard {sid}: shard manifest {field} must be positive, got {val}")
            continue
        count_values[field] = val
        if expected is not None and val != expected:
            errors.append(
                f"shard {sid}: shard manifest {field} mismatch: "
                f"manifest={val}, expected={expected}"
            )

    # failure_rows must be 0
    if "failure_rows" in count_values and count_values["failure_rows"] != 0:
        errors.append(f"shard {sid}: shard manifest failure_rows must equal 0")

    # downstream invariant
    if all(k in count_values for k in ("downstream_rows", "baseline_rows", "governed_rows")):
        dr = count_values["downstream_rows"]
        br = count_values["baseline_rows"]
        gr = count_values["governed_rows"]
        if dr != br + gr:
            errors.append(
                f"shard {sid}: shard manifest downstream_rows mismatch: "
                f"declared={dr}, baseline_plus_governed={br + gr}"
            )

    return errors


def validate_shard_set(
    *,
    plan_manifest: dict,
    plan_manifest_sha256: str,
    plan_dir: Path,
    shard_root: Path,
    expected_mode: str | None = None,
    skip_plan_schema: bool = False,
) -> ShardSetAdmissionResult:
    """Full strict shard-set admission: schema + plan + scope + directories + per-shard + counts."""
    result = ShardSetAdmissionResult(is_valid=False)

    # ── Plan schema ──
    if not skip_plan_schema:
        schema_errors = validate_plan_schema(plan_manifest, expected_mode)
        if schema_errors:
            result.errors.extend(schema_errors)
            return result

    # ── Plan file loading ──
    plan_errors, loaded_keys, loaded_runs = validate_plan(plan_manifest, plan_dir)
    if plan_errors:
        result.errors.extend(plan_errors)
        return result
    result.plan_valid = True

    # ── Global scope closure ──
    scope_errors = validate_global_scope(plan_manifest, loaded_keys, loaded_runs)
    if scope_errors:
        result.errors.extend(scope_errors)
        return result
    result.scope_valid = True

    # Build per-shard key/run mappings
    shard_keys: dict[int, list[dict]] = {}
    shard_runs: dict[int, list[dict]] = {}

    for sid in sorted({k["shard_id"] for k in loaded_keys}):
        shard_keys[sid] = [k for k in loaded_keys if k["shard_id"] == sid]
        shard_runs[sid] = [r for r in loaded_runs if r.get("shard_id") == sid]

    planned_ids = sorted(shard_keys.keys())
    result.planned_shard_ids = planned_ids

    # ── Directory universe ──
    dir_errors = validate_shard_directory_universe(shard_root, planned_ids)
    if dir_errors:
        result.errors.extend(dir_errors)
        return result
    result.directory_universe_valid = True

    # ── Per-shard validation + identity closure ──
    all_valid = True
    for sid in planned_ids:
        sdir = shard_root / f"shard_{sid}"

        # Structured shard manifest loading
        sm_path = sdir / "shard_manifest.json"
        if not sm_path.exists():
            result.errors.append(f"shard {sid}: shard_manifest.json missing")
            all_valid = False; continue
        if sm_path.is_symlink():
            result.errors.append(f"shard {sid}: shard_manifest.json is a symlink")
            all_valid = False; continue
        try:
            raw_sm = sm_path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            result.errors.append(f"shard {sid}: shard_manifest.json read error: {exc}")
            all_valid = False; continue
        try:
            sm = json.loads(raw_sm)
        except json.JSONDecodeError as exc:
            result.errors.append(f"shard {sid}: shard_manifest.json corrupt JSON: {exc}")
            all_valid = False; continue
        if not isinstance(sm, dict):
            result.errors.append(f"shard {sid}: shard_manifest.json is not a JSON object")
            all_valid = False; continue

        # Strict shard-manifest admission schema
        schema_errors = validate_admitted_shard_manifest_schema(
            shard_manifest=sm,
            planned_shard_id=sid,
            planned_key_count=len(shard_keys[sid]),
            plan_mode=plan_manifest.get("mode", "synthetic"),
            plan_sha=plan_manifest_sha256,
        )
        if schema_errors:
            result.errors.extend(schema_errors)
            all_valid = False; continue

        # Real shard validation with structured exception containment
        try:
            val = validate_shard_artifacts(
                output_dir=sdir,
                plan_manifest=plan_manifest,
                plan_manifest_sha256=plan_manifest_sha256,
                shard_key_rows=shard_keys[sid],
                shard_run_rows=shard_runs[sid],
            )
        except (OSError, EOFError, UnicodeDecodeError, ValueError,
                KeyError, gzip.BadGzipFile, zlib.error) as exc:
            result.errors.append(
                f"shard {sid} validator data error: {type(exc).__name__}: {exc}"
            )
            all_valid = False; continue
        if not val.is_valid:
            result.errors.append(f"shard {sid} validation failed:")
            for e in val.errors:
                result.errors.append(f"  {e}")
            all_valid = False; continue

        # Per-shard actual count closure
        if sm["key_count"] != len(shard_keys[sid]):
            result.errors.append(
                f"shard {sid}: key_count mismatch: manifest={sm['key_count']}, planned={len(shard_keys[sid])}"
            )
            all_valid = False; continue
        if sm["baseline_rows"] != val.baseline_rows:
            result.errors.append(
                f"shard {sid}: baseline_rows mismatch: manifest={sm['baseline_rows']}, actual={val.baseline_rows}"
            )
            all_valid = False; continue
        if sm["governed_rows"] != val.governed_rows:
            result.errors.append(
                f"shard {sid}: governed_rows mismatch: manifest={sm['governed_rows']}, actual={val.governed_rows}"
            )
            all_valid = False; continue
        if sm["selection_rows"] != val.selection_rows:
            result.errors.append(
                f"shard {sid}: selection_rows mismatch: manifest={sm['selection_rows']}, actual={val.selection_rows}"
            )
            all_valid = False; continue
        if sm["failure_rows"] != val.failure_rows:
            result.errors.append(
                f"shard {sid}: failure_rows mismatch: manifest={sm['failure_rows']}, actual={val.failure_rows}"
            )
            all_valid = False; continue
        if sm["downstream_rows"] != val.baseline_rows + val.governed_rows:
            result.errors.append(
                f"shard {sid}: downstream_rows mismatch: manifest={sm['downstream_rows']}, actual={val.baseline_rows + val.governed_rows}"
            )
            all_valid = False; continue

        result.validated_shard_ids.append(sid)

        # Accumulate global counts (from validator, not manifest)
        result.baseline_rows += val.baseline_rows
        result.governed_rows += val.governed_rows
        result.selection_rows += val.selection_rows
        result.failure_rows += val.failure_rows
        result.downstream_rows += val.baseline_rows + val.governed_rows

    if all_valid:
        result.all_shards_valid = True
        # Canonical keys from global plan, not shard manifests
        result.canonical_keys = len(loaded_keys)

    # ── Global count closure (mandatory, no silent skip) ──
    if result.all_shards_valid:
        count_specs = [
            ("canonical_keys", result.canonical_keys, "canonical_keys"),
            ("baseline_rows", result.baseline_rows, "baseline_rows"),
            ("governed_rows", result.governed_rows, "governed_rows"),
            ("selection_rows", result.selection_rows, "selection_rows"),
            ("failure_rows", result.failure_rows, "failure_rows"),
            ("downstream_rows", result.downstream_rows, "downstream_rows"),
        ]
        for label, actual, key in count_specs:
            if key not in plan_manifest:
                continue  # skip if plan field absent (legacy mode)
            expected = plan_manifest[key]
            if isinstance(expected, int) and not isinstance(expected, bool):
                if actual != expected:
                    result.errors.append(
                        f"global {label} mismatch: actual={actual}, plan={expected}"
                    )

        # Validated shard count
        if "shard_count" in plan_manifest:
            expected_shard_count = plan_manifest["shard_count"]
            if len(result.validated_shard_ids) != expected_shard_count:
                result.errors.append(
                    f"validated shard count mismatch: actual={len(result.validated_shard_ids)}, plan={expected_shard_count}"
                )

        # failure_rows must be zero
        if result.failure_rows != 0:
            result.errors.append(f"global failure_rows must equal 0, actual={result.failure_rows}")

        if not result.errors:
            result.global_counts_valid = True

    result.is_valid = (
        result.plan_valid
        and result.scope_valid
        and result.directory_universe_valid
        and result.all_shards_valid
        and result.global_counts_valid
    )

    return result


# ═══════════════════════════════════════════════════════════════════
# R10c-2 Global merge contract
# ═══════════════════════════════════════════════════════════════════

MERGE_MANIFEST_SCHEMA_VERSION = 1


def _sorted_digest(values: list[str]) -> str:
    content = "\n".join(sorted(values)) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def _ids_digest(values: list[str]) -> str:
    return _sorted_digest(values)


@dataclass
class GlobalMergeValidation:
    is_valid: bool
    errors: list[str] = field(default_factory=list)

    manifest_valid: bool = False
    artifact_sha_valid: bool = False
    source_aggregate_valid: bool = False
    row_counts_valid: bool = False
    run_universe_valid: bool = False
    selection_multiset_valid: bool = False
    provenance_valid: bool = False

    baseline_rows: int = 0
    governed_rows: int = 0
    selection_rows: int = 0
    failure_rows: int = 0
    downstream_rows: int = 0


def build_source_shard_snapshot(shard_root: Path, planned_ids: list[int]) -> dict:
    """Record immutable source shard manifest SHAs."""
    snapshot = {}
    for sid in sorted(planned_ids):
        sm_path = shard_root / f"shard_{sid}" / "shard_manifest.json"
        sm = json.loads(sm_path.read_text())
        snapshot[sid] = {
            "shard_manifest_sha256": _sha256_file(sm_path),
            "key_count": sm.get("key_count", 0),
            "baseline_rows": sm.get("baseline_rows", 0),
            "governed_rows": sm.get("governed_rows", 0),
            "selection_rows": sm.get("selection_rows", 0),
            "failure_rows": sm.get("failure_rows", 0),
        }
    return snapshot


def source_shard_manifest_set_sha256(snapshot: dict[int, dict]) -> str:
    lines = []
    for sid in sorted(snapshot.keys()):
        lines.append(f"{sid}\t{snapshot[sid]['shard_manifest_sha256']}")
    return _sorted_digest(lines)


def planned_shard_ids_digest(planned_ids: list[int]) -> str:
    return _sorted_digest([str(sid) for sid in sorted(planned_ids)])


def build_merge_manifest(
    *,
    plan_manifest: dict,
    plan_manifest_sha256: str,
    planned_shard_ids: list[int],
    snapshot: dict[int, dict],
    merged_dir: Path,
    key_count: int,
) -> dict:
    """Build deterministic global merge manifest."""
    out = Path(merged_dir)
    ledger_shas = {}
    for name in ["baseline", "governed", "selection", "failure"]:
        fp = out / f"{name}_ledger.csv.gz"
        ledger_shas[name] = _sha256_file(fp)

    return {
        "schema_version": MERGE_MANIFEST_SCHEMA_VERSION,
        "mode": plan_manifest.get("mode", "synthetic"),
        "scientific_freeze_sha": SCIENTIFIC_FREEZE_SHA,
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "plan_manifest_sha256": plan_manifest_sha256,
        "plan_declared_tool_seal_sha": plan_manifest.get("tool_seal_sha", plan_manifest.get("plan_tool_seal_sha", "")),
        "shard_count": len(planned_shard_ids),
        "canonical_keys": key_count,
        "baseline_rows": sum(s["baseline_rows"] for s in snapshot.values()),
        "governed_rows": sum(s["governed_rows"] for s in snapshot.values()),
        "selection_rows": sum(s["selection_rows"] for s in snapshot.values()),
        "failure_rows": sum(s["failure_rows"] for s in snapshot.values()),
        "downstream_rows": sum(s["baseline_rows"] + s["governed_rows"] for s in snapshot.values()),
        "baseline_sha256": ledger_shas["baseline"],
        "governed_sha256": ledger_shas["governed"],
        "selection_sha256": ledger_shas["selection"],
        "failure_sha256": ledger_shas["failure"],
        "completed_key_ids_sha256": "",
        "planned_run_ids_sha256": "",
        "produced_run_ids_sha256": "",
        "selection_hash_multiset_sha256": "",
        "planned_shard_ids_sha256": planned_shard_ids_digest(planned_shard_ids),
        "source_shard_manifest_set_sha256": source_shard_manifest_set_sha256(snapshot),
    }


def validate_global_merge_candidate(
    *,
    merged_dir: Path,
    plan_manifest: dict,
    plan_manifest_sha256: str,
    planned_shard_ids: list[int],
    snapshot: dict[int, dict],
    shard_root: Path,
    run_rows: list[dict],
) -> GlobalMergeValidation:
    """Validate a staged merge candidate against source shards."""
    out = Path(merged_dir)
    errors = []
    result = GlobalMergeValidation(is_valid=False, errors=errors)

    # ── Manifest ──
    mp = out / "merge_manifest.json"
    if not mp.exists():
        errors.append("merge_manifest.json missing")
        return result
    try:
        mm = json.loads(mp.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"merge manifest corrupt JSON: {exc}")
        return result
    if not isinstance(mm, dict):
        errors.append("merge manifest not a JSON object")
        return result

    # Schema
    if not isinstance(mm.get("schema_version"), int) or isinstance(mm.get("schema_version"), bool) or mm["schema_version"] != 1:
        errors.append("merge manifest schema_version invalid")
    if mm.get("mode") != plan_manifest.get("mode", "synthetic"):
        errors.append("merge manifest mode mismatch")
    if mm.get("scientific_freeze_sha") != SCIENTIFIC_FREEZE_SHA:
        errors.append("merge manifest scientific_freeze_sha mismatch")
    if mm.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
        errors.append("merge manifest execution_contract_version mismatch")
    if mm.get("plan_manifest_sha256") != plan_manifest_sha256:
        errors.append("merge manifest plan_manifest_sha256 mismatch")
    tool_seal = plan_manifest.get("tool_seal_sha", plan_manifest.get("plan_tool_seal_sha", ""))
    if mm.get("plan_declared_tool_seal_sha") != tool_seal:
        errors.append("merge manifest plan_declared_tool_seal_sha mismatch")
    if errors:
        return result
    result.manifest_valid = True
    result.provenance_valid = True

    # ── Artifact SHAs ──
    for name in ["baseline", "governed", "selection", "failure"]:
        fp = out / f"{name}_ledger.csv.gz"
        if not fp.exists():
            errors.append(f"{name}_ledger.csv.gz missing")
            continue
        actual = _sha256_file(fp)
        if mm.get(f"{name}_sha256") != actual:
            errors.append(f"merge manifest {name}_sha256 mismatch")
    if errors:
        return result
    result.artifact_sha_valid = True

    # ── Row counts ──
    import gzip as _gz
    for name in ["baseline", "governed", "selection", "failure"]:
        fp = out / f"{name}_ledger.csv.gz"
        text = _gz.decompress(fp.read_bytes()).decode("utf-8")
        rows = [l for l in text.split("\n")[1:] if l != ""]
        setattr(result, f"{name}_rows", len(rows))
    result.downstream_rows = result.baseline_rows + result.governed_rows
    if result.failure_rows != 0:
        errors.append("failure rows present")
    for label, actual, key in [
        ("baseline_rows", result.baseline_rows, "baseline_rows"),
        ("governed_rows", result.governed_rows, "governed_rows"),
        ("selection_rows", result.selection_rows, "selection_rows"),
        ("failure_rows", result.failure_rows, "failure_rows"),
        ("downstream_rows", result.downstream_rows, "downstream_rows"),
    ]:
        expected = mm.get(key)
        if isinstance(expected, int) and actual != expected:
            errors.append(f"merge manifest {key} mismatch: actual={actual}, manifest={expected}")
    if errors:
        return result
    result.row_counts_valid = True

    # ── Run universe ──
    planned_run_ids = sorted(r["run_id"] for r in run_rows)
    produced = []
    for name in ["baseline", "governed"]:
        fp = out / f"{name}_ledger.csv.gz"
        text = _gz.decompress(fp.read_bytes()).decode("utf-8")
        for line in text.split("\n")[1:]:
            if line:
                produced.append(line.split(",")[0])
    produced_set = set(produced)
    planned_set = set(planned_run_ids)
    if len(produced) != len(produced_set):
        errors.append("duplicate produced run IDs")
    missing = planned_set - produced_set
    extra = produced_set - planned_set
    if missing:
        errors.append(f"missing run IDs: {len(missing)}")
    if extra:
        errors.append(f"extra run IDs: {len(extra)}")
    if errors:
        return result
    result.run_universe_valid = True

    # ── Selection multiset ──
    from collections import Counter
    sp = out / "selection_ledger.csv.gz"
    sl_text = _gz.decompress(sp.read_bytes()).decode("utf-8")
    sel_hashes = [l.split(",")[0] for l in sl_text.split("\n")[1:] if l != ""]
    gp = out / "governed_ledger.csv.gz"
    gl_text = _gz.decompress(gp.read_bytes()).decode("utf-8")
    gov_hashes = []
    for line in gl_text.split("\n")[1:]:
        if line:
            parts = line.split(",")
            gov_hashes.append(parts[14])  # selection_hash at index 14
    if Counter(gov_hashes) != Counter(sel_hashes):
        errors.append("selection multiset mismatch between governed and selection ledgers")
        return result
    result.selection_multiset_valid = True

    # ── Source aggregate exactness: compare merged rows against source k-way merge ──
    import heapq as _hq
    for name in ["baseline", "governed", "selection"]:
        # Read merged candidate
        fp = out / f"{name}_ledger.csv.gz"
        merged_text = _gz.decompress(fp.read_bytes()).decode("utf-8")
        merged_lines = merged_text.split("\n")
        merged_header = merged_lines[0]
        merged_data = iter(merged_lines[1:-1] if merged_lines[-1] == "" else merged_lines[1:])

        # K-way merge from source shards
        source_iters = []
        for sid in sorted(planned_shard_ids):
            sfp = shard_root / f"shard_{sid}" / f"{name}_ledger.csv.gz"
            stext = _gz.decompress(sfp.read_bytes()).decode("utf-8")
            slines = stext.split("\n")
            sheader = slines[0]
            if sheader != merged_header:
                errors.append(f"{name} header mismatch in shard {sid}")
            sdata = [l for l in slines[1:] if l != ""]
            source_iters.append(iter(sdata))

        merged_iter = _hq.merge(*source_iters)
        row_idx = 0
        for expected_row in merged_iter:
            try:
                actual_row = next(merged_data)
            except StopIteration:
                errors.append(
                    f"{name} merged ledger ended before source aggregate at row {row_idx}"
                )
                break
            if actual_row != expected_row:
                errors.append(
                    f"{name} merged ledger differs from admitted shard aggregate at row {row_idx}"
                )
                break
            row_idx += 1
        else:
            try:
                extra = next(merged_data)
                errors.append(
                    f"{name} merged ledger has extra row after expected aggregate exhausted"
                )
            except StopIteration:
                pass
        if errors:
            return result

    result.source_aggregate_valid = True
    result.is_valid = (
        result.manifest_valid and result.provenance_valid
        and result.artifact_sha_valid and result.source_aggregate_valid
        and result.row_counts_valid and result.run_universe_valid
        and result.selection_multiset_valid
    )
    return result
