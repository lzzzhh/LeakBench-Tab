"""T0-B R10c Merge Contract — plan validation, scope closure, shard-set admission."""
from __future__ import annotations
from dataclasses import dataclass, field
import gzip, hashlib, json, re
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
    mode = plan_manifest.get("mode")
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
        actual_kp_sha = _sha256_file(kp_path)
        if actual_kp_sha != plan_manifest.get("key_plan_sha256", ""):
            errors.append("key plan SHA mismatch")
        else:
            try:
                keys = _load_jsonl_gzip_strict(kp_path, "key plan")
            except ValueError as exc:
                errors.append(str(exc))

    if not rp_path.exists() or not rp_path.is_file():
        errors.append("run plan file missing")
    else:
        actual_rp_sha = _sha256_file(rp_path)
        if actual_rp_sha != plan_manifest.get("run_plan_sha256", ""):
            errors.append("run plan SHA mismatch")
        else:
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


def validate_shard_set(
    *,
    plan_manifest: dict,
    plan_manifest_sha256: str,
    plan_dir: Path,
    shard_root: Path,
    expected_mode: str | None = None,
) -> ShardSetAdmissionResult:
    """Full strict shard-set admission: schema + plan + scope + directories + per-shard + counts."""
    result = ShardSetAdmissionResult(is_valid=False)

    # ── Plan schema ──
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

        # Identity closure
        if sm.get("shard_id") != sid:
            result.errors.append(
                f"shard identity mismatch: directory={sid}, manifest={sm.get('shard_id')}, planned={sid}"
            )
            all_valid = False; continue
        if sm.get("mode") != plan_manifest.get("mode"):
            result.errors.append(
                f"shard {sid}: mode mismatch: manifest={sm.get('mode')}, plan={plan_manifest.get('mode')}"
            )
            all_valid = False; continue

        # Real shard validation
        val = validate_shard_artifacts(
            output_dir=sdir,
            plan_manifest=plan_manifest,
            plan_manifest_sha256=plan_manifest_sha256,
            shard_key_rows=shard_keys[sid],
            shard_run_rows=shard_runs[sid],
        )
        if not val.is_valid:
            result.errors.append(f"shard {sid} validation failed:")
            for e in val.errors:
                result.errors.append(f"  {e}")
            all_valid = False; continue

        result.validated_shard_ids.append(sid)

        # Accumulate global counts
        result.canonical_keys += sm.get("key_count", len(shard_keys[sid]))
        result.baseline_rows += val.baseline_rows
        result.governed_rows += val.governed_rows
        result.selection_rows += val.selection_rows
        result.failure_rows += val.failure_rows
        result.downstream_rows += val.baseline_rows + val.governed_rows

    if all_valid:
        result.all_shards_valid = True

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
            expected = plan_manifest[key]  # guaranteed to exist and be valid int by schema
            if actual != expected:
                result.errors.append(
                    f"global {label} mismatch: actual={actual}, plan={expected}"
                )

        # Validated shard count
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
