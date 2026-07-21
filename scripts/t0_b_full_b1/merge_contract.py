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


def validate_plan(
    plan_manifest: dict,
    plan_dir: Path,
) -> tuple[list[str], list[dict], list[dict]]:
    """Validate plan manifest and load key/run plans. Returns (errors, keys, runs)."""
    errors = []
    keys = []
    runs = []

    # Plan manifest provenance
    if not isinstance(plan_manifest, dict):
        errors.append("plan manifest not a JSON object")
        return errors, keys, runs
    if plan_manifest.get("scientific_freeze_sha") != SCIENTIFIC_FREEZE_SHA:
        errors.append("plan manifest scientific_freeze_sha mismatch")
    if plan_manifest.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
        errors.append("plan manifest execution_contract_version mismatch")
    if plan_manifest.get("mode") not in ("synthetic", "production"):
        errors.append(f"plan manifest invalid mode: {plan_manifest.get('mode')}")

    shard_count = plan_manifest.get("shard_count")
    if not isinstance(shard_count, int) or isinstance(shard_count, bool) or shard_count < 1:
        errors.append(f"plan manifest invalid shard_count: {shard_count!r}")

    # Load key plan
    kp_path = plan_dir / "full_b1_key_plan.jsonl.gz"
    if not kp_path.exists():
        errors.append("key plan file missing")
    else:
        actual_kp_sha = _sha256_file(kp_path)
        if actual_kp_sha != plan_manifest.get("key_plan_sha256", ""):
            errors.append("key plan SHA mismatch")
        else:
            raw = gzip.decompress(kp_path.read_bytes()).decode("utf-8")
            for line in raw.strip().split("\n"):
                if line:
                    keys.append(json.loads(line))

    # Load run plan
    rp_path = plan_dir / "full_b1_run_plan.jsonl.gz"
    if not rp_path.exists():
        errors.append("run plan file missing")
    else:
        actual_rp_sha = _sha256_file(rp_path)
        if actual_rp_sha != plan_manifest.get("run_plan_sha256", ""):
            errors.append("run plan SHA mismatch")
        else:
            raw = gzip.decompress(rp_path.read_bytes()).decode("utf-8")
            for line in raw.strip().split("\n"):
                if line:
                    runs.append(json.loads(line))

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

    # Canonical key IDs: non-empty, globally unique
    cid_set = set()
    for kp in key_rows:
        cid = kp.get("canonical_key_id")
        if not cid or not isinstance(cid, str) or cid.strip() == "":
            errors.append(f"key plan row has invalid canonical_key_id: {cid!r}")
        elif cid in cid_set:
            errors.append(f"duplicate canonical_key_id in key plan: {cid}")
        else:
            cid_set.add(cid)

    # Run IDs: non-empty, globally unique
    rid_set = set()
    for rp in run_rows:
        rid = rp.get("run_id")
        if not rid or not isinstance(rid, str) or rid.strip() == "":
            errors.append(f"run plan row has invalid run_id: {rid!r}")
        elif rid in rid_set:
            errors.append(f"duplicate run_id in run plan: {rid}")
        else:
            rid_set.add(rid)

    if errors:
        return errors

    # Key-shard assignment
    key_shard = {}
    for kp in key_rows:
        cid = kp["canonical_key_id"]
        sid = kp.get("shard_id")
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 0:
            errors.append(f"key {cid}: invalid shard_id {sid!r}")
        key_shard[cid] = sid

    # Run→key membership + shard consistency
    run_key_shard = {}
    for rp in run_rows:
        rid = rp["run_id"]
        cid = rp.get("canonical_key_id")
        sid = rp.get("shard_id")
        if cid not in cid_set:
            errors.append(f"run {rid}: canonical_key_id {cid} not in key plan")
        run_key_shard.setdefault(cid, set()).add(rid)
        # Shard consistency
        expected_sid = key_shard.get(cid)
        if expected_sid is not None and sid != expected_sid:
            errors.append(
                f"run {rid} assigned to shard {sid} but key {cid} belongs to shard {expected_sid}"
            )

    # Keys without runs
    for cid in cid_set:
        if cid not in run_key_shard:
            errors.append(f"planned key {cid} has no run rows")

    # Derived shard universe
    derived_shard_ids = sorted({v for v in key_shard.values() if v is not None})
    manifest_count = plan_manifest.get("shard_count")
    if isinstance(manifest_count, int) and not isinstance(manifest_count, bool):
        if len(derived_shard_ids) != manifest_count:
            errors.append(
                f"plan shard_count mismatch: manifest={manifest_count}, derived={len(derived_shard_ids)}"
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
            # Non-canonical shard_* entry
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
    key_rows: list[dict],
    run_rows: list[dict],
    shard_root: Path,
) -> ShardSetAdmissionResult:
    """Full strict shard-set admission: plan + scope + directories + per-shard + counts."""
    result = ShardSetAdmissionResult(is_valid=False)

    # ── Plan validation ──
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

        # Load shard manifest
        sm_path = sdir / "shard_manifest.json"
        if not sm_path.exists():
            result.errors.append(f"shard {sid}: shard_manifest.json missing")
            all_valid = False
            continue
        sm = json.loads(sm_path.read_text())

        # Identity closure
        if sm.get("shard_id") != sid:
            result.errors.append(
                f"shard identity mismatch: directory={sid}, manifest={sm.get('shard_id')}, planned={sid}"
            )
            all_valid = False
            continue
        if sm.get("mode") != plan_manifest.get("mode"):
            result.errors.append(
                f"shard {sid}: mode mismatch: manifest={sm.get('mode')}, plan={plan_manifest.get('mode')}"
            )
            all_valid = False
            continue

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
            all_valid = False
            continue

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

    # ── Global count closure ──
    if result.all_shards_valid:
        count_checks = [
            ("canonical_keys", result.canonical_keys, "canonical_keys"),
            ("baseline_rows", result.baseline_rows, "baseline_rows"),
            ("governed_rows", result.governed_rows, "governed_rows"),
            ("selection_rows", result.selection_rows, "selection_rows"),
            ("failure_rows", result.failure_rows, "failure_rows"),
            ("downstream_rows", result.downstream_rows, "downstream_rows"),
        ]
        for label, actual, key in count_checks:
            expected = plan_manifest.get(key)
            if isinstance(expected, int) and not isinstance(expected, bool):
                if actual != expected:
                    result.errors.append(
                        f"global {label} mismatch: actual={actual}, plan={expected}"
                    )
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
