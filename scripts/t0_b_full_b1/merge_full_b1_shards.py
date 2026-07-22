#!/usr/bin/env python3
"""T0-B R10c-2 — Strict deterministic atomic global merge publication."""
import contextlib, csv, gzip, hashlib, heapq, json, os, re, secrets, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_full_b1.io_contract import exclusive_writer_lock, WriterLockError, atomic_write_json
from scripts.t0_b_full_b1.merge_contract import (
    validate_plan_schema, validate_plan, validate_global_scope,
    validate_shard_set, build_source_shard_snapshot,
    build_merge_manifest, validate_global_merge_candidate,
    planned_shard_ids_digest, source_shard_manifest_set_sha256,
    _sorted_digest, _ids_digest, open_strict_sorted_ledger_rows,
)
from scripts.t0_b_full_b1.fragment_contract import SCIENTIFIC_FREEZE_SHA, EXECUTION_CONTRACT_VERSION

_SHARD_DIR_RE = re.compile(r"^shard_(0|[1-9][0-9]*)$")
_SHARD_LEDGER_HEADERS = {
    "baseline": "run_id,dataset_index,mechanism,strength,training_seed,learner,baseline_type,auc",
    "governed": "run_id,dataset_index,mechanism,strength,training_seed,governance_seed,learner,policy,contract,budget_bp,strict_auc,full_auc,governed_auc,legacy_sdr,selection_hash,realized_cost",
    "selection": "selection_hash,policy,contract,budget_bp,removed_encoded_indices,removed_group_ids,realized_encoded_cost",
    "failure": "run_id",
}


def _fatal(*lines):
    print("STRICT_GLOBAL_MERGE_FAIL")
    for line in lines:
        print(f"  {line}")
    sys.exit(1)





def _streaming_merge_write(name, shard_root, planned_ids, staging_dir):
    """K-way streaming merge using shared lazy reader. Returns row count."""
    header = _SHARD_LEDGER_HEADERS[name]
    count = 0
    out_path = staging_dir / f"{name}_ledger.csv.gz"
    with open(str(out_path), "wb") as raw_f:
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=raw_f) as gf:
            gf.write((header + "\n").encode("utf-8"))
            with contextlib.ExitStack() as stack:
                iterators = []
                for sid in sorted(planned_ids):
                    fp = shard_root / f"shard_{sid}" / f"{name}_ledger.csv.gz"
                    it = stack.enter_context(open_strict_sorted_ledger_rows(
                        fp, header, f"source shard_{sid} {name}"))
                    iterators.append(it)
                for row in heapq.merge(*iterators):
                    gf.write((row + "\n").encode("utf-8"))
                    count += 1
            gf.flush()
        raw_f.flush()
        os.fsync(raw_f.fileno())
    return count


def _compute_digests(planned_ids, loaded_keys, loaded_runs, staging_dir):
    """Compute all required merge manifest digests."""
    planned_run_ids = sorted(r["run_id"] for r in loaded_runs)
    planned_run_sha = _ids_digest(planned_run_ids)

    cids = sorted(k["canonical_key_id"] for k in loaded_keys)
    key_sha = _ids_digest(cids)

    produced_ids = []
    sel_hashes = []
    for name in ["baseline", "governed"]:
        fp = staging_dir / f"{name}_ledger.csv.gz"
        with gzip.open(fp, "rt", encoding="utf-8", newline="") as gf:
            gf.readline()
            for line in gf:
                row = line.rstrip("\n")
                if row:
                    produced_ids.append(list(csv.reader([row], strict=True))[0][0])
    produced_sha = _ids_digest(produced_ids)

    fp = staging_dir / "selection_ledger.csv.gz"
    with gzip.open(fp, "rt", encoding="utf-8", newline="") as gf:
        gf.readline()
        for line in gf:
            row = line.rstrip("\n")
            if row:
                sel_hashes.append(list(csv.reader([row], strict=True))[0][0])
    sel_sha = _sorted_digest(sel_hashes)

    return key_sha, planned_run_sha, produced_sha, sel_sha


def _fsync_path(p: Path):
    fd = os.open(str(p), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True)
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    plan_path = Path(args.plan_manifest)
    shard_root = Path(args.shard_root)
    output_dir = Path(args.output_dir)
    expected_mode = "synthetic" if args.synthetic else None

    # ── Path validation ──
    if not plan_path.exists() or not plan_path.is_file() or plan_path.is_symlink():
        _fatal(f"plan manifest invalid: {plan_path}")
    if not shard_root.exists() or not shard_root.is_dir() or shard_root.is_symlink():
        _fatal(f"shard-root invalid: {shard_root}")
    if output_dir.exists():
        _fatal(f"MERGE_OUTPUT_ALREADY_EXISTS: {output_dir}")
    try:
        output_dir.resolve().relative_to(shard_root.resolve())
        _fatal(f"MERGE_OUTPUT_INSIDE_SHARD_ROOT: {output_dir}")
    except ValueError:
        pass  # not inside

    # ── Plan loading ──
    try:
        plan_manifest = json.loads(plan_path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        _fatal(f"plan manifest read error: {exc}")
    if not isinstance(plan_manifest, dict):
        _fatal("plan manifest not a JSON object")

    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_dir = plan_path.parent

    # ── Strict plan validation (no bypass) ──
    schema_errors = validate_plan_schema(plan_manifest, expected_mode)
    if schema_errors:
        _fatal(*schema_errors)

    plan_errs, loaded_keys, loaded_runs = validate_plan(plan_manifest, plan_dir)
    if plan_errs:
        _fatal(*plan_errs)

    scope_errors = validate_global_scope(plan_manifest, loaded_keys, loaded_runs)
    if scope_errors:
        _fatal(*scope_errors)

    planned_ids = sorted({k["shard_id"] for k in loaded_keys})

    # ── Locks: output parent + all source shards ──
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    try:
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(exclusive_writer_lock(output_parent, operation="merge_output"))
            except WriterLockError as exc:
                _fatal(f"OUTPUT_LOCK_FAIL: {exc}")

            for sid in sorted(planned_ids):
                sdir = shard_root / f"shard_{sid}"
                try:
                    stack.enter_context(exclusive_writer_lock(sdir, operation="merge_source"))
                except WriterLockError as exc:
                    _fatal(f"SOURCE_SHARD_LOCK_FAIL: shard_{sid}: {exc}")

            # ── Strict admission under locks ──
            admission = validate_shard_set(
                plan_manifest=plan_manifest, plan_manifest_sha256=plan_sha,
                plan_dir=plan_dir, shard_root=shard_root,
                expected_mode=expected_mode,
            )
            if not admission.is_valid:
                lines = ["admission failed:"]
                lines.extend(f"  {e}" for e in admission.errors)
                _fatal(*lines)

            snapshot = build_source_shard_snapshot(shard_root, planned_ids)

            # ── Staging ──
            nonce = secrets.token_hex(4)
            staging_dir = output_parent / f".{output_dir.name}.staging.{os.getpid()}.{nonce}"
            staging_dir.mkdir(parents=True, exist_ok=False)

            try:
                # Streaming merge
                counts = {}
                for name in ["baseline", "governed", "selection", "failure"]:
                    counts[name] = _streaming_merge_write(name, shard_root, planned_ids, staging_dir)

                # Compute digests
                key_sha, planned_run_sha, produced_sha, sel_sha = _compute_digests(
                    planned_ids, loaded_keys, loaded_runs, staging_dir)

                # Build manifest
                manifest = build_merge_manifest(
                    plan_manifest=plan_manifest, plan_manifest_sha256=plan_sha,
                    planned_shard_ids=planned_ids, snapshot=snapshot,
                    merged_dir=staging_dir, key_count=len(loaded_keys),
                    completed_key_ids_sha256=key_sha,
                    planned_run_ids_sha256=planned_run_sha,
                    produced_run_ids_sha256=produced_sha,
                    selection_hash_multiset_sha256=sel_sha,
                    baseline_rows=counts["baseline"],
                    governed_rows=counts["governed"],
                    selection_rows=counts["selection"],
                    failure_rows=counts["failure"],
                )
                atomic_write_json(staging_dir / "merge_manifest.json", manifest)

                # Candidate validation
                validation = validate_global_merge_candidate(
                    merged_dir=staging_dir, plan_manifest=plan_manifest,
                    plan_manifest_sha256=plan_sha, planned_shard_ids=planned_ids,
                    snapshot=snapshot, shard_root=shard_root, run_rows=loaded_runs,
                    key_rows=loaded_keys,
                )
                if not validation.is_valid:
                    lines = ["candidate validation failed:"]
                    lines.extend(f"  {e}" for e in validation.errors)
                    _fatal(*lines)

                # ── Atomic publication ──
                for fp in staging_dir.iterdir():
                    _fsync_path(fp)
                _fsync_path(staging_dir)
                os.replace(staging_dir, output_dir)
                staging_dir = None
                _fsync_path(output_parent)

                print("STRICT_GLOBAL_MERGE_PASS")
                print(f"planned_shards={len(planned_ids)}")
                print(f"canonical_keys={len(loaded_keys)}")
                print(f"baseline_rows={manifest['baseline_rows']}")
                print(f"governed_rows={manifest['governed_rows']}")
                print(f"selection_rows={manifest['selection_rows']}")
                print(f"failure_rows={manifest['failure_rows']}")
                print(f"downstream_rows={manifest['downstream_rows']}")
                sys.exit(0)

            finally:
                if staging_dir is not None and staging_dir.exists():
                    import shutil; shutil.rmtree(staging_dir, ignore_errors=True)

    except (WriterLockError, OSError, EOFError, UnicodeDecodeError,
            ValueError, KeyError, csv.Error, gzip.BadGzipFile,
            json.JSONDecodeError) as exc:
        _fatal(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
