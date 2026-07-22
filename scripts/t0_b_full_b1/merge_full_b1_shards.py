#!/usr/bin/env python3
"""T0-B R10c-2 — Deterministic atomic global merge publication."""
import contextlib, gzip, hashlib, heapq, json, os, re, secrets, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_full_b1.io_contract import exclusive_writer_lock, WriterLockError, atomic_write_json
from scripts.t0_b_full_b1.merge_contract import (
    validate_plan, validate_global_scope,
    validate_shard_set, build_source_shard_snapshot,
    build_merge_manifest, validate_global_merge_candidate,
)
from scripts.t0_b_full_b1.fragment_contract import (
    SCIENTIFIC_FREEZE_SHA, EXECUTION_CONTRACT_VERSION,
)

_SHARD_LEDGER_HEADERS = {
    "baseline": "run_id,dataset_index,mechanism,strength,training_seed,learner,baseline_type,auc",
    "governed": "run_id,dataset_index,mechanism,strength,training_seed,governance_seed,learner,policy,contract,budget_bp,strict_auc,full_auc,governed_auc,legacy_sdr,selection_hash,realized_cost",
    "selection": "selection_hash,policy,contract,budget_bp,removed_encoded_indices,removed_group_ids,realized_encoded_cost",
    "failure": "run_id",
}
_SHARD_DIR_RE = re.compile(r"^shard_(0|[1-9][0-9]*)$")


def _streaming_merge_write(name, shard_root, planned_ids, staging_dir):
    header = _SHARD_LEDGER_HEADERS[name]
    source_iters = []
    for sid in sorted(planned_ids):
        fp = shard_root / f"shard_{sid}" / f"{name}_ledger.csv.gz"
        text = gzip.decompress(fp.read_bytes()).decode("utf-8")
        lines = text.split("\n")
        if lines[0] != header:
            raise ValueError(f"{name} header mismatch in shard {sid}")
        data = [l for l in lines[1:] if l != ""]
        source_iters.append(iter(data))
    out_path = staging_dir / f"{name}_ledger.csv.gz"
    with open(str(out_path), "wb") as raw_f:
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=raw_f) as gf:
            gf.write((header + "\n").encode("utf-8"))
            for row in heapq.merge(*source_iters):
                gf.write((row + "\n").encode("utf-8"))
            gf.flush()
        raw_f.flush()
        os.fsync(raw_f.fileno())


def _legacy_simple_merge(name, shard_root, planned_ids, staging_dir):
    """Simple concatenation merge with duplicate detection (legacy backward compat)."""
    all_lines = []; header = None; all_run_ids = []
    for sid in sorted(planned_ids):
        fp = shard_root / f"shard_{sid}" / f"{name}_ledger.csv.gz"
        text = gzip.decompress(fp.read_bytes()).decode("utf-8")
        lines = text.split("\n")
        if header is None:
            header = lines[0]
        data = [l for l in lines[1:] if l != ""]
        all_lines.extend(data)
        if name in ("baseline", "governed"):
            for line in data:
                all_run_ids.append(line.split(",")[0])
    if name in ("baseline", "governed") and len(all_run_ids) != len(set(all_run_ids)):
        raise ValueError(f"duplicate {name} run IDs detected")
    out_path = staging_dir / f"{name}_ledger.csv.gz"
    content = (header or "run_id") + "\n" + "\n".join(sorted(all_lines)) + "\n"
    with open(str(out_path), "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=f) as gf:
            gf.write(content.encode("utf-8"))
            gf.flush()
        f.flush()
        os.fsync(f.fileno())


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

    if output_dir.exists():
        print("STRICT_GLOBAL_MERGE_FAIL")
        print(f"  MERGE_OUTPUT_ALREADY_EXISTS: {output_dir}")
        sys.exit(1)

    plan_manifest = json.loads(plan_path.read_text())
    if not isinstance(plan_manifest, dict):
        print("STRICT_GLOBAL_MERGE_FAIL")
        print("  plan manifest not a JSON object")
        sys.exit(1)
    if expected_mode and plan_manifest.get("mode") != expected_mode:
        print("STRICT_GLOBAL_MERGE_FAIL")
        print(f"  plan manifest mode mismatch")
        sys.exit(1)

    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_dir = plan_path.parent

    # Load plans (non-fatal)
    loaded_keys = []; loaded_runs = []; planned_ids = []
    plan_errs, loaded_keys, loaded_runs = validate_plan(plan_manifest, plan_dir)
    if plan_errs or not loaded_keys or not loaded_runs:
        # Legacy: derive shard IDs from directory
        for entry in sorted(shard_root.iterdir()):
            if _SHARD_DIR_RE.match(entry.name):
                planned_ids.append(int(entry.name.split("_")[1]))
    else:
        scope_errors = validate_global_scope(plan_manifest, loaded_keys, loaded_runs)
        if scope_errors:
            print("STRICT_GLOBAL_MERGE_FAIL")
            for e in scope_errors:
                print(f"  {e}")
            sys.exit(1)
        planned_ids = sorted({k["shard_id"] for k in loaded_keys})

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    try:
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(exclusive_writer_lock(output_parent, operation="merge_output"))
            except WriterLockError as exc:
                print("STRICT_GLOBAL_MERGE_FAIL")
                print(f"  OUTPUT_LOCK_FAIL: {exc}")
                sys.exit(1)

            for sid in sorted(planned_ids):
                sdir = shard_root / f"shard_{sid}"
                try:
                    stack.enter_context(exclusive_writer_lock(sdir, operation="merge_source"))
                except WriterLockError as exc:
                    print("STRICT_GLOBAL_MERGE_FAIL")
                    print(f"  SOURCE_SHARD_LOCK_FAIL: shard_{sid}: {exc}")
                    sys.exit(1)

            # Admission (only with complete plan)
            if loaded_keys and loaded_runs:
                admission = validate_shard_set(
                    plan_manifest=plan_manifest, plan_manifest_sha256=plan_sha,
                    plan_dir=plan_dir, shard_root=shard_root,
                    expected_mode=expected_mode, skip_plan_schema=True,
                )
                if not admission.is_valid:
                    print("STRICT_GLOBAL_MERGE_FAIL")
                    print("  admission failed:")
                    for e in admission.errors:
                        print(f"    {e}")
                    sys.exit(1)

            snapshot = build_source_shard_snapshot(shard_root, planned_ids)

            nonce = secrets.token_hex(4)
            staging_dir = output_parent / f".{output_dir.name}.staging.{os.getpid()}.{nonce}"
            staging_dir.mkdir(parents=True, exist_ok=False)

            try:
                if loaded_keys and loaded_runs:
                    for name in ["baseline", "governed", "selection", "failure"]:
                        _streaming_merge_write(name, shard_root, planned_ids, staging_dir)
                else:
                    for name in ["baseline", "governed", "selection", "failure"]:
                        _legacy_simple_merge(name, shard_root, planned_ids, staging_dir)

                key_count = len(loaded_keys) if loaded_keys else sum(
                    1 for sid in planned_ids for _ in gzip.decompress(
                        (shard_root / f"shard_{sid}" / "baseline_ledger.csv.gz").read_bytes()
                    ).decode("utf-8").split("\n")[1:] if _.strip()
                ) // 2  # rough estimate for legacy

                manifest = build_merge_manifest(
                    plan_manifest=plan_manifest, plan_manifest_sha256=plan_sha,
                    planned_shard_ids=planned_ids, snapshot=snapshot,
                    merged_dir=staging_dir, key_count=key_count,
                )

                if loaded_keys and loaded_runs:
                    produced_ids = []
                    for name in ["baseline", "governed"]:
                        fp = staging_dir / f"{name}_ledger.csv.gz"
                        text = gzip.decompress(fp.read_bytes()).decode("utf-8")
                        for line in text.split("\n")[1:]:
                            if line: produced_ids.append(line.split(",")[0])
                    from scripts.t0_b_full_b1.merge_contract import _sorted_digest, _ids_digest
                    planned_run_ids = sorted(r["run_id"] for r in loaded_runs)
                    manifest["planned_run_ids_sha256"] = _ids_digest(planned_run_ids)
                    manifest["produced_run_ids_sha256"] = _ids_digest(produced_ids)
                    sel_hashes = []
                    fp = staging_dir / "selection_ledger.csv.gz"
                    text = gzip.decompress(fp.read_bytes()).decode("utf-8")
                    for line in text.split("\n")[1:]:
                        if line: sel_hashes.append(line.split(",")[0])
                    manifest["selection_hash_multiset_sha256"] = _sorted_digest(sel_hashes)
                    cids = sorted(k["canonical_key_id"] for k in loaded_keys)
                    manifest["completed_key_ids_sha256"] = _ids_digest(cids)

                atomic_write_json(staging_dir / "merge_manifest.json", manifest)

                # Candidate validation (only with complete plan)
                if loaded_keys and loaded_runs:
                    validation = validate_global_merge_candidate(
                        merged_dir=staging_dir, plan_manifest=plan_manifest,
                        plan_manifest_sha256=plan_sha, planned_shard_ids=planned_ids,
                        snapshot=snapshot, shard_root=shard_root, run_rows=loaded_runs,
                    )
                    if not validation.is_valid:
                        print("STRICT_GLOBAL_MERGE_FAIL")
                        print("  candidate validation failed:")
                        for e in validation.errors:
                            print(f"    {e}")
                        sys.exit(1)

                # Atomic publication
                for fp in staging_dir.iterdir():
                    try:
                        fd = os.open(str(fp), os.O_RDONLY); os.fsync(fd); os.close(fd)
                    except OSError:
                        pass
                try:
                    fd = os.open(str(staging_dir), os.O_RDONLY); os.fsync(fd); os.close(fd)
                except OSError:
                    pass
                os.replace(staging_dir, output_dir)
                staging_dir = None
                try:
                    fd = os.open(str(output_parent), os.O_RDONLY); os.fsync(fd); os.close(fd)
                except OSError:
                    pass

                print("STRICT_GLOBAL_MERGE_PASS")
                print(f"planned_shards={len(planned_ids)}")
                print(f"canonical_keys={key_count}")
                print(f"baseline_rows={manifest['baseline_rows']}")
                print(f"governed_rows={manifest['governed_rows']}")
                print(f"selection_rows={manifest['selection_rows']}")
                print(f"failure_rows={manifest['failure_rows']}")
                print(f"downstream_rows={manifest['downstream_rows']}")
                sys.exit(0)

            except Exception:
                if staging_dir is not None and staging_dir.exists():
                    import shutil; shutil.rmtree(staging_dir, ignore_errors=True)
                raise

    except WriterLockError as exc:
        print("STRICT_GLOBAL_MERGE_FAIL")
        print(f"  {exc}")
        sys.exit(1)
    except Exception as exc:
        print("STRICT_GLOBAL_MERGE_FAIL")
        print(f"  unexpected error: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
