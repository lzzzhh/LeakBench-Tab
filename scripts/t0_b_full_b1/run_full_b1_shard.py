#!/usr/bin/env python3
"""T0-B Full-B1 Shard Runner VF — complete resume with validate_completed_key, fixed merge, full validator."""
from __future__ import annotations
from dataclasses import dataclass, field
import gzip, hashlib, io, json, os, sys, time
from pathlib import Path; import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
# Lazy imports for selectors (only in production mode)
# from scripts.t0_b_v3.policy_selectors import ...
from scripts.t0_b_full_b1.io_contract import (
    atomic_write_gzip_text, atomic_write_json, atomic_write_dataframe_gzip,
    exclusive_writer_lock, WriterLockError, cleanup_stale_temp_files,
)
from scripts.t0_b_full_b1.fragment_contract import (
    ProductionGuard, SyntheticCallCounter,
    build_fragment_manifest, validate_completed_key,
)
from scripts.t0_b_full_b1.resume_contract import (
    classify_completed_key_failure, quarantine_invalid_key, ResumeReasonCode,
    decide_repairability, ClassifiedValidationFailure,
)
from scripts.t0_b_full_b1.fragment_contract import (
    validate_missing_receipt_candidate,
)
from scripts.t0_b_full_b1.shard_contract import (
    build_shard_manifest, validate_shard_artifacts,
    build_canonical_shard_ledger_bytes,
)
from scripts.t0_b_full_b1.run_key_contract import (
    baseline_lookup_key, governed_lookup_key, build_run_id_lookup,
)

# ======================================================================
# Dependency injection
# ======================================================================
@dataclass
class ExecutionDependencies:
    mode: str = "production"
    synthetic_call_counter: SyntheticCallCounter = field(default_factory=SyntheticCallCounter)
    production_guard: ProductionGuard = field(default_factory=ProductionGuard)

    def bundle_loader(self, kp):
        if self.mode == "synthetic":
            n = kp.get("n_total_columns", kp["n_original"] + kp["n_injected"])
            rng = np.random.RandomState(kp["training_seed"])
            X = rng.randn(100, n); y = (X[:, 0] > 0).astype(int)
            return X, y, np.arange(60), np.arange(60, 80), np.arange(80, 100)
        self.production_guard.real_bundle_loads += 1
        bundle = np.load(ROOT / kp["bundle_path"], allow_pickle=False)
        X = np.concatenate((bundle["base_X"], bundle[f"block__{kp['bundle_key']}"]), axis=1)
        return X, bundle["y"], bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]

    def model_factory(self, model_id, Xtr, ytr, Xva, yva, Xte, seed):
        if self.mode == "synthetic":
            self.synthetic_call_counter.lr_calls += 1
            rng = np.random.RandomState(seed + len(Xtr))
            class F: probabilities = rng.rand(len(Xte))
            return F()
        self.production_guard.real_model_calls += 1
        from src.leakbench.models.core_models import fit_predict_core_model
        return fit_predict_core_model(model_id, Xtr, ytr, Xva, yva, Xte, seed)

    def mi_scorer(self, Xtr, ytr):
        if self.mode == "synthetic":
            self.synthetic_call_counter.p3_calls += 1
            return np.random.RandomState(42).rand(Xtr.shape[1])
        self.production_guard.real_selector_calls += 1
        from scripts.t0_b_v3.policy_selectors import score_mi; return score_mi(Xtr, ytr)
    def pb_scorer(self, Xtr, ytr):
        if self.mode == "synthetic":
            self.synthetic_call_counter.p4_calls += 1
            return np.abs(np.random.RandomState(43).randn(Xtr.shape[1]))
        self.production_guard.real_selector_calls += 1
        from scripts.t0_b_v3.policy_selectors import score_point_biserial; return score_point_biserial(Xtr, ytr)
    def lr_scorer(self, Xtr, ytr):
        if self.mode == "synthetic":
            self.synthetic_call_counter.p5_calls += 1
            return np.abs(np.random.RandomState(44).randn(Xtr.shape[1]))
        self.production_guard.real_selector_calls += 1
        from scripts.t0_b_v3.policy_selectors import score_lr_coef; return score_lr_coef(Xtr, ytr)
    def rf_scorer(self, Xtr, ytr):
        if self.mode == "synthetic":
            self.synthetic_call_counter.p6_calls += 1
            return np.abs(np.random.RandomState(45).randn(Xtr.shape[1]))
        self.production_guard.real_selector_calls += 1
        from scripts.t0_b_v3.policy_selectors import score_rf_permutation; return score_rf_permutation(Xtr, ytr)

    def mapping_loader(self, gz_name, key_tuple):
        if self.mode == "synthetic":
            synth_name = "synthetic_policy_group_mapping.jsonl.gz" if "policy" in gz_name else "synthetic_semantic_evaluation_mapping.jsonl.gz"
            path = ROOT / "results/edbt_t0_b_full_b1_preflight/synthetic_full_contract" / synth_name
        else:
            path = ROOT / "results/edbt_t0_b" / gz_name
        data = gzip.decompress(path.read_bytes()).decode("utf-8")
        for line in data.strip().split("\n"):
            r = json.loads(line)
            if (r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]) == key_tuple:
                return r
        raise KeyError(f"Mapping not found for {key_tuple}")


def execute_key(kp, deps, run_ids):
    """Unified kernel."""
    from sklearn.metrics import roc_auc_score
    ds, mech, st, ts = kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]
    eval_info = deps.mapping_loader("semantic_evaluation_mapping_v3.jsonl.gz", (ds, mech, st, ts))
    pol_info = deps.mapping_loader("policy_group_mapping_v3.jsonl.gz", (ds, mech, st, ts))
    groups = pol_info["groups"]
    kp["n_total_columns"] = sum(g["group_size"] for g in groups)

    X, y, tr, va, te = deps.bundle_loader(kp)
    n_total = X.shape[1]

    leak_idx = set()
    for gid in eval_info.get("leak_group_ids", []):
        for g in groups:
            if g["opaque_group_id"] == gid: leak_idx.update(g["member_encoded_indices"])
    leak_mask = np.array([i in leak_idx for i in range(n_total)])

    mf = deps.model_factory
    Xs = X[:, ~leak_mask]
    s1 = mf("lr", Xs[tr], y[tr], Xs[va], y[va], Xs[te], ts)
    s2 = mf("lr", X[tr], y[tr], X[va], y[va], X[te], ts)
    sa = float(roc_auc_score(y[te], s1.probabilities)); fa = float(roc_auc_score(y[te], s2.probabilities))

    bl = [{"run_id": run_ids[baseline_lookup_key(bt)], "dataset_index": ds, "mechanism": mech, "strength": st,
           "training_seed": ts, "learner": "lr", "baseline_type": bt, "auc": auc} for bt, auc in [("strict", sa), ("full", fa)]]

    Xtr, ytr = X[tr], y[tr]
    scores = {p: f(Xtr, ytr) for p, f in [("P3", deps.mi_scorer), ("P4", deps.pb_scorer), ("P5", deps.lr_scorer), ("P6", deps.rf_scorer)]}
    from scripts.t0_b_v3.policy_selectors import group_max_score, top_k_groups, top_k_columns
    gscores = {p: group_max_score(scores[p], groups) for p in scores}

    CT = ["semantic_group", "encoded_column"]; BP = [500, 1000, 2000]; G = list(range(2026071700, 2026071720))
    gl = []; sl = []

    for ct in CT:
        for bp in BP:
            ku = compute_k(len(groups) if ct == "semantic_group" else n_total, bp)
            for pid in ["P2", "P3", "P4", "P5", "P6"]:
                seeds = range(20) if pid == "P2" else [-1]
                for gi in seeds:
                    gs_out = gi if pid == "P2" else -1
                    if pid == "P2":
                        p2s = derive_p2_seed(G[gi], ds, mech, st, ts, ct, bp)
                        rng2 = np.random.RandomState(p2s)
                        if ct == "semantic_group":
                            sg = list(rng2.choice(len(groups), ku, replace=False))
                            gids = [groups[i]["opaque_group_id"] for i in sg]; rc = []
                            [rc.extend(groups[i]["member_encoded_indices"]) for i in sg]
                            sh = hash_semantic_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], "P2", ct, bp, gids)
                        else:
                            rc = list(rng2.choice(n_total, ku, replace=False))
                            sh = hash_encoded_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], "P2", ct, bp, np.array(sorted(rc), dtype=np.int64))
                            gids = []
                    else:
                        if ct == "semantic_group":
                            sel = top_k_groups(gscores[pid], ku); rc = []
                            [rc.extend(g["member_encoded_indices"]) for gid in sel for g in groups if g["opaque_group_id"] == gid]
                            gids = sel
                            sh = hash_semantic_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], pid, ct, bp, sel)
                        else:
                            idx = top_k_columns(scores[pid], ku); rc = list(idx)
                            gids = [g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"]) & set(rc)]
                            sh = hash_encoded_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], pid, ct, bp, np.array(sorted(idx), dtype=np.int64))

                    sl.append({"selection_hash": sh, "policy": pid, "contract": ct, "budget_bp": bp,
                               "removed_encoded_indices": json.dumps([int(x) for x in sorted(rc)]),
                               "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(rc)})

                    lk = governed_lookup_key(pid, ct, bp, gi)
                    keep = np.ones(n_total, dtype=bool); keep[rc] = False
                    gov = mf("lr", X[:, keep][tr], y[tr], X[:, keep][va], y[va], X[:, keep][te], ts)
                    ga = float(roc_auc_score(y[te], gov.probabilities))
                    gl.append({"run_id": run_ids[lk], "dataset_index": ds, "mechanism": mech, "strength": st,
                               "training_seed": ts, "governance_seed": gs_out, "learner": "lr", "policy": pid,
                               "contract": ct, "budget_bp": bp, "strict_auc": sa, "full_auc": fa,
                               "governed_auc": ga, "legacy_sdr": abs(fa - sa) - abs(ga - sa),
                               "selection_hash": sh, "realized_cost": len(rc)})
    return {"baseline_rows": bl, "governed_rows": gl, "selection_rows": sl, "failure_rows": []}


def count_execute_result_rows(
    result: dict,
) -> dict[str, int]:
    """Count rows from an execute_key result dict. Fails on missing or non-list fields."""
    required = ["baseline_rows", "governed_rows", "selection_rows", "failure_rows"]
    for field in required:
        if field not in result:
            raise RuntimeError(f"execute_key result missing required field: {field}")
        if not isinstance(result[field], list):
            raise RuntimeError(f"execute_key result field {field} is not a list")
    return {
        "baseline": len(result["baseline_rows"]),
        "governed": len(result["governed_rows"]),
        "selection": len(result["selection_rows"]),
        "failure": len(result["failure_rows"]),
    }


# (validate_completed_key is now imported from fragment_contract)


def validate_all_shard_keys(
    *,
    shard_keys: list[dict],
    planned_ids: dict[str, set[str]],
    output_dir: Path,
    plan_manifest_sha: str,
    deps: ExecutionDependencies,
) -> dict[str, CompletedKeyValidation]:
    """Validate every planned key in the shard using the full fragment contract.

    Returns {canonical_key_id: CompletedKeyValidation} for every key.
    Used for both resume preflight and post-repair re-validation.
    """
    results = {}
    for kp in shard_keys:
        cid = kp["canonical_key_id"]
        fdir = output_dir / "key_fragments" / cid
        planned_for_key = sorted(planned_ids.get(cid, set()))
        pol_info = deps.mapping_loader(
            "policy_group_mapping_v3.jsonl.gz",
            (kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]),
        )
        sem_info = deps.mapping_loader(
            "semantic_evaluation_mapping_v3.jsonl.gz",
            (kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]),
        )
        result = validate_completed_key(
            kp, planned_for_key, fdir, plan_manifest_sha, pol_info, sem_info,
        )
        results[cid] = result
    return results


def _write_gz(path, df, cols):
    """Legacy helper — delegates to atomic_write_dataframe_gzip."""
    atomic_write_dataframe_gzip(path, df, cols)


def _find_duplicates(values):
    """Return list of values that appear more than once."""
    seen = {}; dups = []
    for v in values:
        if v in seen:
            if seen[v] == 1: dups.append(v)
            seen[v] += 1
        else:
            seen[v] = 1
    return dups


def shard_keys_for_validate(args):
    """Check that the shard has planned keys (for validate-only mode)."""
    plan_dir = Path(args.plan_manifest).parent
    keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
    return [k for k in keys if k.get("shard_id") == args.shard_id]


def _audit_run_plan_for_shard(shard_runs, shard_keys):
    """Full run-plan audit: lookup universe, seed validation, schema checks."""
    from scripts.t0_b_full_b1.run_key_contract import (
        baseline_lookup_key, governed_lookup_key, expected_lookup_keys_for_key,
    )
    errors = []
    # Check run IDs unique and non-null
    all_rids = [r.get("run_id") for r in shard_runs]
    if any(rid is None or rid == "" for rid in all_rids):
        errors.append("null or empty run_id found")
    rid_dups = _find_duplicates(all_rids)
    if rid_dups:
        errors.append(f"duplicate run IDs: {len(rid_dups)}")
    # Per-key audit
    expected_keys = expected_lookup_keys_for_key()
    for kp in shard_keys:
        cid = kp["canonical_key_id"]
        key_runs = [r for r in shard_runs if r["canonical_key_id"] == cid]
        if len(key_runs) != 146:
            errors.append(f"key {cid[:12]}: expected 146 runs, got {len(key_runs)}")
            continue
        # Build lookup keys
        actual_lk = set()
        for r in key_runs:
            if r["run_type"] == "baseline":
                lk = baseline_lookup_key(r["baseline_type"])
            else:
                # Validate schema
                if r["policy"] not in ("P2","P3","P4","P5","P6"):
                    errors.append(f"key {cid[:12]}: invalid policy {r['policy']}"); continue
                if r["contract"] not in ("semantic_group","encoded_column"):
                    errors.append(f"key {cid[:12]}: invalid contract {r['contract']}"); continue
                if r["budget_bp"] not in (500,1000,2000):
                    errors.append(f"key {cid[:12]}: invalid budget {r['budget_bp']}"); continue
                gi = r["governance_seed_index"]
                if r["policy"] == "P2" and not (0 <= gi <= 19):
                    errors.append(f"key {cid[:12]}: P2 seed {gi} out of range"); continue
                if r["policy"] != "P2" and gi != -1:
                    errors.append(f"key {cid[:12]}: {r['policy']} seed {gi} must be -1"); continue
                lk = governed_lookup_key(r["policy"], r["contract"], r["budget_bp"], gi)
            if lk in actual_lk:
                errors.append(f"key {cid[:12]}: duplicate lookup key {lk}")
            actual_lk.add(lk)
        missing = expected_keys - actual_lk
        extra = actual_lk - expected_keys
        if missing:
            errors.append(f"key {cid[:12]}: {len(missing)} missing lookup keys")
        if extra:
            errors.append(f"key {cid[:12]}: {len(extra)} extra lookup keys")
    return errors


def _read_gzip_csv(path):
    """Read gzip CSV into DataFrame."""
    import pandas as pd
    return pd.read_csv(pd.io.common.BytesIO(gzip.decompress(path.read_bytes())))


def _require_unique_column(df, column, label):
    """Raise if column has duplicates."""
    dups = df[column].duplicated()
    if dups.any():
        dup_count = dups.sum()
        raise RuntimeError(f"FAIL: duplicate {label} {column}: {dup_count}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True); ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--output-dir", default="/tmp/t0b_shard"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--repair-invalid", action="store_true")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)

    if args.repair_invalid and not args.resume:
        print("REPAIR_INVALID_REQUIRES_RESUME"); sys.exit(2)

    with open(ROOT / args.plan_manifest) as f: pm = json.load(f)
    plan_dir = Path(args.plan_manifest).parent

    if args.validate_only:
        # Full run-plan audit — zero model calls
        errors = []
        for fname, sha_key in [("full_b1_key_plan.jsonl.gz", "key_plan_sha256"),
                                ("full_b1_run_plan.jsonl.gz", "run_plan_sha256")]:
            fp = plan_dir / fname
            if not fp.exists():
                errors.append(f"{fname} missing"); continue
            actual = hashlib.sha256(fp.read_bytes()).hexdigest()
            if actual != pm.get(sha_key, ""):
                errors.append(f"{fname} SHA mismatch")
        keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
        runs = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
        shard_keys = [k for k in keys if k.get("shard_id") == args.shard_id]
        shard_runs = [r for r in runs if r.get("shard_id") == args.shard_id]
        if not shard_keys:
            errors.append("no planned keys for shard")
        # Full lookup universe audit
        errors.extend(_audit_run_plan_for_shard(shard_runs, shard_keys))
        if errors:
            print("VALIDATION_FAIL: " + "; ".join(errors[:5])); sys.exit(1)
        print("VALIDATION_PASS"); return

    deps = ExecutionDependencies(mode="synthetic" if args.synthetic else "production")
    keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
    runs = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]

    shard_keys = [k for k in keys if k.get("shard_id") == args.shard_id]
    shard_runs = [r for r in runs if r.get("shard_id") == args.shard_id]
    rid_lookup = build_run_id_lookup(shard_runs)

    planned_ids = {}
    for kp in shard_keys:
        cid = kp["canonical_key_id"]
        planned_ids[cid] = set()
        for r in shard_runs:
            if r["canonical_key_id"] == cid:
                planned_ids[cid].add(r["run_id"])

    # Execute within writer lock
    try:
        with exclusive_writer_lock(out, operation=f"run_shard_{args.shard_id}"):
            cleanup_stale_temp_files(out, min_age_seconds=3600)
            out.mkdir(parents=True, exist_ok=True)

            # Snapshot counters at shard start
            synth_start = SyntheticCallCounter(**deps.synthetic_call_counter.snapshot())
            prod_start = ProductionGuard(**deps.production_guard.snapshot())

            complete_ids = set(); recomputed = 0; skipped_ids = set()
            plan_manifest_sha = hashlib.sha256(Path(args.plan_manifest).read_bytes()).hexdigest()
            # Snapshot counters at shard start
            pre_validation = {}
            quarantined = {}
            classified = {}
            repairable_key_ids = set()
            unsupported_key_ids = set()

            if args.resume:
                # ─── Preflight: validate all keys via unified helper ───
                pre_validation = validate_all_shard_keys(
                    shard_keys=shard_keys,
                    planned_ids=planned_ids,
                    output_dir=out,
                    plan_manifest_sha=plan_manifest_sha,
                    deps=deps,
                )
                complete_ids = {cid for cid, r in pre_validation.items() if r.is_complete}
                invalid_results = {cid: r for cid, r in pre_validation.items() if not r.is_complete}

                # Handle invalid keys
                if invalid_results:
                    if not args.repair_invalid:
                        print(f"RESUME_VALIDATION_FAIL: {len(invalid_results)} invalid keys")
                        for cid, r in invalid_results.items():
                            print(f"  {cid[:16]}: {r.errors[:3]}")
                        sys.exit(1)
                    # Repair path: classify, evaluate missing-receipt candidates, decide repairability
                    classified = {cid: classify_completed_key_failure(cid, r) for cid, r in invalid_results.items()}

                    # Build candidate validations for receipt-missing keys
                    missing_receipt_candidates = {}
                    for cid, cf in classified.items():
                        if cf.reason_code != ResumeReasonCode.RECEIPT_MISSING:
                            continue
                        kp = next(k for k in shard_keys if k["canonical_key_id"] == cid)
                        planned_for_key = sorted(planned_ids.get(cid, set()))
                        pol_info = deps.mapping_loader(
                            "policy_group_mapping_v3.jsonl.gz",
                            (kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]),
                        )
                        sem_info = deps.mapping_loader(
                            "semantic_evaluation_mapping_v3.jsonl.gz",
                            (kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]),
                        )
                        candidate = validate_missing_receipt_candidate(
                            key_plan_row=kp,
                            planned_run_ids=planned_for_key,
                            fragment_dir=out / "key_fragments" / cid,
                            plan_manifest_sha256=plan_manifest_sha,
                            policy_mapping=pol_info,
                            semantic_mapping=sem_info,
                        )
                        missing_receipt_candidates[cid] = candidate

                    # Produce final repair decisions
                    decisions = {}
                    for cid, cf in classified.items():
                        decisions[cid] = decide_repairability(
                            cf,
                            missing_receipt_candidates.get(cid),
                        )

                    # All-or-nothing: any unrepairable means no quarantine, no recompute
                    unrepairable = {cid: d for cid, d in decisions.items() if not d.repairable}
                    if unrepairable:
                        reasons = []
                        for cid, d in unrepairable.items():
                            reasons.append(f"{cid[:12]}:{d.reason_code.value}")
                            print(f"  {cid[:16]}: {d.reason_code.value} — {list(d.validation_errors)[:3]}")
                            if d.candidate_errors:
                                print(f"    candidate errors: {list(d.candidate_errors)[:3]}")
                        print(f"RESUME_REPAIR_UNSUPPORTED: {', '.join(reasons)}")
                        sys.exit(1)

                    # Quarantine all repairable keys
                    for cid, d in decisions.items():
                        rec = quarantine_invalid_key(out, cid, d.reason_code, d.validation_errors)
                        quarantined[cid] = rec
                        classified[cid] = ClassifiedValidationFailure(  # update classification with final reason
                            canonical_key_id=cid,
                            reason_code=d.reason_code,
                            validation_errors=d.validation_errors,
                            repairable=True,
                        )
                        repairable_key_ids.add(cid)
                        print(f"  Quarantined {cid[:16]} → {rec.quarantine_directory.relative_to(out)}")

            all_bl, all_gl, all_sl = [], [], []; new_keys = 0
            new_bl_count, new_gl_count, new_sl_count, new_fl_count = 0, 0, 0, 0
            for kp in shard_keys:
                cid = kp["canonical_key_id"]; fdir = out / "key_fragments" / cid
                if cid in complete_ids:
                    skipped_ids.add(cid)
                    for name, lst in [("baseline", all_bl), ("governed", all_gl), ("selection", all_sl)]:
                        data = gzip.decompress((fdir / f"{name}.csv.gz").read_bytes()).decode("utf-8")
                        for line in data.strip().split("\n")[1:]:
                            if line: lst.append(line)
                    continue

                fdir.mkdir(parents=True, exist_ok=True)
                # Per-key counter snapshots
                key_synth_before = SyntheticCallCounter(**deps.synthetic_call_counter.snapshot())
                key_prod_before = ProductionGuard(**deps.production_guard.snapshot())
                result = execute_key(kp, deps, rid_lookup.get(cid, {}))
                key_synth_delta = deps.synthetic_call_counter.delta(key_synth_before)
                key_prod_delta = deps.production_guard.delta(key_prod_before)
                new_keys += 1; recomputed += 1
                rc = count_execute_result_rows(result)
                new_bl_count += rc["baseline"]
                new_gl_count += rc["governed"]
                new_sl_count += rc["selection"]
                new_fl_count += rc["failure"]

                for name, rows, cols in [
                    ("baseline", result["baseline_rows"], ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]),
                    ("governed", result["governed_rows"], ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","selection_hash","realized_cost"]),
                    ("selection", result["selection_rows"], ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]),
                ]:
                    atomic_write_dataframe_gzip(fdir / f"{name}.csv.gz", pd.DataFrame(rows, columns=cols), cols)
                atomic_write_dataframe_gzip(fdir / "failure.csv.gz", pd.DataFrame(result["failure_rows"], columns=["run_id"]), ["run_id"])

                # Build fragment manifest
                produced_ids = [r["run_id"] for r in result["baseline_rows"]] + [r["run_id"] for r in result["governed_rows"]]
                planned_for_key = sorted(planned_ids.get(cid, set()))
                manifest = build_fragment_manifest(cid, kp, planned_for_key, produced_ids,
                    fdir/"baseline.csv.gz", fdir/"governed.csv.gz", fdir/"selection.csv.gz", fdir/"failure.csv.gz", plan_manifest_sha)
                atomic_write_json(fdir / "fragment_manifest.json", manifest)
                manifest_sha = hashlib.sha256((fdir/"fragment_manifest.json").read_bytes()).hexdigest()

                # Write completion receipt with PER-KEY delta
                receipt = {"schema_version": 1, "canonical_key_id": cid, "status": "complete",
                    "scientific_freeze_sha": "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845",
                    "execution_contract_version": "v1", "plan_manifest_sha256": plan_manifest_sha,
                    "fragment_manifest_sha256": manifest_sha,
                    "baseline_rows": len(result["baseline_rows"]), "governed_rows": len(result["governed_rows"]),
                    "selection_rows": len(result["selection_rows"]), "failure_rows": len(result["failure_rows"]),
                    "synthetic_call_counter_delta": key_synth_delta,
                    "production_guard_delta": key_prod_delta,
                    "completed_utc": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
                atomic_write_json(fdir / "completion_receipt.json", receipt)

                # Write-then-validate
                pol_info = deps.mapping_loader("policy_group_mapping_v3.jsonl.gz", (kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]))
                sem_info = deps.mapping_loader("semantic_evaluation_mapping_v3.jsonl.gz", (kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]))
                post_result = validate_completed_key(kp, planned_for_key, fdir, plan_manifest_sha, pol_info, sem_info)
                if not post_result.is_complete:
                    print(f"FAIL: post-write validation failed for {cid}: {post_result.errors}"); sys.exit(1)

                for row in result["baseline_rows"]: all_bl.append(json.dumps(row))
                for row in result["governed_rows"]: all_gl.append(json.dumps(row))
                for row in result["selection_rows"]: all_sl.append(json.dumps(row))

            # ─── Validation barrier: post-repair re-validation BEFORE shard publication ───
            final_validation = pre_validation.copy()
            post_repair_all_keys_valid = None
            validation_barrier_passed = False

            if args.resume and args.repair_invalid and recomputed > 0:
                post_validation = validate_all_shard_keys(
                    shard_keys=shard_keys,
                    planned_ids=planned_ids,
                    output_dir=out,
                    plan_manifest_sha=plan_manifest_sha,
                    deps=deps,
                )
                final_validation = post_validation.copy()
                if len(post_validation) != len(shard_keys):
                    print(f"PARTIAL_REPAIR_POST_VALIDATION_FAIL: expected {len(shard_keys)} keys, got {len(post_validation)}")
                    sys.exit(1)
                if not all(r.is_complete for r in post_validation.values()):
                    print("PARTIAL_REPAIR_POST_VALIDATION_FAIL")
                    for cid, r in post_validation.items():
                        if not r.is_complete:
                            print(f"  {cid[:16]}: {r.errors}")
                    sys.exit(1)
                post_repair_all_keys_valid = True
                validation_barrier_passed = True
            elif args.resume:
                # Complete resume: preflight validation already proved all 4 keys valid
                if all(r.is_complete for r in pre_validation.values()):
                    validation_barrier_passed = True
            else:
                # Fresh execution: write-then-validate per-key already guaranteed all passed
                validation_barrier_passed = True

            if not validation_barrier_passed:
                print("SHARD_PUBLICATION_BLOCKED_BY_VALIDATION")
                sys.exit(1)

            # ─── Build canonical shard ledgers from active fragments ───
            canonical_ledgers = build_canonical_shard_ledger_bytes(
                output_dir=out,
                shard_key_rows=shard_keys,
                shard_run_rows=shard_runs,
                plan_manifest_sha256=plan_manifest_sha,
            )
            for name in ["baseline", "governed", "selection", "failure"]:
                fname = "failure_ledger" if name == "failure" else f"{name}_ledger"
                atomic_write_gzip_text(out / f"{fname}.csv.gz", gzip.decompress(canonical_ledgers[name]).decode("utf-8"))

            # Row counts from canonical output (for receipt)
            def _row_count_from_canonical(name: str) -> int:
                text = gzip.decompress(canonical_ledgers[name]).decode("utf-8")
                lines = text.split("\n")
                # header + data rows + trailing newline → count data rows
                return max(0, len([l for l in lines[1:] if l != ""]))

            # ─── Build and validate deterministic shard manifest ───
            plan_manifest_full = json.loads(Path(args.plan_manifest).read_text())
            shard_manifest = build_shard_manifest(
                mode="synthetic" if args.synthetic else "production",
                shard_id=args.shard_id,
                plan_manifest=plan_manifest_full,
                plan_manifest_sha256=plan_manifest_sha,
                shard_key_rows=shard_keys,
                shard_run_rows=shard_runs,
                output_dir=out,
            )
            atomic_write_json(out / "shard_manifest.json", shard_manifest)

            # Validate shard artifacts immediately after write
            shard_val = validate_shard_artifacts(
                output_dir=out,
                plan_manifest=plan_manifest_full,
                plan_manifest_sha256=plan_manifest_sha,
                shard_key_rows=shard_keys,
                shard_run_rows=shard_runs,
            )
            if not shard_val.is_valid:
                print("SHARD_ARTIFACT_VALIDATION_FAIL")
                for e in shard_val.errors:
                    print(f"  {e}")
                sys.exit(1)

            # ─── Dynamic receipts (after deterministic manifest validates) ───
            shard_exec = {"schema_version": 1, "shard_id": args.shard_id, "new_keys": new_keys,
                "synthetic_call_counter_delta": deps.synthetic_call_counter.delta(synth_start),
                "production_guard_delta": deps.production_guard.delta(prod_start)}
            atomic_write_json(out / "shard_execution_receipt.json", shard_exec)

            if args.resume:
                # ─── Construct truthful resume receipt ───
                pre_valid_ids = sorted(cid for cid, r in pre_validation.items() if r.is_complete)
                pre_invalid_ids = sorted(cid for cid, r in pre_validation.items() if not r.is_complete)
                quarantined_ids = sorted(quarantined.keys())
                recomputed_ids = sorted(repairable_key_ids) if recomputed > 0 else []
                skipped_ids_list = sorted(skipped_ids)

                # Determine post_repair_all_keys_valid from real validation
                if post_repair_all_keys_valid is None:
                    post_repair_all_keys_valid = all(r.is_complete for r in pre_validation.values())

                # Build reason codes and quarantine paths
                reason_codes = {
                    cid: cf.reason_code.value
                    for cid, cf in classified.items()
                }
                quarantine_paths = {
                    cid: str(rec.quarantine_directory.relative_to(out))
                    for cid, rec in quarantined.items()
                }

                # Build final validation results snapshot
                final_validation_dict = {
                    cid: {"is_complete": r.is_complete, "errors": list(r.errors)}
                    for cid, r in final_validation.items()
                }

                rr = {
                    "schema_version": 1,
                    "mode": "partial_repair" if (args.repair_invalid and recomputed > 0) else "complete_resume",
                    "shard_id": args.shard_id,
                    "validation_phase": "post_repair" if (args.repair_invalid and recomputed > 0) else "pre_resume",

                    # Count fields (derived from ID lists)
                    "validated_complete": len(pre_valid_ids),
                    "invalid": len(pre_invalid_ids),
                    "repairable_invalid": len(repairable_key_ids),
                    "unsupported_invalid": len(unsupported_key_ids),
                    "recomputed": recomputed,
                    "skipped": len(skipped_ids_list),
                    "quarantined": len(quarantined_ids),

                    # Key ID lists (sorted)
                    "validated_complete_key_ids": pre_valid_ids,
                    "invalid_key_ids": pre_invalid_ids,
                    "repairable_invalid_key_ids": sorted(repairable_key_ids),
                    "unsupported_invalid_key_ids": sorted(unsupported_key_ids),
                    "recomputed_key_ids": recomputed_ids,
                    "skipped_key_ids": skipped_ids_list,
                    "quarantined_key_ids": quarantined_ids,

                    # Reason codes and quarantine paths
                    "reason_codes": reason_codes,
                    "quarantine_paths": quarantine_paths,

                    # Counter deltas
                    "synthetic_call_counter_delta": deps.synthetic_call_counter.delta(synth_start),
                    "production_guard_delta": deps.production_guard.delta(prod_start),

                    # Actual row counts (not hardcoded)
                    "new_rows": {
                        "baseline": new_bl_count,
                        "governed": new_gl_count,
                        "selection": new_sl_count,
                        "failure": new_fl_count,
                    },
                    "final_rows": {
                        "baseline": _row_count_from_canonical("baseline"),
                        "governed": _row_count_from_canonical("governed"),
                        "selection": _row_count_from_canonical("selection"),
                        "failure": _row_count_from_canonical("failure"),
                    },

                    # Final validation results (all keys)
                    "final_validation_results": final_validation_dict,
                    "post_repair_all_keys_valid": post_repair_all_keys_valid,
                }

                # ─── Internal consistency verification ───
                count_checks = [
                    ("validated_complete", rr["validated_complete"], len(rr["validated_complete_key_ids"])),
                    ("invalid", rr["invalid"], len(rr["invalid_key_ids"])),
                    ("repairable_invalid", rr["repairable_invalid"], len(rr["repairable_invalid_key_ids"])),
                    ("unsupported_invalid", rr["unsupported_invalid"], len(rr["unsupported_invalid_key_ids"])),
                    ("recomputed", rr["recomputed"], len(rr["recomputed_key_ids"])),
                    ("skipped", rr["skipped"], len(rr["skipped_key_ids"])),
                    ("quarantined", rr["quarantined"], len(rr["quarantined_key_ids"])),
                ]
                for field, count, list_len in count_checks:
                    if count != list_len:
                        print(f"RESUME_RECEIPT_INTERNAL_INCONSISTENCY: {field}={count}, list_len={list_len}")
                        sys.exit(1)

                atomic_write_json(out / "resume_receipt.json", rr)

            print(f"Shard {args.shard_id}: {new_keys} new, {len(complete_ids)} complete, bl={_row_count_from_canonical('baseline')}, gl={_row_count_from_canonical('governed')}")
    except WriterLockError as exc:
        print(f"FAIL: {exc}"); sys.exit(1)

if __name__ == "__main__": main()
