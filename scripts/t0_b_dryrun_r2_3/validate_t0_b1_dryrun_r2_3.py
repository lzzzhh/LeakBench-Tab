#!/usr/bin/env python3
"""T0-B1R2.3F Validator — testable pure functions, real checks."""
import csv, gzip, hashlib, io, json, sys, subprocess
from pathlib import Path; import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
R2 = ROOT/"results/edbt_t0_b_dryrun_r2"; R23 = ROOT/"results/edbt_t0_b_dryrun_r2_3"
R2_SEAL = "b1fb0041a2e6ebdef817b5be489b9c85993002de"
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# --- Pure validation functions (importable by tests) ---

def validate_selection_receipt(sr):
    errs = []
    if sr.get("downstream_lr_calls",-1)!=0: errs.append(f"LR calls={sr['downstream_lr_calls']}")
    if sr.get("missing_hashes",-1)!=0: errs.append(f"missing={sr['missing_hashes']}")
    if sr.get("extra_hashes",-1)!=0: errs.append(f"extra={sr['extra_hashes']}")
    if sr.get("conflicting_duplicate_payloads",-1)!=0: errs.append(f"conflicts={sr['conflicting_duplicate_payloads']}")
    if sr.get("canonical_payload_mismatches",-1)!=0: errs.append(f"payload_mismatches={sr['canonical_payload_mismatches']}")
    if sr.get("generated_events",0)!=576: errs.append(f"events={sr.get('generated_events')}")
    if sr.get("canonical_unique",0)!=488: errs.append(f"unique={sr.get('canonical_unique')}")
    return errs

def validate_resume_receipt(rr):
    errs = []
    if not rr.get("sha_match",False): errs.append("SHA mismatch")
    if rr.get("duplicate_run_ids",-1)!=0: errs.append("duplicates")
    return errs

def validate_repeat_receipt(rp):
    errs = []
    if not rp.get("all_pass",False): errs.append("not all_pass")
    if len(rp.get("records",[]))!=8: errs.append(f"records={len(rp.get('records',[]))}")
    for rec in rp.get("records",[]):
        if rec.get("auc_abs_diff",1)>1e-12: errs.append(f"auc_diff={rec['auc_abs_diff']}")
        if rec.get("prob_max_diff",1)>1e-12: errs.append(f"prob_diff={rec['prob_max_diff']}")
        for fld in ["model_source_sha256","model_config_sha256","bundle_sha256"]:
            if not rec.get(fld): errs.append(f"missing {fld}")
    return errs

def validate_environment_receipt(env):
    errs = []
    for fld in ["sklearn","numpy","pandas","timezone","validation_scope"]:
        if fld not in env: errs.append(f"missing {fld}")
    if env.get("validation_scope")!="LOCAL_VALIDATION_ONLY": errs.append("wrong scope")
    return errs

def validate_p2_cost(df):
    p2 = df[df.policy=="P2"]
    if (p2.realized_cost<=0).any(): return ["P2 zero cost"]
    return []

def validate_m09_atomicity(df):
    m09 = df[(df.mechanism=="M09")&(df.contract=="semantic_group")]
    if (m09.semantic_partial!=0).any(): return ["M09 partial violation"]
    return []

def validate_manifest(cm):
    errs = []
    for name, info in cm.get("artifacts",{}).items():
        p = info.get("path","")
        if not Path(p).exists(): errs.append(f"manifest path missing: {p}")
        elif s(p)!=info["sha256"]: errs.append(f"manifest SHA mismatch: {name}")
    return errs

# --- Main ---

def main():
    errors = []
    # Seal ancestry
    r = subprocess.run(["git","merge-base","--is-ancestor",R2_SEAL,"HEAD"],capture_output=True,cwd=ROOT)
    if r.returncode: errors.append("R2 seal not ancestor")
    diff = subprocess.run(["git","diff","--name-only",f"{R2_SEAL}...HEAD","--","scripts/t0_b_dryrun_r2","tests/t0_b_dryrun_r2"],capture_output=True,text=True,cwd=ROOT)
    if diff.stdout.strip(): errors.append(f"R2 files modified: {diff.stdout.strip()}")

    # R2 ledger integrity
    refs = {"baseline_ledger.csv.gz":"bd59c32c","governed_ledger.csv.gz":"d636bc1d","selection_ledger.csv.gz":"8f5107c6","failure_ledger.csv.gz":"e185730c"}
    for f,ref in refs.items():
        if s(str(R2/f))[:8]!=ref: errors.append(f"Ledger SHA changed: {f}")

    # Row counts
    for f,exp in [("baseline_ledger.csv.gz",8),("governed_ledger.csv.gz",576)]:
        data = gzip.decompress((R2/f).read_bytes()).decode("utf-8")
        n = len([l for l in data.strip().split("\n") if l])-1
        if n!=exp: errors.append(f"{f}: {n} rows, expected {exp}")

    # Governed integrity
    gl = gzip.decompress((R2/"governed_ledger.csv.gz").read_bytes()).decode("utf-8")
    df = pd.read_csv(io.StringIO(gl))
    if df.run_id.duplicated().any(): errors.append("Duplicate run_ids")
    errors.extend(validate_p2_cost(df))
    errors.extend(validate_m09_atomicity(df))

    # Receipts
    for fname, vfn in [("selection_determinism_receipt.json",validate_selection_receipt),
                        ("resume_hash_receipt.json",validate_resume_receipt),
                        ("repeat_fit_provenance_receipt.json",validate_repeat_receipt)]:
        p = R23/fname
        if not p.exists(): errors.append(f"Missing: {fname}")
        else:
            with open(p) as f: errors.extend([f"{fname}: {e}" for e in vfn(json.load(f))])

    ep = R23/"environment_receipt_r2_3.json"
    if ep.exists():
        with open(ep) as f: errors.extend([f"env: {e}" for e in validate_environment_receipt(json.load(f))])
    else: errors.append("Environment receipt missing")

    # Manifest
    mp = R23/"evidence_closure_manifest.json"
    if mp.exists():
        with open(mp) as f: cm = json.load(f)
        errors.extend([f"manifest: {e}" for e in validate_manifest(cm)])
    else: errors.append("Manifest missing")

    # Scientific config diff
    sci = subprocess.run(["git","diff","--name-only","ff347b...HEAD","--","configs/edbt_t0_b/dryrun_matrix_v4.json"],capture_output=True,text=True,cwd=ROOT)
    if sci.stdout.strip(): errors.append(f"Scientific diff: {sci.stdout}")

    print(f"\n=== T0-B1R2.3F VALIDATOR ===\nErrors: {len(errors)}")
    for e in errors: print(f"  ERROR: {e}")
    sys.exit(1 if errors else 0)

if __name__=="__main__": main()
