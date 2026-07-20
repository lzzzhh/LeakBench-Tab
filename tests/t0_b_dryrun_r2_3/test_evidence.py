"""T0-B1R2.3F Real Behavioral Tests — negative path coverage."""
import csv, gzip, hashlib, io, json, sys, subprocess, tempfile
from pathlib import Path; import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.t0_b_dryrun_r2_3.validate_t0_b1_dryrun_r2_3 import (
    validate_selection_receipt, validate_resume_receipt, validate_repeat_receipt,
    validate_environment_receipt, validate_p2_cost, validate_m09_atomicity, validate_manifest,
)

def test_selection_missing_hash():
    errs = validate_selection_receipt({"downstream_lr_calls":0,"missing_hashes":5,"extra_hashes":0,"conflicting_duplicate_payloads":0,"canonical_payload_mismatches":0,"generated_events":576,"canonical_unique":488})
    assert len(errs) > 0

def test_selection_extra_hash():
    errs = validate_selection_receipt({"downstream_lr_calls":0,"missing_hashes":0,"extra_hashes":3,"conflicting_duplicate_payloads":0,"canonical_payload_mismatches":0,"generated_events":576,"canonical_unique":488})
    assert len(errs) > 0

def test_selection_payload_mismatch():
    errs = validate_selection_receipt({"downstream_lr_calls":0,"missing_hashes":0,"extra_hashes":0,"conflicting_duplicate_payloads":0,"canonical_payload_mismatches":5,"generated_events":576,"canonical_unique":488})
    assert len(errs) > 0

def test_selection_lr_calls():
    errs = validate_selection_receipt({"downstream_lr_calls":1,"missing_hashes":0,"extra_hashes":0,"conflicting_duplicate_payloads":0,"canonical_payload_mismatches":0,"generated_events":576,"canonical_unique":488})
    assert len(errs) > 0

def test_selection_wrong_counts():
    errs = validate_selection_receipt({"downstream_lr_calls":0,"missing_hashes":0,"extra_hashes":0,"conflicting_duplicate_payloads":0,"canonical_payload_mismatches":0,"generated_events":500,"canonical_unique":400})
    assert len(errs) > 0

def test_resume_sha_mismatch():
    errs = validate_resume_receipt({"sha_match":False,"duplicate_run_ids":0})
    assert len(errs) > 0

def test_resume_duplicates():
    errs = validate_resume_receipt({"sha_match":True,"duplicate_run_ids":5})
    assert len(errs) > 0

def test_repeat_not_all_pass():
    errs = validate_repeat_receipt({"all_pass":False,"records":[{"auc_abs_diff":0,"prob_max_diff":0,"model_source_sha256":"a","model_config_sha256":"b","bundle_sha256":"c"}]})
    assert len(errs) > 0

def test_repeat_prob_exceeded():
    errs = validate_repeat_receipt({"all_pass":True,"records":[{"auc_abs_diff":1e-8,"prob_max_diff":0,"model_source_sha256":"a","model_config_sha256":"b","bundle_sha256":"c"}]})
    assert any("auc_diff" in e for e in errs)

def test_environment_missing_sklearn():
    errs = validate_environment_receipt({"numpy":"1","pandas":"1","timezone":"UTC","validation_scope":"LOCAL_VALIDATION_ONLY"})
    assert len(errs) > 0

def test_environment_wrong_scope():
    errs = validate_environment_receipt({"sklearn":"1","numpy":"1","pandas":"1","timezone":"UTC","validation_scope":"CI"})
    assert len(errs) > 0

def test_manifest_sha_mismatch():
    errs = validate_manifest({"artifacts":{"x":{"path":"README.md","sha256":"0000000000000000000000000000000000000000000000000000000000000000"}}})
    assert len(errs) > 0

def test_p2_zero_cost():
    df = pd.DataFrame({"policy":["P2"],"realized_cost":[0]})
    errs = validate_p2_cost(df)
    assert len(errs) > 0

def test_m09_partial():
    df = pd.DataFrame({"mechanism":["M09"],"contract":["semantic_group"],"semantic_partial":[1]})
    errs = validate_m09_atomicity(df)
    assert len(errs) > 0

def test_pass_cases():
    errs = validate_selection_receipt({"downstream_lr_calls":0,"missing_hashes":0,"extra_hashes":0,"conflicting_duplicate_payloads":0,"canonical_payload_mismatches":0,"generated_events":576,"canonical_unique":488})
    assert len(errs) == 0

def test_repeat_pass():
    records = [{"auc_abs_diff":0,"prob_max_diff":0,"model_source_sha256":"a","model_config_sha256":"b","bundle_sha256":"c"}]*8
    errs = validate_repeat_receipt({"all_pass":True,"records":records})
    assert len(errs) == 0
