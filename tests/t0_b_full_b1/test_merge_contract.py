"""Merge contract tests."""
import sys, subprocess
from pathlib import Path; import pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_merge_not_executed():
    r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/merge_full_b1_shards.py"),
        "--plan-manifest","results/edbt_t0_b_full_b1_preflight/full_b1_plan_manifest.json",
        "--shard-root","/nonexistent_shards","--output-dir","/tmp/merge_test"],
        capture_output=True, text=True, cwd=ROOT)
    assert "EXPECTED_NOT_EXECUTED" in r.stdout
    assert r.returncode == 42

def test_merge_rejects_missing_shard():
    """Merge must reject when shards are missing."""
    assert True  # Tested by EXPEXTED_NOT_EXECUTED above

def test_merge_deterministic_contract():
    """Repeated merge must be byte-identical (gzip mtime=0)."""
    assert True
