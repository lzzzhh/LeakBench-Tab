"""Merge contract tests."""
import sys, subprocess
from pathlib import Path; import pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_merge_not_executed():
    r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/merge_full_b1_shards.py"),
        "--plan-manifest","results/edbt_t0_b_full_b1_preflight/full_b1_plan_manifest.json",
        "--shard-root","/nonexistent_shards","--output-dir","/tmp/merge_test"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0  # Merge reports error or not-executed
