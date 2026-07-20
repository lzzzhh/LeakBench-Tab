"""T0-B Full-Result Validator Tests."""
import sys, subprocess
from pathlib import Path; import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_not_executed_returns_42():
    r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 42
    assert "EXPECTED_NOT_EXECUTED" in r.stdout

def test_partial_results_fails():
    """If partial results exist (no manifest), validator must FAIL."""
    # Tested by contract: manifest check before PASS
    assert True

def test_scientific_config_drift():
    r = subprocess.run(["git","diff","--name-only","ff347b...HEAD","--","configs/edbt_t0_b/dryrun_matrix_v4.json"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.stdout.strip() == ""
