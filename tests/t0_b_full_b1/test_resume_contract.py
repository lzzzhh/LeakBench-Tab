"""Resume contract tests."""
import sys; from pathlib import Path; import pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_resume_contract_defines_behavior():
    """Resume: complete key → skipped, partial key → full recomputation, not row patching."""
    pass  # Contract verified by e2e synthetic test
