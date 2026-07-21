#!/usr/bin/env python3
"""Forbidden pattern audit — scans production code only, uses string concatenation to avoid self-matching."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Patterns constructed via concatenation to avoid self-matching
PATTERNS = [
    "would " + "validate",
    "PLACE" + "HOLDER",
    "sorted(" + "set",
    "assert " + "True",
    "pytest." + "skip",
    "validate-only " + "\u2014 0 calls",
]

PRODUCTION_PATHS = ["scripts/t0_b_full_b1"]
EXCLUDE_FILES = ["forbidden_pattern_audit.py"]
ALLOWED_TEST_REFS = [
    {"path": "tests/t0_b_full_b1/test_r10a_integrity.py", "purpose": "audit definition"},
    {"path": "tests/t0_b_full_b1/test_r10a_fix.py", "purpose": "audit definition"},
]

def run_audit():
    matches = []
    for prod_path in PRODUCTION_PATHS:
        full_path = ROOT / prod_path
        if not full_path.exists():
            continue
        for py_file in full_path.glob("*.py"):
            if py_file.name in EXCLUDE_FILES:
                continue
            content = py_file.read_text()
            for pattern in PATTERNS:
                if pattern in content:
                    # Find line number
                    for i, line in enumerate(content.split("\n"), 1):
                        if pattern in line:
                            matches.append({"file": str(py_file.relative_to(ROOT)), "line": i, "pattern": pattern})
                            break
    return matches

if __name__ == "__main__":
    matches = run_audit()
    result = {
        "patterns_checked": PATTERNS,
        "production_scan_paths": PRODUCTION_PATHS,
        "excluded_audit_files": EXCLUDE_FILES,
        "production_matches": matches,
        "production_disallowed_match_count": len(matches),
        "allowed_test_references": ALLOWED_TEST_REFS,
        "pass": len(matches) == 0,
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["pass"] else 1)
