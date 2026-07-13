#!/usr/bin/env python3
"""validate_release.py — LeakBench-Tab Release Validator."""
import sys, subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CHECKS = []

def check(name, condition, detail=""):
    CHECKS.append({"name": name, "pass": bool(condition), "detail": detail})

def exists(rel):
    return (PROJECT / rel).exists()

# Configs
check("config: mechanism_registry", exists("configs/leakbench/mechanism_registry.yaml"))
check("config: audit_template", exists("configs/leakbench/audit_template.yaml"))
check("config: corrected_v2", exists("configs/paper/corrected_v2.yaml"))

# Source
for p in ["src/leakbench/mechanisms/__init__.py", "src/leakbench/diagnostics/__init__.py",
          "src/leakbench/governance/__init__.py", "src/leakbench/capacity/__init__.py",
          "src/leakbench/models/__init__.py", "src/leakbench/cli/worker.py",
          "src/leakbench/cli/plan.py", "src/graph/risk_graph.py"]:
    check(f"src: {p.split('/')[-2]}/{p.split('/')[-1]}", exists(p))

# Reports
for r in ["biq_phase1_kill_test", "ait_kill_test_report", "final_go_no_go_decision",
          "claim_evidence_matrix", "result_consistency_audit", "final_non_paper_readiness"]:
    check(f"report: {r}", exists(f"reports/{r}.md"))

# Results
check("profiles: csv", exists("results/leakbench/profiles/mechanism_profiles.csv"))
check("profiles: json", exists("results/leakbench/profiles/mechanism_profiles.json"))
check("rescue matrix", exists("results/leakbench/structured_rescue/structured_rescue_matrix.csv"))

# Tests
r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], capture_output=True, text=True, cwd=str(PROJECT))
check("tests", r.returncode == 0, r.stdout.strip().split("\n")[-1] if r.stdout else str(r.returncode))

# Summary
passed = sum(1 for c in CHECKS if c["pass"])
total = len(CHECKS)
print(f"\nLeakBench-Tab Release: {passed}/{total}")
for c in CHECKS:
    print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}")
print(f"\n{'ALL PASS' if passed == total else 'FAIL'}")
sys.exit(0 if passed == total else 1)
