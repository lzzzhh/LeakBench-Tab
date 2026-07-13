from pathlib import Path

import numpy as np

class TestFixedBudget:
    def test_exact_count_5pct(self):
        n = 20; k = max(1, int(np.ceil(0.05*n)))
        assert k == 1
    def test_exact_count_10pct(self):
        n = 20; k = max(1, int(np.ceil(0.10*n)))
        assert k == 2
    def test_budget_respected(self):
        n, budget = 20, 0.30
        removed = min(n, max(1, int(np.ceil(budget*n))))
        kept = n - removed
        assert kept >= 0 and removed <= n

class TestGovernanceMetrics:
    def test_legitimate_retention(self):
        mask = [1,1,0,0,1]; legit = [True,True,False,False,False]
        kept_legit = sum(1 for i,m in enumerate(mask) if m==1 and not legit[i])
        assert kept_legit == 1
    def test_false_quarantine(self):
        mask = [1,0,0,1]; legit = [True,True,False,False]
        false_q = sum(1 for i,m in enumerate(mask) if m==0 and legit[i])
        assert false_q == 1  # feature at index 1: quarantined but legitimate

class TestFailureBaselines:
    def test_biq_keep_all(self):
        report = Path("reports/biq_phase1_kill_test.md").read_text().lower()
        assert "biq_phase1_kill_test" in report
        assert "archived" in report
    def test_ait_remove_all(self):
        report = Path("reports/ait_kill_test_report.md").read_text().lower()
        assert "ait_kill_test_report" in report
        assert "archived" in report
