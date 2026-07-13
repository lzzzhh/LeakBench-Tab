import pytest, pandas as pd, os
class TestGovernanceV2:
    def test_results_exist(self):
        assert os.path.exists('results/leakbench/governance_v2.csv')
    def test_cells_480(self):
        df = pd.read_csv('results/leakbench/governance_v2.csv')
        assert len(df) >= 400
    def test_oracle_best(self):
        df = pd.read_csv('results/leakbench/governance_v2.csv')
        oracle = df[df['strategy']=='G2_oracle_remove_all']['regret'].mean()
        assert oracle < 0.001
    def test_fixed_budget_not_best(self):
        df = pd.read_csv('results/leakbench/governance_v2.csv')
        fixed = df[df['strategy']=='G3_fixed_field_budget']['regret'].mean()
        assert fixed > 0.01