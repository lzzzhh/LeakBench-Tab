import pytest, pandas as pd, os
class TestLedger:
    def test_ledger_exists(self):
        assert os.path.exists('results/audits/global_experiment_ledger.csv')
    def test_total_cells_10083(self):
        df = pd.read_csv('results/audits/global_experiment_ledger.csv')
        assert df['completed'].sum() == 10083
    def test_catboost_990(self):
        df = pd.read_csv('results/audits/global_experiment_ledger.csv')
        cb = df[df['experiment']=='catboost_core']['completed'].sum()
        assert cb == 990
    def test_tabm_726_not_792(self):
        df = pd.read_csv('results/audits/global_experiment_ledger.csv')
        tm = df[df['experiment']=='tabm_rep']['completed'].sum()
        assert tm == 726
    def test_tabpfn_is_exploratory(self):
        df = pd.read_csv('results/audits/global_experiment_ledger.csv')
        tf = df[df['experiment']=='tabpfn_audit']['completed'].sum()
        assert tf == 21
    def test_no_fourteen_thousand(self):
        df = pd.read_csv('results/audits/global_experiment_ledger.csv')
        assert df['completed'].sum() < 11000