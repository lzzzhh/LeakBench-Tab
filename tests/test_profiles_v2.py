import pytest, pandas as pd, os
class TestProfilesV2:
    def test_profiles_v2_exists(self):
        assert os.path.exists('results/leakbench/profiles/mechanism_profiles_v2.csv')
    def test_all_11(self):
        df = pd.read_csv('results/leakbench/profiles/mechanism_profiles_v2.csv')
        assert len(df) == 11
    def test_m03_is_xc_not_xl(self):
        df = pd.read_csv('results/leakbench/profiles/mechanism_profiles_v2.csv')
        m03 = df[df['mechanism']=='M03'].iloc[0]
        assert m03['exploitability'] != 'LOW'
        assert m03['profile'] == 'C1-DH-XC'
    def test_c1_dl_xh_empty(self):
        df = pd.read_csv('results/leakbench/profiles/mechanism_profiles_v2.csv')
        assert 'C1-DL-XH' not in df['profile'].values
    def test_simple_all_dh(self):
        df = pd.read_csv('results/leakbench/profiles/mechanism_profiles_v2.csv')
        for m in ['M01','M02','M06','M10']:
            profile = df[df['mechanism']==m]['profile'].values[0]
            assert profile.startswith('C1-DH')
    def test_structured_all_dl(self):
        df = pd.read_csv('results/leakbench/profiles/mechanism_profiles_v2.csv')
        for m in ['M04','M05','M09']:
            profile = df[df['mechanism']==m]['profile'].values[0]
            assert profile.startswith('C1-DL')
    def test_no_negative_harm_as_high(self):
        df = pd.read_csv('results/leakbench/profiles/mechanism_profiles_v2.csv')
        for _,row in df.iterrows():
            if row['core_mean_harm'] < 0.01:
                assert row['exploitability'] != 'HIGH'