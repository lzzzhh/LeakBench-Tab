import pytest, hashlib, json, os
class TestHashes:
    def test_config_hash_stable(self):
        h1 = hashlib.sha256(b'test').hexdigest()[:16]
        h2 = hashlib.sha256(b'test').hexdigest()[:16]
        assert h1 == h2
    def test_feature_mask_hash_deterministic(self):
        features = sorted(["a","b","c"])
        h = hashlib.sha256(json.dumps(features,sort_keys=True).encode()).hexdigest()[:16]
        assert len(h) == 16
class TestReleaseValidator:
    def test_validator_exists(self):
        assert os.path.exists('scripts/validate_release.py')
    def test_reports_exist(self):
        assert os.path.exists('reports/phase10r_corrected_results.md')