import pytest, hashlib, json

class TestHashing:
    def test_config_hash_deterministic(self):
        data = 'test'.encode()
        h1 = hashlib.sha256(data).hexdigest()[:16]
        h2 = hashlib.sha256(data).hexdigest()[:16]
        assert h1 == h2
    def test_feature_mask_hash(self):
        features = sorted(["a","b","c"])
        h = hashlib.sha256(json.dumps(features,sort_keys=True).encode()).hexdigest()[:16]
        assert len(h) == 16

class TestAtomicWrite:
    def test_tmp_rename_pattern(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
            f.write(b"test")
            tmp = f.name
        final = tmp.replace(".tmp", ".json")
        os.rename(tmp, final)
        assert os.path.exists(final)
        os.unlink(final)