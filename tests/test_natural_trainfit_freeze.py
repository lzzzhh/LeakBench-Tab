import hashlib
import json
from pathlib import Path


def test_natural_trainfit_v2_freeze_hashes_and_scope():
    path = Path("results/corrected_v2/public_natural/natural_protocol_v2_freeze.json")
    assert path.exists()
    freeze = json.loads(path.read_text(encoding="utf-8"))
    assert freeze["status"] == "FROZEN_BEFORE_NATURAL_TRAINFIT_V2_RERUN"
    assert freeze["amendment_version"] == "natural_trainfit_categories_v2"
    assert freeze["expected_cells"] == 60
    assert freeze["fit_scope"] == "training rows only"
    for relative, entry in freeze["code_files"].items():
        source = Path(relative)
        assert source.stat().st_size == entry["size_bytes"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
    for entry in freeze["source_files"].values():
        source = Path(entry["path"])
        assert not source.is_absolute()
        assert source.parts[0] == "external_sources"
        assert entry["size_bytes"] > 0
        assert len(entry["sha256"]) == 64
