import json
from pathlib import Path

import pandas as pd


def test_pilot_cells_use_one_frozen_protocol_hash():
    cells = pd.read_csv("results/corrected_v2/pilot_protocol_v2_cells.csv")
    freeze = json.loads(Path("results/corrected_v2/protocol_freeze.json").read_text())
    config_hash = next(item["sha256"] for item in freeze["files"] if item["path"] == "configs/paper/corrected_v2.yaml")
    assert set(cells["config_hash"]) == {config_hash}
    assert (cells["status"] == "SUCCESS").all()


def test_all_eleven_mechanisms_are_present_without_duplicate_cells():
    cells = pd.read_csv("results/corrected_v2/pilot_protocol_v2_cells.csv")
    assert set(cells["mechanism"]) == {f"M{i:02d}" for i in range(1, 12)}
    assert not cells.duplicated(["dataset_id", "mechanism", "strength", "model", "seed"]).any()


def test_negative_harm_is_not_relabelled_as_governance_success():
    source = Path("scripts/analyze_corrected_v2.py").read_text()
    assert "paired_harm" in source
    assert "governance success" not in source.lower()
