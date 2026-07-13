import json
from pathlib import Path


def test_untraceable_final_claims_are_explicitly_superseded():
    manifest = json.loads(Path("results/corrected_v2/superseded_evidence.json").read_text())
    paths = {item["path"] for item in manifest["superseded"]}
    assert "analysis/final_eight_model/conclusions/final_conclusions.json" in paths
    assert "results/ce2r_neural.csv" in paths
    assert "results/corrected_v2/statistics/category_contrasts.csv" in paths
    assert "results/corrected_v2/statistics/correlation_analysis.json" in paths
    assert "results/corrected_v2/statistics/cluster_sensitivity.json" in paths
    assert manifest["status"] == "INTEGRITY_HOLD"


def test_legacy_negative_claims_are_retained_not_deleted():
    legacy = json.loads(Path("results/ce1_corrected/claim_matrix_final.json").read_text())
    text = json.dumps(legacy)
    assert "REFUTED" in text
    assert Path("analysis/final_eight_model/conclusions/final_conclusions.json").exists()
