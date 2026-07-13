from pathlib import Path

import yaml


def test_corrected_model_training_cell_count_is_derived_not_declared_only():
    config = yaml.safe_load(Path("configs/paper/corrected_v2.yaml").read_text())
    protocol = config["protocol"]
    derived = (
        protocol["dataset_count"]
        * len(protocol["mechanisms"])
        * len(protocol["strengths"])
        * len(protocol["core_models"])
        * len(protocol["seeds"])
    )
    assert derived == protocol["expected_model_training_cells"] == 27_500


def test_diagnostic_and_governance_rows_are_not_model_training_cells():
    config = yaml.safe_load(Path("configs/paper/corrected_v2.yaml").read_text())
    assert "diagnostic_rows" not in config["protocol"]["expected_model_training_cells"].__class__.__name__.lower()
    assert isinstance(config["protocol"]["expected_model_training_cells"], int)
