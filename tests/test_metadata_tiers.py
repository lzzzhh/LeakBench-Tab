import inspect

from src.leakbench.diagnostics import (
    OperationalMetadata,
    OracleMetadata,
    compute_operational_diagnostics,
)


def test_oracle_and_operational_types_are_separate():
    assert OperationalMetadata is not OracleMetadata
    assert "leakage_by_feature_id" in OracleMetadata.__dataclass_fields__
    assert "leakage_by_feature_id" not in OperationalMetadata.__dataclass_fields__


def test_operational_scorer_cannot_accept_oracle_metadata():
    parameters = inspect.signature(compute_operational_diagnostics).parameters
    assert "operational_metadata" in parameters
    assert "oracle_metadata" not in parameters
    assert "leak_mask" not in parameters
