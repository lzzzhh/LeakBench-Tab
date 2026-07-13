"""LeakBench-Tab: Mechanism-Centric Leakage Benchmark for Tabular Learning."""

from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
from src.leakbench.diagnostics import (
    DiagnosticScores,
    OperationalFeatureMetadata,
    OperationalMetadata,
    OracleMetadata,
    compute_full_diagnostics,
    compute_operational_diagnostics,
)
from src.leakbench.governance import (
    GovernanceResult,
    GovernanceStatus,
    GovernanceStrategy,
    apply_strategy,
)
from src.leakbench.capacity import evaluate_model_capacity, CapacityProfile
from src.leakbench.models import train_evaluate_rf, train_evaluate_mlp, ModelResult
