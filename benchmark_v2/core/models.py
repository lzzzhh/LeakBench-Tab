"""benchmark_v2/core/models.py — Immutable frozen dataclasses for the benchmark framework."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class InformationRegime(str, Enum):
    O0 = "O0"; O1 = "O1"; O2 = "O2"

class LeakageLabel(str, Enum):
    LEGITIMATE = "LEGITIMATE"
    DIRECT_FORBIDDEN = "DIRECT_FORBIDDEN"
    PROXY = "PROXY"
    POST_OUTCOME = "POST_OUTCOME"
    AMBIGUOUS = "AMBIGUOUS"

class FeatureRole(str, Enum):
    PREDICTOR = "predictor"
    TARGET = "target"
    ID = "id"
    TIMESTAMP = "timestamp"

@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    role: FeatureRole = FeatureRole.PREDICTOR
    dtype: str = "float64"
    name_pool: dict = field(default_factory=lambda: {"natural": ""})
    description: str = ""

@dataclass(frozen=True)
class FeatureAvailability:
    feature_id: str
    available_at_prediction: bool = True
    source_table: str = ""
    event_timestamp_col: str = ""
    prediction_cutoff_col: str = ""

@dataclass(frozen=True)
class LeakageGroundTruth:
    feature_id: str
    label: LeakageLabel = LeakageLabel.LEGITIMATE
    evidence: str = ""
    confidence: str = "medium"

@dataclass
class DetectorInput:
    regime: InformationRegime
    feature_ids: tuple
    feature_names: tuple
    feature_dtypes: tuple
    train_X: np.ndarray
    train_y: np.ndarray
    feature_descriptions: tuple = ()
    feature_availability: tuple = ()

@dataclass
class DetectorOutput:
    regime: InformationRegime
    feature_ids: tuple
    risk_scores: np.ndarray
    audit_ranking: np.ndarray
    run_manifest: dict = field(default_factory=dict)

@dataclass
class RunManifest:
    git_commit: str = ""
    config_hash: str = ""
    task_hash: str = ""
    timestamp: str = ""

@dataclass
class BenchmarkConfig:
    task_id: str = ""
    feature_counts: dict = field(default_factory=dict)
    seed: int = 42
    regimes: list = field(default_factory=lambda: [InformationRegime.O2])
