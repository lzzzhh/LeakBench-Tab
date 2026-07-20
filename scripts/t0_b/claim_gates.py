"""T0-B Claim Gates — deterministic decision tree with precedence order."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MetricEstimate:
    mean: float
    ci_lo: float
    ci_hi: float

    def confirmed_positive(self) -> bool:
        return self.mean > 0 and self.ci_lo > 0

    def confirmed_negative(self) -> bool:
        return self.mean < 0 and self.ci_hi < 0


def determine_claim_status(
    legacy_sdr: MetricEstimate,
    directional_repair: MetricEstimate,
    full_group_recall: MetricEstimate,
    any_hit_recall: MetricEstimate,
    overcorrection: MetricEstimate,
    legit_retention: MetricEstimate,
    introduced_distortion_zero_opp: MetricEstimate,
    evaluable: bool,
) -> str:
    """Apply deterministic claim gates in precedence order.

    Returns one of: NOT_EVALUABLE, SEMANTICALLY_CORROBORATED, TRADEOFF,
    LOCALIZATION_ONLY, SCORE_RECOVERY_ONLY, NEGATIVE.
    """
    # Gate 1: NOT_EVALUABLE
    if not evaluable:
        return "NOT_EVALUABLE"

    # Gate 2: SEMANTICALLY_CORROBORATED
    sem_corroborated = (
        directional_repair.confirmed_positive()
        and full_group_recall.confirmed_positive()
        and overcorrection.mean <= 0
        and legit_retention.mean >= 0
        and introduced_distortion_zero_opp.mean <= 0
    )
    if sem_corroborated:
        return "SEMANTICALLY_CORROBORATED"

    # Gate 3: TRADEOFF
    localization_positive = full_group_recall.confirmed_positive() or directional_repair.confirmed_positive()
    adverse = (
        (overcorrection.mean > 0 and overcorrection.ci_lo > 0)
        or (legit_retention.mean < 0 and legit_retention.ci_hi < 0)
    )
    if localization_positive and adverse:
        return "TRADEOFF"

    # Gate 4: LOCALIZATION_ONLY
    loc_only = full_group_recall.confirmed_positive() or any_hit_recall.confirmed_positive()
    if loc_only:
        return "LOCALIZATION_ONLY"

    # Gate 5: SCORE_RECOVERY_ONLY
    if legacy_sdr.confirmed_positive():
        return "SCORE_RECOVERY_ONLY"

    # Gate 6: NEGATIVE
    return "NEGATIVE"
