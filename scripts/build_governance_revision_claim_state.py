#!/usr/bin/env python3
"""Derive the EDBT governance revision claim state from formal statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excludes_zero_positive(result):
    return float(result["ci_lo"]) > 0


def crosses_zero(result):
    return float(result["ci_lo"]) <= 0 <= float(result["ci_hi"])


def derive(summary, remaining=None):
    if summary.get("primary_budget") != 0.20:
        raise ValueError("claim state requires the 20% primary budget")
    if summary.get("expected_keys_per_model") != 5500:
        raise ValueError("unexpected key coverage")
    models = ["LR", "RF", "LightGBM"]
    for model in models:
        if summary[f"{model}_overall"]["n_keys"] != 5500:
            raise ValueError(f"{model} does not have 5500 primary-budget keys")

    interactions = [
        summary["interaction_LR_vs_RF"],
        summary["interaction_LR_vs_LightGBM"],
        summary["interaction_RF_vs_LightGBM"],
    ]
    all_models_positive = all(excludes_zero_positive(summary[f"{model}_overall"]) for model in models)
    no_detected_interaction = all(crosses_zero(result) for result in interactions)
    structured_crosses_zero = all(crosses_zero(summary[f"{model}_structured"]) for model in models)
    m09_positive = all(excludes_zero_positive(summary[f"{model}_M09"]) for model in models)
    low_gap_negative = all(
        float(summary[f"{model}_{mechanism}"]["ci_hi"]) < 0
        for model in models
        for mechanism in ("M04", "M05", "M08")
    )
    loao_positive = all(
        float(summary[f"archetype_LOAO_{name}"]["paired"]) > 0
        for name in ("linear", "interaction", "nonlinear", "sparse", "drifting")
    )
    sparse_negative = float(summary["archetype_sparse"]["ci_hi"]) < 0

    natural_result = (remaining or {}).get("natural")
    semantic_result = (remaining or {}).get("semantic")
    natural_mixed = bool(
        natural_result
        and natural_result.get("clusters") == 5
        and sum(value > 0 for value in natural_result.get("task_effects", {}).values()) == 4
        and any(value < 0 for value in natural_result.get("task_effects", {}).values())
    )
    semantic_m09_positive = bool(
        semantic_result and float(semantic_result["semantic_M09"]["ci_lo"]) > 0
    )
    semantic_overall_crosses = bool(
        semantic_result
        and float(semantic_result["semantic_recomposed_overall"]["ci_lo"]) <= 0
        <= float(semantic_result["semantic_recomposed_overall"]["ci_hi"])
    )

    claims = {
        "C1_MULTI_LEARNER_GOVERNANCE": {
            "status": "SUPPORTED" if all_models_positive else "INCONCLUSIVE",
            "evidence_tier": "confirmatory_with_disclosed_b2_protocol_deviation",
            "allowed_wording": (
                "At the 20% matched encoded-column budget, blind MI removal improves SDR over "
                "the mean of 20 random-removal realizations for LR, RF, and LightGBM in the controlled registry."
            ),
            "forbidden_wording": "MI removal generally solves tabular leakage.",
            "evidence": [f"{model}_overall" for model in models],
        },
        "C2_NO_DETECTED_LEARNER_INTERACTION": {
            "status": "SUPPORTED" if no_detected_interaction else "INCONCLUSIVE",
            "evidence_tier": "confirmatory_with_disclosed_b2_protocol_deviation",
            "allowed_wording": (
                "Direct paired contrasts do not detect a reliable difference in the governance effect "
                "among LR, RF, and LightGBM."
            ),
            "forbidden_wording": "The governance effect is learner-invariant or equivalent across all learners.",
            "evidence": [
                "interaction_LR_vs_RF", "interaction_LR_vs_LightGBM", "interaction_RF_vs_LightGBM"
            ],
        },
        "C3_STRUCTURED_HETEROGENEITY": {
            "status": "NARROWED" if structured_crosses_zero and m09_positive and low_gap_negative else "INCONCLUSIVE",
            "evidence_tier": "mechanism_level_sensitivity",
            "allowed_wording": (
                "The structured-family average mixes low-gap mechanisms with M09, a structured mechanism "
                "showing substantial initial distortion and a positive MI-guided governance effect."
            ),
            "forbidden_wording": "MI fails on structured leakage or initial gap alone determines repairability.",
            "evidence": [
                "LR_structured", "RF_structured", "LightGBM_structured",
                "LR_M04", "LR_M05", "LR_M08", "LR_M09",
                "RF_M09", "LightGBM_M09",
            ],
        },
        "C4_ARCHETYPE_SENSITIVITY": {
            "status": "SUPPORTED" if loao_positive and sparse_negative else "INCONCLUSIVE",
            "evidence_tier": "designed_registry_sensitivity",
            "allowed_wording": (
                "The overall point estimate remains positive in every leave-one-archetype-out analysis, "
                "while the sparse archetype is a negative regime."
            ),
            "forbidden_wording": "The governance advantage is uniform across archetypes.",
            "evidence": ["archetype_sparse"] + [
                f"archetype_LOAO_{name}" for name in ("linear", "interaction", "nonlinear", "sparse", "drifting")
            ],
        },
        "C5_NATURAL_GOVERNANCE": {
            "status": "MIXED" if natural_mixed else "NOT_RUN",
            "evidence_tier": "descriptive_fixed_case_studies" if natural_mixed else "missing",
            "allowed_wording": (
                "Across five fixed natural case studies, blind MI removal exceeds mean random removal "
                "in four cases, while NYC311 is negative; this is mixed descriptive evidence, not "
                "population-level external validation."
            ) if natural_mixed else "Natural case studies evaluate harm but not the governance policy.",
            "forbidden_wording": "The governance result is validated or generally effective on natural datasets.",
            "evidence": ["remaining_governance.natural"] if natural_mixed else [],
        },
        "C6_SEMANTIC_GROUP_BUDGET": {
            "status": "NARROWED" if semantic_m09_positive and semantic_overall_crosses else "NOT_RUN",
            "evidence_tier": "representation_cost_sensitivity" if semantic_result else "missing",
            "allowed_wording": (
                "Under semantic-group cost, the M09 governance advantage remains positive, but the "
                "recomposed 5,500-key overall interval crosses zero; the overall claim is cost-sensitive."
            ) if semantic_result else "The primary governance cost is encoded-column based.",
            "forbidden_wording": "The overall governance result is invariant to semantic-group cost.",
            "evidence": [
                "remaining_governance.semantic.semantic_M09",
                "remaining_governance.semantic.semantic_recomposed_overall",
            ] if semantic_result else [],
        },
    }
    return {
        "schema_version": 1,
        "derivation": "scripts/build_governance_revision_claim_state.py",
        "primary_budget": 0.20,
        "claims": claims,
        "global_limitations": [
            "B2 strict/full baselines were re-fitted under a disclosed post-run protocol deviation.",
            "Natural governance covers five selected case studies and is mixed, with a negative NYC311 result.",
            "The semantic-group full-panel interval crosses zero even though M09 remains positive.",
            "Gap quartiles and mechanism-level decomposition are sensitivity analyses, not standalone causal tests.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/edbt_eab_revision/analysis_summary.json")
    parser.add_argument("--output", default="results/edbt_eab_revision/claim_state.json")
    parser.add_argument("--remaining-summary", default="results/edbt_eab_revision/remaining_governance_summary.json")
    args = parser.parse_args(argv)
    summary_path = ROOT / args.summary
    output_path = ROOT / args.output
    summary = json.loads(summary_path.read_text())
    remaining_path = ROOT / args.remaining_summary
    remaining = json.loads(remaining_path.read_text())
    payload = derive(summary, remaining)
    payload["analysis_summary_sha256"] = sha256(summary_path)
    payload["remaining_governance_summary_sha256"] = sha256(remaining_path)
    payload["builder_sha256"] = sha256(Path(__file__))
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value["status"] for key, value in payload["claims"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
