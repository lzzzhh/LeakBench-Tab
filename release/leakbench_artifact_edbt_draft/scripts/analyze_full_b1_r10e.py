#!/usr/bin/env python3
"""R10e analysis for the validated T0-B Full-B1 execution.

The estimator is fixed before reading effects: average P2 within key over its
20 governance seeds, form deterministic-policy minus mean-P2 contrasts, reduce
to task means, then cluster-bootstrap tasks.  Semantic metrics are reconstructed
from the oracle-isolated selection ledger and the separately declared evaluation
mapping; they were never policy-visible at execution time.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KEY = ["dataset_index", "mechanism", "strength", "training_seed"]
METRICS = [
    "legacy_sdr",
    "directional_repair",
    "same_side_residual",
    "overcorrection",
    "introduced_distortion",
    "introduced_distortion_zero_opp",
    "leak_recall",
    "deletion_precision",
    "legit_retention",
    "semantic_group_recall_full",
    "semantic_group_recall_any",
    "partial_group_violation_rate",
]
POLICIES = ["P3", "P4", "P5", "P6"]
CONTRACTS = ["semantic_group", "encoded_column"]
BUDGETS = [500, 1000, 2000]
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260722
EPS = 1e-12

ARCHETYPE = {
    **{i: "linear" for i in range(0, 4)},
    **{i: "interaction" for i in range(4, 8)},
    **{i: "nonlinear" for i in range(8, 12)},
    **{i: "sparse" for i in range(12, 16)},
    **{i: "drifting" for i in range(16, 20)},
}
MECHANISM_FAMILY = {
    **{m: "simple" for m in ["M01", "M02", "M03"]},
    **{m: "structured" for m in ["M04", "M05", "M08", "M09"]},
    **{m: "boundary" for m in ["M06", "M07", "M10", "M11"]},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def add_score_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    signed_gap = out["full_auc"] - out["strict_auc"]
    opportunity = signed_gap.abs()
    offset = out["governed_auc"] - out["strict_auc"]
    direction = np.sign(signed_gap)
    zero = opportunity <= EPS
    out["opportunity"] = opportunity
    out["same_side_residual"] = np.where(
        zero, 0.0, np.maximum(direction * offset, 0.0)
    )
    out["overcorrection"] = np.where(
        zero, 0.0, np.maximum(-direction * offset, 0.0)
    )
    out["directional_repair"] = np.where(
        zero, 0.0, opportunity - out["same_side_residual"]
    )
    out["introduced_distortion"] = np.where(zero, offset.abs(), 0.0)
    out["introduced_distortion_zero_opp"] = np.where(
        zero, offset.abs(), np.nan
    )
    recomputed = opportunity - offset.abs()
    if not np.allclose(out["legacy_sdr"], recomputed, atol=1e-12, rtol=0):
        raise RuntimeError("legacy_sdr does not match strict-distance definition")
    return out


def build_semantic_registry(policy_path: Path, eval_path: Path) -> dict:
    policy_rows = {
        (int(r["dataset_index"]), r["mechanism"], r["strength"], int(r["training_seed"])): r
        for r in read_jsonl_gz(policy_path)
    }
    eval_rows = {
        (int(r["dataset_index"]), r["mechanism"], r["strength"], int(r["training_seed"])): r
        for r in read_jsonl_gz(eval_path)
    }
    if set(policy_rows) != set(eval_rows) or len(policy_rows) != 5500:
        raise RuntimeError("policy/evaluation mapping key universe mismatch")
    registry = {}
    for key, prow in policy_rows.items():
        erow = eval_rows[key]
        groups = {
            g["opaque_group_id"]: frozenset(int(i) for i in g["member_encoded_indices"])
            for g in prow["groups"]
        }
        leak_groups = tuple(erow["leak_group_ids"])
        leak_indices = frozenset().union(*(groups[g] for g in leak_groups))
        n_columns = int(erow["n_encoded_columns"])
        registry[key] = {
            "groups": groups,
            "leak_groups": leak_groups,
            "leak_indices": leak_indices,
            "n_leak": len(leak_indices),
            "n_legit": n_columns - len(leak_indices),
        }
    return registry


def semantic_metrics_for_selection(row: pd.Series, registry: dict) -> dict:
    key = (
        int(row.dataset_index), row.mechanism, row.strength,
        int(row.training_seed),
    )
    info = registry[key]
    removed = frozenset(int(i) for i in json.loads(row.removed_encoded_indices))
    removed_leak = len(removed & info["leak_indices"])
    removed_legit = len(removed) - removed_leak
    leak_groups = info["leak_groups"]
    full = sum(info["groups"][gid] <= removed for gid in leak_groups)
    any_hit = sum(bool(info["groups"][gid] & removed) for gid in leak_groups)
    partial = sum(
        bool(info["groups"][gid] & removed)
        and not (info["groups"][gid] <= removed)
        for gid in leak_groups
    )
    n_leak_groups = len(leak_groups)
    return {
        "leak_recall": removed_leak / info["n_leak"] if info["n_leak"] else np.nan,
        "deletion_precision": removed_leak / len(removed) if removed else 0.0,
        "legit_retention": 1.0 - removed_legit / info["n_legit"] if info["n_legit"] else 1.0,
        "semantic_group_recall_full": full / n_leak_groups if n_leak_groups else np.nan,
        "semantic_group_recall_any": any_hit / n_leak_groups if n_leak_groups else np.nan,
        "partial_group_violation_rate": partial / n_leak_groups if n_leak_groups else np.nan,
    }


def deterministic_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}|{label}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def estimate(values: pd.DataFrame, metric: str, label: str) -> dict:
    clean = values[["dataset_index", metric]].dropna()
    task = clean.groupby("dataset_index", sort=True)[metric].mean()
    if task.empty:
        return {
            "metric": metric, "mean": np.nan, "ci_lo": np.nan,
            "ci_hi": np.nan, "p_gt_zero": np.nan, "n_keys": 0,
            "n_tasks": 0, "task_sd": np.nan,
        }
    array = task.to_numpy(float)
    rng = np.random.RandomState(deterministic_seed(label))
    indices = rng.randint(0, len(array), size=(BOOTSTRAP_REPS, len(array)))
    boot = array[indices].mean(axis=1)
    return {
        "metric": metric,
        "mean": float(array.mean()),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_gt_zero": float((boot > 0).mean()),
        "n_keys": int(len(clean)),
        "n_tasks": int(len(task)),
        "task_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def summarize(frame: pd.DataFrame, group_columns: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(group_columns, sort=True, dropna=False)
    for group, part in grouped:
        group = group if isinstance(group, tuple) else (group,)
        identity = dict(zip(group_columns, group))
        for metric in METRICS:
            field = f"delta_{metric}"
            item = estimate(part, field, f"{prefix}|{group}|{metric}")
            item.update(identity)
            rows.append(item)
    return pd.DataFrame(rows)


def lookup(summary: pd.DataFrame, identity: dict, metric: str) -> dict:
    mask = summary.metric.eq(f"delta_{metric}")
    for key, value in identity.items():
        mask &= summary[key].eq(value)
    rows = summary[mask]
    if len(rows) != 1:
        raise RuntimeError(f"summary lookup mismatch: {identity}, {metric}, n={len(rows)}")
    return rows.iloc[0].to_dict()


def fmt(est: dict) -> str:
    return f"{est['mean']:+.3f} [{est['ci_lo']:+.3f}, {est['ci_hi']:+.3f}]"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results/edbt_t0_b_full_b1")
    parser.add_argument("--output-dir", default="results/edbt_t0_b_full_b1_analysis")
    args = parser.parse_args()

    result_root = ROOT / args.result_root
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((result_root / "validation_receipt.json").read_text())
    if receipt.get("status") != "PASS" or receipt.get("errors") != []:
        raise RuntimeError("Full-B1 validation receipt is not PASS")

    merged = result_root / "merged"
    governed = pd.read_csv(merged / "governed_ledger.csv.gz")
    selections = pd.read_csv(merged / "selection_ledger.csv.gz")
    if len(governed) != 792000 or len(selections) != 792000:
        raise RuntimeError("validated ledger row counts changed before analysis")

    payload = [
        "policy", "contract", "budget_bp", "removed_encoded_indices",
        "removed_group_ids", "realized_encoded_cost",
    ]
    conflict = selections.groupby("selection_hash", sort=False)[payload].nunique(dropna=False)
    if (conflict > 1).any().any():
        raise RuntimeError("selection_hash maps to conflicting payloads")
    selection_unique = selections.drop_duplicates("selection_hash").copy()

    key_by_hash = governed[["selection_hash", *KEY]].drop_duplicates()
    if key_by_hash["selection_hash"].duplicated().any():
        raise RuntimeError("selection_hash maps to multiple canonical keys")
    selection_unique = selection_unique.merge(
        key_by_hash, on="selection_hash", how="left", validate="one_to_one"
    )
    if selection_unique[KEY].isna().any().any():
        raise RuntimeError("selection payload lacks canonical-key binding")

    registry = build_semantic_registry(
        ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz",
        ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz",
    )
    semantic = pd.DataFrame(
        [semantic_metrics_for_selection(row, registry) for _, row in selection_unique.iterrows()],
        index=selection_unique.index,
    )
    selection_metrics = pd.concat(
        [selection_unique[["selection_hash"]], semantic], axis=1
    )

    enriched = add_score_metrics(governed).merge(
        selection_metrics, on="selection_hash", how="left", validate="many_to_one"
    )
    if enriched[[m for m in METRICS if m != "introduced_distortion_zero_opp"]].isna().any().any():
        raise RuntimeError("unexpected missing reconstructed metrics")
    enriched["archetype"] = enriched.dataset_index.map(ARCHETYPE)
    enriched["mechanism_family"] = enriched.mechanism.map(MECHANISM_FAMILY)

    p2 = (
        enriched[enriched.policy.eq("P2")]
        .groupby([*KEY, "contract", "budget_bp"], sort=True)[METRICS]
        .mean()
        .add_prefix("p2_")
        .reset_index()
    )
    p2_counts = (
        enriched[enriched.policy.eq("P2")]
        .groupby([*KEY, "contract", "budget_bp"]).size()
    )
    if not p2_counts.eq(20).all() or len(p2_counts) != 5500 * 2 * 3:
        raise RuntimeError("P2 seed closure failed during analysis")

    deterministic = enriched[enriched.policy.isin(POLICIES)].copy()
    det_counts = deterministic.groupby([*KEY, "policy", "contract", "budget_bp"]).size()
    if not det_counts.eq(1).all() or len(det_counts) != 5500 * 4 * 2 * 3:
        raise RuntimeError("deterministic-policy closure failed during analysis")
    paired = deterministic.merge(
        p2, on=[*KEY, "contract", "budget_bp"], how="left", validate="many_to_one"
    )
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"p2_{metric}"]
    paired["archetype"] = paired.dataset_index.map(ARCHETYPE)
    paired["mechanism_family"] = paired.mechanism.map(MECHANISM_FAMILY)

    paired_columns = [
        *KEY, "archetype", "mechanism_family", "policy", "contract", "budget_bp",
        *[f"delta_{metric}" for metric in METRICS],
    ]
    paired[paired_columns].to_csv(
        output / "paired_effects.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )

    overall = summarize(paired, ["policy", "contract", "budget_bp"], "overall")
    mechanism = summarize(
        paired, ["policy", "contract", "budget_bp", "mechanism"], "mechanism"
    )
    archetype = summarize(
        paired, ["policy", "contract", "budget_bp", "archetype"], "archetype"
    )
    family = summarize(
        paired, ["policy", "contract", "budget_bp", "mechanism_family"], "family"
    )
    overall.to_csv(output / "overall_summary.csv", index=False)
    mechanism.to_csv(output / "mechanism_summary.csv", index=False)
    archetype.to_csv(output / "archetype_summary.csv", index=False)
    family.to_csv(output / "family_summary.csv", index=False)

    # Cost-contract sensitivity: semantic effect minus encoded-column effect.
    identity = [*KEY, "policy", "budget_bp"]
    contract_rows = []
    for metric in METRICS:
        pivot = paired.pivot(index=identity, columns="contract", values=f"delta_{metric}")
        diff = (pivot["semantic_group"] - pivot["encoded_column"]).rename(metric).reset_index()
        for group, part in diff.groupby(["policy", "budget_bp"], sort=True):
            item = estimate(part, metric, f"contract|{group}|{metric}")
            item.update({"policy": group[0], "budget_bp": group[1]})
            contract_rows.append(item)
    contract_summary = pd.DataFrame(contract_rows)
    contract_summary.to_csv(output / "contract_sensitivity.csv", index=False)

    # Deterministic policy pairwise contrasts (first policy minus second).
    pairwise_rows = []
    comparisons = [(a, b) for i, a in enumerate(POLICIES) for b in POLICIES[i + 1:]]
    for metric in METRICS:
        pivot = paired.pivot(
            index=[*KEY, "contract", "budget_bp"], columns="policy",
            values=f"delta_{metric}",
        )
        for first, second in comparisons:
            diff = (pivot[first] - pivot[second]).rename(metric).reset_index()
            for group, part in diff.groupby(["contract", "budget_bp"], sort=True):
                item = estimate(part, metric, f"pairwise|{first}|{second}|{group}|{metric}")
                item.update({
                    "first_policy": first, "second_policy": second,
                    "contract": group[0], "budget_bp": group[1],
                })
                pairwise_rows.append(item)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(output / "policy_pairwise.csv", index=False)

    # Apply the frozen claim gate at the primary semantic-group 20% contract.
    from scripts.t0_b.claim_gates import MetricEstimate, determine_claim_status

    claims = {}
    table1 = []
    primary = {"contract": "semantic_group", "budget_bp": 2000}
    for policy in POLICIES:
        ident = {"policy": policy, **primary}
        estimates = {metric: lookup(overall, ident, metric) for metric in METRICS}
        def est(metric: str) -> MetricEstimate:
            value = estimates[metric]
            return MetricEstimate(value["mean"], value["ci_lo"], value["ci_hi"])
        evaluable = estimates["introduced_distortion_zero_opp"]["n_keys"] > 0
        status = determine_claim_status(
            est("legacy_sdr"), est("directional_repair"),
            est("semantic_group_recall_full"), est("semantic_group_recall_any"),
            est("overcorrection"), est("legit_retention"),
            est("introduced_distortion_zero_opp"), evaluable,
        )
        claims[policy] = {
            "status": status,
            "contract": "semantic_group",
            "budget_bp": 2000,
            "evaluable_zero_opportunity": evaluable,
            "allowed_wording": (
                f"Under the semantic-group 20% contract, {policy} is {status.lower().replace('_', ' ')} "
                "relative to the 20-seed matched-random baseline in the controlled LR registry."
            ),
            "forbidden_wording": "The policy generally solves tabular leakage.",
            "metrics": {
                metric: {k: estimates[metric][k] for k in ["mean", "ci_lo", "ci_hi", "n_keys", "n_tasks"]}
                for metric in METRICS
            },
        }
        table1.append({
            "policy": policy,
            "claim_status": status,
            "delta_legacy_sdr": fmt(estimates["legacy_sdr"]),
            "delta_directional_repair": fmt(estimates["directional_repair"]),
            "delta_full_group_recall": fmt(estimates["semantic_group_recall_full"]),
            "delta_overcorrection": fmt(estimates["overcorrection"]),
            "delta_legit_retention": fmt(estimates["legit_retention"]),
            "n_keys": estimates["legacy_sdr"]["n_keys"],
            "n_tasks": estimates["legacy_sdr"]["n_tasks"],
        })
    atomic_json(output / "claim_state.json", {
        "schema_version": 1,
        "evidence_tier": "confirmatory",
        "primary_estimand": "deterministic policy minus within-key mean of 20 matched-random seeds",
        "claims": claims,
    })
    pd.DataFrame(table1).to_csv(output / "paper_table_1_policy.csv", index=False)

    table2 = contract_summary[
        contract_summary.budget_bp.eq(2000)
        & contract_summary.metric.isin(["legacy_sdr", "directional_repair", "semantic_group_recall_full", "overcorrection"])
    ].copy()
    table2["semantic_minus_encoded"] = table2.apply(fmt, axis=1)
    table2[["policy", "metric", "semantic_minus_encoded", "n_keys", "n_tasks"]].to_csv(
        output / "paper_table_2_contract.csv", index=False
    )

    table3 = archetype[
        archetype.contract.eq("semantic_group")
        & archetype.budget_bp.eq(2000)
        & archetype.metric.isin(["delta_legacy_sdr", "delta_directional_repair", "delta_overcorrection"])
    ].copy()
    table3["effect"] = table3.apply(fmt, axis=1)
    table3[["policy", "archetype", "metric", "effect", "n_keys", "n_tasks"]].to_csv(
        output / "paper_table_3_archetype.csv", index=False
    )

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "canonical_keys": 5500,
        "governed_rows": len(governed),
        "selection_rows": len(selections),
        "unique_selection_hashes": len(selection_unique),
        "primary_claim_status": {policy: claims[policy]["status"] for policy in POLICIES},
        "primary_legacy_sdr": {
            policy: claims[policy]["metrics"]["legacy_sdr"] for policy in POLICIES
        },
        "limitations": [
            "Full-B1 evaluates LR downstream repair; RF and LightGBM use the separately frozen cross-learner amendment.",
            "Task-cluster intervals describe the 20-task designed registry and are not population intervals.",
            "Archetype summaries contain four tasks each and are sensitivity analyses.",
        ],
    }
    atomic_json(output / "analysis_summary.json", summary)

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "analysis_manifest.json":
            artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json(output / "analysis_manifest.json", {
        "schema_version": 1,
        "status": "PASS",
        "source_validation_receipt_sha256": sha256(result_root / "validation_receipt.json"),
        "source_merge_manifest_sha256": sha256(merged / "merge_manifest.json"),
        "artifacts": artifacts,
    })
    print("R10E_ANALYSIS_PASS")
    print(json.dumps(summary["primary_claim_status"], sort_keys=True))


if __name__ == "__main__":
    main()
