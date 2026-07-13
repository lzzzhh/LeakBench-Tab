#!/usr/bin/env python3
"""consolidate_three_axis.py — Phase 9: Complete three-axis mechanism profiles.

Loads all existing results (V1 generic inflation + V2 aligned harm + Rescue Matrix),
fills gaps for simple mechanisms, computes I/A/S/E diagnostics,
generates the final three-axis classification table.
"""

import sys, json, glob, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = Path("results/leakbench/profiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("figures/phase9")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Mechanism registry ──
MECHANISMS = {
    "M01": {"name": "Direct Target Copy", "category": "simple",
            "contamination": True, "evidence": "Injected as direct copy of y"},
    "M02": {"name": "Noisy Target Proxy", "category": "simple",
            "contamination": True, "evidence": "Injected as y + noise"},
    "M03": {"name": "Nonlinear Target Transform", "category": "boundary",
            "contamination": True, "evidence": "Injected as nonlinear fn of y"},
    "M04": {"name": "Post-Outcome Aggregation", "category": "structured",
            "contamination": True, "evidence": "Uses future window after prediction time"},
    "M05": {"name": "Temporal Look-Ahead", "category": "structured",
            "contamination": True, "evidence": "Uses cumulative future values"},
    "M06": {"name": "Redundant Leakage Cluster", "category": "simple",
            "contamination": True, "evidence": "Multiple redundant copies of y"},
    "M07": {"name": "Sparse Subgroup Leakage", "category": "boundary",
            "contamination": True, "evidence": "Leakage only in subpopulation"},
    "M08": {"name": "Entity Leakage", "category": "structured",
            "contamination": True, "evidence": "Entity ID encodes label distribution"},
    "M09": {"name": "Source Leakage", "category": "structured",
            "contamination": True, "evidence": "Source ID encodes label prevalence"},
    "M10": {"name": "Mixed Leakage", "category": "simple",
            "contamination": True, "evidence": "Both legit and leak features injected"},
    "M11": {"name": "Graph-Mediated Leakage", "category": "boundary",
            "contamination": True, "evidence": "Distributed over graph components"},
}


def load_v1_results():
    """Load V1 generic inflation results (6,600 cells)."""
    files = glob.glob("results/leakbench/full_matrix/raw/*.json")
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                r = json.load(fh)
            if r.get("status") != "COMPLETED":
                continue
            rows.append(r)
        except:
            pass
    return pd.DataFrame(rows)


def load_rescue_results():
    """Load V2 rescue matrix results (648 cells)."""
    path = "results/leakbench/structured_rescue/structured_rescue_matrix.csv"
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_manifest():
    """Load V1 manifest with mechanism/model/strength."""
    path = "results/manifests/lr_rf_full.parquet"
    if Path(path).exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def classify_detectability(auprc):
    if auprc >= 0.80: return "HIGH"
    if auprc >= 0.30: return "MEDIUM"
    return "LOW"


def classify_exploitability(harm_mean, harm_rate):
    if harm_mean >= 0.02 and harm_rate >= 0.60:
        return "HIGH"
    if harm_mean >= 0.005 or harm_rate >= 0.20:
        return "CONDITIONAL"
    return "LOW"


def profile_name(contamination, detectability, exploitability):
    c = "C1" if contamination else "C0"
    d = {"HIGH": "DH", "MEDIUM": "DM", "LOW": "DL"}[detectability]
    x = {"HIGH": "XH", "CONDITIONAL": "XC", "LOW": "XL"}[exploitability]
    return f"{c}-{d}-{x}"


def main():
    print("=" * 70)
    print("  Phase 9: Three-Axis Mechanism Profile Consolidation")
    print("=" * 70)

    # ── Load all data ──
    v1 = load_v1_results()
    rescue = load_rescue_results()
    manifest = load_manifest()
    print(f"\n  V1 generic: {len(v1)} result cells")
    print(f"  V2 rescue:  {len(rescue)} aligned cells")

    # ── Merge manifest metadata into V1 ──
    if not manifest.empty and "run_id" in manifest.columns and "run_id" in v1.columns:
        v1 = v1.merge(manifest[["run_id", "mechanism_id", "model_id", "strength_id"]],
                      on="run_id", how="left")

    # ── Compute per-mechanism stats ──
    profiles = []
    for mech_id, info in MECHANISMS.items():
        print(f"\n  {mech_id} {info['name']}...")

        # V1 diagnostic stats
        v1_mech = v1[v1.get("mechanism_id", "") == mech_id] if "mechanism_id" in v1.columns else pd.DataFrame()
        if v1_mech.empty and "mechanism" in v1.columns:
            v1_mech = v1[v1["mechanism"] == mech_id]

        diag_auprc = v1_mech["diagnostic_auprc"].mean() if not v1_mech.empty and "diagnostic_auprc" in v1_mech.columns else float("nan")
        diag_auprc_std = v1_mech["diagnostic_auprc"].std(ddof=1) if not v1_mech.empty and "diagnostic_auprc" in v1_mech.columns else 0.0
        top5_recall = v1_mech["top5_recall"].mean() if not v1_mech.empty and "top5_recall" in v1_mech.columns else float("nan")
        legit_fpr = v1_mech["legitimate_fpr"].mean() if not v1_mech.empty and "legitimate_fpr" in v1_mech.columns else float("nan")

        # V1 generic inflation
        v1_infl = v1_mech["inflation_auroc"].dropna() if not v1_mech.empty and "inflation_auroc" in v1_mech.columns else pd.Series(dtype=float)
        generic_infl_mean = v1_infl.mean() if len(v1_infl) > 0 else float("nan")

        # V2 aligned harm (from rescue matrix or compute from V1 for simple mech)
        rescue_mech = rescue[rescue.get("mechanism", "") == mech_id] if not rescue.empty else pd.DataFrame()
        if not rescue_mech.empty:
            harm = rescue_mech["harm_gap"].dropna()
            aligned_harm_mean = harm.mean() if len(harm) > 0 else float("nan")
            aligned_harm_std = harm.std(ddof=1) if len(harm) > 1 else 0.0
            harm_gt_001 = (harm > 0.01).mean() if len(harm) > 0 else 0.0
            harm_gt_002 = (harm > 0.02).mean() if len(harm) > 0 else 0.0
            # Per-model harm
            lr_harm = rescue_mech[rescue_mech["model"] == "lr"]["harm_gap"].mean() if "model" in rescue_mech.columns else float("nan")
            rf_harm = rescue_mech[rescue_mech["model"] == "rf"]["harm_gap"].mean() if "model" in rescue_mech.columns else float("nan")
            cb_harm = rescue_mech[rescue_mech["model"] == "catboost"]["harm_gap"].mean() if "model" in rescue_mech.columns else float("nan")
        elif info["category"] == "simple":
            # For simple mechanisms, aligned harm = generic inflation (remove leakage = clean)
            aligned_harm_mean = generic_infl_mean
            aligned_harm_std = v1_infl.std(ddof=1) if len(v1_infl) > 1 else 0.0
            harm_gt_001 = (v1_infl > 0.01).mean() if len(v1_infl) > 0 else 0.0
            harm_gt_002 = (v1_infl > 0.02).mean() if len(v1_infl) > 0 else 0.0
            lr_harm = v1_mech[v1_mech.get("model_id","")=="logistic_regression"]["inflation_auroc"].mean() if not v1_mech.empty and "model_id" in v1_mech.columns else float("nan")
            rf_harm = v1_mech[v1_mech.get("model_id","")=="random_forest"]["inflation_auroc"].mean() if not v1_mech.empty and "model_id" in v1_mech.columns else float("nan")
            cb_harm = float("nan")
        else:
            aligned_harm_mean = float("nan")
            aligned_harm_std = 0.0
            harm_gt_001 = 0.0
            harm_gt_002 = 0.0
            lr_harm = rf_harm = cb_harm = float("nan")

        # Classify
        detectability = classify_detectability(diag_auprc) if not np.isnan(diag_auprc) else "UNKNOWN"
        exploitability = classify_exploitability(aligned_harm_mean, harm_gt_001) if not np.isnan(aligned_harm_mean) else "UNKNOWN"
        profile = profile_name(info["contamination"], detectability, exploitability)

        profiles.append({
            "mechanism_id": mech_id,
            "mechanism_name": info["name"],
            "category": info["category"],
            "contamination_validity": info["contamination"],
            "contamination_evidence": info["evidence"],
            "i_only_auprc": round(diag_auprc, 4) if not np.isnan(diag_auprc) else None,
            "diag_auprc_std": round(aligned_harm_std, 4),
            "top5_recall": round(top5_recall, 4) if not np.isnan(top5_recall) else None,
            "legitimate_fpr": round(legit_fpr, 4) if not np.isnan(legit_fpr) else None,
            "generic_inflation": round(generic_infl_mean, 4) if not np.isnan(generic_infl_mean) else None,
            "aligned_harm_mean": round(aligned_harm_mean, 4) if not np.isnan(aligned_harm_mean) else None,
            "aligned_harm_std": round(aligned_harm_std, 4),
            "harm_gt_001_rate": round(harm_gt_001, 4),
            "harm_gt_002_rate": round(harm_gt_002, 4),
            "lr_harm": round(lr_harm, 4) if not (isinstance(lr_harm, float) and np.isnan(lr_harm)) else None,
            "rf_harm": round(rf_harm, 4) if not (isinstance(rf_harm, float) and np.isnan(rf_harm)) else None,
            "catboost_harm": round(cb_harm, 4) if not (isinstance(cb_harm, float) and np.isnan(cb_harm)) else None,
            "detectability_class": detectability,
            "exploitability_class": exploitability,
            "final_profile": profile,
        })

        tag = "✓" if detectability == "HIGH" else ("△" if detectability == "MEDIUM" else "✗")
        x_tag = "!" if exploitability == "HIGH" else ("~" if exploitability == "CONDITIONAL" else "-")
        print(f"    {tag}{x_tag} {profile:12s} diag={diag_auprc:.3f} harm={aligned_harm_mean:+.4f} harm>0.01={harm_gt_001:.0%}")

    # ── Save profiles ──
    df = pd.DataFrame(profiles)
    df.to_csv(OUTPUT_DIR / "mechanism_profiles.csv", index=False)
    with open(OUTPUT_DIR / "mechanism_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2, default=str)

    # ── Summary statistics ──
    print(f"\n{'=' * 70}")
    print("  THREE-AXIS SUMMARY")
    print("=" * 70)

    for cat in ["simple", "boundary", "structured"]:
        cat_df = df[df["category"] == cat]
        if cat_df.empty: continue
        d = cat_df["aligned_harm_mean"].dropna()
        print(f"\n  {cat.upper()} ({len(cat_df)} mechanisms):")
        print(f"    AUPRC: {cat_df['i_only_auprc'].mean():.3f} ± {cat_df['i_only_auprc'].std(ddof=1):.3f}")
        print(f"    Harm:  {d.mean():+.4f} ± {d.std(ddof=1):.4f}" if len(d) > 0 else "    Harm: N/A")
        print(f"    Profiles: {cat_df['final_profile'].value_counts().to_dict()}")

    # ── Correlations ──
    valid = df.dropna(subset=["i_only_auprc", "aligned_harm_mean"])
    if len(valid) >= 5:
        r_diag_harm, p_diag_harm = spearmanr(valid["i_only_auprc"], valid["aligned_harm_mean"])
        print(f"\n  Correlation: AUPRC vs Harm: r={r_diag_harm:.3f} (p={p_diag_harm:.3f})")

    # ── Model heterogeneity ──
    print(f"\n  Model Harm (mean):")
    for col in ["lr_harm", "rf_harm", "catboost_harm"]:
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"    {col}: {vals.mean():+.4f}")

    # ── Profile distribution ──
    print(f"\n  Profile distribution:")
    for p in sorted(df["final_profile"].unique()):
        count = (df["final_profile"] == p).sum()
        mechs = df[df["final_profile"] == p]["mechanism_id"].tolist()
        print(f"    {p}: {count} mechanisms ({', '.join(mechs)})")

    # ── Source data for figures ──
    source = df[["mechanism_id", "mechanism_name", "category", "i_only_auprc",
                  "aligned_harm_mean", "harm_gt_001_rate", "detectability_class",
                  "exploitability_class", "final_profile"]].copy()
    source.to_csv(FIG_DIR.parent / "source_data" / "three_axis_map.csv", index=False)

    print(f"\n  Profiles saved to: {OUTPUT_DIR}")
    print(f"  Source data: figures/source_data/three_axis_map.csv")
    print("=" * 70)

    return df


if __name__ == "__main__":
    main()
