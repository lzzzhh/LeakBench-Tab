#!/usr/bin/env python3
"""render_sp5_figures.py — figures from claim_ledger_v2 (no manual numbers)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SP5 = ROOT / "artifacts/sp5"
FIG = SP5 / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CATC = {"simple": "#2ca02c", "structured": "#d62728", "boundary": "#1f77b4"}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    led = pd.read_csv(SP5 / "claim_ledger_v2.csv")
    lineage = {}

    # CL2: structured AUPRC forest + simple vs structured
    ms = pd.read_csv(SP5 / "cl2/cl2_mechanism_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ms2 = ms.sort_values("mean")
    ax.barh(ms2["mechanism"], ms2["mean"], color=[CATC[c] for c in ms2["category"]])
    ax.set_xlabel("Detector AUPRC (diagnostic_ap)"); ax.set_title("CL2: Leak detectability by mechanism")
    ax.axvline(0.5, ls="--", c="gray", lw=0.8)
    plt.tight_layout(); p = FIG / "cl2_structured_auprc_forest.png"; plt.savefig(p, dpi=120); plt.close()
    lineage["cl2_structured_auprc_forest"] = {"source": "cl2/cl2_mechanism_summary.csv"}

    # CL3: detectability vs exploitability global + by category
    pts = pd.read_csv(SP5 / "cl3/cl3_mechanism_points.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in ["simple", "structured", "boundary"]:
        s = pts[pts.category == c]
        ax.scatter(s["detectability"], s["exploitability"], c=CATC[c], label=c, s=80)
        for _, r in s.iterrows():
            ax.annotate(r["mechanism"], (r["detectability"], r["exploitability"]), fontsize=7)
    ax.set_xlabel("Detectability (AUPRC)"); ax.set_ylabel("Exploitability (paired_harm)")
    ax.set_title("CL3: Detectability vs Exploitability"); ax.legend()
    plt.tight_layout(); p = FIG / "cl3_detectability_vs_exploitability_global.png"; plt.savefig(p, dpi=120); plt.close()
    lineage["cl3_detectability_vs_exploitability_global"] = {"source": "cl3/cl3_mechanism_points.csv"}

    # CL3 by category (small multiples)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, c in zip(axes, ["simple", "structured", "boundary"]):
        s = pts[pts.category == c]
        ax.scatter(s["detectability"], s["exploitability"], c=CATC[c], s=80)
        for _, r in s.iterrows():
            ax.annotate(r["mechanism"], (r["detectability"], r["exploitability"]), fontsize=7)
        ax.set_title(f"{c}"); ax.set_xlabel("Detectability")
    axes[0].set_ylabel("Exploitability")
    plt.tight_layout(); p = FIG / "cl3_detectability_vs_exploitability_by_category.png"; plt.savefig(p, dpi=120); plt.close()
    lineage["cl3_detectability_vs_exploitability_by_category"] = {"source": "cl3/cl3_mechanism_points.csv"}

    # CL4: model harm forest
    cs = pd.read_csv(SP5 / "cl4/cl4_model_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(cs["mean"], range(len(cs)),
                xerr=[cs["mean"] - cs["ci_lo"], cs["ci_hi"] - cs["mean"]], fmt="o", capsize=4)
    ax.set_yticks(range(len(cs))); ax.set_yticklabels(cs["model"])
    ax.set_xlabel("Mean paired_harm (95% CI)"); ax.set_title("CL4: Exploitation by model")
    plt.tight_layout(); p = FIG / "cl4_model_harm_forest.png"; plt.savefig(p, dpi=120); plt.close()
    lineage["cl4_model_harm_forest"] = {"source": "cl4/cl4_model_summary.csv"}

    # CL4: model x mechanism heatmap
    mm = pd.read_csv(SP5 / "cl4/cl4_model_mechanism_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(mm.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(mm.columns))); ax.set_xticklabels(mm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mm.index))); ax.set_yticklabels(mm.index)
    fig.colorbar(im, ax=ax, label="paired_harm"); ax.set_title("CL4: model × mechanism")
    plt.tight_layout(); p = FIG / "cl4_model_mechanism_heatmap.png"; plt.savefig(p, dpi=120); plt.close()
    lineage["cl4_model_mechanism_heatmap"] = {"source": "cl4/cl4_model_mechanism_matrix.csv"}

    # CL10: model similarity heatmap
    sim = pd.read_csv(SP5 / "cl10/cl10_rank_correlations.csv")
    models = sorted(set(sim["model_a"]) | set(sim["model_b"]))
    M = pd.DataFrame(np.eye(len(models)), index=models, columns=models)
    for _, r in sim.iterrows():
        M.loc[r["model_a"], r["model_b"]] = r["spearman"]
        M.loc[r["model_b"], r["model_a"]] = r["spearman"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M.values, cmap="viridis", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M.values[i, j] < 0.85 else "black")
    fig.colorbar(im, ax=ax, label="Spearman"); ax.set_title("CL10: cross-model profile similarity")
    plt.tight_layout(); p = FIG / "cl10_model_similarity_heatmap.png"; plt.savefig(p, dpi=120); plt.close()
    lineage["cl10_model_similarity_heatmap"] = {"source": "cl10/cl10_rank_correlations.csv"}

    # CL10: three-axis profile (detectability vs exploitability, mechanism-avg)
    prof = pd.read_csv(SP5 / "cl10/cl10_three_axis_profiles.csv")
    avg = prof.groupby("mechanism").agg(detect=("detectability", "mean"),
                                        exploit=("exploitability", "mean")).reset_index()
    cat = led.drop_duplicates("mechanism").set_index("mechanism")["mechanism_category"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in avg.iterrows():
        c = CATC[cat[r["mechanism"]]]
        ax.scatter(r["detect"], r["exploit"], c=c, s=90)
        ax.annotate(r["mechanism"], (r["detect"], r["exploit"]), fontsize=7)
    ax.set_xlabel("Detectability"); ax.set_ylabel("Exploitability")
    ax.set_title("CL10: three-axis mechanism map (C=invalid for all)")
    plt.tight_layout(); p = FIG / "cl10_three_axis_profiles.png"; plt.savefig(p, dpi=120); plt.close()
    lineage["cl10_three_axis_profiles"] = {"source": "cl10/cl10_three_axis_profiles.csv"}

    for k in lineage:
        lineage[k]["figure_sha256"] = sha(FIG / f"{k}.png")
        lineage[k]["ledger_sha256"] = sha(SP5 / "claim_ledger_v2.csv")
    (FIG / "figure_lineage.json").write_text(json.dumps(lineage, indent=2))
    print(f"rendered {len(lineage)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
