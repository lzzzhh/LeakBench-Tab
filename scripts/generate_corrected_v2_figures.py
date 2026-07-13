#!/usr/bin/env python3
"""Generate submission figures from complete corrected_v2 evidence only."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATISTICS = ROOT / "results/corrected_v2/statistics"
OUTPUT = ROOT / "paper/aaai27/figures/generated"
MECHANISMS = [f"M{index:02d}" for index in range(1, 12)]
MODELS = ["lr", "rf", "catboost", "lightgbm", "tabm"]
MODEL_LABELS = {"lr": "LR", "rf": "RF", "catboost": "CatBoost", "lightgbm": "LightGBM", "tabm": "TabM"}
CATEGORIES = {
    "M01": "simple", "M02": "simple", "M03": "boundary",
    "M04": "structured", "M05": "structured", "M06": "simple",
    "M07": "boundary", "M08": "structured", "M09": "structured",
    "M10": "simple", "M11": "boundary",
}
COLORS = {"simple": "#D55E00", "boundary": "#0072B2", "structured": "#009E73"}
MARKERS = {"simple": "o", "boundary": "s", "structured": "^"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_final_integrity():
    required_json = {
        "canonical": ROOT / "results/corrected_v2/canonical_manifest.json",
        "primary": STATISTICS / "integrity_summary.json",
        "diagnostic": STATISTICS / "diagnostic_integrity.json",
        "secondary": STATISTICS / "secondary_integrity.json",
        "claims": ROOT / "results/corrected_v2/paper_claims.json",
    }
    payloads = {}
    for name, path in required_json.items():
        if not path.exists():
            raise FileNotFoundError(f"missing final-only input: {path}")
        if "pilot" in str(path).lower():
            raise RuntimeError("pilot input is forbidden for submission figures")
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    canonical = payloads["canonical"]
    if canonical.get("status") != "CANONICAL" or canonical.get("cells") != 27500 or canonical.get("successful_cells") != 27500:
        raise RuntimeError("canonical 27,500-cell matrix is not complete")
    primary = payloads["primary"]
    if primary.get("namespace") != "confirmatory" or primary.get("rows_success") != 27500 or primary.get("completion_rate") != 1.0:
        raise RuntimeError("primary confirmatory statistics are incomplete")
    diagnostic = payloads["diagnostic"]
    if (
        diagnostic.get("evidence_tier") != "confirmatory"
        or diagnostic.get("rows_success") != 22000
        or diagnostic.get("rows_failure") != 0
        or diagnostic.get("expected_cells") != 22000
    ):
        raise RuntimeError("diagnostic confirmatory statistics are incomplete")
    secondary = payloads["secondary"]
    if secondary.get("evidence_tier") != "confirmatory" or secondary.get("rows_success") != 27500:
        raise RuntimeError("secondary confirmatory statistics are incomplete")
    claims = payloads["claims"]
    if claims.get("evidence_tier") != "confirmatory":
        raise RuntimeError("paper claim release is not confirmatory")
    return required_json


def _read_csv(name, required_columns):
    path = STATISTICS / name
    if not path.exists() or "pilot" in str(path).lower():
        raise FileNotFoundError(f"missing final-only table: {path}")
    frame = pd.read_csv(path)
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    return frame, path


def _save(fig, filename):
    path = OUTPUT / filename
    fixed_time = datetime(2026, 7, 13, tzinfo=timezone.utc)
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Creator": "LeakBench-Tab deterministic figure generator",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    plt.close(fig)
    return path


def cdx_scatter():
    harm, harm_path = _read_csv(
        "mechanism_summary.csv",
        ["mechanism", "category", "paired_harm", "paired_harm_ci_low", "paired_harm_ci_high"],
    )
    detect, detect_path = _read_csv(
        "detectability_mechanism_summary.csv",
        ["mechanism", "diagnostic_normalized_ap", "diagnostic_normalized_ap_ci_low", "diagnostic_normalized_ap_ci_high"],
    )
    frame = harm.merge(detect, on="mechanism", validate="one_to_one").set_index("mechanism").loc[MECHANISMS].reset_index()
    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    for category in ("simple", "boundary", "structured"):
        block = frame[frame["category"] == category]
        x = block["diagnostic_normalized_ap"].to_numpy()
        y = block["paired_harm"].to_numpy()
        xerr = np.vstack([
            x - block["diagnostic_normalized_ap_ci_low"].to_numpy(),
            block["diagnostic_normalized_ap_ci_high"].to_numpy() - x,
        ])
        yerr = np.vstack([
            y - block["paired_harm_ci_low"].to_numpy(),
            block["paired_harm_ci_high"].to_numpy() - y,
        ])
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt=MARKERS[category], markersize=4.2,
            color=COLORS[category], ecolor=COLORS[category], elinewidth=0.7,
            capsize=1.5, alpha=0.88, label=category.capitalize(),
        )
        for row in block.itertuples():
            ax.annotate(row.mechanism, (row.diagnostic_normalized_ap, row.paired_harm),
                        xytext=(3, 3), textcoords="offset points", fontsize=6.3)
    ax.axhline(0, color="0.45", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Primary MI detectability (normalized AP)")
    ax.set_ylabel("Paired AUROC harm")
    ax.legend(frameon=False, fontsize=6.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(True, color="0.9", linewidth=0.5)
    return _save(fig, "cdx_scatter.pdf"), [harm_path, detect_path]


def mechanism_model_heatmap():
    frame, source = _read_csv(
        "mechanism_model_summary.csv", ["mechanism", "model", "paired_harm"]
    )
    matrix = frame.pivot(index="mechanism", columns="model", values="paired_harm").reindex(index=MECHANISMS, columns=MODELS)
    if matrix.isna().any().any():
        raise ValueError("mechanism-model heatmap matrix is incomplete")
    values = matrix.to_numpy()
    maximum = max(abs(float(values.min())), abs(float(values.max())), 0.01)
    fig, ax = plt.subplots(figsize=(3.45, 4.25))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum))
    ax.set_xticks(np.arange(len(MODELS)), [MODEL_LABELS[model] for model in MODELS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(MECHANISMS)), MECHANISMS)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if abs(values[row, column]) > 0.55 * maximum else "black"
            ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=5.6, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Paired AUROC harm")
    return _save(fig, "mechanism_model_heatmap.pdf"), [source]


def strength_diagnostic_robustness():
    dose, dose_path = _read_csv(
        "strength_dose_response.csv",
        ["mechanism", "harm_S1", "harm_S2", "harm_S3", "harm_S4", "harm_S5"],
    )
    diagnostic, diagnostic_path = _read_csv(
        "diagnostic_method_by_mechanism.csv",
        ["method", "mechanism", "diagnostic_normalized_ap"],
    )
    dose = dose.set_index("mechanism").loc[MECHANISMS]
    methods = ["absolute_correlation", "lr_coefficient", "mutual_information", "rf_permutation"]
    diagnostic_matrix = (
        diagnostic.pivot(index="mechanism", columns="method", values="diagnostic_normalized_ap")
        .reindex(index=MECHANISMS, columns=methods)
    )
    if diagnostic_matrix.isna().any().any():
        raise ValueError("diagnostic robustness matrix is incomplete")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.45, 5.3), gridspec_kw={"height_ratios": [1.05, 1.35]})
    x = np.arange(1, 6)
    for mechanism in MECHANISMS:
        category = CATEGORIES[mechanism]
        y = dose.loc[mechanism, [f"harm_S{i}" for i in range(1, 6)]].to_numpy(dtype=float)
        top.plot(x, y, color=COLORS[category], linewidth=0.8, alpha=0.76, marker="o", markersize=1.8)
        top.annotate(mechanism, (x[-1], y[-1]), xytext=(2, 0), textcoords="offset points", fontsize=5.5, color=COLORS[category])
    top.axhline(0, color="0.45", linewidth=0.6, linestyle="--")
    top.set_xticks(x, [f"S{i}" for i in x])
    top.set_ylabel("Paired AUROC harm")
    top.set_title("(a) Pre-ordered strength response", loc="left", fontsize=8)
    top.grid(True, color="0.92", linewidth=0.4)

    values = diagnostic_matrix.to_numpy()
    image = bottom.imshow(values, aspect="auto", cmap="viridis", vmin=min(0.0, float(values.min())), vmax=max(1.0, float(values.max())))
    bottom.set_xticks(np.arange(len(methods)), ["Abs. corr.", "LR coef.", "MI", "RF perm."], rotation=30, ha="right")
    bottom.set_yticks(np.arange(len(MECHANISMS)), MECHANISMS)
    bottom.set_title("(b) Diagnostic-conditional normalized AP", loc="left", fontsize=8)
    colorbar = fig.colorbar(image, ax=bottom, fraction=0.045, pad=0.03)
    colorbar.set_label("Normalized AP")
    fig.subplots_adjust(hspace=0.38)
    return _save(fig, "strength_diagnostic_robustness.pdf"), [dose_path, diagnostic_path]


def main():
    integrity_paths = _require_final_integrity()
    mpl.rcParams.update({
        "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 8,
        "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "legend.fontsize": 6.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    sources = list(integrity_paths.values())
    for builder in (cdx_scatter, mechanism_model_heatmap, strength_diagnostic_robustness):
        output, used = builder()
        outputs.append(output)
        sources.extend(used)
    manifest = {
        "schema_version": 1,
        "evidence_tier": "confirmatory",
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__)),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(sources))},
        "figure_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "pilot_inputs_forbidden": True,
    }
    (OUTPUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
