#!/usr/bin/env python3
"""recompute_sp5_claims.py — CL2/CL3/CL4/CL10 from claim_ledger_v2.

Cluster bootstrap over datasets. Mechanism-level aggregation for CL3/CL10.
All outputs written under artifacts/sp5/cl{2,3,4,10}/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SP5 = ROOT / "artifacts/sp5"
SEED = 20260714
NBOOT = 10000
NBOOT_SENS = 2000

SIMPLE = ["M01", "M02", "M06", "M10"]
STRUCTURED = ["M04", "M05", "M08", "M09"]
BOUNDARY = ["M03", "M07", "M11"]
MODELS = ["lr", "rf", "lightgbm", "catboost", "tabm"]


def load():
    return pd.read_csv(SP5 / "claim_ledger_v2.csv")


def cluster_bootstrap(df, valcol, unit="dataset_index", nboot=NBOOT, seed=SEED, agg="mean"):
    """Resample clusters (datasets) with replacement; recompute stat each draw."""
    rng = np.random.RandomState(seed)
    units = df[unit].unique()
    means = []
    grp = {u: df[df[unit] == u][valcol].values for u in units}
    for _ in range(nboot):
        pick = rng.choice(units, len(units), replace=True)
        vals = np.concatenate([grp[u] for u in pick])
        means.append(np.mean(vals) if agg == "mean" else np.median(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(means)), float(lo), float(hi)


# ---------------- CL2 ----------------
def cl2(df):
    out = SP5 / "cl2"; out.mkdir(exist_ok=True)
    # detectability = diagnostic_ap; task-level, so dedup per mechanism-task
    task = df.drop_duplicates(["dataset_index", "mechanism", "strength", "seed"])
    # per-mechanism AUPRC
    ms = task.groupby("mechanism")["detectability_value"].agg(["mean", "median",
         lambda s: s.quantile(.25), lambda s: s.quantile(.75), "min", "max"]).reset_index()
    ms.columns = ["mechanism", "mean", "median", "q25", "q75", "min", "max"]
    ms["category"] = ms["mechanism"].map(lambda m: "simple" if m in SIMPLE else
                                          "structured" if m in STRUCTURED else "boundary")
    ms.to_csv(out / "cl2_mechanism_summary.csv", index=False)
    # model summary (detectability is model-indep, but report by-strength/model coverage)
    task.groupby("strength")["detectability_value"].agg(["mean", "median"]).to_csv(out / "cl2_strength_summary.csv")
    (task.groupby("mechanism")["detectability_value"].mean()).to_frame().to_csv(out / "cl2_model_summary.csv")

    struct = task[task["mechanism"].isin(STRUCTURED)]
    simple = task[task["mechanism"].isin(SIMPLE)]
    s_mean, s_lo, s_hi = cluster_bootstrap(struct, "detectability_value")
    p_mean, p_lo, p_hi = cluster_bootstrap(simple, "detectability_value")
    # diff simple - structured (cluster bootstrap on paired datasets)
    rng = np.random.RandomState(SEED)
    units = task["dataset_index"].unique()
    diffs = []
    sg = {u: simple[simple["dataset_index"] == u]["detectability_value"].mean() for u in units}
    tg = {u: struct[struct["dataset_index"] == u]["detectability_value"].mean() for u in units}
    for _ in range(NBOOT):
        pick = rng.choice(units, len(units), replace=True)
        diffs.append(np.mean([sg[u] for u in pick]) - np.mean([tg[u] for u in pick]))
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
    cmp = pd.DataFrame([{
        "simple_auprc_mean": p_mean, "simple_ci": [p_lo, p_hi],
        "structured_auprc_mean": s_mean, "structured_ci": [s_lo, s_hi],
        "diff_simple_minus_structured": p_mean - s_mean, "diff_ci": [float(d_lo), float(d_hi)],
        "structured_range": [float(ms[ms.category == "structured"]["mean"].min()),
                             float(ms[ms.category == "structured"]["mean"].max())],
    }])
    cmp.to_csv(out / "cl2_simple_structured_comparison.csv", index=False)

    boot = {"structured_auprc": {"mean": s_mean, "lo": s_lo, "hi": s_hi},
            "simple_auprc": {"mean": p_mean, "lo": p_lo, "hi": p_hi},
            "n_replicates": NBOOT, "seed": SEED, "unit": "dataset_index"}
    (out / "cl2_bootstrap.json").write_text(json.dumps(boot, indent=2))

    # LOMO / LODO sensitivity (structured mean)
    lomo = []
    for m in STRUCTURED:
        sub = struct[struct["mechanism"] != m]
        lomo.append({"left_out": m, "structured_mean": float(sub["detectability_value"].mean())})
    pd.DataFrame(lomo).to_csv(out / "cl2_lomo.csv", index=False)
    lodo = []
    for d in sorted(struct["dataset_index"].unique()):
        sub = struct[struct["dataset_index"] != d]
        lodo.append({"left_out_dataset": int(d), "structured_mean": float(sub["detectability_value"].mean())})
    pd.DataFrame(lodo).to_csv(out / "cl2_lodo.csv", index=False)

    return {"structured_mean": s_mean, "structured_ci": [s_lo, s_hi],
            "structured_range": cmp["structured_range"].iloc[0],
            "simple_mean": p_mean, "diff": p_mean - s_mean, "diff_ci": [float(d_lo), float(d_hi)],
            "per_mechanism": ms.set_index("mechanism")["mean"].round(4).to_dict()}


# ---------------- CL3 ----------------
def cl3(df):
    out = SP5 / "cl3"; out.mkdir(exist_ok=True)
    # mechanism-level points: mean detectability (task-level) + mean exploitability (over models)
    det = df.drop_duplicates(["dataset_index", "mechanism", "strength", "seed"]) \
            .groupby("mechanism")["detectability_value"].mean()
    exp = df.groupby("mechanism")["paired_harm"].mean()
    pts = pd.DataFrame({"mechanism": det.index, "detectability": det.values,
                        "exploitability": exp.reindex(det.index).values})
    pts["category"] = pts["mechanism"].map(lambda m: "simple" if m in SIMPLE else
                                            "structured" if m in STRUCTURED else "boundary")
    pts.to_csv(out / "cl3_mechanism_points.csv", index=False)

    def corrs(sub):
        if len(sub) < 3:
            return {"pearson": None, "spearman": None, "kendall": None, "slope": None, "r2": None, "n": len(sub)}
        pr = stats.pearsonr(sub["detectability"], sub["exploitability"])
        sp = stats.spearmanr(sub["detectability"], sub["exploitability"])
        kt = stats.kendalltau(sub["detectability"], sub["exploitability"])
        sl, ic, r, pv, se = stats.linregress(sub["detectability"], sub["exploitability"])
        return {"pearson": float(pr[0]), "pearson_p": float(pr[1]),
                "spearman": float(sp.correlation), "kendall": float(kt.correlation),
                "slope": float(sl), "r2": float(r**2), "n": len(sub)}

    glob = corrs(pts)
    # bootstrap global pearson over mechanisms
    rng = np.random.RandomState(SEED)
    prs = []
    for _ in range(NBOOT):
        idx = rng.choice(len(pts), len(pts), replace=True)
        s = pts.iloc[idx]
        if s["detectability"].nunique() > 1 and s["exploitability"].nunique() > 1:
            prs.append(stats.pearsonr(s["detectability"], s["exploitability"])[0])
    glob["pearson_ci"] = [float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5))]
    pd.DataFrame([glob]).to_csv(out / "cl3_global_correlations.csv", index=False)

    within = {"simple": corrs(pts[pts.category == "simple"]),
              "structured": corrs(pts[pts.category == "structured"]),
              "boundary": corrs(pts[pts.category == "boundary"])}
    pd.DataFrame(within).T.to_csv(out / "cl3_within_category.csv")

    # regression models A-D
    import statsmodels.formula.api as smf
    pts2 = pts.copy()
    pts2["cat"] = pts2["category"]
    models = {}
    try:
        rA = smf.ols("exploitability ~ detectability", pts2).fit()
        rB = smf.ols("exploitability ~ C(cat)", pts2).fit()
        rC = smf.ols("exploitability ~ C(cat) + detectability", pts2).fit()
        rD = smf.ols("exploitability ~ C(cat) * detectability", pts2).fit()
        models = {"R2_A_detect": float(rA.rsquared), "R2_B_category": float(rB.rsquared),
                  "R2_C_both": float(rC.rsquared), "R2_D_interaction": float(rD.rsquared),
                  "incr_detect_after_category": float(rC.rsquared - rB.rsquared),
                  "incr_category_after_detect": float(rC.rsquared - rA.rsquared)}
    except Exception as e:
        models = {"error": str(e)}
    pd.DataFrame([models]).to_csv(out / "cl3_regression_models.csv", index=False)

    # partial correlation detect vs exploit controlling category (via residuals)
    try:
        import statsmodels.formula.api as smf
        pts2["cat"] = pts2["category"]
        re = smf.ols("exploitability ~ C(cat)", pts2).fit().resid
        rd = smf.ols("detectability ~ C(cat)", pts2).fit().resid
        pcorr = float(stats.pearsonr(rd, re)[0])
    except Exception:
        pcorr = None
    pd.DataFrame([{"partial_corr_detect_exploit_given_category": pcorr}]).to_csv(
        out / "cl3_partial_correlation.csv", index=False)

    # influence: Cook's distance for global OLS
    infl = {}
    try:
        import statsmodels.api as sm
        X = sm.add_constant(pts["detectability"].values)
        m = sm.OLS(pts["exploitability"].values, X).fit()
        cooks = m.get_influence().cooks_distance[0]
        infl = {pts.iloc[i]["mechanism"]: float(cooks[i]) for i in range(len(pts))}
    except Exception as e:
        infl = {"error": str(e)}
    pd.DataFrame([infl]).T.to_csv(out / "cl3_influence.csv")

    # leave-one-mechanism-out global pearson
    lomo = []
    for m in pts["mechanism"]:
        sub = pts[pts["mechanism"] != m]
        lomo.append({"left_out": m, "global_pearson": float(stats.pearsonr(
            sub["detectability"], sub["exploitability"])[0])})
    pd.DataFrame(lomo).to_csv(out / "cl3_lomo.csv", index=False)
    # leave-M08-out, leave-M10-out explicit
    for lm in ["M08", "M10"]:
        sub = pts[pts["mechanism"] != lm]
        r = stats.pearsonr(sub["detectability"], sub["exploitability"])[0]
        pd.DataFrame([{"left_out": lm, "global_pearson": float(r),
                       "within_structured": corrs(sub[sub.category == "structured"])["pearson"]}]).to_csv(
            out / f"cl3_leave_{lm.lower()}_out.csv", index=False)
    # leave-one-model-out (recompute exploit with one model dropped)
    lmom = []
    for mdl in MODELS:
        e = df[df["model"] != mdl].groupby("mechanism")["paired_harm"].mean()
        p = pd.DataFrame({"detectability": det, "exploitability": e.reindex(det.index)}).dropna()
        lmom.append({"left_out_model": mdl, "global_pearson": float(stats.pearsonr(
            p["detectability"], p["exploitability"])[0])})
    pd.DataFrame(lmom).to_csv(out / "cl3_lmom.csv", index=False)

    (out / "cl3_bootstrap.json").write_text(json.dumps(
        {"global_pearson": glob["pearson"], "pearson_ci": glob["pearson_ci"],
         "n_replicates": NBOOT, "seed": SEED, "unit": "mechanism"}, indent=2))

    return {"global": glob, "within": within, "regression": models,
            "partial_corr": pcorr, "influence": infl,
            "points": pts.round(4).to_dict("records")}


# ---------------- CL4 ----------------
def cl4(df):
    out = SP5 / "cl4"; out.mkdir(exist_ok=True)
    rows = []
    for mdl in MODELS:
        s = df[df["model"] == mdl]["paired_harm"]
        mean, lo, hi = cluster_bootstrap(df[df["model"] == mdl], "paired_harm")
        rows.append({"model": mdl, "mean": float(s.mean()), "median": float(s.median()),
                     "std": float(s.std()), "iqr": float(s.quantile(.75) - s.quantile(.25)),
                     "ci_lo": lo, "ci_hi": hi,
                     "pos_rate": float((s > 0.005).mean()), "neg_rate": float((s < -0.005).mean())})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "cl4_model_summary.csv", index=False)

    # matrices
    df.groupby(["model", "mechanism"])["paired_harm"].mean().unstack().to_csv(out / "cl4_model_mechanism_matrix.csv")
    df.groupby(["model", "strength"])["paired_harm"].mean().unstack().to_csv(out / "cl4_model_strength_matrix.csv")
    df.groupby(["model", "mechanism_category"])["paired_harm"].mean().unstack().to_csv(out / "cl4_model_category_matrix.csv")

    # paired comparisons on identical keys
    piv = df.pivot_table(index=["dataset_index", "mechanism", "strength", "seed"],
                         columns="model", values="paired_harm")
    pairs = [("rf", "lr"), ("lightgbm", "lr"), ("catboost", "lr"), ("tabm", "lr"),
             ("catboost", "rf"), ("catboost", "lightgbm"), ("catboost", "tabm"), ("rf", "tabm")]
    pc = []
    rng = np.random.RandomState(SEED)
    dsets = df["dataset_index"].unique()
    for a, b in pairs:
        d = (piv[a] - piv[b]).dropna()
        # cluster bootstrap on dataset
        dd = df[df["model"] == a][["dataset_index"]].copy()
        diff_by_key = (piv[a] - piv[b]).dropna().reset_index()
        diff_by_key.columns = list(diff_by_key.columns[:-1]) + ["diff"]
        g = {u: diff_by_key[diff_by_key["dataset_index"] == u]["diff"].values for u in dsets}
        bs = []
        for _ in range(NBOOT):
            pick = rng.choice(dsets, len(dsets), replace=True)
            bs.append(np.concatenate([g[u] for u in pick]).mean())
        pc.append({"comparison": f"{a}-{b}", "mean_diff": float(d.mean()),
                   "median_diff": float(d.median()),
                   "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                   "cohen_d": float(d.mean() / d.std()) if d.std() > 0 else None})
    pd.DataFrame(pc).to_csv(out / "cl4_pairwise_comparisons.csv", index=False)

    # ratios (guarded)
    lr_mean = summ[summ.model == "lr"]["mean"].iloc[0]
    ratios = []
    for mdl in ["rf", "lightgbm", "catboost", "tabm"]:
        mm = summ[summ.model == mdl]["mean"].iloc[0]
        ratios.append({"model": mdl, "ratio_vs_lr": float(mm / lr_mean) if abs(lr_mean) > 0.01 else None,
                       "lr_denominator_stable": bool(abs(lr_mean) > 0.01)})
    pd.DataFrame(ratios).to_csv(out / "cl4_ratios.csv", index=False)

    # interaction models
    inter = {}
    try:
        import statsmodels.formula.api as smf
        samp = df.sample(n=min(8000, len(df)), random_state=SEED)
        m1 = smf.ols("paired_harm ~ C(model) + C(mechanism) + C(strength)", samp).fit()
        m2 = smf.ols("paired_harm ~ C(model) * C(mechanism) + C(strength)", samp).fit()
        inter = {"main_R2": float(m1.rsquared), "interaction_R2": float(m2.rsquared),
                 "model_x_mechanism_incr_R2": float(m2.rsquared - m1.rsquared)}
    except Exception as e:
        inter = {"error": str(e)}
    pd.DataFrame([inter]).to_csv(out / "cl4_interaction_models.csv", index=False)

    (out / "cl4_bootstrap.json").write_text(json.dumps(
        {"models": rows, "n_replicates": NBOOT, "seed": SEED, "unit": "dataset_index"}, indent=2))

    return {"summary": summ.round(4).to_dict("records"), "pairwise": pc,
            "ratios": ratios, "lr_mean": float(lr_mean), "interaction": inter}


# ---------------- CL10 ----------------
def cl10(df):
    out = SP5 / "cl10"; out.mkdir(exist_ok=True)
    det = df.drop_duplicates(["dataset_index", "mechanism", "strength", "seed"]) \
            .groupby("mechanism")["detectability_value"].mean()
    # 55 profiles: mechanism x model, three axes
    prof = df.groupby(["mechanism", "model"])["paired_harm"].mean().reset_index()
    prof["detectability"] = prof["mechanism"].map(det)
    prof["construction"] = 1.0
    prof.rename(columns={"paired_harm": "exploitability"}, inplace=True)
    prof.to_csv(out / "cl10_raw_profiles.csv", index=False)
    assert len(prof) == 55, f"expected 55 profiles got {len(prof)}"

    # normalize axes to [0,1] over the mechanism-model population (fixed method)
    def norm(s): 
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0
    prof["exploit_norm"] = norm(prof["exploitability"])
    prof["detect_norm"] = norm(prof["detectability"])
    prof.to_csv(out / "cl10_three_axis_profiles.csv", index=False)
    (out / "cl10_normalization.json").write_text(json.dumps(
        {"method": "min-max over mechanism-model population", "axes": ["detectability", "exploitability"],
         "construction": "constant 1.0 (all invalid)"}, indent=2))

    # per-model mechanism vector (exploitability ranking) -> cross-model consistency
    piv = prof.pivot(index="mechanism", columns="model", values="exploitability")
    # pairwise Spearman across models
    from itertools import combinations
    rc = []
    for a, b in combinations(MODELS, 2):
        rho = stats.spearmanr(piv[a], piv[b]).correlation
        rc.append({"model_a": a, "model_b": b, "spearman": float(rho),
                   "kendall": float(stats.kendalltau(piv[a], piv[b]).correlation),
                   "cosine": float(np.dot(piv[a], piv[b]) / (np.linalg.norm(piv[a]) * np.linalg.norm(piv[b])))})
    rcdf = pd.DataFrame(rc)
    rcdf.to_csv(out / "cl10_rank_correlations.csv", index=False)
    rcdf.to_csv(out / "cl10_model_similarity.csv", index=False)

    # Kendall W (concordance across models over mechanism rankings)
    ranks = piv.rank(axis=0)
    n, k = ranks.shape
    Rj = ranks.sum(axis=1)
    S = ((Rj - Rj.mean())**2).sum()
    W = 12 * S / (k**2 * (n**3 - n))
    pd.DataFrame([{"kendall_w": float(W), "n_mechanisms": n, "n_models": k}]).to_csv(
        out / "cl10_kendall_w.csv", index=False)

    # quadrant agreement: HIGH/LOW detect x HIGH/LOW exploit (median split)
    dmed = prof["detectability"].median()
    emed = prof["exploitability"].median()
    prof["quadrant"] = prof.apply(lambda r: f"{'DH' if r.detectability>=dmed else 'DL'}-"
                                  f"{'XH' if r.exploitability>=emed else 'XL'}", axis=1)
    qa = prof.pivot(index="mechanism", columns="model", values="quadrant")
    agree = (qa.apply(lambda row: row.nunique() == 1, axis=1))
    qa["all_models_agree"] = agree
    qa.to_csv(out / "cl10_quadrant_agreement.csv")

    # leave-one-model-out consistency
    lomo = []
    for mdl in MODELS:
        rest = [m for m in MODELS if m != mdl]
        rhos = [stats.spearmanr(piv[a], piv[b]).correlation for a, b in combinations(rest, 2)]
        lomo.append({"left_out_model": mdl, "mean_pairwise_spearman": float(np.mean(rhos))})
    pd.DataFrame(lomo).to_csv(out / "cl10_lomo.csv", index=False)

    # M08 / M10 profile change vs superseded profiles_v2
    try:
        old = pd.read_csv(ROOT / "results/leakbench/profiles/mechanism_profiles_v2.csv")
        chg = []
        for m in ["M08", "M10"]:
            new_exp = prof[prof.mechanism == m]["exploitability"].mean()
            old_exp = old[old.mechanism == m]["core_mean_harm"].iloc[0] if m in old["mechanism"].values else None
            chg.append({"mechanism": m, "old_core_mean_harm": float(old_exp) if old_exp is not None else None,
                        "new_mean_exploitability": float(new_exp)})
        pd.DataFrame([c for c in chg if c["mechanism"] == "M08"]).to_csv(out / "cl10_m08_change.csv", index=False)
        pd.DataFrame([c for c in chg if c["mechanism"] == "M10"]).to_csv(out / "cl10_m10_change.csv", index=False)
    except Exception:
        pass

    # TabM diagnostics: negative harm
    tabm = df[df["model"] == "tabm"]
    tdiag = tabm.groupby("mechanism")["paired_harm"].mean().reset_index()
    tdiag["negative"] = tdiag["paired_harm"] < 0
    tdiag.to_csv(out / "cl10_tabm_diagnostics.csv", index=False)

    mean_rho = float(rcdf["spearman"].mean())
    min_rho = float(rcdf["spearman"].min())
    return {"n_profiles": len(prof), "mean_pairwise_spearman": mean_rho,
            "min_pairwise_spearman": min_rho, "kendall_w": float(W),
            "quadrant_all_agree": int(agree.sum()), "quadrant_total": len(agree),
            "tabm_negative_mechs": tdiag[tdiag.negative]["mechanism"].tolist(),
            "lomo": lomo}


def main():
    df = load()
    res = {"cl2": cl2(df), "cl3": cl3(df), "cl4": cl4(df), "cl10": cl10(df)}
    (SP5 / "sp5_claim_results.json").write_text(json.dumps(res, indent=2, default=str))
    print("=== CL2 ==="); print("structured AUPRC:", round(res["cl2"]["structured_mean"],4),
          res["cl2"]["structured_ci"], "| range", [round(x,3) for x in res["cl2"]["structured_range"]])
    print("simple AUPRC:", round(res["cl2"]["simple_mean"],4), "| diff", round(res["cl2"]["diff"],4), res["cl2"]["diff_ci"])
    print("=== CL3 ==="); print("global pearson:", round(res["cl3"]["global"]["pearson"],3),
          "CI", [round(x,3) for x in res["cl3"]["global"]["pearson_ci"]], "spearman", round(res["cl3"]["global"]["spearman"],3))
    print("within simple pearson:", res["cl3"]["within"]["simple"]["pearson"],
          "| structured:", res["cl3"]["within"]["structured"]["pearson"])
    print("regression:", res["cl3"]["regression"])
    print("=== CL4 ==="); 
    for r in res["cl4"]["summary"]: print(f"  {r['model']}: mean {r['mean']:+.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")
    print("lr_mean:", round(res["cl4"]["lr_mean"],4))
    print("=== CL10 ==="); print("mean pairwise spearman:", round(res["cl10"]["mean_pairwise_spearman"],3),
          "min", round(res["cl10"]["min_pairwise_spearman"],3), "Kendall W", round(res["cl10"]["kendall_w"],3))
    print("quadrant agree:", res["cl10"]["quadrant_all_agree"], "/", res["cl10"]["quadrant_total"])
    print("TabM negative mechs:", res["cl10"]["tabm_negative_mechs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
