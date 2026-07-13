"""plan.py — LeakBench-Tab manifest generator."""
import hashlib, pandas as pd
from pathlib import Path


def generate_manifest(datasets, mechanisms, strengths, models, seeds, output_path):
    rows = []
    for ds in datasets:
        for mech in mechanisms:
            for strength in strengths:
                for model in models:
                    for seed in seeds:
                        key = f"{ds}_{mech}_{strength}_{model}_{seed}"
                        run_id = hashlib.md5(key.encode()).hexdigest()[:12]
                        rows.append({"run_id": run_id, "dataset_id": ds, "mechanism_id": mech,
                                     "strength_id": strength, "model_id": model, "seed": seed,
                                     "status": "PENDING", "device": "auto", "attempt": 0,
                                     "result_path": "", "error_type": ""})
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def build_pilot_manifest(output_path="results/manifests/pilot_manifest.parquet"):
    return generate_manifest([f"pilot_ds_{i}" for i in range(5)],
        ["M01","M05","M08"], ["S1_very_weak","S3_medium","S5_extreme"],
        ["logistic_regression","random_forest","catboost","mlp","tabm"], [42], output_path)


def build_full_manifest(output_path="results/manifests/full_matrix_manifest.parquet"):
    return generate_manifest([f"base_{i}" for i in range(20)],
        [f"M{m:02d}" for m in range(1,12)],
        ["S1_very_weak","S2_weak","S3_medium","S4_strong","S5_extreme"],
        ["logistic_regression","random_forest","catboost","mlp","tabm"],
        [13,42,2026,3407,7777], output_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--wave", choices=["pilot","full"], default="pilot")
    args = p.parse_args()
    if args.wave == "pilot": build_pilot_manifest()
    else: build_full_manifest()
