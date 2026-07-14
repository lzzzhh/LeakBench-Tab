#!/usr/bin/env python3
"""build_sp6_bundle_manifest.py — resolve all 11 mechanisms to frozen SP5 bundles.

Replacement precedence identical to SP5:
  M04/M05/M08 -> structured_prior_replacement_v1/task_bundles (corrected)
  M10         -> corrected_v2/task_bundles (base bundle) + amendment strict view
  others      -> corrected_v2/task_bundles (base injector, correct)

Strict view for every mechanism = X[:, ~leakage_mask]; full view = X.
Emits artifacts/sp6/sp6_bundle_manifest.csv with per-cell bundle resolution +
frozen hashes. No injection, no data regeneration.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/sp6/sp6_bundle_manifest.csv"
CORE = ROOT / "results/corrected_v2/task_bundles/task_manifest.csv"
SP4 = ROOT / "results/structured_prior_replacement_v1/task_bundles/task_manifest.csv"

SP4_MECHS = {"M04", "M05", "M08"}


def main():
    core = pd.read_csv(CORE)
    sp4 = pd.read_csv(SP4)
    keep_cols = ["dataset_index", "mechanism", "strength", "seed", "bundle_key",
                 "task_hash", "split_hash", "n_original", "n_injected", "n_leak",
                 "bundle_path", "bundle_sha256"]
    # base bundles for the 7 non-amended + M10 (M10 uses base bundle, amendment strict derived in-runner)
    base = core[~core["mechanism"].isin(SP4_MECHS)][keep_cols].copy()
    base["bundle_source"] = base["mechanism"].map(
        lambda m: "m10_amendment" if m == "M10" else "corrected_v2_base")
    base["strict_policy"] = "X[:, ~leakage_mask]"
    # corrected M04/M05/M08
    rep = sp4[sp4["mechanism"].isin(SP4_MECHS)][keep_cols].copy()
    rep["bundle_source"] = "structured_prior_v1"
    rep["strict_policy"] = "X[:, ~leakage_mask]"

    man = pd.concat([base, rep], ignore_index=True)
    assert sorted(man["mechanism"].unique()) == [f"M{i:02d}" for i in range(1, 12)]
    # expected 11 mech x 20 ds x 5 str x 5 seed = 5500 task cells (model-independent)
    assert len(man) == 5500, f"expected 5500 got {len(man)}"
    assert man.duplicated(["dataset_index", "mechanism", "strength", "seed"]).sum() == 0
    man = man.sort_values(["dataset_index", "mechanism", "strength", "seed"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT, index=False)
    print(f"wrote {len(man)} task-cell bundle resolutions -> {OUT.relative_to(ROOT)}")
    print("by source:", man["bundle_source"].value_counts().to_dict())
    print("sha256:", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
