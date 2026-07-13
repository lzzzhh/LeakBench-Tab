#!/usr/bin/env python3
"""Freeze the independent M10 strict-view amendment before confirmation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.leakbench.run_m10_amendment import (  # noqa: E402
    AMENDMENT_VERSION,
    FULL_POLICY,
    STRICT_POLICY,
    derive_strict_contract,
    file_sha256,
    load_amendment_config,
    load_bundle_contract,
    load_verified_task,
)


OUTPUT = ROOT / "results/corrected_v2/m10_amendment_protocol_freeze.json"
CPU_CONFIRMATORY_OUTPUT = (
    "results/corrected_v2/m10_amendment_confirmatory/cpu_cells.csv"
)
TABM_CONFIRMATORY_OUTPUT = (
    "results/corrected_v2/m10_amendment_confirmatory/tabm_cells.csv"
)


def _entry(relative):
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for relative in (CPU_CONFIRMATORY_OUTPUT, TABM_CONFIRMATORY_OUTPUT):
        if (ROOT / relative).exists():
            raise RuntimeError(
                f"Confirmatory output already exists; amendment was not frozen first: {relative}"
            )

    config_relative = "configs/paper/m10_amendment_v1.yaml"
    config_path = ROOT / config_relative
    _, amendment, _, _, base_config_hash = load_amendment_config(config_path)
    full_manifest_path = ROOT / amendment["confirmatory_task_manifest_path"]
    full_manifest, full_manifest_hash, full_summary_path, _ = load_bundle_contract(
        full_manifest_path, base_config_hash
    )
    if full_manifest_hash != amendment["confirmatory_task_manifest_sha256"]:
        raise RuntimeError("Confirmatory task manifest differs from amendment binding")
    if file_sha256(full_summary_path) != amendment["confirmatory_bundle_summary_sha256"]:
        raise RuntimeError("Confirmatory bundle summary differs from amendment binding")
    full_m10 = full_manifest.loc[full_manifest["mechanism"].astype(str) == "M10"]
    expected_tasks = int(amendment["expected_confirmatory_tasks"])
    if len(full_m10) != expected_tasks:
        raise RuntimeError(f"Expected {expected_tasks} confirmatory M10 tasks, got {len(full_m10)}")

    verified_tasks = 0
    for _, row in full_m10.iterrows():
        task, _ = load_verified_task(row, ROOT)
        derive_strict_contract(task, row, amendment)
        verified_tasks += 1
    if verified_tasks != expected_tasks:
        raise RuntimeError("Not all confirmatory M10 tasks cleared the strict-view contract")

    pilot_manifest_path = ROOT / amendment["pilot_task_manifest_path"]
    pilot_tasks, pilot_manifest_hash, pilot_summary_path, _ = load_bundle_contract(
        pilot_manifest_path, base_config_hash
    )
    if len(pilot_tasks) != int(amendment["expected_pilot_tasks"]):
        raise RuntimeError("Standalone M10 pilot bundle is incomplete")
    for _, row in pilot_tasks.iterrows():
        task, _ = load_verified_task(row, ROOT)
        derive_strict_contract(task, row, amendment)

    pilot_output_path = ROOT / amendment["pilot_cpu_output_path"]
    pilot_output_manifest_path = pilot_output_path.with_name(
        f"{pilot_output_path.stem}_manifest.json"
    )
    pilot_output_manifest = json.loads(
        pilot_output_manifest_path.read_text(encoding="utf-8")
    )
    expected_pilot_cells = int(amendment["expected_pilot_cpu_cells"])
    if (
        pilot_output_manifest.get("amendment_version") != AMENDMENT_VERSION
        or pilot_output_manifest.get("strict_policy") != STRICT_POLICY
        or int(pilot_output_manifest.get("requested_cells", -1)) != expected_pilot_cells
        or int(pilot_output_manifest.get("success_cells", -1)) != expected_pilot_cells
        or int(pilot_output_manifest.get("failure_cells", -1)) != 0
        or int(pilot_output_manifest.get("integrity_verified_cells", -1))
        != expected_pilot_cells
    ):
        raise RuntimeError("M10 CPU pilot did not clear the pre-freeze behavior gate")
    if pilot_output_manifest.get("result_sha256") != file_sha256(pilot_output_path):
        raise RuntimeError("M10 CPU pilot result SHA256 differs from its manifest")

    frozen_paths = [
        "configs/paper/corrected_v2.yaml",
        config_relative,
        "src/leakbench/models/core_models.py",
        "src/leakbench/models/official_tabm.py",
        "experiments/leakbench/run_m10_amendment.py",
        "scripts/build_m10_amendment_pilot_bundle.py",
        "scripts/freeze_m10_amendment_protocol.py",
        str(full_manifest_path.relative_to(ROOT)),
        str(full_summary_path.relative_to(ROOT)),
        str(pilot_manifest_path.relative_to(ROOT)),
        str(pilot_summary_path.relative_to(ROOT)),
        str(pilot_output_path.relative_to(ROOT)),
        str(pilot_output_manifest_path.relative_to(ROOT)),
    ]
    bundle_paths = sorted(set(full_manifest["bundle_path"].astype(str)))
    pilot_bundle_paths = sorted(set(pilot_tasks["bundle_path"].astype(str)))
    frozen_paths.extend(bundle_paths)
    frozen_paths.extend(pilot_bundle_paths)
    if len(bundle_paths) != 20:
        raise RuntimeError(f"Expected 20 confirmatory bundles, got {len(bundle_paths)}")

    frozen_files = {}
    for relative in frozen_paths:
        if relative in frozen_files:
            continue
        frozen_files[relative] = _entry(relative)
    for bundle_relative in bundle_paths:
        declared = set(
            full_manifest.loc[
                full_manifest["bundle_path"].astype(str) == bundle_relative,
                "bundle_sha256",
            ].astype(str)
        )
        if declared != {frozen_files[bundle_relative]["sha256"]}:
            raise RuntimeError(f"Bundle manifest/hash conflict: {bundle_relative}")

    expected_cpu = int(amendment["expected_confirmatory_cpu_cells"])
    expected_tabm = int(amendment["expected_confirmatory_tabm_cells"])
    expected_replacement = int(amendment["expected_replacement_cells"])
    if expected_cpu + expected_tabm != expected_replacement:
        raise RuntimeError("M10 replacement-cell arithmetic is inconsistent")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_M10_AMENDMENT_CONFIRMATORY_RUN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_version": AMENDMENT_VERSION,
        "reason": (
            "M10 strict comparator must retain the injected legitimate duplicate and "
            "remove only the mask-identified contamination feature"
        ),
        "strict_policy": STRICT_POLICY,
        "full_policy": FULL_POLICY,
        "expected_replacement_cells": expected_replacement,
        "expected_cpu_cells": expected_cpu,
        "expected_tabm_cells": expected_tabm,
        "expected_confirmatory_tasks": expected_tasks,
        "verified_confirmatory_tasks": verified_tasks,
        "expected_pilot_cpu_cells": expected_pilot_cells,
        "pilot_cpu_success_cells": pilot_output_manifest["success_cells"],
        "pilot_cpu_integrity_verified_cells": pilot_output_manifest[
            "integrity_verified_cells"
        ],
        "outputs": {
            "cpu": CPU_CONFIRMATORY_OUTPUT,
            "tabm": TABM_CONFIRMATORY_OUTPUT,
        },
        "output_manifests": {
            "cpu": CPU_CONFIRMATORY_OUTPUT.replace(".csv", "_manifest.json"),
            "tabm": TABM_CONFIRMATORY_OUTPUT.replace(".csv", "_manifest.json"),
        },
        "source_task_manifest": {
            "path": str(full_manifest_path.relative_to(ROOT)),
            "sha256": full_manifest_hash,
        },
        "source_bundle_summary": {
            "path": str(full_summary_path.relative_to(ROOT)),
            "sha256": file_sha256(full_summary_path),
        },
        "pilot_task_manifest": {
            "path": str(pilot_manifest_path.relative_to(ROOT)),
            "sha256": pilot_manifest_hash,
        },
        "base_config_sha256": base_config_hash,
        "amendment_config_sha256": file_sha256(config_path),
        "frozen_files": frozen_files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "freeze": str(OUTPUT.relative_to(ROOT)),
        "status": payload["status"],
        "frozen_files": len(frozen_files),
        "verified_confirmatory_tasks": verified_tasks,
        "expected_replacement_cells": expected_replacement,
    }, indent=2))


if __name__ == "__main__":
    main()
