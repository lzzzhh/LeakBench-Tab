#!/usr/bin/env python3
"""Project private natural-case lineage into a deterministic public evidence chain.

The projection never reads or redistributes raw natural datasets.  It replaces
private source locations with repository-relative acquisition placeholders,
proves that all non-path scientific fields are unchanged, and recomputes the
public statistics file so its input hashes bind the public CSVs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VERSION = "natural_public_provenance_v1"
PRIVATE_ROOT = ROOT / "results/corrected_v2"
DEFAULT_OUTPUT = PRIVATE_ROOT / "public_natural"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def public_source_path(task: str, private_path: str) -> str:
    name = Path(private_path).name
    candidate = PurePosixPath("external_sources") / task / name
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Public source placeholder escaped repository scope: {candidate}")
    return str(candidate)


def _normalize_lineage(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    payload["source_path"] = "<SOURCE_PATH>"
    return payload


def _scientific_task_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in frame.sort_values("task").to_dict(orient="records"):
        row = {}
        for key, value in raw.items():
            if key == "source":
                continue
            if key == "lineage":
                row[key] = _normalize_lineage(str(value))
            elif pd.isna(value):
                row[key] = None
            elif isinstance(value, np.generic):
                row[key] = value.item()
            else:
                row[key] = value
        rows.append(row)
    return rows


def _normalized_freeze(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    normalized.pop("public_projection", None)
    normalized["output"] = "<NATURAL_CELLS>"
    normalized["task_summary"] = "<NATURAL_TASK_SUMMARY>"
    for task, entry in normalized["source_files"].items():
        entry["path"] = f"<SOURCE_PATH:{task}>"
    return normalized


def recompute_statistics(cells_path: Path, tasks_path: Path) -> dict[str, Any]:
    cells = pd.read_csv(cells_path)
    tasks = pd.read_csv(tasks_path)
    expected_tasks = {
        "BankMarketing", "LendingClub", "BTSFlights", "ChicagoFood", "NYC311",
    }
    if set(cells["task"]) != expected_tasks or set(tasks["task"]) != expected_tasks:
        raise ValueError("Public natural task set changed")
    if (
        len(cells) != 60
        or not (cells["status"].astype(str) == "SUCCESS").all()
        or cells.duplicated(["task", "model", "seed"]).any()
    ):
        raise ValueError("Public natural matrix is incomplete")
    task_effects = cells.groupby("task")["paired_harm"].mean()
    model_effects = cells.groupby("model")["paired_harm"].mean()
    values = task_effects.to_numpy(dtype=float)
    rng = np.random.RandomState(20260713)
    bootstrap = np.empty(20_000, dtype=float)
    for repetition in range(len(bootstrap)):
        bootstrap[repetition] = rng.choice(
            values, size=len(values), replace=True
        ).mean()
    signs = np.array(
        np.meshgrid(*[[-1.0, 1.0]] * len(values))
    ).T.reshape(-1, len(values))
    observed = abs(values.mean())
    return {
        "schema_version": 1,
        "interpretation": "fixed real-data case studies; not a population-level dataset sample",
        "cells": len(cells),
        "tasks": len(tasks),
        "models": sorted(cells["model"].astype(str).unique()),
        "seeds": sorted(int(seed) for seed in cells["seed"].unique()),
        "all_task_effects_positive": bool((task_effects > 0).all()),
        "mean_paired_harm": float(values.mean()),
        "task_bootstrap_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "exact_two_sided_sign_flip_p": float(
            np.mean(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15)
        ),
        "task_effects": {str(key): float(value) for key, value in task_effects.items()},
        "model_effects": {str(key): float(value) for key, value in model_effects.items()},
        "diagnostic_normalized_ap": {
            str(row.task): float(row.diagnostic_normalized_ap)
            for row in tasks.itertuples()
        },
        "cells_sha256": sha256(cells_path),
        "task_summary_sha256": sha256(tasks_path),
        "public_projection_version": VERSION,
    }


def _normalized_statistics(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    normalized.pop("public_projection_version", None)
    normalized["cells_sha256"] = "<NATURAL_CELLS_SHA256>"
    normalized["task_summary_sha256"] = "<NATURAL_TASK_SUMMARY_SHA256>"
    return normalized


def build_projection(output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    private_paths = {
        "freeze": PRIVATE_ROOT / "natural_protocol_v2_freeze.json",
        "cells": PRIVATE_ROOT / "natural_cells.csv",
        "tasks": PRIVATE_ROOT / "natural_task_summary.csv",
        "statistics": PRIVATE_ROOT / "natural_statistics.json",
    }
    missing = [relative(path) for path in private_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Private natural provenance is incomplete: {missing}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    public_paths = {
        "freeze": output_dir / "natural_protocol_v2_freeze.json",
        "cells": output_dir / "natural_cells.csv",
        "tasks": output_dir / "natural_task_summary.csv",
        "statistics": output_dir / "natural_statistics.json",
    }

    private_freeze = json.loads(private_paths["freeze"].read_text(encoding="utf-8"))
    if private_freeze.get("amendment_version") != "natural_trainfit_categories_v2":
        raise ValueError("Unexpected private natural protocol version")
    source_mapping = {
        str(entry["path"]): public_source_path(task, str(entry["path"]))
        for task, entry in private_freeze["source_files"].items()
    }

    # Cells contain no paths; keep the public bytes exactly identical.
    shutil.copyfile(private_paths["cells"], public_paths["cells"])
    if private_paths["cells"].read_bytes() != public_paths["cells"].read_bytes():
        raise RuntimeError("Natural cells changed during public projection")

    # Path-only substitutions preserve every other byte in the task summary.
    private_task_bytes = private_paths["tasks"].read_bytes()
    public_task_bytes = private_task_bytes
    replacement_counts: dict[str, int] = {}
    for private_source, public_source in sorted(source_mapping.items()):
        count = public_task_bytes.count(private_source.encode("utf-8"))
        if count != 2:
            raise ValueError(
                f"Expected source and lineage path for {private_source}, observed {count}"
            )
        public_task_bytes = public_task_bytes.replace(
            private_source.encode("utf-8"), public_source.encode("utf-8")
        )
        replacement_counts[private_source] = count
    public_paths["tasks"].write_bytes(public_task_bytes)
    private_tasks = pd.read_csv(private_paths["tasks"])
    public_tasks = pd.read_csv(public_paths["tasks"])
    private_science = _scientific_task_rows(private_tasks)
    public_science = _scientific_task_rows(public_tasks)
    if private_science != public_science:
        raise ValueError("Natural task scientific fields changed during path projection")
    if any(Path(value).is_absolute() for value in public_tasks["source"].astype(str)):
        raise ValueError("Public natural task summary still contains an absolute source path")

    public_freeze = copy.deepcopy(private_freeze)
    public_freeze["output"] = relative(public_paths["cells"])
    public_freeze["task_summary"] = relative(public_paths["tasks"])
    for task, entry in public_freeze["source_files"].items():
        entry["path"] = source_mapping[str(private_freeze["source_files"][task]["path"])]
    public_freeze["public_projection"] = {
        "version": VERSION,
        "generator_path": relative(Path(__file__)),
        "generator_sha256": sha256(Path(__file__)),
        "private_freeze_sha256": sha256(private_paths["freeze"]),
        "raw_source_files_included": False,
        "source_path_policy": "repo_relative_external_sources_placeholders",
        "scientific_fields_unchanged": True,
    }
    if _normalized_freeze(private_freeze) != _normalized_freeze(public_freeze):
        raise ValueError("Natural freeze scientific fields changed during public projection")
    public_paths["freeze"].write_text(
        json.dumps(public_freeze, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    public_statistics = recompute_statistics(public_paths["cells"], public_paths["tasks"])
    private_statistics = json.loads(
        private_paths["statistics"].read_text(encoding="utf-8")
    )
    if _normalized_statistics(private_statistics) != _normalized_statistics(public_statistics):
        raise ValueError("Recomputed public natural statistics differ scientifically")
    public_paths["statistics"].write_text(
        json.dumps(public_statistics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    invariants = {
        "natural_cells": {
            "mode": "byte_identical",
            "passed": True,
        },
        "natural_task_summary": {
            "mode": "byte_identical_except_typed_source_paths",
            "changed_fields": ["source", "lineage.source_path"],
            "private_scientific_sha256": canonical_sha256(private_science),
            "public_scientific_sha256": canonical_sha256(public_science),
            "replacement_count": sum(replacement_counts.values()),
            "passed": True,
        },
        "natural_protocol_freeze": {
            "mode": "semantic_equal_except_paths_and_public_projection",
            "private_scientific_sha256": canonical_sha256(
                _normalized_freeze(private_freeze)
            ),
            "public_scientific_sha256": canonical_sha256(
                _normalized_freeze(public_freeze)
            ),
            "passed": True,
        },
        "natural_statistics": {
            "mode": "fixed_seed_recomputed_equal_except_public_input_hashes",
            "bootstrap_seed": 20260713,
            "bootstrap_repetitions": 20000,
            "sign_flip_enumeration": 32,
            "private_scientific_sha256": canonical_sha256(
                _normalized_statistics(private_statistics)
            ),
            "public_scientific_sha256": canonical_sha256(
                _normalized_statistics(public_statistics)
            ),
            "passed": True,
        },
    }
    manifest = {
        "schema_version": 1,
        "status": "PUBLIC_NATURAL_PROVENANCE_PROJECTED",
        "projection_version": VERSION,
        "generator": {
            "path": relative(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "raw_natural_data_included": False,
        "private_provenance": {
            "distribution": "EXCLUDED_FROM_PUBLIC_ARTIFACT",
            "artifacts": {
                logical: {
                    "artifact_type": "private_natural_provenance",
                    "path": relative(path),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for logical, path in private_paths.items()
            },
        },
        "private_to_public": [
            {
                "logical_name": logical,
                "artifact_type": (
                    "csv" if public_paths[logical].suffix == ".csv" else "json"
                ),
                "private_sha256": sha256(private_paths[logical]),
                "public_sha256": sha256(public_paths[logical]),
                "public_path": relative(public_paths[logical]),
                "private_distribution": "EXCLUDED_FROM_PUBLIC_ARTIFACT",
            }
            for logical in ("freeze", "cells", "tasks", "statistics")
        ],
        "public_outputs": {
            logical: {
                "path": relative(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for logical, path in public_paths.items()
        },
        "scientific_invariants": invariants,
        "all_scientific_invariants_passed": all(
            item["passed"] for item in invariants.values()
        ),
    }
    manifest_path = output_dir / "public_natural_provenance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest_path, "outputs": public_paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=relative(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    try:
        output.resolve().relative_to((ROOT / "results/corrected_v2").resolve())
    except ValueError as error:
        raise ValueError("Public natural output must remain under corrected_v2") from error
    result = build_projection(output, overwrite=args.overwrite)
    print(json.dumps({
        "status": "PUBLIC_NATURAL_PROVENANCE_PROJECTED",
        "manifest": relative(result["manifest"]),
        "manifest_sha256": sha256(result["manifest"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
