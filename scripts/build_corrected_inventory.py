#!/usr/bin/env python3
"""Build a read-only inventory of evidence that predates corrected_v2.

The command intentionally excludes corrected_v2 itself so that rerunning it does
not change the legacy evidence boundary.  It records hashes and table coverage;
it never rewrites an existing legacy artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "corrected_v2" / "inventory"
EVIDENCE_DIRS = ("analysis", "configs", "results", "paper", "reports", "src", "benchmark_v2", "experiments")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_legacy_files():
    for directory in EVIDENCE_DIRS:
        base = ROOT / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if (ROOT / "results" / "corrected_v2") in path.parents:
                continue
            yield path


def table_summary(path: Path) -> dict:
    row = {"rows": None, "columns": [], "coverage": {}, "read_error": ""}
    try:
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            return row
        row["rows"] = int(len(frame))
        row["columns"] = list(frame.columns)
        aliases = {
            "dataset": ("dataset", "dataset_id", "ds", "task"),
            "mechanism": ("mechanism", "mechanism_id", "mech"),
            "strength": ("strength", "strength_id", "str"),
            "model": ("model", "model_id"),
            "seed": ("seed",),
            "method": ("method", "strategy"),
        }
        for name, candidates in aliases.items():
            column = next((candidate for candidate in candidates if candidate in frame), None)
            if column is not None:
                values = frame[column].dropna().astype(str).unique().tolist()
                row["coverage"][name] = {"n": len(values), "values": sorted(values)[:100]}
    except Exception as exc:  # Inventory must retain unreadable artifacts too.
        row["read_error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    entries = []
    table_rows = []
    for path in iter_legacy_files():
        relative = path.relative_to(ROOT).as_posix()
        entry = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        entries.append(entry)
        if path.suffix in {".csv", ".parquet"}:
            table_rows.append({"path": relative, **table_summary(path)})

    manifest = {
        "schema_version": 1,
        "boundary": "all listed files predate corrected_v2",
        "root": str(ROOT),
        "file_count": len(entries),
        "files": entries,
    }
    (OUTPUT / "legacy_sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUTPUT / "legacy_table_inventory.json").write_text(
        json.dumps(table_rows, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# corrected_v2 legacy evidence inventory",
        "",
        f"- Files hashed: {len(entries)}",
        f"- Tables inspected: {len(table_rows)}",
        "- Boundary: all listed files existed before corrected_v2 and must not be overwritten.",
        "",
        "| table | rows | dataset | mechanism | strength | model | seed | read error |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in table_rows:
        coverage = row["coverage"]
        count = lambda key: coverage.get(key, {}).get("n", "")
        lines.append(
            f"| `{row['path']}` | {row['rows'] if row['rows'] is not None else ''} | "
            f"{count('dataset')} | {count('mechanism')} | {count('strength')} | "
            f"{count('model')} | {count('seed')} | {row['read_error']} |"
        )
    (ROOT / "reports" / "corrected_v2_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
