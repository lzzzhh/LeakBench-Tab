#!/usr/bin/env python3
"""Build a fail-closed, EDBT-only draft artifact from an explicit allowlist."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "release/leakbench_artifact_edbt_draft"
ARCHIVE = DESTINATION.with_suffix(".zip")

PAPER_FILES = (
    "paper/edbt_eab/main.tex",
    "paper/edbt_eab/references.bib",
    "paper/edbt_eab/README.md",
    "paper/edbt_eab/ASSET_GOVERNANCE.md",
    "paper/edbt_eab/acmart.cls",
    "paper/edbt_eab/ACM-Reference-Format.bst",
    "paper/edbt_eab/edbt-macros.tex",
    "paper/edbt_eab/libertinusmath-regular.otf",
    "paper/edbt_eab/OFL-Libertinus.txt",
    "paper/edbt_eab/output/official/main.pdf",
    "paper/edbt_eab/generated/paper_artifact_manifest.json",
    "paper/edbt_eab/generated/result_macros.tex",
    "paper/edbt_eab/generated/table_governance.tex",
    "paper/edbt_eab/generated/table_measurement.tex",
    "paper/edbt_eab/generated/table_natural.tex",
    "paper/edbt_eab/figures/generated/cdx_profiles.pdf",
    "paper/edbt_eab/figures/generated/governance_tradeoff.pdf",
    "paper/edbt_eab/source_data/build_paper_assets.py",
    "paper/edbt_eab/source_data/generate_paper_artifacts.py",
    "paper/edbt_eab/source_data/generated/main_results.csv",
    "paper/edbt_eab/source_data/generated/governance_results.csv",
    "paper/edbt_eab/source_data/generated/natural_cases.csv",
    "paper/edbt_eab/source_data/generated/paper_asset_manifest.json",
)

EVIDENCE_FILES = (
    "results/corrected_v2/canonical_cells.csv",
    "results/corrected_v2/canonical_manifest.json",
    "results/corrected_v2/paper_claims.json",
    "results/corrected_v2/claim_state.json",
    "results/corrected_v2/statistics/mechanism_summary.csv",
    "results/corrected_v2/statistics/detectability_mechanism_summary.csv",
    "results/corrected_v2/statistics/strength_dose_response.csv",
    "results/corrected_v2/public_natural/natural_cells.csv",
    "results/corrected_v2/public_natural/natural_task_summary.csv",
    "results/corrected_v2/public_natural/natural_statistics.json",
    "results/corrected_v2/public_natural/natural_protocol_v2_freeze.json",
    "results/corrected_v2/public_natural/public_natural_provenance_manifest.json",
    "artifacts/sp8/governance_clean.csv",
    "artifacts/sp8/governance_clean_manifest.json",
    "artifacts/sp8/bootstrap_analysis.json",
    "artifacts/sp8/claims/claim_evidence_matrix_sp8.json",
    "artifacts/sp8/claims/claim_evidence_matrix_sp8.csv",
    "artifacts/sp8/protocol/policy_registry.yaml",
    "scripts/run_sp8_clean.py",
    "scripts/analyze_sp8_governance.py",
    "results/edbt_t0_b_full_b1/merged/merge_manifest.json",
    "results/edbt_t0_b_full_b1/validation_receipt.json",
    "results/edbt_t0_b_full_b1_analysis/analysis_manifest.json",
    "results/edbt_t0_b_full_b1_analysis/analysis_summary.json",
    "results/edbt_t0_b_full_b1_analysis/claim_state.json",
    "results/edbt_t0_b_full_b1_analysis/paper_table_1_policy.csv",
    "results/edbt_t0_b_full_b1_analysis/paper_table_2_contract.csv",
    "results/edbt_t0_b_full_b1_analysis/paper_table_3_archetype.csv",
    "scripts/analyze_full_b1_r10e.py",
)

ALLOWLIST = tuple(PurePosixPath(path) for path in PAPER_FILES + EVIDENCE_FILES)
FORBIDDEN_PARTS = {"pilot", "superseded_snapshots", "_excluded_smoke"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_allowlist() -> None:
    if len(ALLOWLIST) != len(set(ALLOWLIST)):
        raise RuntimeError("EDBT draft allowlist contains duplicates")
    for relative in ALLOWLIST:
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe allowlist path: {relative}")
        if FORBIDDEN_PARTS.intersection(relative.parts):
            raise RuntimeError(f"forbidden historical path in EDBT package: {relative}")
        source = ROOT.joinpath(*relative.parts)
        if not source.is_file():
            raise FileNotFoundError(f"required EDBT draft input is absent: {relative}")


def _validate_governed_assets() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "paper/edbt_eab/source_data/build_paper_assets.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
    claims = ROOT / "results/corrected_v2/paper_claims.json"
    state = ROOT / "results/corrected_v2/claim_state.json"
    if claims.read_bytes() != state.read_bytes():
        raise RuntimeError("paper_claims.json and claim_state.json are not byte-identical")


def _copy_exact(relative: PurePosixPath) -> dict[str, object]:
    source = ROOT.joinpath(*relative.parts)
    destination = DESTINATION.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    if sha256(destination) != source_hash:
        raise RuntimeError(f"byte-exact copy failed: {relative}")
    return {
        "path": relative.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": source_hash,
    }


def _write_readme() -> None:
    text = """# LeakBench-Tab EDBT 2027 draft artifact

Status: `EDBT_DRAFT` -- not a final submission package.

This package deliberately contains only the EDBT manuscript, compact
paper-facing tables, the Full-B1 claim state, and their allowlisted evidence
chain. It excludes pilots, per-shard execution cache, superseded snapshots,
excluded smoke runs, and the non-redistributable Lending Club source file.

The remaining submission blocker is author front matter: replace the pending
author, affiliation, email, and ORCID fields before creating a final package.

From the repository root, verify governed paper inputs with:

```bash
python paper/edbt_eab/source_data/build_paper_assets.py --check
```

Compile from `paper/edbt_eab/` with:

```bash
tectonic -X compile main.tex --outdir output/official
```
"""
    (DESTINATION / "ARTIFACT_README.md").write_text(text, encoding="utf-8")


def build() -> Path:
    _validate_allowlist()
    _validate_governed_assets()
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    records = [_copy_exact(relative) for relative in ALLOWLIST]
    _write_readme()
    readme = DESTINATION / "ARTIFACT_README.md"
    records.append(
        {
            "path": "ARTIFACT_README.md",
            "bytes": readme.stat().st_size,
            "sha256": sha256(readme),
        }
    )
    manifest = {
        "schema_version": 1,
        "status": "EDBT_DRAFT",
        "venue": "EDBT 2027 Experiments, Analysis & Benchmarks",
        "file_count": len(records),
        "blockers": [
            "Replace pending author, affiliation, email, and ORCID fields."
        ],
        "exclusions": {
            "pilots": True,
            "superseded_snapshots": True,
            "excluded_smoke": True,
            "raw_lending_club_source": True,
        },
        "files": sorted(records, key=lambda item: str(item["path"])),
    }
    manifest_path = DESTINATION / "ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DESTINATION.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(DESTINATION.parent))
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = tuple(PurePosixPath(name) for name in archive.namelist())
        if any(FORBIDDEN_PARTS.intersection(name.parts) for name in names):
            raise RuntimeError("forbidden historical material entered EDBT draft ZIP")
    return ARCHIVE


if __name__ == "__main__":
    archive = build()
    print(
        json.dumps(
            {
                "status": "EDBT_DRAFT_BUILT",
                "zip": str(archive.relative_to(ROOT)),
                "sha256": sha256(archive),
            },
            sort_keys=True,
        )
    )
