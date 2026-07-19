#!/usr/bin/env python3
"""Build the complete CDXR EDBT paper bundle, including sources and evidence."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper/edbt_eab"
PACKAGE_NAME = "CDXR-EDBT-2027-complete"
DESTINATION = ROOT / "release" / PACKAGE_NAME
ARCHIVE = DESTINATION.with_suffix(".zip")

EXCLUDED_PARTS = {"__pycache__", ".DS_Store"}
EVIDENCE_FILES = {
    "results/edbt_eab_revision/analysis_summary.json": "evidence/analysis_summary.json",
    "results/edbt_eab_revision/a1_mechanism_level.csv": "evidence/a1_mechanism_level.csv",
    "results/edbt_eab_revision/a2_gap_stratification.csv": "evidence/a2_gap_stratification.csv",
    "results/edbt_eab_revision/a3_archetype.csv": "evidence/a3_archetype.csv",
    "results/edbt_eab_revision/natural_governance_summary.csv": "evidence/natural_governance_summary.csv",
    "results/edbt_eab_revision/semantic_budget_summary.csv": "evidence/semantic_budget_summary.csv",
    "results/edbt_eab_revision/remaining_governance_summary.json": "evidence/remaining_governance_summary.json",
    "results/edbt_eab_revision/claim_state.json": "evidence/claim_state.json",
    "results/edbt_eab_revision/manifest.json": "evidence/revision_manifest.json",
    "results/edbt_eab_revision/failure_anatomy/failure_anatomy_summary.json": "evidence/failure_anatomy_summary.json",
    "results/edbt_eab_revision/failure_anatomy/failure_anatomy_manifest.json": "evidence/failure_anatomy_manifest.json",
    "results/edbt_eab_revision/failure_anatomy/sparse_failure_anatomy.csv": "evidence/sparse_failure_anatomy.csv",
    "results/edbt_eab_revision/failure_anatomy/nyc311_selection_diagnostic.csv": "evidence/nyc311_selection_diagnostic.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paper_files() -> list[Path]:
    files = []
    for path in PAPER.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(PAPER)
        if relative.parts[:1] == ("output",) and relative.as_posix() != "output/official/main.pdf":
            continue
        files.append(path)
    return sorted(files)


def copy_bound(source: Path, destination: Path, records: list[dict[str, object]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    if sha256(destination) != source_hash:
        raise RuntimeError(f"byte-exact copy failed: {source}")
    records.append({
        "path": destination.relative_to(DESTINATION).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": source_hash,
    })


def build() -> Path:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    records: list[dict[str, object]] = []
    for source in paper_files():
        copy_bound(source, DESTINATION / source.relative_to(PAPER), records)
    for source_name, destination_name in EVIDENCE_FILES.items():
        source = ROOT / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        copy_bound(source, DESTINATION / destination_name, records)

    readme = DESTINATION / "COMPLETE_PACKAGE_README.md"
    readme.write_text(
        """# CDXR EDBT 2027 complete paper package

This bundle contains the complete current paper workspace: LaTeX sources,
bibliography, EDBT template, fonts and license, generated tables/macros,
figures in PDF/PNG/SVG/TIFF formats, paper-facing CSV assets, generation and
packaging scripts, the compiled PDF, and compact revision evidence tables.

The paper presents CDXR, a contract-grounded evaluation architecture that
separates construction validity, blind detectability, learner-conditional
exploitability, and matched-cost repair response. LeakBench-Tab is the
controlled experimental instantiation rather than the paper's sole claim.

Compile from this directory with:

```bash
tectonic -X compile main.tex --outdir output
```

The manuscript still contains placeholder author metadata. Replace the author,
affiliation, city, country, email, and ORCID fields before submission.

The included paper-facing assets are sufficient to compile the paper. Full
regeneration of evidence-derived assets requires the complete repository and
is intentionally not claimed to be standalone in this paper-only bundle.
""",
        encoding="utf-8",
    )
    records.append({
        "path": readme.relative_to(DESTINATION).as_posix(),
        "bytes": readme.stat().st_size,
        "sha256": sha256(readme),
    })
    manifest = {
        "schema_version": 1,
        "status": "CDXR_EDBT_COMPLETE_PACKAGE_READY",
        "entrypoint": "main.tex",
        "compiled_pdf": "output/official/main.pdf",
        "file_count": len(records),
        "files": sorted(records, key=lambda item: str(item["path"])),
    }
    manifest_path = DESTINATION / "COMPLETE_PACKAGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DESTINATION.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(DESTINATION.parent))
    return ARCHIVE


if __name__ == "__main__":
    archive = build()
    print(json.dumps({
        "status": "CDXR_EDBT_COMPLETE_PACKAGE_BUILT",
        "zip": str(archive.relative_to(ROOT)),
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
    }, sort_keys=True))
