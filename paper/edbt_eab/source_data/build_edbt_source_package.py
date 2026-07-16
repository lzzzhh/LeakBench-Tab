#!/usr/bin/env python3
"""Build the minimal self-contained EDBT LaTeX submission source package."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper/edbt_eab"
PACKAGE_NAME = "LeakBench-Tab-EDBT-2027-source"
DESTINATION = ROOT / "release" / PACKAGE_NAME
ARCHIVE = DESTINATION.with_suffix(".zip")

SOURCE_FILES = tuple(
    PurePosixPath(path)
    for path in (
        "main.tex",
        "references.bib",
        "acmart.cls",
        "ACM-Reference-Format.bst",
        "edbt-macros.tex",
        "libertinusmath-regular.otf",
        "OFL-Libertinus.txt",
        "generated/result_macros.tex",
        "generated/table_measurement.tex",
        "generated/table_natural.tex",
        "generated/table_governance.tex",
        "figures/generated/cdx_profiles.pdf",
        "figures/generated/governance_budget.pdf",
        "figures/generated/governance_by_category.pdf",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sources() -> None:
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)):
        raise RuntimeError("EDBT source allowlist contains duplicate paths")
    for relative in SOURCE_FILES:
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe EDBT source path: {relative}")
        if not PAPER.joinpath(*relative.parts).is_file():
            raise FileNotFoundError(f"required EDBT source is missing: {relative}")


def _write_readme() -> None:
    readme = """# LeakBench-Tab EDBT 2027 LaTeX source

This directory is a self-contained compilation package. It includes the EDBT
2027 template, bibliography, generated tables/macros, figure PDFs, and the local
Libertinus Math font needed by Tectonic's XeTeX path.

Compile with:

```bash
tectonic -X compile main.tex --outdir output
```

The current source contains placeholder author metadata. Replace the author,
affiliation, email, and ORCID details before submission.
"""
    (DESTINATION / "README.md").write_text(readme, encoding="utf-8")


def build() -> Path:
    _validate_sources()
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    records: list[dict[str, object]] = []
    for relative in SOURCE_FILES:
        source = PAPER.joinpath(*relative.parts)
        destination = DESTINATION.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"byte-exact EDBT source copy failed: {relative}")
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    _write_readme()
    readme = DESTINATION / "README.md"
    records.append(
        {"path": "README.md", "bytes": readme.stat().st_size, "sha256": sha256(readme)}
    )
    manifest = {
        "schema_version": 1,
        "status": "EDBT_SOURCE_READY",
        "entrypoint": "main.tex",
        "file_count": len(records),
        "files": sorted(records, key=lambda item: str(item["path"])),
    }
    (DESTINATION / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DESTINATION.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(DESTINATION.parent))
    return ARCHIVE


if __name__ == "__main__":
    archive = build()
    print(
        json.dumps(
            {
                "status": "EDBT_SOURCE_PACKAGE_BUILT",
                "zip": str(archive.relative_to(ROOT)),
                "sha256": sha256(archive),
            },
            sort_keys=True,
        )
    )
