#!/usr/bin/env python3
"""Build and attest the anonymous AAAI-27 PDFs.

The default mode is intentionally fail closed.  It accepts only final evidence
macros, machine-generated tables and figures whose manifests still match their
files.  It then builds twice in the pinned Docker toolchain, requires byte-for-
byte reproducibility, checks the AAAI page limit, fonts, references, anonymity,
and placeholder leakage, and only then replaces ``output/pdf`` PDFs and writes
``paper_build_manifest.json``.

``--draft-only`` is a structural LaTeX check.  It never writes submission PDFs
or a PASS manifest and may render the explicit RESULTS BLOCKED branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/aaai27"
OUTPUT = PAPER / "output/pdf"
TMP = PAPER / "tmp/pdfs"
MANIFEST = OUTPUT / "paper_build_manifest.json"

IMAGE = "leakbench-aaai27-texlive:20260713"
SOURCE_DATE_EPOCH = 1_783_900_800  # 2026-07-13 00:00:00 UTC
EXPECTED_BASE_DIGEST = (
    "sha256:e36cbf0b9561c4c975858410abe815c6df7a73e5b7b687d36cdd4a1606164cee"
)
EXPECTED_FIGURES = (
    "paper/aaai27/figures/generated/cdx_scatter.pdf",
    "paper/aaai27/figures/generated/mechanism_model_heatmap.pdf",
    "paper/aaai27/figures/generated/strength_diagnostic_robustness.pdf",
)
EXPECTED_TABLES = (
    "paper/aaai27/generated/result_tables.tex",
    "paper/aaai27/generated/table_task_registry.tex",
    "paper/aaai27/generated/table_mechanism_profiles.tex",
    "paper/aaai27/generated/table_mechanism_models.tex",
    "paper/aaai27/generated/table_diagnostic_methods.tex",
    "paper/aaai27/generated/table_strength_response.tex",
    "paper/aaai27/generated/table_natural_cases.tex",
    "paper/aaai27/generated/table_claim_scope.tex",
)
STATIC_PAPER_INPUTS = (
    "paper/aaai27/main.tex",
    "paper/aaai27/supplement.tex",
    "paper/aaai27/ReproducibilityChecklist.tex",
    "paper/aaai27/references.bib",
    "paper/aaai27/aaai2027.sty",
    "paper/aaai27/aaai2027.bst",
    "paper/aaai27/source_data/result_macros_base.tex",
    "paper/aaai27/source_data/generate_result_macros.py",
    "paper/aaai27/source_data/generate_result_tables.py",
    "paper/aaai27/Dockerfile",
    "scripts/build_aaai27_paper.py",
    "scripts/generate_corrected_v2_figures.py",
)

PRIVATE_PATTERNS = {
    "local_user_path": re.compile(r"/(?:Users|home)/[^/\s]+/", re.I),
    "windows_drive_path": re.compile(r"\b[A-Za-z]:[\\/][^\s]+"),
    "private_host": re.compile(
        r"\b(?:LAPTOP-[A-Z0-9]+|10(?:\.\d{1,3}){3}|"
        r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b",
        re.I,
    ),
    "email_address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
}


class BuildError(RuntimeError):
    """Raised when a submission-readiness condition is not satisfied."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise BuildError(f"Required JSON file is missing: {relative(path)}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"Invalid JSON file {relative(path)}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BuildError(f"JSON root must be an object: {relative(path)}")
    return value


def run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command), cwd=cwd, text=True, capture_output=capture, check=False
    )
    if completed.returncode != 0:
        output = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        tail = "\n".join(output.splitlines()[-30:])
        raise BuildError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{tail}"
        )
    return completed


def _verify_hash_map(
    manifest_path: Path,
    values: Any,
    *,
    expected_paths: set[str] | None = None,
) -> list[Path]:
    if not isinstance(values, Mapping) or not values:
        raise BuildError(f"Missing hash map in {relative(manifest_path)}")
    observed = {str(key) for key in values}
    if expected_paths is not None and observed != expected_paths:
        raise BuildError(
            f"Unexpected file set in {relative(manifest_path)}: "
            f"expected {sorted(expected_paths)}, observed {sorted(observed)}"
        )
    paths: list[Path] = []
    for name, expected_hash in sorted(values.items()):
        path = ROOT / str(name)
        if not path.is_file():
            raise BuildError(f"Manifest-bound file is missing: {name}")
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            raise BuildError(
                f"Manifest-bound file changed: {name}; "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        paths.append(path)
    return paths


def require_byte_stable_regeneration(
    command: Sequence[str], paths: Sequence[Path], label: str
) -> None:
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise BuildError(f"Cannot regenerate {label}; files are missing: {missing}")
    before = {relative(path): sha256(path) for path in paths}
    run(command)
    after = {relative(path): sha256(path) for path in paths}
    if before != after:
        changed = sorted(name for name in before if before[name] != after.get(name))
        raise BuildError(
            f"{label} was stale or non-deterministic under fresh regeneration: {changed}"
        )


def verify_final_inputs() -> list[Path]:
    claims = ROOT / "results/corrected_v2/paper_claims.json"
    macros = PAPER / "generated/result_macros.tex"
    figure_manifest_path = PAPER / "figures/generated/figure_manifest.json"
    table_manifest_path = PAPER / "generated/result_tables_manifest.json"

    if not claims.is_file() or not macros.is_file():
        raise BuildError("Final paper claims/macros are absent; submission build is blocked")
    macro_text = macros.read_text(encoding="utf-8")
    if r"\LBResultsReadytrue" not in macro_text:
        raise BuildError("Generated macros do not enable the final-results branch")
    expected_marker = f"% paper_claims.json sha256: {sha256(claims)}"
    if expected_marker not in macro_text:
        raise BuildError("Generated macros are stale relative to paper_claims.json")
    run(
        [
            sys.executable,
            str(PAPER / "source_data/generate_result_macros.py"),
            "--input",
            str(claims),
            "--output",
            str(macros),
            "--check-only",
        ]
    )

    figure_manifest = read_json(figure_manifest_path)
    if (
        figure_manifest.get("schema_version") != 1
        or figure_manifest.get("evidence_tier") != "confirmatory"
        or figure_manifest.get("pilot_inputs_forbidden") is not True
        or figure_manifest.get("generator")
        != "scripts/generate_corrected_v2_figures.py"
        or figure_manifest.get("generator_sha256")
        != sha256(ROOT / "scripts/generate_corrected_v2_figures.py")
    ):
        raise BuildError("Figure manifest is not a final confirmatory manifest")
    figure_paths = _verify_hash_map(
        figure_manifest_path,
        figure_manifest.get("figure_sha256"),
        expected_paths=set(EXPECTED_FIGURES),
    )
    figure_source_paths = _verify_hash_map(
        figure_manifest_path, figure_manifest.get("source_sha256")
    )

    table_manifest = read_json(table_manifest_path)
    if (
        table_manifest.get("schema_version") != 1
        or table_manifest.get("status") != "PASS"
        or table_manifest.get("evidence_tier") != "confirmatory"
        or table_manifest.get("pilot_inputs_forbidden") is not True
        or table_manifest.get("table_count") != 7
        or table_manifest.get("generator")
        != "paper/aaai27/source_data/generate_result_tables.py"
        or table_manifest.get("generator_sha256")
        != sha256(PAPER / "source_data/generate_result_tables.py")
        or table_manifest.get("paper_claims_sha256") != sha256(claims)
    ):
        raise BuildError("Result-table manifest is not final and confirmatory")
    table_paths = _verify_hash_map(
        table_manifest_path,
        table_manifest.get("table_sha256"),
        expected_paths=set(EXPECTED_TABLES),
    )
    table_source_paths = _verify_hash_map(
        table_manifest_path, table_manifest.get("source_sha256")
    )
    run(
        [
            sys.executable,
            str(PAPER / "source_data/generate_result_tables.py"),
            "--check-only",
        ]
    )
    require_byte_stable_regeneration(
        [sys.executable, str(PAPER / "source_data/generate_result_tables.py")],
        [table_manifest_path, *table_paths],
        "result tables",
    )
    require_byte_stable_regeneration(
        [sys.executable, str(ROOT / "scripts/generate_corrected_v2_figures.py")],
        [figure_manifest_path, *figure_paths],
        "paper figures",
    )

    required = [ROOT / name for name in STATIC_PAPER_INPUTS]
    required += [claims, macros, figure_manifest_path, table_manifest_path]
    required += figure_paths + table_paths
    required += figure_source_paths + table_source_paths
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise BuildError(f"Paper input files are missing: {missing}")
    return sorted(set(required))


def image_id(image: str) -> str:
    value = run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"]
    ).stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise BuildError(f"Unexpected Docker image ID: {value!r}")
    return value


def build_image(image: str) -> str:
    run(
        [
            "docker",
            "build",
            "--pull=false",
            "-t",
            image,
            "-f",
            str(PAPER / "Dockerfile"),
            str(PAPER),
        ],
        capture=True,
    )
    return image_id(image)


def build_once(image: str, directory: Path) -> None:
    if directory.parent != TMP:
        raise BuildError(f"Refusing to use a build directory outside {TMP}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    target = f"tmp/pdfs/{directory.name}"
    uid = str(os.getuid())
    gid = str(os.getgid())
    script = f"""
set -euo pipefail
for document in main supplement; do
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error \\
    -file-line-error -recorder -output-directory={target} ${{document}}.tex
  bibtex {target}/${{document}}
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error \\
    -file-line-error -recorder -output-directory={target} ${{document}}.tex
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error \\
    -file-line-error -recorder -output-directory={target} ${{document}}.tex
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error \\
    -file-line-error -recorder -output-directory={target} ${{document}}.tex
done
""".strip()
    run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{uid}:{gid}",
            "-e",
            f"SOURCE_DATE_EPOCH={SOURCE_DATE_EPOCH}",
            "-e",
            "FORCE_SOURCE_DATE=1",
            "-e",
            "TZ=UTC",
            "-v",
            f"{PAPER}:/work",
            "-w",
            "/work",
            image,
            "bash",
            "-lc",
            script,
        ],
        capture=True,
    )


def find_poppler_tool(name: str, explicit_dir: Path | None) -> Path:
    if explicit_dir is not None:
        candidate = explicit_dir / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise BuildError(f"Poppler tool is missing or not executable: {candidate}")
    found = shutil.which(name)
    if found:
        return Path(found)
    candidates = sorted(
        Path.home().glob(
            ".cache/codex-runtimes/*/dependencies/native/poppler/poppler/bin/" + name
        )
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise BuildError(
        f"Could not locate {name}; pass --poppler-bin with the Poppler bin directory"
    )


def parse_pdfinfo(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def parse_fonts(text: str) -> list[dict[str, str]]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or "object ID" not in lines[0]:
        raise BuildError("Could not parse pdffonts output")
    header = lines[0]
    starts = {
        name: header.index(name)
        for name in ("name", "type", "encoding", "emb", "sub", "uni", "object ID")
    }
    fonts: list[dict[str, str]] = []
    for line in lines[2:]:
        fonts.append(
            {
                "name": line[starts["name"] : starts["type"]].strip(),
                "type": line[starts["type"] : starts["encoding"]].strip(),
                "encoding": line[starts["encoding"] : starts["emb"]].strip(),
                "embedded": line[starts["emb"] : starts["sub"]].strip(),
                "subset": line[starts["sub"] : starts["uni"]].strip(),
                "unicode": line[starts["uni"] : starts["object ID"]].strip(),
            }
        )
    if not fonts:
        raise BuildError("PDF contains no discoverable fonts")
    return fonts


def parse_main_content_page(aux_text: str) -> int:
    match = re.search(
        r"\\newlabel\{lb:last-main-content-page\}\{\{[^{}]*\}\{(\d+)\}",
        aux_text,
    )
    if not match:
        raise BuildError("Main-content page label is missing from main.aux")
    return int(match.group(1))


def scan_private_text(text: str, label: str) -> None:
    findings = [name for name, pattern in PRIVATE_PATTERNS.items() if pattern.search(text)]
    local_user = os.environ.get("USER", "").strip()
    if len(local_user) >= 3 and local_user.lower() not in {"root", "runner"}:
        if re.search(rf"\b{re.escape(local_user)}\b", text, re.I):
            findings.append("local_username")
    if findings:
        raise BuildError(f"Private/anonymity-sensitive text in {label}: {findings}")


def validate_log(path: Path, *, allow_overfull: bool = False) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    forbidden = {
        "undefined_reference_or_citation": re.compile(
            r"(?:Citation|Reference).+undefined|There were undefined (?:citations|references)",
            re.I,
        ),
        "rerun_required": re.compile(r"Label\(s\) may have changed", re.I),
        "overfull_box": re.compile(r"Overfull \\[hv]box", re.I),
    }
    findings = [name for name, pattern in forbidden.items() if pattern.search(text)]
    if allow_overfull and "overfull_box" in findings:
        findings.remove("overfull_box")
    if findings:
        raise BuildError(f"LaTeX quality failure in {relative(path)}: {findings}")
    return {
        "undefined_references_or_citations": False,
        "rerun_required": False,
        "overfull_boxes": False,
    }


def validate_pdf(
    pdf: Path,
    *,
    pdfinfo: Path,
    pdffonts: Path,
    pdftotext: Path,
    final: bool,
) -> dict[str, Any]:
    info = parse_pdfinfo(run([str(pdfinfo), str(pdf)]).stdout)
    try:
        pages = int(info["Pages"])
    except (KeyError, ValueError) as exc:
        raise BuildError(f"Could not determine page count for {relative(pdf)}") from exc
    if info.get("Page size") != "612 x 792 pts (letter)":
        raise BuildError(f"PDF is not US letter: {relative(pdf)}")
    if info.get("Encrypted") != "no":
        raise BuildError(f"PDF must not be encrypted: {relative(pdf)}")

    fonts = parse_fonts(run([str(pdffonts), str(pdf)]).stdout)
    not_embedded = [font["name"] for font in fonts if font["embedded"] != "yes"]
    type3 = [font["name"] for font in fonts if font["type"].lower() == "type 3"]
    if not_embedded or type3:
        raise BuildError(
            f"Font policy failure in {relative(pdf)}: "
            f"not_embedded={not_embedded}, type3={type3}"
        )

    extracted = run([str(pdftotext), str(pdf), "-"]).stdout
    scan_private_text(extracted, relative(pdf))
    if "Anonymous submission" not in extracted:
        raise BuildError(f"Anonymous author line is missing from {relative(pdf)}")
    if final:
        blocked_markers = ("INTERNAL DRAFT", "RESULTS BLOCKED")
        present = [marker for marker in blocked_markers if marker in extracted]
        if pdf.name == "main.pdf" and re.search(r"\bPENDING\b", extracted):
            present.append("PENDING")
        if present:
            raise BuildError(f"Blocked-result markers remain in {relative(pdf)}: {present}")

    return {
        "path": relative(pdf),
        "sha256": sha256(pdf),
        "bytes": pdf.stat().st_size,
        "pages": pages,
        "letter_page_size": True,
        "encrypted": False,
        "fonts_embedded": True,
        "type3_fonts": False,
        "font_count": len(fonts),
        "anonymous_author_line": True,
        "private_text_scan": "PASS",
        "blocked_marker_scan": "PASS" if final else "NOT_APPLICABLE_DRAFT",
    }


def validate_source_anonymity(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.suffix.lower() not in {".tex", ".bib", ".json", ".py", ".sty", ".bst"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        scan_private_text(text, relative(path))


def tool_version(image: str, command: str) -> str:
    return run(
        ["docker", "run", "--rm", image, command, "--version"]
    ).stdout.splitlines()[0].strip()


def build_manifest(
    *,
    image: str,
    docker_id: str,
    inputs: Sequence[Path],
    main_result: Mapping[str, Any],
    supplement_result: Mapping[str, Any],
    main_content_page: int,
) -> dict[str, Any]:
    return {
        "schema_version": "leakbench.paper-build.v1",
        "status": "PASS",
        "submission_ready": True,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "toolchain": {
            "docker_image": image,
            "docker_image_id": docker_id,
            "dockerfile": "paper/aaai27/Dockerfile",
            "dockerfile_sha256": sha256(PAPER / "Dockerfile"),
            "expected_base_image_digest": EXPECTED_BASE_DIGEST,
            "pdflatex": tool_version(image, "pdflatex"),
            "bibtex": tool_version(image, "bibtex"),
        },
        "inputs": {relative(path): sha256(path) for path in sorted(set(inputs))},
        "outputs": {
            "main": dict(main_result),
            "supplement": dict(supplement_result),
        },
        "checks": {
            "independent_build_count": 2,
            "byte_identical_rebuilds": True,
            "undefined_references_or_citations": False,
            "rerun_required": False,
            "overfull_boxes": False,
            "main_content_last_page": main_content_page,
            "main_content_page_limit": 7,
            "main_total_page_limit": 9,
            "anonymous_submission": True,
            "private_text_scan": "PASS",
            "blocked_marker_scan": "PASS",
            "fonts_embedded": True,
            "type3_fonts": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--draft-only", action="store_true")
    parser.add_argument("--poppler-bin", type=Path)
    args = parser.parse_args()

    inputs = [ROOT / name for name in STATIC_PAPER_INPUTS]
    if not args.draft_only:
        inputs = verify_final_inputs()
        validate_source_anonymity(inputs)

    docker_id = build_image(args.image) if args.build_image else image_id(args.image)
    pdfinfo = find_poppler_tool("pdfinfo", args.poppler_bin)
    pdffonts = find_poppler_tool("pdffonts", args.poppler_bin)
    pdftotext = find_poppler_tool("pdftotext", args.poppler_bin)

    build_a = TMP / ("paper-build-a" if not args.draft_only else "draft-build")
    build_once(args.image, build_a)
    for document in ("main", "supplement"):
        validate_log(
            build_a / f"{document}.log", allow_overfull=args.draft_only
        )
        validate_pdf(
            build_a / f"{document}.pdf",
            pdfinfo=pdfinfo,
            pdffonts=pdffonts,
            pdftotext=pdftotext,
            final=not args.draft_only,
        )

    if args.draft_only:
        print(
            json.dumps(
                {
                    "status": "DRAFT_STRUCTURE_PASS",
                    "submission_ready": False,
                    "directory": relative(build_a),
                    "docker_image_id": docker_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    build_b = TMP / "paper-build-b"
    build_once(args.image, build_b)
    for document in ("main", "supplement"):
        validate_log(build_b / f"{document}.log")
        if sha256(build_a / f"{document}.pdf") != sha256(build_b / f"{document}.pdf"):
            raise BuildError(f"Independent {document} PDF builds are not byte-identical")

    main_content_page = parse_main_content_page(
        (build_a / "main.aux").read_text(encoding="utf-8", errors="replace")
    )
    if main_content_page > 7:
        raise BuildError(
            f"Main-paper content ends on page {main_content_page}; AAAI limit is 7"
        )

    main_result = validate_pdf(
        build_a / "main.pdf",
        pdfinfo=pdfinfo,
        pdffonts=pdffonts,
        pdftotext=pdftotext,
        final=True,
    )
    supplement_result = validate_pdf(
        build_a / "supplement.pdf",
        pdfinfo=pdfinfo,
        pdffonts=pdffonts,
        pdftotext=pdftotext,
        final=True,
    )
    if int(main_result["pages"]) > 9:
        raise BuildError(
            f"Main PDF has {main_result['pages']} pages; AAAI total limit is 9"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for document in ("main", "supplement"):
        shutil.copyfile(build_a / f"{document}.pdf", OUTPUT / f"{document}.pdf")
    main_result = {**main_result, "path": "paper/aaai27/output/pdf/main.pdf"}
    supplement_result = {
        **supplement_result,
        "path": "paper/aaai27/output/pdf/supplement.pdf",
    }
    manifest = build_manifest(
        image=args.image,
        docker_id=docker_id,
        inputs=inputs,
        main_result=main_result,
        supplement_result=supplement_result,
        main_content_page=main_content_page,
    )
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"PAPER BUILD BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
