#!/usr/bin/env python3
"""Verify an unpacked corrected_v2 public artifact without private natural data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import sys
from typing import Any
import zipfile

sys.dont_write_bytecode = True

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or re.match(r"^[A-Za-z]:", value)
        or path.is_absolute()
        or not path.parts
        or ".." in path.parts
    ):
        raise ValueError(f"Unsafe artifact path: {value}")
    return path


def verify_npz(path: Path, *, secret_patterns: Any, private_patterns: Any) -> None:
    patterns = (*secret_patterns, *private_patterns)

    def scan_member(data: bytes, member_name: str) -> None:
        for name, pattern in patterns:
            if pattern.search(data):
                raise ValueError(
                    f"Private/secret pattern {name} in NPZ member: "
                    f"{path}:{member_name}"
                )

    with zipfile.ZipFile(path) as archive:
        if not archive.infolist():
            raise ValueError(f"Empty NPZ archive: {path}")
        scan_member(
            b"\n".join(re.findall(rb"[\x09\x0a\x0d\x20-\x7e]{4,}", archive.comment)),
            "<archive-comment>",
        )
        if sum(member.file_size for member in archive.infolist()) > 1_000_000_000:
            raise ValueError(f"NPZ uncompressed payload exceeds 1 GB: {path}")
        for member in archive.infolist():
            member_path = safe_relative(member.filename)
            scan_member(member.filename.encode("utf-8"), "<member-name>")
            metadata = member.comment + b"\n" + member.extra
            scan_member(
                b"\n".join(re.findall(rb"[\x09\x0a\x0d\x20-\x7e]{4,}", metadata)),
                f"{member.filename}:<zip-metadata>",
            )
            if member_path.parts[0] in {"", "."}:
                raise ValueError(f"Unsafe NPZ member: {path}:{member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"NPZ symlink member is forbidden: {path}:{member.filename}")
            if member.flag_bits & 0x1:
                raise ValueError(f"Encrypted NPZ member is forbidden: {path}:{member.filename}")
            if (
                member.file_size > 512_000_000
                or member.file_size / max(1, member.compress_size) > 10_000
            ):
                raise ValueError(f"NPZ member exceeds safety limits: {path}:{member.filename}")
            # Numeric NPY payload bytes can coincidentally spell path patterns.
            # Scan the structured NPY header, then parse arrays below; numeric
            # payload bytes themselves are deliberately not searched.
            if member.filename.lower().endswith(".npy"):
                with archive.open(member) as handle:
                    if handle.read(6) != b"\x93NUMPY":
                        raise ValueError(f"Invalid NPY magic: {path}:{member.filename}")
                    version = tuple(handle.read(2))
                    length_size = 2 if version == (1, 0) else 4
                    if version not in {(1, 0), (2, 0), (3, 0)}:
                        raise ValueError(
                            f"Unsupported NPY version: {path}:{member.filename}:{version}"
                        )
                    raw_length = handle.read(length_size)
                    if len(raw_length) != length_size:
                        raise ValueError(f"Truncated NPY header: {path}:{member.filename}")
                    header_length = struct.unpack(
                        "<H" if length_size == 2 else "<I", raw_length
                    )[0]
                    if not 0 < header_length <= 1_000_000:
                        raise ValueError(f"Unsafe NPY header size: {path}:{member.filename}")
                    header = handle.read(header_length)
                    if len(header) != header_length:
                        raise ValueError(f"Truncated NPY header: {path}:{member.filename}")
                    scan_member(header, f"{member.filename}:<npy-header>")
            else:
                scan_member(archive.read(member), member.filename)
    try:
        with np.load(path, allow_pickle=False) as payload:
            for name in payload.files:
                array = np.asarray(payload[name])
                if array.dtype.hasobject:
                    raise ValueError(f"object dtype is forbidden: {name}")
                dtype_metadata = (
                    repr(array.dtype.descr) if array.dtype.fields else str(array.dtype)
                )
                scan_member(dtype_metadata.encode("utf-8"), f"{name}.npy:<dtype>")
                if array.dtype.kind in {"S", "U"}:
                    text = "\n".join(str(value) for value in array.reshape(-1))
                    scan_member(text.encode("utf-8"), f"{name}.npy")
    except ValueError as error:
        raise ValueError(f"NPZ validation failed: {path}: {error}") from error


def find_poppler_tool(name: str) -> str:
    found = shutil.which(name)
    if found is not None:
        return found
    candidates = sorted(
        Path.home().glob(
            ".cache/codex-runtimes/*/dependencies/native/poppler/poppler/bin/" + name
        )
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(
        f"{name} is required for public PDF verification; install Poppler or "
        "provide it through the bundled Codex runtime"
    )


def verify_public_natural(root: Path) -> None:
    base = root / "results/corrected_v2/public_natural"
    manifest_path = base / "public_natural_provenance_manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "PUBLIC_NATURAL_PROVENANCE_PROJECTED"
        or manifest.get("projection_version") != "natural_public_provenance_v1"
        or manifest.get("raw_natural_data_included") is not False
        or manifest.get("all_scientific_invariants_passed") is not True
    ):
        raise ValueError("Public natural projection manifest identity changed")
    expected_outputs = {
        "freeze": "results/corrected_v2/public_natural/natural_protocol_v2_freeze.json",
        "cells": "results/corrected_v2/public_natural/natural_cells.csv",
        "tasks": "results/corrected_v2/public_natural/natural_task_summary.csv",
        "statistics": "results/corrected_v2/public_natural/natural_statistics.json",
    }
    outputs = manifest.get("public_outputs", {})
    if set(outputs) != set(expected_outputs):
        raise ValueError("Public natural output set changed")
    for logical, expected in expected_outputs.items():
        entry = outputs[logical]
        if entry.get("path") != expected:
            raise ValueError(f"Public natural output path changed: {logical}")
        path = root / expected
        if (
            not path.is_file()
            or sha256(path) != entry.get("sha256")
            or path.stat().st_size != entry.get("size_bytes")
        ):
            raise ValueError(f"Public natural output hash mismatch: {logical}")
    private = manifest.get("private_provenance", {})
    if private.get("distribution") != "EXCLUDED_FROM_PUBLIC_ARTIFACT":
        raise ValueError("Private natural provenance is not excluded")
    for entry in private.get("artifacts", {}).values():
        private_path = safe_relative(entry.get("path", ""))
        if (root / str(private_path)).exists():
            raise ValueError(f"Private natural provenance was packaged: {private_path}")
    mappings = manifest.get("private_to_public", [])
    if len(mappings) != 4 or {entry.get("logical_name") for entry in mappings} != set(expected_outputs):
        raise ValueError("Typed natural private-to-public mappings are incomplete")
    for entry in mappings:
        logical = entry["logical_name"]
        if (
            entry.get("private_distribution") != "EXCLUDED_FROM_PUBLIC_ARTIFACT"
            or entry.get("public_path") != expected_outputs[logical]
            or entry.get("public_sha256") != outputs[logical]["sha256"]
            or not isinstance(entry.get("private_sha256"), str)
            or len(entry["private_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in entry["private_sha256"]
            )
        ):
            raise ValueError(f"Typed natural mapping changed: {logical}")
    freeze = load_json(root / expected_outputs["freeze"])
    for entry in freeze.get("source_files", {}).values():
        source = safe_relative(entry.get("path", ""))
        if source.parts[:1] != ("external_sources",):
            raise ValueError(f"Public natural lineage is not repo-relative: {source}")
    statistics = load_json(root / expected_outputs["statistics"])
    if (
        statistics.get("public_projection_version") != "natural_public_provenance_v1"
        or statistics.get("cells_sha256") != outputs["cells"]["sha256"]
        or statistics.get("task_summary_sha256") != outputs["tasks"]["sha256"]
    ):
        raise ValueError("Public natural statistics do not bind public inputs")


def verify_artifact(root: Path, *, deep_archives: bool = True) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "ARTIFACT_MANIFEST.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("ARTIFACT_MANIFEST.json is absent from the unpacked artifact")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Public artifact contains a symlink")
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"Public artifact contains a special filesystem node: {path}")

    builder_path = root / "scripts/build_corrected_v2_artifact.py"
    spec = importlib.util.spec_from_file_location(
        "_leakbench_packaged_artifact_builder", builder_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the packaged artifact policy: {builder_path}")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("evidence_tier") != "confirmatory"
        or manifest.get("selection_policy", {}).get("results_are_explicit_allowlist") is not True
    ):
        raise ValueError("Artifact manifest schema/tier/allowlist policy changed")
    declared = manifest.get("files", {})
    if not isinstance(declared, dict) or not declared:
        raise ValueError("Artifact manifest has no file inventory")
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(declared):
        missing = sorted(set(declared) - set(actual))
        extra = sorted(set(actual) - set(declared))
        raise ValueError(f"Artifact tree differs from manifest: missing={missing}, extra={extra}")
    for name, path in actual.items():
        safe_relative(name)
        entry = declared[name]
        if sha256(path) != entry.get("sha256") or path.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"Artifact hash/size mismatch: {name}")

    scan_hits = builder.scan_public_files([*actual.values(), manifest_path], root=root)
    if any(scan_hits.values()):
        raise ValueError(f"Artifact privacy/secret scan failed: {scan_hits}")
    if deep_archives:
        for name, path in actual.items():
            if path.suffix.lower() == ".npz":
                verify_npz(
                    path, secret_patterns=builder.SECRET_PATTERNS,
                    private_patterns=builder.PRIVATE_BYTE_PATTERNS,
                )

    verify_public_natural(root)
    claims_path = root / "results/corrected_v2/paper_claims.json"
    state_path = root / "results/corrected_v2/claim_state.json"
    if claims_path.read_bytes() != state_path.read_bytes():
        raise ValueError("Packaged paper claims and claim state differ")
    claims = load_json(claims_path)
    for name, digest in claims.get("provenance", {}).get("input_sha256", {}).items():
        path = root / str(safe_relative(name))
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Packaged claim provenance mismatch: {name}")

    inventory, selection, paper_build, disclosures = builder.compute_pre_release_inventory()
    expected_inventory = {
        "policy_version": inventory["policy_version"],
        "sha256": inventory["sha256"],
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
    }
    if manifest.get("artifact_input_inventory") != expected_inventory:
        raise ValueError("Artifact manifest input inventory differs from fresh recomputation")
    expected_actual_files = {
        *(str(path) for path in selection),
        "results/corrected_v2/release_validation.json",
        "ARTIFACT_README.md",
    }
    if set(actual) != expected_actual_files:
        missing = sorted(expected_actual_files - set(actual))
        extra = sorted(set(actual) - expected_actual_files)
        raise ValueError(
            f"Artifact contains files outside the full allowlist: missing={missing}, extra={extra}"
        )
    report = load_json(root / "results/corrected_v2/release_validation.json")
    tests = report.get("tests", {})
    test_checks = [
        entry for entry in report.get("checks", [])
        if isinstance(entry, dict) and entry.get("name") == "tests"
    ]
    if (
        report.get("status") != "PASS"
        or report.get("tests_skipped") is not False
        or tests.get("status") != "PASS"
        or tests.get("command") != "python -m pytest tests -q"
        or not re.search(r"\b[1-9][0-9]* passed\b", str(tests.get("tail", "")))
        or test_checks != [{"name": "tests", "status": "PASS"}]
        or report.get("artifact_input_inventory") != expected_inventory
        or report.get("paper_build_manifest_sha256")
        != sha256(root / str(builder.PAPER_BUILD_MANIFEST))
    ):
        raise ValueError("Packaged release report is stale, skipped, or not PASS")
    table_manifest = load_json(
        root / "paper/aaai27/generated/result_tables_manifest.json"
    )
    if (
        table_manifest.get("status") != "PASS"
        or table_manifest.get("table_count") != 7
        or table_manifest.get("pilot_inputs_forbidden") is not True
        or len(table_manifest.get("table_sha256", {})) != 8
    ):
        raise ValueError("Packaged final result-table manifest is incomplete")
    figure_manifest = load_json(
        root / "paper/aaai27/figures/generated/figure_manifest.json"
    )
    if (
        figure_manifest.get("evidence_tier") != "confirmatory"
        or figure_manifest.get("pilot_inputs_forbidden") is not True
        or len(figure_manifest.get("figure_sha256", {})) != 3
    ):
        raise ValueError("Packaged final figure manifest is incomplete")
    selected_results = sorted(
        str(path) for path in selection
        if path.parts[:2] == builder.RESULT_ROOT.parts
    )
    expected_result_files = sorted(
        selected_results + ["results/corrected_v2/release_validation.json"]
    )
    if manifest.get("selection_policy", {}).get("result_files") != expected_result_files:
        raise ValueError("Artifact result allowlist differs from fresh manifest closure")
    if manifest.get("selection_policy", {}).get("raw_provenance_only") != disclosures:
        raise ValueError("Artifact selector-level supersession disclosures changed")
    if manifest.get("selection_policy", {}).get("non_numerical_attestations") != [
        {
            "path": str(builder.GPU_INTERIM_INCIDENT),
            "status": "DOCUMENTED_NO_PROTOCOL_OR_CLAIM_POLICY_CHANGE",
            "numerical_claim_source_allowed": False,
        }
    ]:
        raise ValueError("GPU interim incident lost its non-numerical attestation role")
    selection_policy = manifest.get("selection_policy", {})
    required_false_flags = {
        "private_natural_provenance_included", "raw_natural_data_included",
        "pilot_results_included", "preflight_results_included",
        "checkpoints_included", "inventory_results_included",
        "whole_file_superseded_results_included",
    }
    if any(selection_policy.get(name) is not False for name in required_false_flags):
        raise ValueError("Artifact exclusion-policy attestations changed")
    privacy = manifest.get("privacy_scan", {})
    if (
        privacy.get("status") != "PASS"
        or privacy.get("independent_deep_verifier_required") is not True
        or privacy.get("generic_pattern_literal_exemptions") != []
    ):
        raise ValueError("Artifact privacy-scan policy changed")

    submission_pdfs = {
        root / paper_build["outputs"][name]["path"] for name in ("main", "supplement")
    }
    pdfs = sorted(
        path for path in actual.values() if path.suffix.lower() == ".pdf"
    )
    if not submission_pdfs.issubset(pdfs):
        raise ValueError("Submission PDFs are absent from the artifact PDF set")
    pdftotext = find_poppler_tool("pdftotext")
    pdfinfo = find_poppler_tool("pdfinfo")
    pdffonts = find_poppler_tool("pdffonts")
    submission_entries = {
        root / paper_build["outputs"][name]["path"]: paper_build["outputs"][name]
        for name in ("main", "supplement")
    }
    for pdf in pdfs:
        completed = subprocess.run(
            [pdftotext, str(pdf), "-"], capture_output=True, text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"pdftotext failed for {pdf}: {completed.stderr}")
        text = completed.stdout
        if pdf in submission_pdfs and "Anonymous submission" not in text:
            raise ValueError(f"Anonymous author marker absent from {pdf}")
        if any(marker in text for marker in ("INTERNAL DRAFT", "RESULTS BLOCKED")):
            raise ValueError(f"Blocked paper marker remains in {pdf}")
        encoded = text.encode("utf-8")
        metadata = subprocess.run(
            [pdfinfo, str(pdf)], capture_output=True, text=True, timeout=60,
        )
        if metadata.returncode != 0:
            raise RuntimeError(f"pdfinfo failed for {pdf}: {metadata.stderr}")
        info = {
            key.strip(): value.strip()
            for line in metadata.stdout.splitlines() if ":" in line
            for key, value in [line.split(":", 1)]
        }
        if info.get("Encrypted") != "no":
            raise ValueError(f"PDF is encrypted or has unknown encryption state: {pdf}")
        font_text = ""
        if pdf in submission_entries:
            entry = submission_entries[pdf]
            try:
                observed_pages = int(info["Pages"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"PDF page count is unavailable: {pdf}") from error
            if (
                observed_pages != entry["pages"]
                or info.get("Page size") != "612 x 792 pts (letter)"
            ):
                raise ValueError(f"Submission PDF geometry/page attestation differs: {pdf}")
            fonts = subprocess.run(
                [pdffonts, str(pdf)], capture_output=True, text=True, timeout=60,
            )
            if fonts.returncode != 0:
                raise RuntimeError(f"pdffonts failed for {pdf}: {fonts.stderr}")
            font_text = fonts.stdout
            font_lines = [line.strip() for line in font_text.splitlines() if line.strip()]
            if len(font_lines) < 3:
                raise ValueError(f"Submission PDF has no independently parsed fonts: {pdf}")
            for line in font_lines[2:]:
                columns = line.split()
                if (
                    len(columns) < 7
                    or columns[-5].lower() != "yes"
                    or re.search(r"\bType\s+3\b", line, re.IGNORECASE)
                ):
                    raise ValueError(f"Submission PDF font policy failed: {pdf}: {line}")
        encoded += b"\n" + metadata.stdout.encode("utf-8")
        encoded += b"\n" + font_text.encode("utf-8")
        for rule, pattern in (*builder.SECRET_PATTERNS, *builder.PRIVATE_BYTE_PATTERNS):
            if pattern.search(encoded):
                raise ValueError(f"PDF text matched private/secret rule {rule}: {pdf}")

    return {
        "status": "PASS" if deep_archives else "PARTIAL_DEEP_ARCHIVE_SCAN_SKIPPED",
        "artifact": str(root),
        "file_count": len(actual) + 1,
        "artifact_manifest_sha256": sha256(manifest_path),
        "artifact_input_inventory_sha256": inventory["sha256"],
        "prediction_count": 5000,
        "bundle_count": 20,
        "deep_archive_scan": deep_archives,
        "public_natural_projection": "PASS",
        "paper_build": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="?", default=".")
    parser.add_argument("--skip-deep-archive-scan", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args(argv)
    if args.run_tests and args.skip_deep_archive_scan:
        parser.error("--run-tests cannot be combined with --skip-deep-archive-scan")
    artifact = Path(args.artifact).resolve()
    result = verify_artifact(
        artifact, deep_archives=not args.skip_deep_archive_scan,
    )
    if args.run_tests:
        before_files = {
            path.relative_to(artifact).as_posix()
            for path in artifact.rglob("*") if path.is_file()
        }
        completed = subprocess.run(
            [
                sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider",
                "tests/public", "tests/test_statistical_amendment.py",
                "tests/test_statistical_amendment_v2.py", "-q",
            ],
            cwd=artifact, capture_output=True, text=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Public artifact tests failed:\n"
                + "\n".join((completed.stdout or completed.stderr).splitlines()[-20:])
            )
        if not re.search(r"\b[1-9][0-9]* passed\b", completed.stdout):
            raise RuntimeError("Public artifact tests did not execute a passing test")
        after_files = {
            path.relative_to(artifact).as_posix()
            for path in artifact.rglob("*") if path.is_file()
        }
        if after_files != before_files:
            raise RuntimeError(
                "Public tests mutated the exact artifact tree: "
                f"added={sorted(after_files - before_files)}, "
                f"removed={sorted(before_files - after_files)}"
            )
        result["public_tests"] = {
            "status": "PASS",
            "command": (
                f"{sys.executable} -B -m pytest -p no:cacheprovider tests/public "
                "tests/test_statistical_amendment.py "
                "tests/test_statistical_amendment_v2.py -q"
            ),
            "tail": "\n".join(completed.stdout.splitlines()[-3:]),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
