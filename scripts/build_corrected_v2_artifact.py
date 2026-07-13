#!/usr/bin/env python3
"""Build the byte-exact, allowlisted corrected_v2 public submission artifact."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PurePosixPath("results/corrected_v2")
DEFAULT_DESTINATION = ROOT / "release/leakbench_tab_corrected_v2_aaai27"
PAPER_BUILD_MANIFEST = PurePosixPath(
    "paper/aaai27/output/pdf/paper_build_manifest.json"
)
CANONICAL_BUILDER = PurePosixPath("scripts/build_canonical_corrected_v2.py")
INVENTORY_POLICY_VERSION = "corrected_v2_public_allowlist_v1"

PAPER_BUILD_STATIC_INPUTS = {
    PurePosixPath("paper/aaai27/main.tex"),
    PurePosixPath("paper/aaai27/supplement.tex"),
    PurePosixPath("paper/aaai27/ReproducibilityChecklist.tex"),
    PurePosixPath("paper/aaai27/references.bib"),
    PurePosixPath("paper/aaai27/aaai2027.sty"),
    PurePosixPath("paper/aaai27/aaai2027.bst"),
    PurePosixPath("paper/aaai27/source_data/result_macros_base.tex"),
    PurePosixPath("paper/aaai27/source_data/generate_result_macros.py"),
    PurePosixPath("paper/aaai27/source_data/generate_result_tables.py"),
    PurePosixPath("paper/aaai27/Dockerfile"),
    PurePosixPath("scripts/build_aaai27_paper.py"),
    PurePosixPath("scripts/generate_corrected_v2_figures.py"),
}
EXPECTED_PAPER_FIGURES = {
    PurePosixPath("paper/aaai27/figures/generated/cdx_scatter.pdf"),
    PurePosixPath("paper/aaai27/figures/generated/mechanism_model_heatmap.pdf"),
    PurePosixPath("paper/aaai27/figures/generated/strength_diagnostic_robustness.pdf"),
}
EXPECTED_PAPER_TABLES = {
    PurePosixPath("paper/aaai27/generated/result_tables.tex"),
    PurePosixPath("paper/aaai27/generated/table_task_registry.tex"),
    PurePosixPath("paper/aaai27/generated/table_mechanism_profiles.tex"),
    PurePosixPath("paper/aaai27/generated/table_mechanism_models.tex"),
    PurePosixPath("paper/aaai27/generated/table_diagnostic_methods.tex"),
    PurePosixPath("paper/aaai27/generated/table_strength_response.tex"),
    PurePosixPath("paper/aaai27/generated/table_natural_cases.tex"),
    PurePosixPath("paper/aaai27/generated/table_claim_scope.tex"),
}

PRIVATE_NATURAL_PATHS = {
    PurePosixPath("results/corrected_v2/natural_protocol_freeze.json"),
    PurePosixPath("results/corrected_v2/natural_protocol_v2_freeze.json"),
    PurePosixPath("results/corrected_v2/natural_cells.csv"),
    PurePosixPath("results/corrected_v2/natural_task_summary.csv"),
    PurePosixPath("results/corrected_v2/natural_statistics.json"),
}
PUBLIC_NATURAL_PATHS = {
    PurePosixPath("results/corrected_v2/public_natural/natural_protocol_v2_freeze.json"),
    PurePosixPath("results/corrected_v2/public_natural/natural_cells.csv"),
    PurePosixPath("results/corrected_v2/public_natural/natural_task_summary.csv"),
    PurePosixPath("results/corrected_v2/public_natural/natural_statistics.json"),
    PurePosixPath(
        "results/corrected_v2/public_natural/public_natural_provenance_manifest.json"
    ),
}
FORBIDDEN_RESULT_PARTS = {
    "checkpoints",
    "inventory",
    "superseded_snapshots",
    "pilot_statistics",
    "diagnostic_pilot_statistics",
    "diagnostic_pilot_tasks",
    "m10_amendment_pilot",
    "m10_amendment_pilot_tasks",
    "tabm_bundle_pilot",
    "tabm_bundle_pilot_tasks",
    "tabm_pilot",
}
FORBIDDEN_RESULT_NAME_FRAGMENTS = ("pilot", "preflight", "checkpoint")
ALLOWED_PREDICTION_PREFIXES = (
    PurePosixPath("results/corrected_v2/predictions"),
    PurePosixPath(
        "results/corrected_v2/tabm_bundle_confirmatory/tabm_cells_predictions"
    ),
)
RAW_PROVENANCE_ONLY = {
    PurePosixPath("results/corrected_v2/core_cpu_cells.csv"): {
        "selector": {"mechanism": "M10"},
        "replacement": "results/corrected_v2/m10_amendment_confirmatory/cpu_cells.csv",
    },
    PurePosixPath("results/corrected_v2/tabm_bundle_confirmatory/tabm_cells.csv"): {
        "selector": {"mechanism": "M10"},
        "replacement": "results/corrected_v2/m10_amendment_confirmatory/tabm_cells.csv",
    },
    PurePosixPath("results/corrected_v2/diagnostic_confirmatory_cells.csv"): {
        "selector": {"method": "mutual_information"},
        "replacement": "results/corrected_v2/diagnostic_canonical_cells.csv",
    },
}
GPU_INTERIM_INCIDENT = PurePosixPath(
    "results/corrected_v2/gpu_interim_access_incident.json"
)

STATIC_REPRODUCIBILITY_FILES = {
    PurePosixPath("requirements-corrected-v2.txt"),
    PurePosixPath("benchmark_v2/core/models.py"),
    PurePosixPath("benchmark_v2/datasets/adapters.py"),
    PurePosixPath("benchmark_v2/datasets/confirmatory_adapters.py"),
    PurePosixPath("tests/test_statistical_amendment.py"),
    PurePosixPath("tests/test_statistical_amendment_v2.py"),
}
TREE_RULES = {
    PurePosixPath("src/leakbench"): {".py"},
    PurePosixPath("experiments/leakbench"): {".py"},
    PurePosixPath("scripts"): {".py", ".sh"},
    PurePosixPath("tests/public"): {".py"},
}
PAPER_STATIC_FILES = {
    PurePosixPath("paper/aaai27/ReproducibilityChecklist.tex"),
    PurePosixPath("paper/aaai27/aaai2027.bst"),
    PurePosixPath("paper/aaai27/aaai2027.sty"),
    PurePosixPath("paper/aaai27/main.tex"),
    PurePosixPath("paper/aaai27/references.bib"),
    PurePosixPath("paper/aaai27/supplement.tex"),
    PurePosixPath("paper/aaai27/Dockerfile"),
    PurePosixPath("paper/aaai27/source_data/audit_numbers.py"),
    PurePosixPath("paper/aaai27/source_data/generate_result_macros.py"),
    PurePosixPath("paper/aaai27/source_data/paper_claims.schema.json"),
    PurePosixPath("paper/aaai27/source_data/result_macros_base.tex"),
}

# Scanner patterns are written so their source literals do not self-match.  No
# complete file is exempt: an exemption would also hide an unrelated private path.
GENERIC_PATTERN_LITERAL_EXEMPTIONS: set[PurePosixPath] = set()
SECRET_PATTERNS = (
    ("openai_or_generic_sk", re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic", re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("huggingface", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("github", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google_api", re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("stripe_live", re.compile(rb"\b(?:sk|rk)_live_[A-Za-z0-9]{20,}\b")),
    ("jwt", re.compile(rb"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("basic_auth_url", re.compile(rb"https?://[^\s/:@]+:[^\s/@]+@[^\s/]+")),
    (
        "credential_assignment",
        re.compile(
            rb"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\b"
            rb"\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"
        ),
    ),
)
PRIVATE_BYTE_PATTERNS = (
    ("macos_home", re.compile(rb"/Users/[A-Za-z0-9._-]+")),
    ("linux_home", re.compile(rb"/home/[A-Za-z0-9._-]+")),
    ("windows_drive", re.compile(rb"(?i)(?<![A-Za-z0-9])(?:[A-Z]):[\\/][^\r\n]*")),
    (
        "windows_unc",
        re.compile(
            rb"(?i)\\\\[A-Z0-9][A-Z0-9._-]{0,62}"
            rb"\\[A-Z0-9$._ -]+(?:\\[^\r\n]*)?"
        ),
    ),
    ("rfc1918_10", re.compile(rb"\b10(?:\.\d{1,3}){3}\b")),
    ("rfc1918_172", re.compile(rb"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b")),
    ("rfc1918_192", re.compile(rb"\b192\.168(?:\.\d{1,3}){2}\b")),
    ("loopback_ipv4", re.compile(rb"\b127(?:\.\d{1,3}){3}\b")),
    ("link_local_ipv4", re.compile(rb"\b169\.254(?:\.\d{1,3}){2}\b")),
    (
        "loopback_ipv6",
        re.compile(rb"(?i)(?<![0-9a-f:]):" rb":1(?![0-9a-f:])"),
    ),
    ("local_ipv6", re.compile(rb"(?i)\b(?:fc|fd|fe8|fe9|fea|feb)[0-9a-f:]*:[0-9a-f:]+\b")),
    (
        "local_endpoint_name",
        re.compile(
            rb"(?i)(?<![A-Z0-9._-])local" rb"host(?![A-Z0-9._-])"
        ),
    ),
    ("local_host", re.compile(rb"(?i)\b(?:LAPTOP|DESKTOP)-[A-Za-z0-9-]+\b")),
    ("ssh_endpoint", re.compile(rb"(?i)\bssh\s+[^\s@]+@(?:10\.|172\.(?:1[6-9]|2\d|3[01])\.|192\.168\.)")),
    ("email_address", re.compile(rb"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    ("mac_address", re.compile(rb"(?i)\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b")),
)
CREDENTIAL_FILENAMES = {
    ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
    "id_dsa", "id_ecdsa", ".netrc", ".npmrc", ".pypirc",
    "authorized_keys", "credentials", "credentials.json", "secrets.json",
    "service-account.json",
}
CREDENTIAL_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _safe_relative(value: str | PurePosixPath) -> PurePosixPath:
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        "\\" in raw
        or re.match(r"^[A-Za-z]:", raw)
        or path.is_absolute()
        or not path.parts
        or ".." in path.parts
    ):
        raise ValueError(f"Unsafe repository-relative path: {value}")
    return path


def _source(path: PurePosixPath) -> Path:
    source = ROOT.joinpath(*path.parts)
    root_resolved = ROOT.resolve()
    try:
        lexical_relative = source.relative_to(ROOT)
    except ValueError as error:
        raise FileNotFoundError(source) from error
    cursor = ROOT
    for part in lexical_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise FileNotFoundError(f"symlinked source path is forbidden: {source}")
    try:
        source.resolve(strict=True).relative_to(root_resolved)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise FileNotFoundError(source) from error
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _assert_bound_file(
    path_value: str | PurePosixPath,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> PurePosixPath:
    relative = _safe_relative(path_value)
    source = _source(relative)
    if expected_sha256 is not None and sha256(source) != expected_sha256:
        raise ValueError(f"Manifest SHA-256 mismatch: {relative}")
    if expected_size is not None and source.stat().st_size != expected_size:
        raise ValueError(f"Manifest size mismatch: {relative}")
    return relative


def _add(
    selection: dict[PurePosixPath, set[str]],
    path: PurePosixPath,
    role: str,
) -> None:
    _source(path)
    selection.setdefault(path, set()).add(role)


def _iter_code_tree(root: PurePosixPath, suffixes: set[str]) -> Iterable[PurePosixPath]:
    directory = ROOT.joinpath(*root.parts)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    for source in sorted(directory.rglob("*")):
        if source.is_symlink() or not source.is_file():
            continue
        relative = PurePosixPath(source.relative_to(ROOT).as_posix())
        if "__pycache__" in relative.parts or source.suffix not in suffixes:
            continue
        yield relative


def _bound_hash_map_paths(
    manifest_path: PurePosixPath,
    field: str,
    *,
    expected_paths: set[PurePosixPath] | None = None,
) -> set[PurePosixPath]:
    payload = _load_json(_source(manifest_path))
    mapping = payload.get(field)
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"{manifest_path} lacks a non-empty {field} hash map")
    paths: set[PurePosixPath] = set()
    for name, digest in mapping.items():
        path = _safe_relative(name)
        if str(path) != name or path in paths or not _is_sha256(digest):
            raise RuntimeError(f"Invalid or duplicate {field} entry: {name}")
        _assert_bound_file(path, expected_sha256=digest)
        paths.add(path)
    if expected_paths is not None and paths != expected_paths:
        raise RuntimeError(
            f"{manifest_path} {field} set changed: "
            f"expected={sorted(map(str, expected_paths))}, "
            f"observed={sorted(map(str, paths))}"
        )
    return paths


def _expected_paper_build_inputs() -> set[PurePosixPath]:
    figure_manifest = PurePosixPath(
        "paper/aaai27/figures/generated/figure_manifest.json"
    )
    table_manifest = PurePosixPath(
        "paper/aaai27/generated/result_tables_manifest.json"
    )
    figures = _bound_hash_map_paths(
        figure_manifest, "figure_sha256", expected_paths=EXPECTED_PAPER_FIGURES
    )
    figure_sources = _bound_hash_map_paths(figure_manifest, "source_sha256")
    tables = _bound_hash_map_paths(
        table_manifest, "table_sha256", expected_paths=EXPECTED_PAPER_TABLES
    )
    table_sources = _bound_hash_map_paths(table_manifest, "source_sha256")
    return {
        *PAPER_BUILD_STATIC_INPUTS,
        PurePosixPath("results/corrected_v2/paper_claims.json"),
        PurePosixPath("paper/aaai27/generated/result_macros.tex"),
        figure_manifest,
        table_manifest,
        *figures,
        *figure_sources,
        *tables,
        *table_sources,
    }


def _validate_paper_build_manifest() -> tuple[dict[str, Any], set[PurePosixPath]]:
    path = _source(PAPER_BUILD_MANIFEST)
    payload = _load_json(path)
    if (
        payload.get("schema_version") != "leakbench.paper-build.v1"
        or payload.get("status") != "PASS"
        or payload.get("submission_ready") is not True
    ):
        raise RuntimeError("paper build manifest is not submission-ready PASS")
    checks = payload.get("checks", {})
    required_checks = {
        "independent_build_count": 2,
        "byte_identical_rebuilds": True,
        "undefined_references_or_citations": False,
        "rerun_required": False,
        "overfull_boxes": False,
        "main_content_last_page": 7,
        "main_content_page_limit": 7,
        "main_total_page_limit": 9,
        "anonymous_submission": True,
        "private_text_scan": "PASS",
        "blocked_marker_scan": "PASS",
        "fonts_embedded": True,
        "type3_fonts": False,
    }
    for name, expected in required_checks.items():
        observed = checks.get(name)
        if name == "main_content_last_page":
            if type(observed) is not int or not 1 <= observed <= expected:
                raise RuntimeError("paper main content exceeds seven pages")
        elif observed != expected:
            raise RuntimeError(f"paper build check failed or changed: {name}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise RuntimeError("paper build manifest has no bound inputs")
    observed_inputs: dict[PurePosixPath, str] = {}
    for name, digest in inputs.items():
        path = _safe_relative(name)
        if str(path) != name or path in observed_inputs:
            raise RuntimeError(f"paper build manifest has an ambiguous input path: {name}")
        observed_inputs[path] = digest
    expected_inputs = _expected_paper_build_inputs()
    if set(observed_inputs) != expected_inputs:
        raise RuntimeError(
            "paper build manifest input closure changed: "
            f"missing={sorted(map(str, expected_inputs - set(observed_inputs)))}, "
            f"extra={sorted(map(str, set(observed_inputs) - expected_inputs))}"
        )
    selected = {PAPER_BUILD_MANIFEST}
    for path, digest in observed_inputs.items():
        if not _is_sha256(digest):
            raise ValueError(f"Invalid paper input digest: {path}")
        selected.add(_assert_bound_file(path, expected_sha256=digest))
    outputs = payload.get("outputs", {})
    if set(outputs) != {"main", "supplement"}:
        raise RuntimeError("paper build must bind exactly main and supplement PDFs")
    expected_output_paths = {
        "main": PurePosixPath("paper/aaai27/output/pdf/main.pdf"),
        "supplement": PurePosixPath("paper/aaai27/output/pdf/supplement.pdf"),
    }
    for name, expected_path in expected_output_paths.items():
        entry = outputs[name]
        if (
            not isinstance(entry, dict)
            or _safe_relative(entry.get("path", "")) != expected_path
            or type(entry.get("pages")) is not int
            or entry["pages"] <= 0
            or entry.get("fonts_embedded") is not True
            or entry.get("type3_fonts") is not False
            or entry.get("anonymous_author_line") is not True
            or entry.get("private_text_scan") != "PASS"
            or entry.get("blocked_marker_scan") != "PASS"
            or not _is_sha256(entry.get("sha256"))
            or type(entry.get("bytes")) is not int
            or entry["bytes"] <= 0
        ):
            raise RuntimeError(f"paper PDF attestation changed: {name}")
        selected.add(_assert_bound_file(
            expected_path,
            expected_sha256=entry.get("sha256"),
            expected_size=entry.get("bytes"),
        ))
        if name == "main" and entry["pages"] > checks["main_total_page_limit"]:
            raise RuntimeError("main PDF exceeds the nine-page total submission limit")
    return payload, selected


def _validate_supersession_policy(
    selected_results: set[PurePosixPath],
) -> list[dict[str, Any]]:
    manifest_path = PurePosixPath("results/corrected_v2/superseded_evidence.json")
    payload = _load_json(_source(manifest_path))
    if payload.get("status") != "INTEGRITY_HOLD":
        raise RuntimeError("superseded evidence policy is not active")
    whole_file: set[PurePosixPath] = set()
    observed_selector: dict[PurePosixPath, dict[str, str]] = {}
    selector_entry_count = 0
    for entry in payload.get("superseded", []):
        path = _safe_relative(entry["path"])
        selector = entry.get("selector")
        if selector is None:
            whole_file.add(path)
        else:
            selector_entry_count += 1
            if path in observed_selector:
                raise RuntimeError(f"duplicate selector-level supersession: {path}")
            observed_selector[path] = selector
    forbidden = selected_results.intersection(whole_file)
    if forbidden:
        raise RuntimeError(
            f"whole-file superseded evidence entered artifact: {sorted(map(str, forbidden))}"
        )
    if selector_entry_count != 3 or observed_selector != {
        path: details["selector"] for path, details in RAW_PROVENANCE_ONLY.items()
    }:
        raise RuntimeError("selector-level supersession policy changed")
    missing = set(RAW_PROVENANCE_ONLY) - selected_results
    if missing:
        raise RuntimeError(f"required raw provenance is absent: {sorted(map(str, missing))}")
    return [
        {
            "path": str(path),
            "artifact_role": "RAW_PROVENANCE_ONLY",
            "excluded_selector": details["selector"],
            "replacement_path": details["replacement"],
            "claim_facing_use_allowed": False,
        }
        for path, details in sorted(RAW_PROVENANCE_ONLY.items(), key=lambda item: str(item[0]))
    ]


def _whole_file_superseded_paths() -> set[PurePosixPath]:
    payload = _load_json(
        _source(PurePosixPath("results/corrected_v2/superseded_evidence.json"))
    )
    return {
        _safe_relative(entry["path"])
        for entry in payload.get("superseded", [])
        if entry.get("selector") is None
    }


def _validate_result_path_policy(paths: set[PurePosixPath]) -> None:
    private = paths.intersection(PRIVATE_NATURAL_PATHS)
    if private:
        raise RuntimeError(f"private natural provenance selected: {sorted(map(str, private))}")
    missing_public = PUBLIC_NATURAL_PATHS - paths
    if missing_public:
        raise RuntimeError(f"public natural projection incomplete: {sorted(map(str, missing_public))}")
    for path in paths:
        if path.parts[:2] != RESULT_ROOT.parts:
            raise RuntimeError(f"non-corrected_v2 result selected: {path}")
        if set(path.parts).intersection(FORBIDDEN_RESULT_PARTS):
            raise RuntimeError(f"forbidden result component selected: {path}")
        name = path.name.lower()
        if any(fragment in name for fragment in FORBIDDEN_RESULT_NAME_FRAGMENTS):
            raise RuntimeError(f"pilot/preflight/checkpoint result selected: {path}")


def _validate_gpu_interim_incident() -> None:
    payload = _load_json(_source(GPU_INTERIM_INCIDENT))
    if (
        payload.get("schema_version") != 1
        or payload.get("status")
        != "DOCUMENTED_NO_PROTOCOL_OR_CLAIM_POLICY_CHANGE"
        or payload.get("access_scope") != {
            "aggregate_final_outputs_accessed": False,
            "checkpoint_files_modified": False,
            "gpu_process_modified_or_interrupted": False,
            "display_was_truncated": True,
            "some_cell_level_metrics_were_visible": True,
        }
    ):
        raise RuntimeError("GPU interim-access incident attestation changed")
    locked = payload.get("locked_response", {})
    required_false = {
        "protocol_changes_allowed", "mechanism_or_dataset_exclusions_allowed",
        "estimand_or_resampling_changes_allowed", "support_threshold_changes_allowed",
        "new_thresholded_claims_allowed",
        "interim_metric_use_for_manuscript_wording_allowed",
    }
    if any(locked.get(name) is not False for name in required_false):
        raise RuntimeError("GPU interim-access incident does not preserve frozen decisions")
    attestations = payload.get("pre_incident_attestations", {})
    if not isinstance(attestations, dict) or len(attestations) != 6:
        raise RuntimeError("GPU incident lacks the six pre-incident attestations")
    for name, digest in attestations.items():
        if not _is_sha256(digest):
            raise RuntimeError(f"GPU incident has invalid attestation digest: {name}")
        _assert_bound_file(name, expected_sha256=digest)


def _normalized_integer_text(value: Any, *, field: str) -> str:
    try:
        number = float(str(value))
    except ValueError as error:
        raise RuntimeError(f"cluster lineage has a non-numeric {field}: {value}") from error
    if not number.is_integer():
        raise RuntimeError(f"cluster lineage has a non-integral {field}: {value}")
    return str(int(number))


def _load_canonical_cluster_contract(
    path: Path,
) -> dict[str, tuple[str, str, str, str, str]]:
    required = {"run_id", "dataset_id", "mechanism", "strength", "seed", "model"}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise RuntimeError("canonical cells lack the full cluster lineage columns")
        result: dict[str, tuple[str, str, str, str, str]] = {}
        for row in reader:
            if str(row["mechanism"]) not in {"M08", "M09"}:
                continue
            run_id = str(row["run_id"])
            if not run_id or run_id in result:
                raise RuntimeError(f"canonical cluster run ID is empty or duplicated: {run_id}")
            result[run_id] = (
                str(row["dataset_id"]), str(row["mechanism"]),
                str(row["strength"]),
                _normalized_integer_text(row["seed"], field="seed"),
                str(row["model"]),
            )
    return result


def _load_frozen_task_lineage(
    path: Path,
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    required = {
        "dataset_id", "dataset_namespace", "mechanism", "strength", "seed",
        "task_hash", "split_hash", "bundle_path", "bundle_sha256",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise RuntimeError("task manifest lacks the full frozen lineage columns")
        result: dict[tuple[str, str, str, str], dict[str, str]] = {}
        for row in reader:
            mechanism = str(row["mechanism"])
            if mechanism not in {"M08", "M09"}:
                continue
            if str(row["dataset_namespace"]) != "confirmatory":
                raise RuntimeError("cluster task lineage escaped the confirmatory namespace")
            key = (
                str(row["dataset_id"]), mechanism, str(row["strength"]),
                _normalized_integer_text(row["seed"], field="seed"),
            )
            if key in result:
                raise RuntimeError(f"frozen task lineage key is duplicated: {key}")
            lineage = {
                "task_hash": str(row["task_hash"]),
                "split_hash": str(row["split_hash"]),
                "bundle_path": str(_safe_relative(row["bundle_path"])),
                "bundle_sha256": str(row["bundle_sha256"]),
            }
            if any(
                not _is_sha256(lineage[field])
                for field in ("task_hash", "split_hash", "bundle_sha256")
            ):
                raise RuntimeError(f"frozen task lineage has an invalid digest: {key}")
            result[key] = lineage
    return result


def _select_result_sources() -> tuple[dict[PurePosixPath, set[str]], list[dict[str, Any]]]:
    selection: dict[PurePosixPath, set[str]] = {}
    claims_path = PurePosixPath("results/corrected_v2/paper_claims.json")
    state_path = PurePosixPath("results/corrected_v2/claim_state.json")
    claims = _load_json(_source(claims_path))
    if _source(claims_path).read_bytes() != _source(state_path).read_bytes():
        raise RuntimeError("paper claims and claim state are not byte-identical")
    provenance = claims.get("provenance", {}).get("input_sha256")
    if not isinstance(provenance, dict) or not provenance:
        raise RuntimeError("paper claims lack a complete provenance hash map")
    if str(GPU_INTERIM_INCIDENT) in provenance:
        raise RuntimeError("GPU interim incident cannot be a numerical claim source")
    for name, digest in provenance.items():
        if not _is_sha256(digest):
            raise RuntimeError(f"claim provenance lacks typed SHA-256: {name}")
        path = _assert_bound_file(name, expected_sha256=digest)
        if path.parts[:1] not in {
            ("results",), ("configs",), ("scripts",), ("experiments",),
        }:
            raise RuntimeError(f"claim provenance escaped public source roots: {path}")
        _add(selection, path, "CLAIM_PROVENANCE")
    _add(selection, claims_path, "CLAIM_OUTPUT")
    _add(selection, state_path, "CLAIM_OUTPUT")
    _validate_gpu_interim_incident()
    _add(selection, GPU_INTERIM_INCIDENT, "OPERATIONAL_INCIDENT_ATTESTATION_ONLY")

    canonical_manifest_path = PurePosixPath("results/corrected_v2/canonical_manifest.json")
    canonical_manifest = _load_json(_source(canonical_manifest_path))
    if canonical_manifest.get("builder") != {
        "path": str(CANONICAL_BUILDER),
        "sha256": sha256(_source(CANONICAL_BUILDER)),
    }:
        raise RuntimeError("canonical manifest is not bound to the packaged canonical builder")
    expected_sources = {
        "cpu": PurePosixPath("results/corrected_v2/core_cpu_cells.csv"),
        "tabm": PurePosixPath("results/corrected_v2/tabm_bundle_confirmatory/tabm_cells.csv"),
        "tasks": PurePosixPath("results/corrected_v2/task_bundles/task_manifest.csv"),
        "m10_cpu": PurePosixPath("results/corrected_v2/m10_amendment_confirmatory/cpu_cells.csv"),
        "m10_tabm": PurePosixPath("results/corrected_v2/m10_amendment_confirmatory/tabm_cells.csv"),
    }
    source_hashes = canonical_manifest.get("source_sha256", {})
    if set(source_hashes) != set(expected_sources):
        raise RuntimeError("canonical source dependency set changed")
    for key, path in expected_sources.items():
        if not _is_sha256(source_hashes[key]):
            raise RuntimeError(f"canonical source digest is invalid: {key}")
        _assert_bound_file(path, expected_sha256=source_hashes[key])
        _add(selection, path, "CANONICAL_RAW_DEPENDENCY")
    source_manifest_paths = {
        "cpu": PurePosixPath("results/corrected_v2/core_cpu_cells_manifest.json"),
        "tabm": PurePosixPath("results/corrected_v2/tabm_bundle_confirmatory/tabm_cells_manifest.json"),
        "m10_cpu": PurePosixPath("results/corrected_v2/m10_amendment_confirmatory/cpu_cells_manifest.json"),
        "m10_tabm": PurePosixPath("results/corrected_v2/m10_amendment_confirmatory/tabm_cells_manifest.json"),
    }
    if canonical_manifest.get("source_manifest_sha256") != {
        name: sha256(_source(path)) for name, path in source_manifest_paths.items()
    }:
        raise RuntimeError("canonical raw source-manifest hashes changed")
    for path in (
        *source_manifest_paths.values(),
        PurePosixPath("results/corrected_v2/task_bundles/bundle_summary.json"),
        PurePosixPath("results/corrected_v2/diagnostic_confirmatory_cells.manifest.json"),
    ):
        _add(selection, path, "BOUND_RAW_MANIFEST")

    freeze = _load_json(_source(PurePosixPath(
        "results/corrected_v2/statistical_amendment_protocol_v2_freeze.json"
    )))
    cluster_manifest_path = _safe_relative(freeze["outputs"]["cluster_manifest"])
    cluster = _load_json(_source(cluster_manifest_path))
    if (
        cluster.get("schema_version") != 2
        or cluster.get("status") != "SYNCHRONIZED_CLUSTER_ANALYSIS_COMPLETE"
        or cluster.get("consumed_prediction_count") != 5000
    ):
        raise RuntimeError("final cluster manifest is incomplete")
    frozen_bundles = cluster.get("frozen_bundles", [])
    expected_bundle_paths = {
        PurePosixPath(f"results/corrected_v2/task_bundles/panel_{index:02d}.npz")
        for index in range(20)
    }
    observed_bundle_paths = {
        _safe_relative(entry.get("path", "")) for entry in frozen_bundles
    }
    if len(frozen_bundles) != 20 or observed_bundle_paths != expected_bundle_paths:
        raise RuntimeError("cluster manifest must bind exactly panel_00..panel_19")
    frozen_hashes = freeze.get("bundle_sha256_by_path", {})
    if set(map(str, expected_bundle_paths)) != set(frozen_hashes):
        raise RuntimeError("statistical freeze bundle set changed")
    if any(not _is_sha256(digest) for digest in frozen_hashes.values()):
        raise RuntimeError("statistical freeze contains an invalid bundle digest")
    for entry in frozen_bundles:
        if not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] <= 0:
            raise RuntimeError(f"frozen bundle lacks typed size_bytes: {entry.get('path')}")
        path = _assert_bound_file(
            entry["path"], expected_sha256=entry.get("sha256"),
            expected_size=entry.get("size_bytes"),
        )
        if entry.get("sha256") != frozen_hashes[str(path)]:
            raise RuntimeError(f"cluster/freeze bundle hash differs: {path}")
        _add(selection, path, "FROZEN_TASK_BUNDLE")

    predictions = cluster.get("consumed_predictions", [])
    if len(predictions) != 5000:
        raise RuntimeError("cluster manifest must enumerate 5,000 predictions")
    run_ids = [str(entry.get("run_id", "")) for entry in predictions]
    prediction_paths = [_safe_relative(entry.get("path", "")) for entry in predictions]
    if len(set(run_ids)) != 5000 or len(set(prediction_paths)) != 5000:
        raise RuntimeError("cluster prediction run_id/path identities are not unique")
    mechanism_counts = Counter(str(entry.get("mechanism")) for entry in predictions)
    if mechanism_counts != Counter({"M08": 2500, "M09": 2500}):
        raise RuntimeError("cluster predictions are not exactly 2,500 M08 + 2,500 M09")
    canonical_path = _source(PurePosixPath("results/corrected_v2/canonical_cells.csv"))
    canonical_cluster = _load_canonical_cluster_contract(canonical_path)
    if len(canonical_cluster) != 5000 or set(canonical_cluster) != set(run_ids):
        raise RuntimeError("cluster prediction run IDs differ from canonical M08/M09 rows")
    task_lineage = _load_frozen_task_lineage(
        _source(PurePosixPath("results/corrected_v2/task_bundles/task_manifest.csv"))
    )
    expected_task_keys = {details[:4] for details in canonical_cluster.values()}
    if len(task_lineage) != 1000 or set(task_lineage) != expected_task_keys:
        raise RuntimeError("frozen task lineage differs from canonical M08/M09 task keys")
    models_by_task: dict[tuple[str, str, str, str], set[str]] = {}
    for details in canonical_cluster.values():
        models_by_task.setdefault(details[:4], set()).add(details[4])
    expected_models = {"lr", "rf", "lightgbm", "catboost", "tabm"}
    if any(models != expected_models for models in models_by_task.values()):
        raise RuntimeError("canonical cluster tasks do not each contain the five frozen models")
    directory_counts: Counter[str] = Counter()
    for entry, path in zip(predictions, prediction_paths):
        if path.suffix != ".npz":
            raise RuntimeError(f"prediction is not an NPZ archive: {path}")
        parent = path.parent
        if parent not in ALLOWED_PREDICTION_PREFIXES:
            raise RuntimeError(f"prediction escaped final directories: {path}")
        canonical_details = canonical_cluster[str(entry["run_id"])]
        expected_mechanism, expected_model = canonical_details[1], canonical_details[4]
        expected_parent = (
            ALLOWED_PREDICTION_PREFIXES[1]
            if expected_model == "tabm" else ALLOWED_PREDICTION_PREFIXES[0]
        )
        if entry.get("mechanism") != expected_mechanism or parent != expected_parent:
            raise RuntimeError(f"prediction/canonical model or mechanism mismatch: {path}")
        for field in ("sha256", "task_hash", "split_hash", "bundle_sha256"):
            digest = str(entry.get(field, ""))
            if not _is_sha256(digest):
                raise RuntimeError(f"prediction lineage lacks typed {field}: {path}")
        if not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] <= 0:
            raise RuntimeError(f"prediction lineage lacks typed size_bytes: {path}")
        bundle_path = _safe_relative(entry.get("bundle_path", ""))
        expected_lineage = task_lineage[canonical_details[:4]]
        if (
            bundle_path not in expected_bundle_paths
            or entry.get("bundle_sha256") != frozen_hashes[str(bundle_path)]
            or entry.get("task_hash") != expected_lineage["task_hash"]
            or entry.get("split_hash") != expected_lineage["split_hash"]
            or str(bundle_path) != expected_lineage["bundle_path"]
            or entry.get("bundle_sha256") != expected_lineage["bundle_sha256"]
        ):
            raise RuntimeError(f"prediction bundle lineage differs from freeze: {path}")
        directory_counts[str(parent)] += 1
        _assert_bound_file(
            path, expected_sha256=entry.get("sha256"),
            expected_size=entry.get("size_bytes"),
        )
        _add(selection, path, "CLUSTER_PREDICTION")
    if directory_counts != Counter({
        str(ALLOWED_PREDICTION_PREFIXES[0]): 4000,
        str(ALLOWED_PREDICTION_PREFIXES[1]): 1000,
    }):
        raise RuntimeError("cluster prediction CPU/TabM split is not 4,000/1,000")

    result_paths = {
        path for path in selection if path.parts[:2] == RESULT_ROOT.parts
    }
    _validate_result_path_policy(result_paths)
    disclosures = _validate_supersession_policy(result_paths)
    return selection, disclosures


def select_pre_release_sources() -> tuple[
    dict[PurePosixPath, set[str]], dict[str, Any], list[dict[str, Any]]
]:
    selection, disclosures = _select_result_sources()
    whole_file_superseded = _whole_file_superseded_paths()
    for path in STATIC_REPRODUCIBILITY_FILES:
        if path in whole_file_superseded:
            raise RuntimeError(f"static source is whole-file superseded: {path}")
        _add(selection, path, "REPRODUCIBILITY_SOURCE")
    for root, suffixes in TREE_RULES.items():
        for path in _iter_code_tree(root, suffixes):
            if path in whole_file_superseded:
                continue
            _add(selection, path, "REPRODUCIBILITY_SOURCE")
    paper_manifest, paper_paths = _validate_paper_build_manifest()
    for path in PAPER_STATIC_FILES:
        if (ROOT.joinpath(*path.parts)).is_file():
            if path in whole_file_superseded:
                raise RuntimeError(f"paper source is whole-file superseded: {path}")
            _add(selection, path, "PAPER_SOURCE")
    for path in paper_paths:
        if path in whole_file_superseded:
            raise RuntimeError(f"paper build consumes whole-file superseded input: {path}")
        if any(
            fragment in str(path).lower()
            for fragment in FORBIDDEN_RESULT_NAME_FRAGMENTS
        ):
            raise RuntimeError(f"paper build consumes pilot/preflight/checkpoint input: {path}")
        if path.parts[:1] == ("results",) and (
            path.parts[:2] != RESULT_ROOT.parts or path not in selection
        ):
            raise RuntimeError(
                f"paper build introduced a result outside the claim allowlist: {path}"
            )
        if path not in selection and path.parts[:2] != ("paper", "aaai27"):
            raise RuntimeError(f"paper build input escaped the public source closure: {path}")
        _add(
            selection, path,
            "PAPER_BUILD_MANIFEST" if path == PAPER_BUILD_MANIFEST else "PAPER_BOUND_INPUT",
        )
    forbidden = set(selection).intersection(whole_file_superseded)
    if forbidden:
        raise RuntimeError(
            f"whole-file superseded path entered final selection: {sorted(map(str, forbidden))}"
        )
    return selection, paper_manifest, disclosures


def compute_pre_release_inventory() -> tuple[
    dict[str, Any], dict[PurePosixPath, set[str]], dict[str, Any], list[dict[str, Any]]
]:
    selection, paper_manifest, disclosures = select_pre_release_sources()
    files = {
        str(path): {
            "sha256": sha256(_source(path)),
            "size_bytes": _source(path).stat().st_size,
            "roles": sorted(selection[path]),
        }
        for path in sorted(selection, key=str)
    }
    inventory = {
        "schema_version": 1,
        "policy_version": INVENTORY_POLICY_VERSION,
        "file_count": len(files),
        "total_bytes": sum(entry["size_bytes"] for entry in files.values()),
        "files": files,
    }
    inventory["sha256"] = _json_sha256(inventory)
    return inventory, selection, paper_manifest, disclosures


def _require_release_pass() -> tuple[
    Path, dict[str, Any], dict[str, Any], dict[PurePosixPath, set[str]],
    dict[str, Any], list[dict[str, Any]],
]:
    report = ROOT / "results/corrected_v2/release_validation.json"
    if not report.is_file():
        raise FileNotFoundError("final release_validation.json is absent")
    payload = _load_json(report)
    if payload.get("status") != "PASS":
        raise RuntimeError("release validator has not returned PASS")
    tests = payload.get("tests", {})
    if (
        payload.get("tests_skipped") is not False
        or tests.get("status") != "PASS"
        or tests.get("command") != "python -m pytest tests -q"
        or not re.search(r"\b[1-9][0-9]* passed\b", str(tests.get("tail", "")))
    ):
        raise RuntimeError("release validation tests were skipped or did not pass")
    test_checks = [
        entry for entry in payload.get("checks", []) if entry.get("name") == "tests"
    ]
    if test_checks != [{"name": "tests", "status": "PASS"}]:
        raise RuntimeError("release validation does not attest one non-skipped test run")

    claims = ROOT / "results/corrected_v2/paper_claims.json"
    canonical = ROOT / "results/corrected_v2/canonical_manifest.json"
    if not claims.is_file() or not canonical.is_file():
        raise FileNotFoundError("final claims or canonical manifest is absent")
    if _load_json(claims).get("evidence_tier") != "confirmatory":
        raise RuntimeError("paper claims are not confirmatory")
    claims_hash = sha256(claims)
    if payload.get("paper_claims_sha256") != claims_hash:
        raise RuntimeError("release report is stale relative to paper claims")
    macros = ROOT / "paper/aaai27/generated/result_macros.tex"
    if (
        payload.get("result_macros_path") != "paper/aaai27/generated/result_macros.tex"
        or not macros.is_file()
        or payload.get("result_macros_sha256") != sha256(macros)
        or f"% paper_claims.json sha256: {claims_hash}"
        not in macros.read_text(encoding="utf-8")
    ):
        raise RuntimeError("generated result macros are stale or unbound")
    canonical_payload = _load_json(canonical)
    if canonical_payload.get("status") != "CANONICAL" or canonical_payload.get("cells") != 27500:
        raise RuntimeError("canonical manifest is not the final 27,500-cell release")

    inventory, selection, paper_manifest, disclosures = compute_pre_release_inventory()
    expected_inventory = {
        "policy_version": inventory["policy_version"],
        "sha256": inventory["sha256"],
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
    }
    if payload.get("artifact_input_inventory") != expected_inventory:
        raise RuntimeError("release report lacks a fresh full artifact input inventory")
    if payload.get("paper_build_manifest_sha256") != sha256(_source(PAPER_BUILD_MANIFEST)):
        raise RuntimeError("release report is stale relative to the paper build manifest")
    return report, payload, inventory, selection, paper_manifest, disclosures


def _is_utf8_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _sanitize_text_bytes(data: bytes) -> tuple[bytes, int]:
    """Compatibility helper only; artifact assembly never mutates evidence bytes."""
    if not _is_utf8_text(data):
        return data, 0
    text = data.decode("utf-8")
    replacements = 0
    rules = (
        (re.compile(r"/Users/[^/\s,\"']+"), "<LOCAL_HOME>"),
        (re.compile(r"/home/[^/\s,\"']+"), "<LOCAL_HOME>"),
        (re.compile(r"(?i)(?<![A-Za-z0-9])(?:[A-Z]):[\\/][^\r\n]*"), "<WINDOWS_PATH>"),
        (re.compile(r"\b(?:10|192\.168)(?:\.\d{1,3}){2,3}\b"), "<PRIVATE_IP>"),
        (re.compile(r"\bLAPTOP-[A-Za-z0-9-]+\b", re.IGNORECASE), "<LOCAL_HOST>"),
    )
    for pattern, replacement in rules:
        text, count = pattern.subn(replacement, text)
        replacements += count
    return text.encode("utf-8"), replacements


def _binary_ascii_projection(data: bytes) -> bytes:
    """Expose printable binary metadata without treating numeric bytes as text."""
    return b"\n".join(re.findall(rb"[\x09\x0a\x0d\x20-\x7e]{4,}", data))


def _literal_identity_match(data: bytes, literal: bytes) -> bool:
    if not literal:
        return False
    return re.search(
        rb"(?<![A-Za-z0-9._-])" + re.escape(literal) + rb"(?![A-Za-z0-9._-])",
        data,
        re.IGNORECASE,
    ) is not None


def scan_public_files(paths: Iterable[Path], *, root: Path) -> dict[str, list[dict[str, str]]]:
    hits: dict[str, list[dict[str, str]]] = {
        "secret": [], "private_identity": [], "credential_filename": [],
    }
    local_user = Path.home().name.encode("utf-8").lower()
    if len(local_user) < 3 or local_user in {b"root", b"runner", b"nobody"}:
        local_user = b""
    local_host = socket.gethostname().encode("utf-8").lower()
    generic_local_host = b"local" + b"host"
    if len(local_host) < 4 or local_host in {
        generic_local_host, generic_local_host + b".localdomain",
    }:
        local_host = b""
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            hits["credential_filename"].append({"path": str(path), "rule": "non_regular_file"})
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.name.lower() in CREDENTIAL_FILENAMES or path.suffix.lower() in CREDENTIAL_SUFFIXES:
            hits["credential_filename"].append({"path": str(relative), "rule": "credential_filename"})
        data = path.read_bytes()
        if _is_utf8_text(data):
            content = data
        elif path.suffix.lower() == ".npz":
            # The independent verifier parses each NPZ member and string array.
            # Scanning compressed/numeric bytes directly creates false positives.
            content = b""
        else:
            content = _binary_ascii_projection(data)
        relative_bytes = str(relative).encode("utf-8")
        searchable = relative_bytes + b"\n" + content
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(searchable):
                hits["secret"].append({"path": str(relative), "rule": name})
        for name, pattern in PRIVATE_BYTE_PATTERNS:
            if pattern.search(searchable):
                hits["private_identity"].append({"path": str(relative), "rule": name})
        if _literal_identity_match(searchable, local_user):
            hits["private_identity"].append({"path": str(relative), "rule": "local_username"})
        if _literal_identity_match(searchable, local_host):
            hits["private_identity"].append({"path": str(relative), "rule": "local_hostname"})
    return hits


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination) or source.stat().st_size != destination.stat().st_size:
        raise RuntimeError(f"byte-exact artifact copy failed: {source}")


def build_artifact(destination: Path) -> dict[str, Any]:
    (
        report, validation, inventory, selection, paper_manifest, disclosures,
    ) = _require_release_pass()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    try:
        for relative in sorted(selection, key=str):
            _copy_exact(_source(relative), destination.joinpath(*relative.parts))
        report_relative = PurePosixPath("results/corrected_v2/release_validation.json")
        _copy_exact(report, destination.joinpath(*report_relative.parts))

        readme = destination / "ARTIFACT_README.md"
        readme.write_text(
            "# LeakBench-Tab corrected_v2 public artifact\n\n"
            "This byte-exact package is generated only after the fail-closed release, "
            "non-skipped test suite, fresh full input inventory, and independently rebuilt "
            "AAAI submission PDFs all pass. Raw natural datasets and their private local "
            "provenance files are excluded; deterministic public projections preserve every "
            "scientific field and bind typed private-to-public hashes.\n\n"
            "Three accepted ledgers are retained only as raw provenance. Their selector-scoped "
            "superseded rows are disclosed in ARTIFACT_MANIFEST.json and are replaced by the "
            "frozen M10 and diagnostic amendments. Pilot, preflight, checkpoint, inventory, "
            "snapshot, and whole-file superseded results are excluded.\n"
            "\nVerify an unpacked copy with `python scripts/run_corrected_v2_public_ci.py .`. "
            "Verification requires the pinned Python dependencies in "
            "`requirements-corrected-v2.txt` plus Poppler's `pdftotext` and `pdfinfo` "
            "and `pdffonts` (either on PATH or in the bundled Codex runtime).\n",
            encoding="utf-8",
        )
        scan_paths = [path for path in destination.rglob("*") if path.is_file()]
        scan_hits = scan_public_files(scan_paths, root=destination)
        if any(scan_hits.values()):
            raise RuntimeError(f"public artifact privacy/secret scan failed: {scan_hits}")

        files = {
            path.relative_to(destination).as_posix(): {
                "sha256": sha256(path), "size_bytes": path.stat().st_size,
            }
            for path in sorted(scan_paths)
        }
        result_files = sorted(
            name for name in files if PurePosixPath(name).parts[:2] == RESULT_ROOT.parts
        )
        manifest = {
            "schema_version": 2,
            "artifact": "LeakBench-Tab corrected_v2 AAAI-27 public submission evidence",
            "evidence_tier": "confirmatory",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_validation": {
                "path": str(report_relative),
                "sha256": sha256(report),
                "status": validation["status"],
                "tests_skipped": False,
            },
            "paper_build_manifest": {
                "path": str(PAPER_BUILD_MANIFEST),
                "sha256": sha256(_source(PAPER_BUILD_MANIFEST)),
                "schema_version": paper_manifest["schema_version"],
                "status": paper_manifest["status"],
            },
            "artifact_input_inventory": {
                "policy_version": inventory["policy_version"],
                "sha256": inventory["sha256"],
                "file_count": inventory["file_count"],
                "total_bytes": inventory["total_bytes"],
            },
            "selection_policy": {
                "version": INVENTORY_POLICY_VERSION,
                "results_are_explicit_allowlist": True,
                "result_files": result_files,
                "raw_provenance_only": disclosures,
                "non_numerical_attestations": [
                    {
                        "path": str(GPU_INTERIM_INCIDENT),
                        "status": "DOCUMENTED_NO_PROTOCOL_OR_CLAIM_POLICY_CHANGE",
                        "numerical_claim_source_allowed": False,
                    }
                ],
                "private_natural_provenance_included": False,
                "raw_natural_data_included": False,
                "pilot_results_included": False,
                "preflight_results_included": False,
                "checkpoints_included": False,
                "inventory_results_included": False,
                "whole_file_superseded_results_included": False,
            },
            "privacy_scan": {
                "status": "PASS",
                "independent_deep_verifier_required": True,
                "secret_rule_count": len(SECRET_PATTERNS),
                "private_identity_rule_count": len(PRIVATE_BYTE_PATTERNS),
                "generic_pattern_literal_exemptions": sorted(
                    str(path) for path in GENERIC_PATTERN_LITERAL_EXEMPTIONS
                    if str(path) in files
                ),
            },
            "files": files,
        }
        manifest_path = destination / "ARTIFACT_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        final_hits = scan_public_files([manifest_path], root=destination)
        if any(final_hits.values()):
            raise RuntimeError(f"artifact manifest privacy scan failed: {final_hits}")
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(destination / "scripts/verify_corrected_v2_public_artifact.py"),
                str(destination),
            ],
            cwd=destination,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "fresh artifact failed independent deep verification:\n"
                + "\n".join((completed.stderr or completed.stdout).splitlines()[-20:])
            )
        return manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default=str(DEFAULT_DESTINATION.relative_to(ROOT)))
    parser.add_argument("--zip", action="store_true", dest="make_zip")
    args = parser.parse_args(argv)
    destination = (ROOT / args.destination).resolve()
    try:
        destination.relative_to((ROOT / "release").resolve())
    except ValueError as exc:
        raise RuntimeError("artifact destination must remain under release/") from exc
    manifest = build_artifact(destination)
    manifest_path = destination / "ARTIFACT_MANIFEST.json"
    print(json.dumps({
        "destination": str(destination.relative_to(ROOT)),
        "files": len(manifest["files"]) + 1,
        "bytes": sum(item["size_bytes"] for item in manifest["files"].values()),
        "manifest_sha256": sha256(manifest_path),
    }, indent=2))
    if args.make_zip:
        archive = Path(shutil.make_archive(
            str(destination), "zip", root_dir=destination.parent,
            base_dir=destination.name,
        ))
        try:
            with tempfile.TemporaryDirectory(
                prefix="artifact-unpack-", dir=destination.parent
            ) as temporary:
                unpack_root = Path(temporary)
                with zipfile.ZipFile(archive) as package:
                    for member in package.infolist():
                        try:
                            _safe_relative(member.filename)
                        except ValueError as error:
                            raise RuntimeError(
                                f"unsafe generated ZIP member: {member.filename}"
                            ) from error
                        if stat.S_ISLNK(member.external_attr >> 16):
                            raise RuntimeError(
                                f"generated ZIP contains symlink: {member.filename}"
                            )
                    package.extractall(unpack_root)
                unpacked = unpack_root / destination.name
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(unpacked / "scripts/verify_corrected_v2_public_artifact.py"),
                        str(unpacked),
                        "--run-tests",
                    ],
                    cwd=unpacked,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "generated ZIP failed independent unpack verification:\n"
                        + "\n".join(
                            (completed.stderr or completed.stdout).splitlines()[-20:]
                        )
                    )
        except Exception:
            archive.unlink(missing_ok=True)
            raise
        print(json.dumps({
            "zip": str(archive.relative_to(ROOT)), "sha256": sha256(archive),
            "independent_unpack_verification": "PASS",
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
