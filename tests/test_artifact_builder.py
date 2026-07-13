import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
import scripts.build_corrected_v2_artifact as builder
import scripts.verify_corrected_v2_public_artifact as public_verifier
from scripts.verify_corrected_v2_public_artifact import verify_npz

from scripts.build_corrected_v2_artifact import (
    PRIVATE_BYTE_PATTERNS,
    PRIVATE_NATURAL_PATHS,
    PUBLIC_NATURAL_PATHS,
    RAW_PROVENANCE_ONLY,
    SECRET_PATTERNS,
    _copy_exact,
    _sanitize_text_bytes,
    _whole_file_superseded_paths,
    _validate_gpu_interim_incident,
    scan_public_files,
    sha256,
)


def test_artifact_builder_is_allowlisted_fail_closed_and_requires_fresh_release():
    source = Path("scripts/build_corrected_v2_artifact.py").read_text(encoding="utf-8")
    assert "release validator has not returned PASS" in source
    assert "release validation tests were skipped or did not pass" in source
    assert "release report lacks a fresh full artifact input inventory" in source
    assert "paper build manifest is not submission-ready PASS" in source
    assert "results_are_explicit_allowlist" in source
    assert "_iter_tree(\"results/corrected_v2\"" not in source
    assert "_sanitize_text_bytes(original)" not in source
    assert len(PRIVATE_NATURAL_PATHS) == 5
    assert len(PUBLIC_NATURAL_PATHS) == 5


def test_selector_level_supersession_is_typed_raw_provenance_only():
    assert RAW_PROVENANCE_ONLY == {
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
    assert PurePosixPath(
        "experiments/leakbench/run_meta_tier.py"
    ) in _whole_file_superseded_paths()


def test_cluster_lineage_loaders_bind_canonical_task_identity(tmp_path):
    canonical = tmp_path / "canonical.csv"
    canonical.write_text(
        "run_id,dataset_id,mechanism,strength,seed,model\n"
        "run-1,panel_00,M08,S1,13.0,lr\n",
        encoding="utf-8",
    )
    digest_a, digest_b, digest_c = "a" * 64, "b" * 64, "c" * 64
    tasks = tmp_path / "tasks.csv"
    tasks.write_text(
        "dataset_id,dataset_namespace,mechanism,strength,seed,task_hash,"
        "split_hash,bundle_path,bundle_sha256\n"
        f"panel_00,confirmatory,M08,S1,13,{digest_a},{digest_b},"
        f"results/corrected_v2/task_bundles/panel_00.npz,{digest_c}\n",
        encoding="utf-8",
    )
    assert builder._load_canonical_cluster_contract(canonical) == {
        "run-1": ("panel_00", "M08", "S1", "13", "lr")
    }
    assert builder._load_frozen_task_lineage(tasks) == {
        ("panel_00", "M08", "S1", "13"): {
            "task_hash": digest_a,
            "split_hash": digest_b,
            "bundle_path": "results/corrected_v2/task_bundles/panel_00.npz",
            "bundle_sha256": digest_c,
        }
    }


def test_gpu_interim_access_incident_locks_all_adaptive_responses():
    _validate_gpu_interim_incident()


def test_artifact_copy_is_byte_exact_and_never_redacts(tmp_path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "out/destination.bin"
    payload = b"\x00\xffscientific evidence\n" * 17
    source.write_bytes(payload)
    _copy_exact(source, destination)
    assert destination.read_bytes() == payload
    assert sha256(destination) == sha256(source)


def test_source_rejects_symlinked_parent_that_escapes_root(tmp_path, monkeypatch):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "private.txt").write_text("private", encoding="utf-8")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="symlinked source path"):
        builder._source(PurePosixPath("linked/private.txt"))


def test_expanded_scanner_detects_multiple_secret_and_private_families(tmp_path):
    sample = tmp_path / "payload.txt"
    values = [
        "hf" + "_" + "a" * 24,
        "github" + "_pat_" + "a" * 42,
        "AK" + "IA" + "A" * 16,
        "ssh user@" + "10" + ".1.2.3",
    ]
    sample.write_text("\n".join(values), encoding="utf-8")
    hits = scan_public_files([sample], root=tmp_path)
    assert {entry["rule"] for entry in hits["secret"]} >= {
        "huggingface", "github", "aws_access_key",
    }
    assert any(entry["rule"] == "rfc1918_10" for entry in hits["private_identity"])
    assert len(SECRET_PATTERNS) >= 10
    assert len(PRIVATE_BYTE_PATTERNS) >= 8


def test_scanner_has_no_whole_file_exemptions_and_scans_binary_metadata(tmp_path):
    scanner = tmp_path / "scripts/build_corrected_v2_artifact.py"
    scanner.parent.mkdir()
    scanner.write_text(
        "/Users/anotherperson/project\nanother.person@example.com\n192.168.50.7\n",
        encoding="utf-8",
    )
    opaque = tmp_path / "opaque.bin"
    opaque.write_bytes(
        b"\x00hf_" + b"a" * 24 + b"\x00/Users/anotherperson/project\x00"
    )
    hits = scan_public_files([scanner, opaque], root=tmp_path)
    assert builder.GENERIC_PATTERN_LITERAL_EXEMPTIONS == set()
    assert {entry["rule"] for entry in hits["private_identity"]} >= {
        "macos_home", "email_address", "rfc1918_192",
    }
    assert any(entry["rule"] == "huggingface" for entry in hits["secret"])


def test_scanner_is_portable_when_public_verifier_runs_as_root(tmp_path, monkeypatch):
    sample = tmp_path / "source.py"
    sample.write_text("scratch_root = root / value\n", encoding="utf-8")

    class RootHome:
        @staticmethod
        def home():
            return Path("/root")

    monkeypatch.setattr(builder, "Path", RootHome)
    hits = scan_public_files([sample], root=tmp_path)
    assert not any(
        entry["rule"] == "local_username" for entry in hits["private_identity"]
    )


def test_deep_npz_scanner_rejects_secret_inside_string_array(tmp_path):
    path = tmp_path / "payload.npz"
    token = "hf" + "_" + "a" * 24
    np.savez(path, metadata=np.asarray([token]))
    with pytest.raises(ValueError, match="huggingface"):
        verify_npz(
            path, secret_patterns=SECRET_PATTERNS,
            private_patterns=PRIVATE_BYTE_PATTERNS,
        )


def test_deep_npz_scanner_rejects_secret_in_member_name(tmp_path):
    path = tmp_path / "payload.npz"
    token = "hf" + "_" + "a" * 24
    with public_verifier.zipfile.ZipFile(path, "w") as archive:
        archive.writestr(token + ".txt", b"metadata")
    with pytest.raises(ValueError, match="huggingface"):
        verify_npz(
            path, secret_patterns=SECRET_PATTERNS,
            private_patterns=PRIVATE_BYTE_PATTERNS,
        )


def test_deep_npz_scanner_rejects_secret_in_zip_comment_and_numeric_dtype(tmp_path):
    token = "hf" + "_" + "a" * 24
    comment_path = tmp_path / "comment.npz"
    np.savez(comment_path, values=np.asarray([1.0]))
    with public_verifier.zipfile.ZipFile(comment_path, "a") as archive:
        archive.comment = token.encode("utf-8")
    with pytest.raises(ValueError, match="huggingface"):
        verify_npz(
            comment_path, secret_patterns=SECRET_PATTERNS,
            private_patterns=PRIVATE_BYTE_PATTERNS,
        )

    dtype_path = tmp_path / "dtype.npz"
    np.savez(dtype_path, values=np.zeros(1, dtype=[(token, "f8")]))
    with pytest.raises(ValueError, match="huggingface"):
        verify_npz(
            dtype_path, secret_patterns=SECRET_PATTERNS,
            private_patterns=PRIVATE_BYTE_PATTERNS,
        )


def test_poppler_tool_falls_back_to_bundled_codex_runtime(tmp_path, monkeypatch):
    executable = (
        tmp_path
        / ".cache/codex-runtimes/runtime/dependencies/native/poppler/poppler/bin/pdftotext"
    )
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(public_verifier.shutil, "which", lambda _: None)
    assert public_verifier.find_poppler_tool("pdftotext") == str(executable)


def test_compatibility_sanitizer_is_not_used_for_artifact_copy():
    original = (
        (
            "/" + "Users/alice/project /" + "home/bob/work C:"
            + "\\Users\\carol\\repo"
        ).encode()
    )
    public, replacements = _sanitize_text_bytes(original)
    assert replacements >= 3
    assert b"alice" not in public and b"bob" not in public and b"carol" not in public


def test_paper_build_manifest_contract_is_strictly_consumed(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    claims = PurePosixPath("results/corrected_v2/paper_claims.json")
    macros = PurePosixPath("paper/aaai27/generated/result_macros.tex")
    generated = {
        *builder.PAPER_BUILD_STATIC_INPUTS,
        claims,
        macros,
        *builder.EXPECTED_PAPER_FIGURES,
        *builder.EXPECTED_PAPER_TABLES,
    }
    for relative in generated:
        name = str(relative)
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    figure_manifest_path = PurePosixPath(
        "paper/aaai27/figures/generated/figure_manifest.json"
    )
    table_manifest_path = PurePosixPath(
        "paper/aaai27/generated/result_tables_manifest.json"
    )
    figure_manifest = {
        "figure_sha256": {
            str(path): hashlib.sha256((tmp_path / str(path)).read_bytes()).hexdigest()
            for path in builder.EXPECTED_PAPER_FIGURES
        },
        "source_sha256": {
            str(claims): hashlib.sha256((tmp_path / str(claims)).read_bytes()).hexdigest()
        },
    }
    table_manifest = {
        "table_sha256": {
            str(path): hashlib.sha256((tmp_path / str(path)).read_bytes()).hexdigest()
            for path in builder.EXPECTED_PAPER_TABLES
        },
        "source_sha256": {
            str(claims): hashlib.sha256((tmp_path / str(claims)).read_bytes()).hexdigest()
        },
    }
    for relative, payload in (
        (figure_manifest_path, figure_manifest),
        (table_manifest_path, table_manifest),
    ):
        path = tmp_path / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    required = {
        *generated,
        figure_manifest_path,
        table_manifest_path,
    }
    outputs = {}
    for logical in ("main", "supplement"):
        path = tmp_path / f"paper/aaai27/output/pdf/{logical}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((logical + "-pdf").encode())
        outputs[logical] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "pages": 9 if logical == "main" else 4,
            "fonts_embedded": True,
            "type3_fonts": False,
            "anonymous_author_line": True,
            "private_text_scan": "PASS",
            "blocked_marker_scan": "PASS",
        }
    manifest = {
        "schema_version": "leakbench.paper-build.v1",
        "status": "PASS",
        "submission_ready": True,
        "inputs": {
            str(name): hashlib.sha256((tmp_path / str(name)).read_bytes()).hexdigest()
            for name in required
        },
        "outputs": outputs,
        "checks": {
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
        },
    }
    path = tmp_path / str(builder.PAPER_BUILD_MANIFEST)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    observed, selected = builder._validate_paper_build_manifest()
    assert observed == manifest
    assert builder.PAPER_BUILD_MANIFEST in selected
    assert len(selected) == len(required) + 3

    unexpected = tmp_path / "paper/aaai27/private_notes.txt"
    unexpected.write_text("not an input to the reproducible build", encoding="utf-8")
    manifest["inputs"]["paper/aaai27/private_notes.txt"] = hashlib.sha256(
        unexpected.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="input closure changed"):
        builder._validate_paper_build_manifest()
