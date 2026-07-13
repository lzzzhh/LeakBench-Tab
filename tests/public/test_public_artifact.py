import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.build_corrected_v2_artifact import scan_public_files


ROOT = Path.cwd()
PUBLIC = ROOT / "results/corrected_v2/public_natural"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_natural_projection_is_self_contained_and_hash_bound():
    manifest = json.loads(
        (PUBLIC / "public_natural_provenance_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "PUBLIC_NATURAL_PROVENANCE_PROJECTED"
    assert manifest["raw_natural_data_included"] is False
    assert manifest["all_scientific_invariants_passed"] is True
    for entry in manifest["public_outputs"].values():
        path = ROOT / entry["path"]
        assert path.is_file()
        assert digest(path) == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]
    for entry in manifest["private_provenance"]["artifacts"].values():
        assert not (ROOT / entry["path"]).exists() or not (
            ROOT / "ARTIFACT_MANIFEST.json"
        ).exists()


def test_private_and_public_natural_scientific_values_match_when_private_inputs_exist():
    private_cells = ROOT / "results/corrected_v2/natural_cells.csv"
    private_tasks = ROOT / "results/corrected_v2/natural_task_summary.csv"
    if not private_cells.is_file() or not private_tasks.is_file():
        pytest.skip("private natural provenance is intentionally unavailable in public artifact")
    pd.testing.assert_frame_equal(
        pd.read_csv(private_cells), pd.read_csv(PUBLIC / "natural_cells.csv"),
        check_exact=True,
    )
    private = pd.read_csv(private_tasks)
    public = pd.read_csv(PUBLIC / "natural_task_summary.csv")
    scientific = [column for column in private.columns if column not in {"source", "lineage"}]
    pd.testing.assert_frame_equal(private[scientific], public[scientific], check_exact=True)


def test_expanded_scanner_rejects_composed_secret_and_private_endpoint(tmp_path):
    sample = tmp_path / "sample.txt"
    secret = ("AK" + "IA" + "A" * 16).encode()
    endpoint = ("ssh user@" + "192" + ".168.1.111").encode()
    sample.write_bytes(secret + b"\n" + endpoint)
    hits = scan_public_files([sample], root=tmp_path)
    assert hits["secret"]
    assert hits["private_identity"]


def test_actual_unpacked_artifact_reverifies_without_private_sources():
    if not (ROOT / "ARTIFACT_MANIFEST.json").is_file():
        pytest.skip("runs after the real public artifact has been unpacked")
    before = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*") if path.is_file()
    }
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_corrected_v2_public_artifact.py",
            ".",
            "--skip-deep-archive-scan",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert (
        json.loads(completed.stdout)["status"]
        == "PARTIAL_DEEP_ARCHIVE_SCAN_SKIPPED"
    )
    after = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*") if path.is_file()
    }
    assert after == before
    assert not any("__pycache__" in path.split("/") for path in after)
