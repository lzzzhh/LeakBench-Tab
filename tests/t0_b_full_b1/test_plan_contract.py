"""Plan contract tests."""
import gzip, hashlib, io, json, shutil, subprocess, sys, tempfile
from pathlib import Path; import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT / "scripts/t0_b_full_b1/run_full_b1_shard.py")
SYNTHETIC_PLAN_DIR = ROOT / "results/edbt_t0_b_full_b1_preflight/synthetic_full_contract"


def _copy_complete_synthetic_plan(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for name in [
        "full_b1_plan_manifest.json",
        "full_b1_key_plan.jsonl.gz",
        "full_b1_run_plan.jsonl.gz",
        "full_b1_shard_plan.json",
        "synthetic_policy_group_mapping.jsonl.gz",
        "synthetic_semantic_evaluation_mapping.jsonl.gz",
    ]:
        shutil.copy2(SYNTHETIC_PLAN_DIR / name, destination / name)
    return destination / "full_b1_plan_manifest.json"

def test_plan_5500_keys():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    assert len([l for l in data.strip().split("\n")]) == 5500

def test_plan_full_sha_ids():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    for line in data.strip().split("\n")[:10]:
        assert len(json.loads(line)["canonical_key_id"]) == 64

def test_plan_counts():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    runs = [json.loads(l) for l in data.strip().split("\n")]
    assert len(runs) == 803000
    assert len([r for r in runs if r["run_type"]=="baseline"]) == 11000
    assert len([r for r in runs if r["run_type"]=="governed"]) == 792000

def test_run_ids_unique():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    ids = set()
    for line in data.strip().split("\n"):
        rid = json.loads(line)["run_id"]
        assert rid not in ids; ids.add(rid)

def test_gzip_deterministic():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    s1 = hashlib.sha256((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
    s2 = hashlib.sha256((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
    assert s1 == s2

def test_shard_balance():
    with open(ROOT/"results/edbt_t0_b_full_b1_preflight/full_b1_shard_plan.json") as f:
        sp = json.load(f)
    counts = [v["count"] for v in sp["shard_stats"].values()]
    assert max(counts) - min(counts) <= 1


def test_formal_plan_passes_shared_production_contract():
    from scripts.t0_b_full_b1.merge_contract import (
        validate_plan_schema, validate_plan, validate_global_scope,
    )
    pref = ROOT / "results/edbt_t0_b_full_b1_preflight"
    manifest = json.loads((pref / "full_b1_plan_manifest.json").read_text())
    errors = validate_plan_schema(manifest, "production")
    plan_errors, keys, runs = validate_plan(manifest, pref)
    errors.extend(plan_errors)
    if not plan_errors:
        errors.extend(validate_global_scope(manifest, keys, runs))
    assert errors == []
    assert len(keys) == 5500
    assert len(runs) == 803000
    assert {row["execution_contract_version"] for row in runs} == {"v1"}
    declared_files = {
        "policy_mapping_sha256": ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz",
        "semantic_mapping_sha256": ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz",
        "shard_plan_sha256": pref / "full_b1_shard_plan.json",
    }
    for field, path in declared_files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[field]


def test_formal_plan_receipt_binds_manifest_and_tool_seal():
    pref = ROOT / "results/edbt_t0_b_full_b1_preflight"
    manifest_path = pref / "full_b1_plan_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((pref / "full_b1_plan_receipt.json").read_text())
    assert receipt["plan_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert receipt["tool_seal_sha"] == manifest["tool_seal_sha"]
    assert len(manifest["tool_seal_sha"]) == 40


@pytest.mark.parametrize(
    ("filename", "expected_error"),
    [
        ("synthetic_policy_group_mapping.jsonl.gz", "policy mapping SHA mismatch"),
        ("synthetic_semantic_evaluation_mapping.jsonl.gz", "semantic mapping SHA mismatch"),
        ("full_b1_shard_plan.json", "shard plan SHA mismatch"),
    ],
)
def test_runner_validate_only_rejects_declared_input_mutation(
    filename, expected_error,
):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        plan_path = _copy_complete_synthetic_plan(root / "plan")
        target = plan_path.parent / filename
        payload = bytearray(target.read_bytes())
        assert payload
        payload[0] ^= 1
        target.write_bytes(payload)
        output = root / "output"

        result = subprocess.run(
            [
                sys.executable, RUNNER,
                "--plan-manifest", str(plan_path),
                "--shard-id", "0",
                "--output-dir", str(output),
                "--synthetic", "--validate-only",
            ],
            capture_output=True, text=True, cwd=ROOT,
        )

        assert result.returncode != 0
        assert expected_error in result.stdout
        assert not output.exists()


@pytest.mark.parametrize(
    "field",
    ["policy_mapping_sha256", "semantic_mapping_sha256", "shard_plan_sha256"],
)
def test_runner_validate_only_rejects_missing_declared_sha(field):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        plan_path = _copy_complete_synthetic_plan(root / "plan")
        manifest = json.loads(plan_path.read_text())
        del manifest[field]
        plan_path.write_text(json.dumps(manifest))
        output = root / "output"

        result = subprocess.run(
            [
                sys.executable, RUNNER,
                "--plan-manifest", str(plan_path),
                "--shard-id", "0",
                "--output-dir", str(output),
                "--synthetic", "--validate-only",
            ],
            capture_output=True, text=True, cwd=ROOT,
        )

        assert result.returncode != 0
        assert f"plan manifest missing required field: {field}" in result.stdout
        assert not output.exists()


@pytest.mark.parametrize(
    "field",
    [
        "key_plan_sha256", "run_plan_sha256", "shard_plan_sha256",
        "policy_mapping_sha256", "semantic_mapping_sha256",
    ],
)
@pytest.mark.parametrize("invalid", ["a" * 63, "A" * 64, 7, None])
def test_declared_plan_sha_schema_is_exact_lowercase_hex64(field, invalid):
    from scripts.t0_b_full_b1.merge_contract import validate_plan_schema
    manifest = json.loads(
        (SYNTHETIC_PLAN_DIR / "full_b1_plan_manifest.json").read_text()
    )
    manifest[field] = invalid
    errors = validate_plan_schema(manifest, "synthetic")
    assert any(
        f"plan manifest {field} must be a 64-char hex string" in error
        for error in errors
    )
