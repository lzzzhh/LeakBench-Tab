"""Tests for atomic I/O contract — real behavioral assertions, no stubs."""
import gzip, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_full_b1.io_contract import (
    atomic_write_bytes, atomic_write_text, atomic_write_gzip_text,
    atomic_write_json, atomic_write_dataframe_gzip,
    exclusive_writer_lock, WriterLockError,
    cleanup_stale_temp_files, validate_written_artifact,
    fsync_parent_directory, _tmp_name,
)


# ─── temp filename tests ───────────────────────────────────────

def test_tmp_name_contains_pid():
    target = Path("/tmp/test.csv.gz")
    tmp = _tmp_name(target)
    assert str(os.getpid()) in tmp.name

def test_tmp_name_contains_nonce():
    target = Path("/tmp/test.csv.gz")
    t1 = _tmp_name(target)
    t2 = _tmp_name(target)
    assert t1 != t2  # unique nonce each call

def test_tmp_name_starts_with_dot():
    target = Path("/tmp/data.json")
    tmp = _tmp_name(target)
    assert tmp.name.startswith(".")

def test_tmp_name_same_directory():
    target = Path("/some/dir/file.csv.gz")
    tmp = _tmp_name(target)
    assert tmp.parent == target.parent

def test_tmp_name_contains_tmp_marker():
    target = Path("/tmp/test.csv.gz")
    tmp = _tmp_name(target)
    assert ".tmp." in tmp.name


# ─── atomic write tests ────────────────────────────────────────

def test_atomic_write_bytes_creates_file():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        sha = atomic_write_bytes(p, b"hello")
        assert p.exists()
        assert p.read_bytes() == b"hello"
        assert len(sha) == 64

def test_atomic_write_bytes_no_tmp_left():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        atomic_write_bytes(p, b"data")
        tmps = [f for f in os.listdir(td) if ".tmp." in f]
        assert len(tmps) == 0

def test_atomic_write_bytes_sha_returned():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        sha = atomic_write_bytes(p, b"hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert sha == expected

def test_atomic_write_bytes_expected_sha_passes():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        expected = hashlib.sha256(b"hello").hexdigest()
        sha = atomic_write_bytes(p, b"hello", expected_sha=expected)
        assert sha == expected

def test_atomic_write_bytes_expected_sha_mismatch_raises():
    from scripts.t0_b_full_b1.io_contract import ArtifactIntegrityError
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        with pytest.raises(ArtifactIntegrityError):
            atomic_write_bytes(p, b"hello", expected_sha="0" * 64)

def test_atomic_write_text_utf8():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.txt"
        atomic_write_text(p, "héllo 世界")
        assert p.read_text(encoding="utf-8") == "héllo 世界"

def test_atomic_write_gzip_text_mtime_zero():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.csv.gz"
        atomic_write_gzip_text(p, "a,b\n1,2\n")
        with open(p, "rb") as f:
            f.read(2)  # magic (2 bytes: 0x1f 0x8b)
            f.read(1)  # method
            f.read(1)  # flags
            mtime = int.from_bytes(f.read(4), "little")
        assert mtime == 0

def test_atomic_write_gzip_text_decompresses():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.csv.gz"
        atomic_write_gzip_text(p, "a,b\n1,2\n")
        data = gzip.decompress(p.read_bytes()).decode("utf-8")
        assert data == "a,b\n1,2\n"

def test_atomic_write_json_sorted_keys():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.json"
        atomic_write_json(p, {"b": 2, "a": 1})
        text = p.read_text()
        assert text.index('"a"') < text.index('"b"')

def test_atomic_write_json_trailing_newline():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.json"
        atomic_write_json(p, {"a": 1})
        assert p.read_text().endswith("\n")

def test_atomic_write_dataframe_gzip():
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.csv.gz"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        sha = atomic_write_dataframe_gzip(p, df, ["a", "b"])
        assert len(sha) == 64
        data = gzip.decompress(p.read_bytes()).decode("utf-8")
        assert "a,b" in data
        assert "1,3" in data


# ─── exception cleanup ─────────────────────────────────────────

def test_exception_cleans_tmp():
    from scripts.t0_b_full_b1.io_contract import ArtifactIntegrityError
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        with pytest.raises(ArtifactIntegrityError):
            atomic_write_bytes(p, b"hello", expected_sha="0" * 64)
        tmps = [f for f in os.listdir(td) if ".tmp." in f]
        assert len(tmps) == 0

def test_interrupted_write_no_final_artifact():
    """If write fails, no final artifact should exist."""
    from scripts.t0_b_full_b1.io_contract import ArtifactIntegrityError
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        with pytest.raises(ArtifactIntegrityError):
            atomic_write_bytes(p, b"data", expected_sha="wrong")
        assert not p.exists()


# ─── stale tmp cleanup ─────────────────────────────────────────

def test_cleanup_stale_temp_files():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # create stale temp files
        (tdp / ".data.csv.gz.tmp.123.abc12345").write_bytes(b"stale")
        (tdp / ".other.tmp.456.def67890").write_bytes(b"stale2")
        # create a normal file
        (tdp / "normal.txt").write_text("keep me")
        removed = cleanup_stale_temp_files(tdp)
        assert removed == 2
        assert (tdp / "normal.txt").exists()

def test_cleanup_stale_no_normal_files_removed():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "data.csv.gz").write_bytes(b"keep")
        (tdp / "manifest.json").write_text("{}")
        removed = cleanup_stale_temp_files(tdp)
        assert removed == 0
        assert (tdp / "data.csv.gz").exists()
        assert (tdp / "manifest.json").exists()


# ─── writer lock tests ─────────────────────────────────────────

def test_first_writer_lock_succeeds():
    with tempfile.TemporaryDirectory() as td:
        with exclusive_writer_lock(Path(td), "test"):
            assert (Path(td) / ".writer.lock").exists()

def test_lock_released_after_success():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with exclusive_writer_lock(tdp, "test"):
            pass
        assert not (tdp / ".writer.lock").exists()

def test_lock_released_after_exception():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        try:
            with exclusive_writer_lock(tdp, "test"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert not (tdp / ".writer.lock").exists()

def test_second_writer_lock_fails():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with exclusive_writer_lock(tdp, "first"):
            with pytest.raises(WriterLockError):
                with exclusive_writer_lock(tdp, "second"):
                    pass

def test_lock_can_be_reacquired_after_release():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with exclusive_writer_lock(tdp, "first"):
            pass
        # Should succeed now that first lock is released
        with exclusive_writer_lock(tdp, "second"):
            assert (tdp / ".writer.lock").exists()


# ─── validate_written_artifact ─────────────────────────────────

def test_validate_written_artifact_correct():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        sha = atomic_write_bytes(p, b"hello")
        assert validate_written_artifact(p, sha)

def test_validate_written_artifact_wrong_sha():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.bin"
        atomic_write_bytes(p, b"hello")
        assert not validate_written_artifact(p, "0" * 64)

def test_validate_written_artifact_missing_file():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nonexistent.bin"
        assert not validate_written_artifact(p, "0" * 64)


# ─── determinism ───────────────────────────────────────────────

def test_gzip_deterministic_same_content_same_sha():
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.csv.gz"
        p2 = Path(td) / "b.csv.gz"
        s1 = atomic_write_gzip_text(p1, "a,b\n1,2\n")
        s2 = atomic_write_gzip_text(p2, "a,b\n1,2\n")
        assert s1 == s2

def test_json_deterministic_same_obj_same_sha():
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.json"
        p2 = Path(td) / "b.json"
        s1 = atomic_write_json(p1, {"b": 2, "a": 1})
        s2 = atomic_write_json(p2, {"b": 2, "a": 1})
        assert s1 == s2
