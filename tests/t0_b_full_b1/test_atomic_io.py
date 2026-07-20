"""T0-B Atomic I/O Tests."""
import hashlib, io, os, sys, tempfile
from pathlib import Path; import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_atomic_write_creates_file():
    from scripts.t0_b_full_b1.io_contract import atomic_write_gz
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.csv.gz"
        sha = atomic_write_gz(p, "a,b\n1,2\n")
        assert p.exists()
        assert p.stat().st_size > 0

def test_atomic_write_no_tmp_left():
    from scripts.t0_b_full_b1.io_contract import atomic_write_gz
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.csv.gz"
        atomic_write_gz(p, "x\n")
        assert not Path(td).joinpath("test.csv.gz.tmp").exists()

def test_atomic_write_deterministic():
    from scripts.t0_b_full_b1.io_contract import atomic_write_gz
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv.gz"
        s1 = atomic_write_gz(p, "a,b\n1,2\n")
        s2 = atomic_write_gz(p, "a,b\n1,2\n")
        assert s1 == s2

def test_atomic_write_json():
    from scripts.t0_b_full_b1.io_contract import atomic_write_json
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.json"
        sha = atomic_write_json(p, {"k":"v"})
        assert len(sha) == 64
