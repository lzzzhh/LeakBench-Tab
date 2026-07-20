"""T0-B Full-B1 I/O Contract — atomic writes with tmp/flush/fsync/os.replace."""
from __future__ import annotations
import gzip, hashlib, io, os, shutil
from pathlib import Path


def atomic_write_gz(path: Path, data: str) -> str:
    """Write gzip atomically: tmp → flush → fsync → os.replace → parent fsync.

    Returns SHA-256 of the final compressed file.
    """
    compressed = gzip.compress(data.encode("utf-8"), mtime=0)
    sha = hashlib.sha256(compressed).hexdigest()

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(compressed)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)
    # fsync parent directory
    fd = os.open(str(path.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

    return sha


def atomic_write_json(path: Path, obj: dict) -> str:
    """Write JSON atomically."""
    import json
    data = json.dumps(obj, indent=2)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
