"""T0-B Full-B1 I/O Contract — atomic writes, writer locks, stale cleanup.

All formal artifacts MUST be written through these helpers.
Direct Path.write_bytes() or open(path, "w") is forbidden for formal artifacts.
"""
from __future__ import annotations

import gzip, hashlib, io, json, os, secrets, socket, time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


class WriterLockError(RuntimeError):
    """Raised when exclusive writer lock cannot be acquired."""
    pass


class ArtifactIntegrityError(RuntimeError):
    """Raised when a written artifact fails integrity verification."""
    pass


# ─── temp file naming ───────────────────────────────────────────

def _tmp_name(target: Path) -> Path:
    """Generate unique temp filename: .<name>.tmp.<pid>.<nonce>"""
    nonce = secrets.token_hex(4)  # 8-char hex
    return target.parent / f".{target.name}.tmp.{os.getpid()}.{nonce}"


# ─── fsync helpers ──────────────────────────────────────────────

def fsync_parent_directory(path: Path) -> None:
    """fsync the parent directory of *path* to ensure rename durability."""
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(fh) -> None:
    """fsync an open file handle."""
    fh.flush()
    os.fsync(fh.fileno())


# ─── atomic write primitives ────────────────────────────────────

def atomic_write_bytes(target: Path, data: bytes, expected_sha: str | None = None) -> str:
    """Atomically write bytes to *target*.

    Returns SHA-256 of the written file.
    Raises ArtifactIntegrityError if expected_sha is provided and doesn't match.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(target)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            _fsync_file(f)
        # verify SHA before replace
        actual_sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if expected_sha is not None and actual_sha != expected_sha:
            raise ArtifactIntegrityError(f"SHA mismatch: expected {expected_sha[:16]}… got {actual_sha[:16]}…")
        os.replace(tmp, target)
        fsync_parent_directory(target)
        # re-verify after replace
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def atomic_write_text(target: Path, text: str, expected_sha: str | None = None) -> str:
    """Atomically write UTF-8 text."""
    return atomic_write_bytes(target, text.encode("utf-8"), expected_sha)


def atomic_write_gzip_text(target: Path, text: str, expected_sha: str | None = None) -> str:
    """Atomically write gzip-compressed text with mtime=0."""
    compressed = gzip.compress(text.encode("utf-8"), mtime=0)
    return atomic_write_bytes(target, compressed, expected_sha)


def atomic_write_dataframe_gzip(
    target: Path, df, columns: list[str], expected_sha: str | None = None
) -> str:
    """Atomically write a DataFrame as gzip CSV with mtime=0."""
    import pandas as pd
    buf = io.StringIO()
    df.to_csv(buf, columns=columns, index=False, header=True)
    return atomic_write_gzip_text(target, buf.getvalue(), expected_sha)


def atomic_write_json(target: Path, obj: Any, expected_sha: str | None = None) -> str:
    """Atomically write JSON with sort_keys=True, indent=2, trailing newline."""
    text = json.dumps(obj, sort_keys=True, indent=2) + "\n"
    return atomic_write_text(target, text, expected_sha)


def validate_written_artifact(path: Path, expected_sha: str) -> bool:
    """Verify that a file on disk has the expected SHA-256."""
    if not path.exists():
        return False
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return actual == expected_sha


# ─── stale temp cleanup ────────────────────────────────────────

def cleanup_stale_temp_files(directory: Path, prefix: str = ".") -> int:
    """Remove stale temp files (starting with . and containing .tmp.) in *directory*.

    Returns count of files removed.
    Only removes files matching pattern: .*.tmp.*
    Does NOT remove files that don't match the temp pattern.
    """
    removed = 0
    if not directory.exists():
        return removed
    for f in directory.iterdir():
        if f.name.startswith(".") and ".tmp." in f.name:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# ─── exclusive writer lock ─────────────────────────────────────

@contextmanager
def exclusive_writer_lock(
    directory: Path, operation: str = "write"
) -> Generator[Path, None, None]:
    """Acquire an exclusive writer lock for *directory*.

    Uses O_CREAT | O_EXCL for atomic lock acquisition.
    Raises WriterLockError if the lock is already held.
    Lock is released in finally block.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".writer.lock"
    try:
        fd = os.open(
            str(lock_path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError:
        raise WriterLockError(
            f"Writer lock held for {directory} (operation: {operation})"
        )
    try:
        lock_content = json.dumps({
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "operation": operation,
            "target_directory": str(directory),
        }, sort_keys=True)
        os.write(fd, lock_content.encode("utf-8"))
        _fsync_file(os.fdopen(fd, "w"))
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
