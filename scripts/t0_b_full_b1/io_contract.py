"""T0-B Full-B1 I/O Contract — atomic writes, writer locks, safe stale cleanup.

All formal artifacts MUST be written through these helpers.
Direct Path.write_bytes() or open(path, "w") is forbidden for formal artifacts.
"""
from __future__ import annotations

import gzip, hashlib, io, json, os, re, secrets, socket, time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator


class WriterLockError(RuntimeError):
    pass


class ArtifactIntegrityError(RuntimeError):
    pass


# ─── temp file naming ───────────────────────────────────────────

def _tmp_name(target: Path) -> Path:
    nonce = secrets.token_hex(4)
    return target.parent / f".{target.name}.tmp.{os.getpid()}.{nonce}"


def parse_temp_owner_pid(path: Path) -> int | None:
    """Extract PID from temp filename: .<name>.tmp.<pid>.<nonce>"""
    m = re.search(r"\.tmp\.(\d+)\.[0-9a-f]+$", path.name)
    return int(m.group(1)) if m else None


# ─── process liveness ──────────────────────────────────────────

def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# ─── fsync helpers ──────────────────────────────────────────────

def fsync_parent_directory(path: Path) -> None:
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(fh) -> None:
    fh.flush()
    os.fsync(fh.fileno())


# ─── atomic write primitives ────────────────────────────────────

def atomic_write_bytes(target: Path, data: bytes, expected_sha: str | None = None) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(target)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            _fsync_file(f)
        actual_sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if expected_sha is not None and actual_sha != expected_sha:
            raise ArtifactIntegrityError(f"SHA mismatch: expected {expected_sha[:16]}… got {actual_sha[:16]}…")
        os.replace(tmp, target)
        fsync_parent_directory(target)
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def atomic_write_text(target: Path, text: str, expected_sha: str | None = None) -> str:
    return atomic_write_bytes(target, text.encode("utf-8"), expected_sha)


def atomic_write_gzip_text(target: Path, text: str, expected_sha: str | None = None) -> str:
    compressed = gzip.compress(text.encode("utf-8"), mtime=0)
    return atomic_write_bytes(target, compressed, expected_sha)


def atomic_write_dataframe_gzip(
    target: Path, df, columns: list[str], expected_sha: str | None = None
) -> str:
    import pandas as pd
    buf = io.StringIO()
    df.to_csv(buf, columns=columns, index=False, header=True)
    return atomic_write_gzip_text(target, buf.getvalue(), expected_sha)


def atomic_write_json(target: Path, obj: Any, expected_sha: str | None = None) -> str:
    text = json.dumps(obj, sort_keys=True, indent=2) + "\n"
    return atomic_write_text(target, text, expected_sha)


def validate_written_artifact(path: Path, expected_sha: str) -> bool:
    if not path.exists():
        return False
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return actual == expected_sha


# ─── stale temp cleanup (safe: age + PID liveness) ────────────

@dataclass
class StaleCleanupReport:
    removed_files: list[str] = field(default_factory=list)
    removed_count: int = 0
    skipped_young_files: list[str] = field(default_factory=list)
    skipped_alive_pid_files: list[str] = field(default_factory=list)
    skipped_unrecognized_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def cleanup_stale_temp_files(
    directory: Path,
    min_age_seconds: float = 3600,
    now: float | None = None,
) -> StaleCleanupReport:
    """Safely remove stale temp files.

    Rules:
    - File age < min_age_seconds: skip (young)
    - PID parseable and process alive: skip (active writer)
    - PID parseable, process dead, age >= min_age: remove
    - PID unparseable, age >= min_age * 2: remove (stricter for unrecognized)
    - .writer.lock: NEVER remove
    - Normal files: NEVER remove
    - Current process PID: NEVER remove
    """
    report = StaleCleanupReport()
    if not directory.exists():
        return report
    now = now if now is not None else time.time()
    current_pid = os.getpid()
    for f in directory.iterdir():
        if f.name == ".writer.lock":
            continue
        if not (f.name.startswith(".") and ".tmp." in f.name):
            continue
        # Check age
        try:
            mtime = f.stat().st_mtime
        except OSError as e:
            report.errors.append(f"{f.name}: stat error: {e}")
            continue
        age = now - mtime
        # Parse PID
        owner_pid = parse_temp_owner_pid(f)
        if owner_pid == current_pid:
            report.skipped_alive_pid_files.append(f.name)
            continue
        if owner_pid is not None:
            if is_process_alive(owner_pid):
                report.skipped_alive_pid_files.append(f.name)
                continue
            if age < min_age_seconds:
                report.skipped_young_files.append(f.name)
                continue
        else:
            # Unrecognized: stricter threshold
            if age < min_age_seconds * 2:
                report.skipped_unrecognized_files.append(f.name)
                continue
        # Safe to remove
        try:
            f.unlink()
            report.removed_files.append(f.name)
            report.removed_count += 1
        except OSError as e:
            report.errors.append(f"{f.name}: unlink error: {e}")
    return report


# ─── exclusive writer lock (token-based, fd-safe) ──────────────

@contextmanager
def exclusive_writer_lock(
    directory: Path, operation: str = "write"
) -> Generator[Path, None, None]:
    """Acquire exclusive writer lock with owner token for safe release."""
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".writer.lock"
    token = secrets.token_hex(8)
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise WriterLockError(f"Writer lock held for {directory} (operation: {operation})")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "operation": operation,
                "target_directory": str(directory),
                "token": token,
            }, fh, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        fd = None  # fd closed by context manager
        fsync_parent_directory(lock_path)
        yield lock_path
    finally:
        # Only remove if token matches (prevents removing foreign re-acquired lock)
        if lock_path.exists():
            try:
                content = lock_path.read_text()
                lock_data = json.loads(content)
                if lock_data.get("token") == token:
                    lock_path.unlink()
                    fsync_parent_directory(lock_path)
            except (json.JSONDecodeError, OSError):
                pass
