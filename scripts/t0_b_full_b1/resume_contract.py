"""T0-B Resume Contract — validation classification, quarantine, controlled repair."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import hashlib, json, os, secrets
from datetime import datetime, timezone
from pathlib import Path

from scripts.t0_b_full_b1.fragment_contract import CompletedKeyValidation
from scripts.t0_b_full_b1.io_contract import atomic_write_json


class ResumeReasonCode(str, Enum):
    FRAGMENT_SHA_MISMATCH = "fragment_sha_mismatch"
    RECEIPT_MISSING = "receipt_missing"
    RECEIPT_CORRUPT = "receipt_corrupt"
    MANIFEST_MISSING = "manifest_missing"
    MANIFEST_CORRUPT = "manifest_corrupt"
    RUN_ID_MISMATCH = "run_id_mismatch"
    SELECTION_CLOSURE_FAILURE = "selection_closure_failure"
    REALIZED_COST_FAILURE = "realized_cost_failure"
    SEMANTIC_ATOMICITY_FAILURE = "semantic_atomicity_failure"
    FAILURE_ROWS_PRESENT = "failure_rows_present"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedValidationFailure:
    canonical_key_id: str
    reason_code: ResumeReasonCode
    validation_errors: tuple[str, ...]
    repairable: bool


def classify_completed_key_failure(
    canonical_key_id: str,
    validation: CompletedKeyValidation,
) -> ClassifiedValidationFailure:
    """Classify validation failure. Only pure fragment SHA mismatch is repairable."""
    errors = tuple(validation.errors)

    sha_patterns = ["baseline sha mismatch", "governed sha mismatch", "selection sha mismatch", "failure sha mismatch"]
    sha_errors = [e for e in errors if any(p in e.lower() for p in sha_patterns)]
    non_sha_errors = [e for e in errors if e not in sha_errors]

    # Repairable only if ALL errors are fragment SHA mismatches (no mixed errors)
    if sha_errors and not non_sha_errors:
        return ClassifiedValidationFailure(
            canonical_key_id=canonical_key_id,
            reason_code=ResumeReasonCode.FRAGMENT_SHA_MISMATCH,
            validation_errors=errors,
            repairable=True,
        )

    # Classify non-SHA errors
    if non_sha_errors:
        for err in non_sha_errors:
            err_lower = err.lower()
            if "receipt missing" in err_lower:
                return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.RECEIPT_MISSING, errors, False)
            if "receipt corrupt" in err_lower or "receipt" in err_lower:
                return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.RECEIPT_CORRUPT, errors, False)
            if "manifest missing" in err_lower:
                return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.MANIFEST_MISSING, errors, False)
            if "manifest corrupt" in err_lower:
                return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.MANIFEST_CORRUPT, errors, False)
            if "run id" in err_lower or "duplicate run" in err_lower or "missing run" in err_lower or "extra run" in err_lower or "null run" in err_lower:
                return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.RUN_ID_MISMATCH, errors, False)
            if "selection multiset" in err_lower or "selection closure" in err_lower:
                return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.SELECTION_CLOSURE_FAILURE, errors, False)
            if "realized" in err_lower and "cost" in err_lower:
                return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.REALIZED_COST_FAILURE, errors, False)
            if "semantic" in err_lower or "partial group" in err_lower or "atomic" in err_lower or "m09" in err_lower or "leak union" in err_lower:
                return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.SEMANTIC_ATOMICITY_FAILURE, errors, False)
            if "failure row" in err_lower:
                return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.FAILURE_ROWS_PRESENT, errors, False)

    return ClassifiedValidationFailure(canonical_key_id, ResumeReasonCode.UNKNOWN, errors, False)


@dataclass(frozen=True)
class QuarantineRecord:
    canonical_key_id: str
    reason_code: ResumeReasonCode
    source_directory: Path
    quarantine_directory: Path
    quarantined_utc: str
    artifact_shas: dict[str, str | None]


def quarantine_invalid_key(
    output_dir: Path,
    canonical_key_id: str,
    reason_code: ResumeReasonCode,
    validation_errors: Sequence[str],
) -> QuarantineRecord:
    """Atomically move entire key fragment directory to quarantine via os.replace."""
    source_dir = output_dir / "key_fragments" / canonical_key_id
    if not source_dir.exists():
        raise RuntimeError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise RuntimeError(f"Source is not a directory: {source_dir}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    nonce = secrets.token_hex(4)
    target_parent = output_dir / "quarantine" / canonical_key_id / reason_code.value
    target_parent.mkdir(parents=True, exist_ok=True)
    target_dir = target_parent / f"{ts}_{nonce}"

    # Record artifact SHAs before moving
    artifact_shas = {}
    for fname in ["baseline.csv.gz", "governed.csv.gz", "selection.csv.gz", "failure.csv.gz",
                  "fragment_manifest.json", "completion_receipt.json"]:
        fp = source_dir / fname
        artifact_shas[fname] = hashlib.sha256(fp.read_bytes()).hexdigest() if fp.exists() else None

    # Single atomic directory rename
    os.replace(source_dir, target_dir)

    # Verify post-move state
    if source_dir.exists():
        raise RuntimeError(f"Source directory still exists after os.replace: {source_dir}")
    if not target_dir.exists():
        raise RuntimeError(f"Target directory does not exist after os.replace: {target_dir}")
    if not target_dir.is_dir():
        raise RuntimeError(f"Target is not a directory: {target_dir}")

    # Fsync parent directories
    for d in [source_dir.parent, target_dir.parent]:
        try:
            fd = os.open(str(d), os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass  # Some platforms don't support directory fsync

    # Write quarantine receipt in target_dir
    receipt = {
        "schema_version": 1,
        "canonical_key_id": canonical_key_id,
        "reason_code": reason_code.value,
        "validation_errors": list(validation_errors),
        "source_directory": str(source_dir),
        "quarantine_directory": str(target_dir),
        "quarantined_utc": datetime.now(timezone.utc).isoformat(),
        "original_artifact_sha256": artifact_shas,
    }
    try:
        atomic_write_json(target_dir / "quarantine_receipt.json", receipt)
    except Exception:
        raise RuntimeError(f"Failed to write quarantine receipt in {target_dir}")

    return QuarantineRecord(
        canonical_key_id=canonical_key_id,
        reason_code=reason_code,
        source_directory=source_dir,
        quarantine_directory=target_dir,
        quarantined_utc=receipt["quarantined_utc"],
        artifact_shas=artifact_shas,
    )
