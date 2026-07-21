"""T0-B Resume Contract — validation classification, quarantine, controlled repair."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import hashlib, json, os, secrets
from datetime import datetime, timezone
from pathlib import Path

from scripts.t0_b_full_b1.fragment_contract import (
    CompletedKeyValidation, MissingReceiptCandidateValidation,
)
from scripts.t0_b_full_b1.io_contract import atomic_write_json


class QuarantineIntegrityError(RuntimeError):
    """Raised when artifact SHA differs before/after os.replace."""


class QuarantineReceiptWriteError(RuntimeError):
    """Raised when quarantine receipt write fails, preserving original exception."""


@dataclass(frozen=True)
class RepairDecision:
    canonical_key_id: str
    reason_code: ResumeReasonCode
    repairable: bool
    validation_errors: tuple[str, ...]
    candidate_errors: tuple[str, ...]


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


def decide_repairability(
    classified_failure: ClassifiedValidationFailure,
    missing_receipt_candidate: MissingReceiptCandidateValidation | None,
) -> RepairDecision:
    """Produce final repair decision combining classification with candidate validation.

    - Fragment SHA mismatch: always repairable.
    - Receipt missing: repairable only if candidate validation passes all non-receipt checks.
    - All other reasons (including receipt corrupt, mixed errors, etc.): NOT repairable.
    """
    errors = tuple(classified_failure.validation_errors)
    candidate_errors = (
        tuple(missing_receipt_candidate.errors) if missing_receipt_candidate is not None
        else ()
    )

    # Fragment SHA mismatch: always repairable
    if classified_failure.repairable:
        return RepairDecision(
            canonical_key_id=classified_failure.canonical_key_id,
            reason_code=classified_failure.reason_code,
            repairable=True,
            validation_errors=errors,
            candidate_errors=candidate_errors,
        )

    # Receipt missing: repairable only if all non-receipt artifacts pass validation
    if (classified_failure.reason_code == ResumeReasonCode.RECEIPT_MISSING
            and missing_receipt_candidate is not None
            and missing_receipt_candidate.is_repairable):
        return RepairDecision(
            canonical_key_id=classified_failure.canonical_key_id,
            reason_code=ResumeReasonCode.RECEIPT_MISSING,
            repairable=True,
            validation_errors=errors,
            candidate_errors=(),
        )

    # All other cases: NOT repairable
    return RepairDecision(
        canonical_key_id=classified_failure.canonical_key_id,
        reason_code=classified_failure.reason_code,
        repairable=False,
        validation_errors=errors,
        candidate_errors=candidate_errors,
    )


@dataclass(frozen=True)
class QuarantineRecord:
    canonical_key_id: str
    reason_code: ResumeReasonCode
    source_directory: Path
    quarantine_directory: Path
    quarantined_utc: str
    artifact_shas: dict[str, str | None]
    verified_artifact_shas: dict[str, str | None]
    integrity_verified: bool


def _fsync_directory(path: Path) -> None:
    """Fsync a directory, failing loudly on error. fd is always closed."""
    fd = None
    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.fsync(fd)
    except OSError as exc:
        raise RuntimeError(
            f"directory fsync failed for {path}: {exc}"
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)


def quarantine_invalid_key(
    output_dir: Path,
    canonical_key_id: str,
    reason_code: ResumeReasonCode,
    validation_errors: Sequence[str],
) -> QuarantineRecord:
    """Atomically move entire key fragment directory to quarantine via os.replace.

    Verifies artifact SHA integrity before and after the move.
    Fails loudly on fsync errors or SHA mismatches.
    """
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

    artifact_names = [
        "baseline.csv.gz", "governed.csv.gz", "selection.csv.gz", "failure.csv.gz",
        "fragment_manifest.json", "completion_receipt.json",
    ]

    # Record artifact SHAs before moving
    original_artifact_shas = {}
    for fname in artifact_names:
        fp = source_dir / fname
        original_artifact_shas[fname] = (
            hashlib.sha256(fp.read_bytes()).hexdigest() if fp.exists() else None
        )

    # Single atomic directory rename
    os.replace(source_dir, target_dir)

    # Verify post-move state
    if source_dir.exists():
        raise RuntimeError(f"Source directory still exists after os.replace: {source_dir}")
    if not target_dir.exists():
        raise RuntimeError(f"Target directory does not exist after os.replace: {target_dir}")
    if not target_dir.is_dir():
        raise RuntimeError(f"Target is not a directory: {target_dir}")

    # Fsync parent directories — fail-loud
    _fsync_directory(source_dir.parent)
    _fsync_directory(target_dir.parent)

    # Recompute artifact SHAs in target directory, compare
    moved_artifact_shas = {}
    for fname in artifact_names:
        fp = target_dir / fname
        moved_artifact_shas[fname] = (
            hashlib.sha256(fp.read_bytes()).hexdigest() if fp.exists() else None
        )

    for fname in artifact_names:
        before = original_artifact_shas[fname]
        after = moved_artifact_shas[fname]
        if before != after:
            raise QuarantineIntegrityError(
                f"Quarantine SHA mismatch for {canonical_key_id}/{fname}: "
                f"before={before}, after={after}"
            )

    # Write quarantine receipt in target_dir
    receipt = {
        "schema_version": 1,
        "canonical_key_id": canonical_key_id,
        "reason_code": reason_code.value,
        "validation_errors": list(validation_errors),
        "source_directory": str(source_dir),
        "quarantine_directory": str(target_dir),
        "quarantined_utc": datetime.now(timezone.utc).isoformat(),
        "original_artifact_sha256": original_artifact_shas,
        "moved_artifact_sha256": moved_artifact_shas,
        "integrity_verified": True,
    }
    try:
        atomic_write_json(target_dir / "quarantine_receipt.json", receipt)
    except (OSError, RuntimeError) as exc:
        raise QuarantineReceiptWriteError(
            f"Failed to write quarantine receipt in {target_dir}"
        ) from exc

    return QuarantineRecord(
        canonical_key_id=canonical_key_id,
        reason_code=reason_code,
        source_directory=source_dir,
        quarantine_directory=target_dir,
        quarantined_utc=receipt["quarantined_utc"],
        artifact_shas=original_artifact_shas,
        verified_artifact_shas=moved_artifact_shas,
        integrity_verified=True,
    )
