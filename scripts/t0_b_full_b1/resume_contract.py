"""T0-B Resume Contract — validation classification, quarantine, controlled repair."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import hashlib, json, secrets
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
    """Classify validation failure into reason code. Only fragment SHA mismatch is repairable."""
    errors = tuple(validation.errors)

    # Check for fragment SHA mismatch (most specific first)
    sha_mismatch_patterns = [
        "baseline SHA mismatch",
        "governed SHA mismatch",
        "selection SHA mismatch",
        "failure SHA mismatch",
    ]
    for err in errors:
        for pattern in sha_mismatch_patterns:
            if pattern.lower() in err.lower():
                return ClassifiedValidationFailure(
                    canonical_key_id=canonical_key_id,
                    reason_code=ResumeReasonCode.FRAGMENT_SHA_MISMATCH,
                    validation_errors=errors,
                    repairable=True,
                )

    # Classify other errors (none repairable in R10b-3)
    for err in errors:
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
    """Move entire key fragment directory to quarantine. Returns quarantine record."""
    source_dir = output_dir / "key_fragments" / canonical_key_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    nonce = secrets.token_hex(4)
    target_dir = output_dir / "quarantine" / canonical_key_id / reason_code.value / f"{ts}_{nonce}"
    target_dir.mkdir(parents=True, exist_ok=False)

    # Record original artifact SHAs
    artifact_shas = {}
    for fname in ["baseline.csv.gz", "governed.csv.gz", "selection.csv.gz", "failure.csv.gz",
                  "fragment_manifest.json", "completion_receipt.json"]:
        fp = source_dir / fname
        if fp.exists():
            artifact_shas[fname] = hashlib.sha256(fp.read_bytes()).hexdigest()
        else:
            artifact_shas[fname] = None

    # Move entire source directory to quarantine
    for item in source_dir.iterdir():
        item.rename(target_dir / item.name)

    # Verify source is empty, then remove it
    remaining = list(source_dir.iterdir())
    assert len(remaining) == 0, f"Source directory not empty after move: {remaining}"
    source_dir.rmdir()

    # Write quarantine receipt
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
    atomic_write_json(target_dir / "quarantine_receipt.json", receipt)

    return QuarantineRecord(
        canonical_key_id=canonical_key_id,
        reason_code=reason_code,
        source_directory=source_dir,
        quarantine_directory=target_dir,
        quarantined_utc=receipt["quarantined_utc"],
        artifact_shas=artifact_shas,
    )
