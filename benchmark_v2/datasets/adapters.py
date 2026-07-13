"""Lending Club and Bank Marketing natural-task adapters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_v2.core.models import (
    FeatureAvailability,
    FeatureRole,
    FeatureSpec,
    LeakageGroundTruth,
    LeakageLabel,
)
from benchmark_v2.datasets.confirmatory_adapters import (
    NaturalTask,
    _encode_cats,
    _file_lineage,
    _time_lineage,
    _time_split,
)

DATA_DIR = Path("data")


def _select_lending_file(raw: Path) -> Path | None:
    supported = [
        path for path in raw.rglob("*")
        if path.is_file()
        and path.name.lower().startswith("accepted")
        and (path.suffix.lower() in {".csv", ".parquet"} or path.name.lower().endswith(".csv.gz"))
    ] if raw.exists() else []
    if not supported:
        return None

    def selection_key(path: Path) -> tuple:
        format_rank = 0 if path.suffix.lower() == ".parquet" else 1 if path.name.lower().endswith(".csv.gz") else 2
        return (-path.stat().st_size, format_rank, path.name.lower(), str(path.resolve()))

    return sorted(supported, key=selection_key)[0]


def _select_bank_file(raw: Path) -> Path | None:
    csv_files = sorted(
        {path.resolve() for path in raw.rglob("**/*.csv", recursive=True) if path.is_file() and not path.name.startswith("._")},
        key=lambda path: str(path).lower(),
    ) if raw.exists() else []
    full_files = [path for path in csv_files if "full" in path.stem.lower()]
    if not full_files:
        return None

    def selection_key(path: Path) -> tuple:
        name = path.name.lower()
        exact_rank = 0 if name == "bank-additional-full.csv" else 1 if name == "bank-full.csv" else 2
        return (exact_rank, -path.stat().st_size, str(path).lower())

    return sorted(full_files, key=selection_key)[0]


def _target_column(columns) -> str:
    for candidate in ("target", "loan_status", "bad_loan", "is_bad"):
        if candidate in columns:
            return candidate
    raise ValueError("Lending Club input has no supported target column")


def _binary_labels(values: pd.Series, target_col: str) -> pd.Series:
    if not (values.dtype == object or str(values.dtype).startswith("string") or str(values.dtype) == "category"):
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.where(numeric.isin([0, 1])).astype(float)

    normalized = values.astype("string").str.strip().str.lower()
    labels = pd.Series(np.nan, index=values.index, dtype=float)
    positive = {
        "1", "true", "charged off", "default", "late (31-120 days)", "late (16-30 days)",
        "does not meet the credit policy. status:charged off",
    }
    negative = {
        "0", "false", "fully paid", "does not meet the credit policy. status:fully paid",
    }
    labels.loc[normalized.isin(positive)] = 1.0
    labels.loc[normalized.isin(negative)] = 0.0
    if target_col != "loan_status" and labels.notna().sum() == 0:
        raise ValueError(f"Unsupported categorical labels in {target_col}")
    return labels


def _parse_lending_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="%b-%Y", errors="coerce")


def _read_lending(path: Path, max_rows: int) -> tuple[pd.DataFrame, str, pd.Series, dict]:
    """Select a bounded, deterministic temporal sample over the whole accepted file."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    if path.suffix.lower() == ".parquet":
        full = pd.read_parquet(path)
        target_col = _target_column(full.columns)
        labels = _binary_labels(full[target_col], target_col)
        metadata = pd.DataFrame({
            "__source_row": np.arange(len(full)),
            "__label": labels,
            "__event_time": _parse_lending_dates(full["issue_d"]) if "issue_d" in full else pd.NaT,
        }).dropna(subset=["__label"])
        rows_scanned = len(full)
    else:
        header = pd.read_csv(path, nrows=0)
        target_col = _target_column(header.columns)
        metadata_columns = [target_col] + (["issue_d"] if "issue_d" in header.columns else [])
        metadata_parts = []
        offset = 0
        for chunk in pd.read_csv(path, usecols=metadata_columns, chunksize=100000, low_memory=False):
            labels = _binary_labels(chunk[target_col], target_col)
            event_time = _parse_lending_dates(chunk["issue_d"]) if "issue_d" in chunk else pd.Series(pd.NaT, index=chunk.index)
            part = pd.DataFrame({
                "__source_row": offset + np.arange(len(chunk)),
                "__label": labels.to_numpy(),
                "__event_time": event_time.to_numpy(),
            }).dropna(subset=["__label"])
            metadata_parts.append(part)
            offset += len(chunk)
        metadata = pd.concat(metadata_parts, ignore_index=True) if metadata_parts else pd.DataFrame()
        rows_scanned = offset

    if metadata.empty:
        raise ValueError(f"No resolved binary outcomes found in {path}")

    has_time = metadata["__event_time"].notna().any()
    order = ["__event_time", "__source_row"] if has_time else ["__source_row"]
    metadata = metadata.sort_values(order, kind="mergesort", na_position="last").reset_index(drop=True)
    if len(metadata) > max_rows:
        positions = np.linspace(0, len(metadata) - 1, num=max_rows, dtype=np.int64)
        selected = metadata.iloc[positions].copy()
    else:
        selected = metadata.copy()
    selected_rows = np.sort(selected["__source_row"].to_numpy(dtype=np.int64))

    if path.suffix.lower() == ".parquet":
        df = full.iloc[selected_rows].copy()
        df["__source_row"] = selected_rows
    else:
        selected_parts = []
        offset = 0
        for chunk in pd.read_csv(path, chunksize=100000, low_memory=False):
            source_rows = offset + np.arange(len(chunk))
            keep = np.isin(source_rows, selected_rows, assume_unique=True)
            if keep.any():
                part = chunk.loc[keep].copy()
                part["__source_row"] = source_rows[keep]
                selected_parts.append(part)
            offset += len(chunk)
        df = pd.concat(selected_parts, ignore_index=True)

    df["label"] = _binary_labels(df[target_col], target_col)
    if has_time:
        df["__event_time"] = _parse_lending_dates(df["issue_d"])
        df = df.sort_values(["__event_time", "__source_row"], kind="mergesort", na_position="last")
        event_time = df["__event_time"].reset_index(drop=True)
    else:
        df = df.sort_values("__source_row", kind="mergesort")
        event_time = pd.Series(pd.NaT, index=np.arange(len(df)))
    df = df.drop(columns=["__event_time", "__source_row"], errors="ignore").reset_index(drop=True)
    details = {
        "rows_scanned": rows_scanned,
        "resolved_outcome_rows": len(metadata),
        "rows_loaded": len(df),
        "row_limit": max_rows,
        "sampling_rule": "systematic_over_chronologically_sorted_resolved_outcomes" if has_time else "systematic_over_source_order_resolved_outcomes",
        "chronological_sort": has_time,
        "sort_column": "issue_d" if has_time else None,
    }
    if has_time:
        details.update({key: value for key, value in _time_lineage(event_time, "issue_d").items() if key not in details})
    return df, target_col, event_time, details


def _synthetic_lending() -> NaturalTask:
    rng = np.random.RandomState(42)
    n = 50000
    X = rng.randn(n, 25).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] > 0).astype(float) * 0.8 + rng.rand(n) * 0.4 > 0.5).astype(np.float32)
    specs = [FeatureSpec(feature_id=f"lc_feat_{i}", role=FeatureRole.PREDICTOR) for i in range(25)]
    avail = [FeatureAvailability(feature_id=f"lc_feat_{i}") for i in range(25)]
    gt = [LeakageGroundTruth(feature_id=f"lc_feat_{i}", label=LeakageLabel.POST_OUTCOME if i >= 19 else LeakageLabel.LEGITIMATE) for i in range(25)]
    tr, va, te = _time_split(n)
    lineage = {"lineage_schema_version": 1, "is_synthetic": True, "generator": "RandomState(42)", "rows_loaded": n}
    return NaturalTask("LendingClub_SYNTHETIC", X, y, specs, avail, gt, tr, va, te,
                       source="synthetic://lending-club", lineage=lineage)


def build_lending_club(*, allow_synthetic: bool = False, max_rows: int = 50000) -> NaturalTask:
    raw = DATA_DIR / "raw" / "lending_club"
    use_file = _select_lending_file(raw)
    if use_file is None:
        if allow_synthetic:
            return _synthetic_lending()
        raise FileNotFoundError(f"No accepted Lending Club parquet/CSV found under {raw}")

    df, target_col, event_time, read_details = _read_lending(use_file, max_rows=max_rows)
    pred_cols = [column for column in df.columns if column not in {target_col, "label"}]
    df = _encode_cats(df, {target_col, "label"})
    pred_cols = [column for column in pred_cols if column in df.columns]
    X = df[pred_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    specs = [FeatureSpec(feature_id=f"lc_{column}", role=FeatureRole.PREDICTOR, name_pool={"natural": column}) for column in pred_cols]
    post_patterns = (
        "recoveries", "collection_recovery_fee", "last_pymnt", "next_pymnt", "total_rec",
        "out_prncp", "total_pymnt", "last_fico", "last_credit_pull", "hardship",
        "deferral_term", "payment_plan_start_date", "orig_projected_additional_accrued_interest",
        "settlement", "debt_settlement",
    )
    is_post = {column: any(pattern in column.lower() for pattern in post_patterns) for column in pred_cols}
    avail = [FeatureAvailability(feature_id=f"lc_{column}", available_at_prediction=not is_post[column]) for column in pred_cols]
    gt = [LeakageGroundTruth(feature_id=f"lc_{column}", label=LeakageLabel.POST_OUTCOME if is_post[column] else LeakageLabel.LEGITIMATE) for column in pred_cols]
    tr, va, te = _time_split(len(y))
    lineage = _file_lineage(
        use_file, dataset="LendingClub", selected_file_rule="largest_accepted_file_then_format_then_name",
        target_column=target_col, target_definition="resolved_good_vs_bad_loan_status", **read_details,
    )
    return NaturalTask("LendingClub", X, y, specs, avail, gt, tr, va, te,
                       source=str(use_file.resolve()), lineage=lineage)


def _synthetic_bank() -> NaturalTask:
    rng = np.random.RandomState(42)
    n = 5000
    X = rng.randn(n, 20).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    specs = [FeatureSpec(feature_id=f"bm_{i}", role=FeatureRole.PREDICTOR) for i in range(20)]
    avail = [FeatureAvailability(feature_id=f"bm_{i}") for i in range(20)]
    gt = [LeakageGroundTruth(feature_id=f"bm_{i}", label=LeakageLabel.POST_OUTCOME if i == 19 else LeakageLabel.LEGITIMATE) for i in range(20)]
    tr, va, te = _time_split(n)
    lineage = {"lineage_schema_version": 1, "is_synthetic": True, "generator": "RandomState(42)", "rows_loaded": n}
    return NaturalTask("BankMarketing_SYNTHETIC", X, y, specs, avail, gt, tr, va, te,
                       source="synthetic://bank-marketing", lineage=lineage)


def build_bank_marketing(*, allow_synthetic: bool = False) -> NaturalTask:
    raw = DATA_DIR / "raw" / "bank_marketing"
    use_file = _select_bank_file(raw)
    if use_file is None:
        if allow_synthetic:
            return _synthetic_bank()
        raise FileNotFoundError(f"No full Bank Marketing CSV found under {raw}")

    df = pd.read_csv(use_file, sep=";", low_memory=False)
    delimiter = ";"
    if len(df.columns) == 1:
        df = pd.read_csv(use_file, low_memory=False)
        delimiter = ","
    if "y" not in df.columns:
        raise ValueError(f"Bank Marketing input {use_file} has no y target column")
    normalized_target = df["y"].astype("string").str.strip().str.lower()
    if not normalized_target.isin({"yes", "no"}).all():
        raise ValueError(f"Bank Marketing input {use_file} contains unsupported y labels")
    df["label"] = (normalized_target == "yes").astype(float)
    pred_cols = [column for column in df.columns if column not in {"y", "label"}]
    df = _encode_cats(df, {"y", "label"})
    pred_cols = [column for column in pred_cols if column in df.columns]
    X = df[pred_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    specs = [FeatureSpec(feature_id=f"bm_{column}", role=FeatureRole.PREDICTOR, name_pool={"natural": column}) for column in pred_cols]
    is_post = {column: column == "duration" for column in pred_cols}
    avail = [FeatureAvailability(feature_id=f"bm_{column}", available_at_prediction=not is_post[column]) for column in pred_cols]
    gt = [LeakageGroundTruth(feature_id=f"bm_{column}", label=LeakageLabel.POST_OUTCOME if is_post[column] else LeakageLabel.LEGITIMATE) for column in pred_cols]
    tr, va, te = _time_split(len(y))
    lineage = _file_lineage(
        use_file, dataset="BankMarketing", selected_file_rule="named_full_csv_with_canonical_name_priority",
        delimiter=delimiter, rows_loaded=len(y), target_column="y", target_definition="subscription_yes",
        chronological_sort=False, sort_column=None, source_order_preserved=True,
        chronology_note="month/day_of_week lack a year, so no fabricated timestamp was used",
    )
    return NaturalTask("BankMarketing", X, y, specs, avail, gt, tr, va, te,
                       source=str(use_file.resolve()), lineage=lineage)
