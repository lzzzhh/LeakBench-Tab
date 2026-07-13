"""benchmark_v2/datasets/confirmatory_adapters.py — BTS, Chicago, NYC 311 adapters."""
from __future__ import annotations
import hashlib
import numpy as np, pandas as pd, zipfile
from pathlib import Path
from dataclasses import dataclass, field
from benchmark_v2.core.models import FeatureSpec, FeatureAvailability, LeakageGroundTruth, LeakageLabel, FeatureRole

DATA_DIR = Path("data")

@dataclass
class NaturalTask:
    name: str; X: np.ndarray; y: np.ndarray
    feature_specs: list; availability: list; ground_truth: list
    train_idx: np.ndarray; val_idx: np.ndarray; test_idx: np.ndarray
    source: str = ""; license_info: str = ""
    lineage: dict = field(default_factory=dict)


def _file_lineage(path: Path, **details) -> dict:
    """Return serializable provenance that identifies the exact local input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "lineage_schema_version": 1,
        "source_path": str(path.resolve()),
        "source_sha256": digest.hexdigest(),
        "source_size_bytes": path.stat().st_size,
        "is_synthetic": False,
        **details,
    }


def _time_lineage(values: pd.Series, column: str) -> dict:
    observed = values.dropna()
    return {
        "chronological_sort": True,
        "sort_column": column,
        "time_min": observed.min().isoformat() if not observed.empty else None,
        "time_max": observed.max().isoformat() if not observed.empty else None,
    }

def _time_split(n, train_frac=0.6, val_frac=0.2):
    te, ve = int(n*train_frac), int(n*(train_frac+val_frac))
    return np.arange(te), np.arange(te,ve), np.arange(ve,n)

def _encode_cats(df, exclude):
    df = df.copy()
    for c in df.columns:
        if c in exclude: continue
        if df[c].dtype == object or str(df[c].dtype) == 'category':
            df[c] = df[c].astype('category').cat.codes.astype(float)
        elif 'datetime' in str(df[c].dtype):
            df[c] = df[c].astype('int64').astype(float)
        else:
            try: df[c] = df[c].fillna(-1).astype(float)
            except: df.drop(columns=[c], inplace=True)
    return df

def build_bts_flights():
    zip_path = DATA_DIR / "bts" / "bts_2023_1.csv.zip"
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = sorted(n for n in zf.namelist() if n.lower().endswith('.csv'))
        if not csv_names:
            raise ValueError(f"No CSV member found in {zip_path}")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, low_memory=False, nrows=100000)
    flight_time = pd.to_datetime(df['FlightDate'], errors='coerce')
    df = (df.assign(__event_time=flight_time, __source_row=np.arange(len(df)))
            .sort_values(['__event_time', '__source_row'], kind='mergesort', na_position='last')
            .drop(columns=['__event_time', '__source_row'])
            .reset_index(drop=True))
    flight_time = pd.to_datetime(df['FlightDate'], errors='coerce')
    # Prediction boundary: immediately before scheduled departure.  Use an
    # auditable allowlist for fields known at that boundary; every actual
    # departure/arrival/diversion field is post-event.  A blacklist previously
    # missed DepDelay and many Div* outcome columns.
    schedule_allowlist = {
        'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
        'Reporting_Airline', 'DOT_ID_Reporting_Airline', 'IATA_CODE_Reporting_Airline',
        'Tail_Number', 'Flight_Number_Reporting_Airline',
        'OriginAirportID', 'OriginAirportSeqID', 'OriginCityMarketID', 'Origin',
        'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName', 'OriginWac',
        'DestAirportID', 'DestAirportSeqID', 'DestCityMarketID', 'Dest', 'DestCityName',
        'DestState', 'DestStateFips', 'DestStateName', 'DestWac',
        'CRSDepTime', 'DepTimeBlk', 'CRSArrTime', 'ArrTimeBlk', 'CRSElapsedTime',
        'Flights', 'Distance', 'DistanceGroup',
    }
    usable_cols = [c for c in df.columns if c not in {'Unnamed: 109', 'label'}]
    schedule_cols = [c for c in usable_cols if c in schedule_allowlist]
    outcome_cols = [c for c in usable_cols if c not in schedule_allowlist]
    # Build label
    df['label'] = ((df['ArrDel15'].fillna(0) >= 1) | (df['Cancelled'].fillna(0) == 1) | (df['Diverted'].fillna(0) == 1)).astype(float)
    df = _encode_cats(df, {'label'})
    pred_cols = [c for c in schedule_cols + outcome_cols if c in df.columns]
    X = df[pred_cols].values.astype(np.float32); y = df['label'].values.astype(np.float32)
    specs = []; avail = []; gt = []
    for c in pred_cols:
        specs.append(FeatureSpec(feature_id=f"bts_{c}", role=FeatureRole.PREDICTOR, name_pool={"natural": c}))
        is_post = c in outcome_cols
        avail.append(FeatureAvailability(feature_id=f"bts_{c}", available_at_prediction=not is_post))
        gt.append(LeakageGroundTruth(feature_id=f"bts_{c}",
            label=LeakageLabel.POST_OUTCOME if is_post else LeakageLabel.LEGITIMATE))
    tr, va, te = _time_split(len(y))
    lineage = _file_lineage(
        zip_path, dataset="BTSFlights", archive_member=csv_names[0],
        rows_loaded=len(y), row_limit=100000, target_definition="ArrDel15_or_Cancelled_or_Diverted",
        prediction_boundary="immediately_before_scheduled_departure",
        availability_rule="schedule_allowlist_v2_all_other_operational_fields_post_event",
        **_time_lineage(flight_time, "FlightDate"),
    )
    return NaturalTask("BTSFlights", X, y, specs, avail, gt, tr, va, te,
                       source=str(zip_path.resolve()), lineage=lineage)

def build_chicago_food():
    cache = DATA_DIR / "chicago_food" / "chicago_food_cache.csv"
    df = pd.read_csv(cache)
    rows_scanned = len(df)
    # Remap columns
    col_map = {'dba_name': 'chi_dba_name', 'aka_name': 'chi_aka_name', 'license_': 'chi_license_',
               'facility_type': 'chi_facility_type', 'risk': 'chi_risk', 'address': 'chi_address',
               'city': 'chi_city', 'state': 'chi_state', 'zip': 'chi_zip',
               'inspection_date': 'chi_inspection_date', 'inspection_type': 'chi_inspection_type',
               'results': 'chi_results', 'violations': 'chi_violations'}
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
    inspection_time = pd.to_datetime(df.get('chi_inspection_date'), errors='coerce')
    df = (df.assign(__event_time=inspection_time, __source_row=np.arange(len(df)))
            .sort_values(['__event_time', '__source_row'], kind='mergesort', na_position='last')
            .reset_index(drop=True))
    if len(df) > 20000:
        positions = np.linspace(0, len(df) - 1, num=20000, dtype=np.int64)
        df = df.iloc[positions].copy()
    df = df.drop(columns=['__event_time', '__source_row']).reset_index(drop=True)
    inspection_time = pd.to_datetime(df.get('chi_inspection_date'), errors='coerce')
    if 'chi_results' in df.columns:
        df['label'] = df['chi_results'].apply(lambda x: 1 if str(x).lower().startswith('fail') else 0)
    else:
        df['label'] = 0
    pred_cols = [c for c in df.columns if c not in ['label'] and df[c].dtype in [np.float64,np.int64,object]]
    df = df.dropna(subset=['label'])
    df = _encode_cats(df, {'label'})
    X = df[pred_cols].values.astype(np.float32) if pred_cols else np.zeros((len(df),1),dtype=np.float32)
    y = df['label'].values.astype(np.float32)
    specs = [FeatureSpec(feature_id=f"chi_{c}", role=FeatureRole.PREDICTOR, name_pool={"natural":c}) for c in pred_cols]
    post_cols = {'chi_results', 'chi_violations'}
    avail = [FeatureAvailability(feature_id=f"chi_{c}", available_at_prediction=c not in post_cols)
             for c in pred_cols]
    gt = [LeakageGroundTruth(feature_id=f"chi_{c}",
        label=LeakageLabel.POST_OUTCOME if c in post_cols else LeakageLabel.LEGITIMATE) for c in pred_cols]
    tr, va, te = _time_split(len(y))
    lineage = _file_lineage(
        cache, dataset="ChicagoFood", rows_scanned=rows_scanned, rows_loaded=len(y), row_limit=20000,
        sampling_rule="systematic_over_chronologically_sorted_rows",
        target_definition="inspection_result_starts_with_fail",
        **_time_lineage(inspection_time, "inspection_date"),
    )
    return NaturalTask("ChicagoFood", X, y, specs, avail, gt, tr, va, te,
                       source=str(cache.resolve()), lineage=lineage)

def build_nyc_311():
    cache = DATA_DIR / "nyc311" / "nyc311_cache.csv"
    df = pd.read_csv(cache, nrows=100000)
    if 'closed_date' in df.columns and 'created_date' in df.columns:
        df['created_dt'] = pd.to_datetime(df['created_date'], errors='coerce')
        df['closed_dt'] = pd.to_datetime(df['closed_date'], errors='coerce')
        df['label'] = ((df['closed_dt'] - df['created_dt']).dt.days <= 7).fillna(0).astype(float)
    else:
        df['label'] = 0
    created_time = df.get('created_dt', pd.Series(pd.NaT, index=df.index))
    df = (df.assign(__event_time=created_time, __source_row=np.arange(len(df)))
            .sort_values(['__event_time', '__source_row'], kind='mergesort', na_position='last')
            .drop(columns=['__event_time', '__source_row'])
            .reset_index(drop=True))
    created_time = df.get('created_dt', pd.Series(pd.NaT, index=df.index))
    pred_cols = [c for c in df.columns if c not in ['label','created_date','closed_date','created_dt','closed_dt'] and df[c].dtype in [np.float64,np.int64,object]]
    df = df.dropna(subset=['label'])
    df = _encode_cats(df, {'label'})
    X = df[pred_cols].values.astype(np.float32) if pred_cols else np.zeros((len(df),1),dtype=np.float32)
    y = df['label'].values.astype(np.float32)
    specs = [FeatureSpec(feature_id=f"nyc_{c}", role=FeatureRole.PREDICTOR, name_pool={"natural":c}) for c in pred_cols]
    post = {'nyc_status', 'nyc_resolution_description', 'nyc_resolution_action_updated_date'}
    avail = [FeatureAvailability(feature_id=f"nyc_{c}", available_at_prediction=f"nyc_{c}" not in post)
             for c in pred_cols]
    gt = [LeakageGroundTruth(feature_id=f"nyc_{c}",
        label=LeakageLabel.POST_OUTCOME if f"nyc_{c}" in post else LeakageLabel.LEGITIMATE) for c in pred_cols]
    tr, va, te = _time_split(len(y))
    lineage = _file_lineage(
        cache, dataset="NYC311", rows_loaded=len(y), row_limit=100000,
        target_definition="closed_within_7_days_of_created_date",
        **_time_lineage(created_time, "created_date"),
    )
    return NaturalTask("NYC311", X, y, specs, avail, gt, tr, va, te,
                       source=str(cache.resolve()), lineage=lineage)
