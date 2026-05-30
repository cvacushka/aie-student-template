from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "co2_risk_next_30min"

NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "room_area_m2",
    "capacity",
    "occupancy_count",
    "occupancy_ratio",
    "hvac_airflow_m3h",
    "ventilation_level",
    "outdoor_temp_c",
    "indoor_temp_c",
    "humidity_pct",
    "co2_ppm_current",
    "minutes_since_lesson_start",
    "minutes_until_break",
    "noise_level_db",
    "is_exam_period",
    "floor",
    "ventilation_per_person",
    "co2_pressure",
    "thermal_discomfort",
]

CATEGORICAL_FEATURES = ["room_type", "building_zone", "lesson_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "timestamp" in result.columns:
        timestamp = pd.to_datetime(result["timestamp"], errors="coerce")
        result["hour"] = timestamp.dt.hour
        result["day_of_week"] = timestamp.dt.dayofweek
        result["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
        result["month"] = timestamp.dt.month

    if "occupancy_ratio" not in result.columns:
        capacity = result["capacity"].clip(lower=1)
        result["occupancy_ratio"] = result["occupancy_count"] / capacity

    occupancy = result["occupancy_count"].clip(lower=1)
    result["ventilation_per_person"] = result["hvac_airflow_m3h"] / occupancy
    result["co2_pressure"] = (result["co2_ppm_current"] - 800).clip(lower=0)
    result["thermal_discomfort"] = (
        (result["indoor_temp_c"] - 24).clip(lower=0) * 3.0
        + (result["humidity_pct"] - 55).clip(lower=0) * 0.35
    )
    result["is_exam_period"] = result["is_exam_period"].astype(int)
    return result


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = add_derived_features(df)
    missing = [column for column in FEATURE_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return result[FEATURE_COLUMNS]
