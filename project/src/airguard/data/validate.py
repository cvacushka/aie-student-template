from __future__ import annotations

import pandas as pd

from airguard.features import TARGET_COLUMN


REQUIRED_COLUMNS = {
    "timestamp",
    "room_id",
    "room_type",
    "building_zone",
    "lesson_type",
    "room_area_m2",
    "capacity",
    "occupancy_count",
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
    TARGET_COLUMN,
}


def validate_dataset(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors

    if df.empty:
        errors.append("Dataset is empty.")
    if df[TARGET_COLUMN].nunique() < 2:
        errors.append("Target must contain both classes.")
    if df.isna().mean().max() > 0:
        errors.append("Dataset contains missing values.")
    if (df["occupancy_count"] > df["capacity"] + 20).any():
        errors.append("Occupancy is unrealistically higher than capacity.")
    if not df["ventilation_level"].between(0, 1).all():
        errors.append("Ventilation level must be in [0, 1].")
    if not df["co2_ppm_current"].between(350, 3000).all():
        errors.append("Current CO2 is outside the accepted range.")
    return errors
