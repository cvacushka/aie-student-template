from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from airguard.config import PROJECT_DIR


ROOM_TYPES = np.array(["lecture", "seminar", "lab", "coworking"])
BUILDING_ZONES = np.array(["north", "south", "east", "west", "central"])
LESSON_TYPES = np.array(["lecture", "practice", "lab", "exam", "none"])


def _sample_capacity(room_type: str, rng: np.random.Generator) -> int:
    if room_type == "lecture":
        return int(rng.integers(55, 130))
    if room_type == "lab":
        return int(rng.integers(18, 45))
    if room_type == "coworking":
        return int(rng.integers(20, 80))
    return int(rng.integers(20, 65))


def generate_dataset(rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-02-09 08:00:00")

    records: list[dict[str, object]] = []
    for idx in range(rows):
        day_offset = int(rng.integers(0, 100))
        hour = int(rng.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]))
        minute = int(rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))
        timestamp = base + pd.Timedelta(days=day_offset, hours=hour - 8, minutes=minute)

        room_type = str(rng.choice(ROOM_TYPES, p=[0.34, 0.31, 0.23, 0.12]))
        lesson_type = str(rng.choice(LESSON_TYPES, p=[0.33, 0.27, 0.18, 0.08, 0.14]))
        building_zone = str(rng.choice(BUILDING_ZONES))
        capacity = _sample_capacity(room_type, rng)
        room_area_m2 = round(capacity * rng.uniform(0.9, 1.45), 1)

        is_exam_period = timestamp.month in [5, 6] or rng.random() < 0.08
        occupancy_base = {
            "lecture": 0.76,
            "practice": 0.68,
            "lab": 0.72,
            "exam": 0.93,
            "none": 0.22,
        }[lesson_type]
        if hour in [12, 13]:
            occupancy_base -= 0.10
        if is_exam_period:
            occupancy_base += 0.10
        occupancy_ratio = float(np.clip(rng.normal(occupancy_base, 0.16), 0.02, 1.18))
        occupancy_count = int(np.clip(round(capacity * occupancy_ratio), 0, capacity + 15))

        ventilation_level = float(np.clip(rng.beta(2.4, 2.8), 0.05, 1.0))
        if room_type == "lab":
            ventilation_level = float(np.clip(ventilation_level + 0.15, 0.05, 1.0))
        hvac_airflow_m3h = round(
            capacity * rng.uniform(5.0, 10.5) * (0.55 + ventilation_level), 1
        )

        seasonal_temp = 7 + 11 * np.sin((timestamp.dayofyear - 80) / 365 * 2 * np.pi)
        outdoor_temp_c = round(float(rng.normal(seasonal_temp, 5.0)), 1)
        indoor_temp_c = round(
            float(
                rng.normal(22.5, 1.4)
                + occupancy_ratio * 2.4
                - ventilation_level * 1.0
                + max(outdoor_temp_c - 24, 0) * 0.12
            ),
            1,
        )
        humidity_pct = round(float(np.clip(rng.normal(45 + occupancy_ratio * 18, 8), 18, 88)), 1)
        minutes_since_lesson_start = int(np.clip(rng.normal(42, 27), 0, 90))
        minutes_until_break = int(np.clip(90 - minutes_since_lesson_start + rng.normal(0, 8), 0, 90))
        noise_level_db = round(float(np.clip(rng.normal(43 + occupancy_ratio * 23, 5), 25, 85)), 1)

        co2_ppm_current = round(
            float(
                430
                + occupancy_count * rng.uniform(5.4, 8.3)
                + minutes_since_lesson_start * rng.uniform(1.2, 2.4)
                - ventilation_level * rng.uniform(90, 210)
                - hvac_airflow_m3h * 0.025
                + rng.normal(0, 55)
            ),
            1,
        )
        co2_ppm_current = float(np.clip(co2_ppm_current, 380, 2200))

        future_co2_ppm = round(
            float(
                co2_ppm_current
                + occupancy_count * rng.uniform(3.0, 5.8)
                + max(minutes_since_lesson_start - 35, 0) * rng.uniform(0.8, 1.7)
                - hvac_airflow_m3h * ventilation_level * rng.uniform(0.035, 0.075)
                - max(15 - minutes_until_break, 0) * rng.uniform(4.5, 8.0)
                + rng.normal(0, 65)
            ),
            1,
        )
        discomfort = future_co2_ppm >= 1000 and indoor_temp_c >= 25.5 and humidity_pct >= 57
        target = int(future_co2_ppm >= 1100 or discomfort)

        records.append(
            {
                "sample_id": f"ag-{idx:05d}",
                "timestamp": timestamp.isoformat(),
                "room_id": f"{building_zone[:1].upper()}-{int(rng.integers(101, 599))}",
                "room_type": room_type,
                "building_zone": building_zone,
                "lesson_type": lesson_type,
                "room_area_m2": room_area_m2,
                "capacity": capacity,
                "occupancy_count": occupancy_count,
                "hvac_airflow_m3h": hvac_airflow_m3h,
                "ventilation_level": round(ventilation_level, 3),
                "outdoor_temp_c": outdoor_temp_c,
                "indoor_temp_c": indoor_temp_c,
                "humidity_pct": humidity_pct,
                "co2_ppm_current": co2_ppm_current,
                "minutes_since_lesson_start": minutes_since_lesson_start,
                "minutes_until_break": minutes_until_break,
                "noise_level_db": noise_level_db,
                "is_exam_period": bool(is_exam_period),
                "floor": int(rng.integers(1, 8)),
                "future_co2_ppm": future_co2_ppm,
                "co2_risk_next_30min": target,
            }
        )

    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic AirGuard dataset.")
    parser.add_argument("--rows", type=int, default=6000, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data" / "processed" / "airguard_dataset.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(
        f"Saved {len(dataset)} rows to {args.output}. "
        f"Positive class share: {dataset['co2_risk_next_30min'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
