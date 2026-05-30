from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("AIRGUARD_APP_NAME", "Smart Campus AirGuard")
    model_version: str = os.getenv("AIRGUARD_MODEL_VERSION", "0.1.0")
    log_level: str = os.getenv("AIRGUARD_LOG_LEVEL", "INFO")
    model_artifact_path: Path = _path_from_env(
        "AIRGUARD_MODEL_PATH", PROJECT_DIR / "models" / "airguard_model.joblib"
    )
    metrics_path: Path = _path_from_env(
        "AIRGUARD_METRICS_PATH", PROJECT_DIR / "models" / "metrics.json"
    )
    experiments_path: Path = _path_from_env(
        "AIRGUARD_EXPERIMENTS_PATH", PROJECT_DIR / "models" / "experiments.csv"
    )
    dataset_path: Path = _path_from_env(
        "AIRGUARD_DATASET_PATH",
        PROJECT_DIR / "data" / "processed" / "airguard_dataset.csv",
    )


def get_settings() -> Settings:
    return Settings()
