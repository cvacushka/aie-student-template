from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from airguard.features import build_feature_frame


def risk_level(probability: float) -> str:
    if probability >= 0.65:
        return "high"
    if probability >= 0.35:
        return "medium"
    return "low"


def recommendation(level: str) -> str:
    if level == "high":
        return "Increase ventilation now, check occupancy, and consider a short break."
    if level == "medium":
        return "Monitor the room and raise ventilation before CO2 reaches the limit."
    return "Keep the current mode and continue periodic monitoring."


class AirGuardModel:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])
        self.model_name = str(artifact["model_name"])
        self.model_version = str(artifact.get("model_version", "unknown"))
        self.prediction_horizon_minutes = int(artifact.get("prediction_horizon_minutes", 30))

    @classmethod
    def load(cls, path: Path) -> "AirGuardModel":
        return cls(joblib.load(path))

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([payload])
        features = build_feature_frame(frame)
        probability = float(self.pipeline.predict_proba(features)[0, 1])
        level = risk_level(probability)
        return {
            "risk_probability": round(probability, 4),
            "risk_level": level,
            "model_threshold": self.threshold,
            "model_name": self.model_name,
            "prediction_horizon_minutes": self.prediction_horizon_minutes,
            "recommendation": recommendation(level),
        }
