from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from airguard.data.generate_dataset import generate_dataset
from airguard.models.train import run_training
from airguard.service.app import app


def local_artifact_dir() -> Path:
    path = Path(".test-artifacts") / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_predict_endpoint_uses_loaded_model() -> None:
    artifact_dir = local_artifact_dir()
    data_path = artifact_dir / "dataset.csv"
    model_path = artifact_dir / "model.joblib"
    generate_dataset(rows=700, seed=21).to_csv(data_path, index=False)
    run_training(
        data_path=data_path,
        model_path=model_path,
        metrics_path=artifact_dir / "metrics.json",
        experiments_path=artifact_dir / "experiments.csv",
        candidate_names=["logistic_regression"],
    )

    from airguard.models.inference import AirGuardModel

    with TestClient(app) as client:
        client.app.state.model = AirGuardModel.load(model_path)
        payload = {
            "timestamp": "2026-03-17T10:35:00",
            "room_id": "B-421",
            "room_type": "lecture",
            "building_zone": "central",
            "lesson_type": "lecture",
            "room_area_m2": 72.0,
            "capacity": 70,
            "occupancy_count": 61,
            "hvac_airflow_m3h": 360.0,
            "ventilation_level": 0.38,
            "outdoor_temp_c": 6.0,
            "indoor_temp_c": 25.8,
            "humidity_pct": 58.0,
            "co2_ppm_current": 1040.0,
            "minutes_since_lesson_start": 48,
            "minutes_until_break": 32,
            "noise_level_db": 59.0,
            "is_exam_period": False,
            "floor": 4,
        }

        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["room_id"] == "B-421"
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}
