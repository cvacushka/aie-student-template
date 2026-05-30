from pathlib import Path
from uuid import uuid4

from airguard.data.generate_dataset import generate_dataset
from airguard.models.inference import AirGuardModel
from airguard.models.train import run_training


def local_artifact_dir() -> Path:
    path = Path(".test-artifacts") / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_training_creates_real_model_artifact() -> None:
    artifact_dir = local_artifact_dir()
    data_path = artifact_dir / "dataset.csv"
    model_path = artifact_dir / "model.joblib"
    metrics_path = artifact_dir / "metrics.json"
    experiments_path = artifact_dir / "experiments.csv"
    generate_dataset(rows=700, seed=12).to_csv(data_path, index=False)

    metrics = run_training(
        data_path=data_path,
        model_path=model_path,
        metrics_path=metrics_path,
        experiments_path=experiments_path,
        candidate_names=["dummy_majority", "logistic_regression"],
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert experiments_path.exists()
    assert metrics["best_model"] == "logistic_regression"

    model = AirGuardModel.load(model_path)
    sample = generate_dataset(rows=1, seed=99).drop(columns=["future_co2_ppm", "co2_risk_next_30min"]).iloc[0]
    prediction = model.predict_one(sample.to_dict())
    assert 0 <= prediction["risk_probability"] <= 1
    assert prediction["risk_level"] in {"low", "medium", "high"}
