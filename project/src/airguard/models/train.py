from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from airguard.config import get_settings
from airguard.data.validate import validate_dataset
from airguard.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_feature_frame,
)


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def make_model_candidates(random_state: int = 42) -> dict[str, object]:
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def make_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", estimator),
        ]
    )


def predict_scores(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(frame)[:, 1]
    decision = pipeline.decision_function(frame)
    return 1 / (1 + np.exp(-decision))


def binary_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return {key: round(float(value), 4) for key, value in metrics.items()}


def find_best_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.2, 0.8, 61):
        score = f1_score(y_true, y_score >= threshold, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return round(best_threshold, 3), round(best_f1, 4)


def run_training(
    data_path: Path | None = None,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    experiments_path: Path | None = None,
    random_state: int = 42,
    candidate_names: Iterable[str] | None = None,
) -> dict[str, object]:
    settings = get_settings()
    data_path = data_path or settings.dataset_path
    model_path = model_path or settings.model_artifact_path
    metrics_path = metrics_path or settings.metrics_path
    experiments_path = experiments_path or settings.experiments_path

    df = pd.read_csv(data_path)
    validation_errors = validate_dataset(df)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    X = build_feature_frame(df)
    y = df[TARGET_COLUMN].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=random_state,
    )

    candidates = make_model_candidates(random_state=random_state)
    if candidate_names:
        candidates = {name: candidates[name] for name in candidate_names}

    records: list[dict[str, object]] = []
    trained: dict[str, tuple[Pipeline, float, dict[str, float]]] = {}

    for name, estimator in candidates.items():
        pipeline = make_pipeline(deepcopy(estimator))
        pipeline.fit(X_train, y_train)

        val_scores = predict_scores(pipeline, X_val)
        threshold, _ = find_best_threshold(y_val, val_scores)
        val_metrics = binary_metrics(y_val, val_scores, threshold)
        test_metrics = binary_metrics(y_test, predict_scores(pipeline, X_test), threshold)

        records.append({"model": name, "split": "validation", "threshold": threshold, **val_metrics})
        records.append({"model": name, "split": "test", "threshold": threshold, **test_metrics})
        trained[name] = (pipeline, threshold, val_metrics)

    experiments = pd.DataFrame(records)
    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    experiments.to_csv(experiments_path, index=False)

    best_name = max(
        trained,
        key=lambda item: (
            trained[item][2]["f1"],
            trained[item][2]["recall"],
            trained[item][2].get("roc_auc", 0),
        ),
    )
    best_threshold = trained[best_name][1]
    final_pipeline = make_pipeline(deepcopy(candidates[best_name]))
    final_pipeline.fit(X_train_val, y_train_val)
    final_test_metrics = binary_metrics(y_test, predict_scores(final_pipeline, X_test), best_threshold)

    artifact = {
        "pipeline": final_pipeline,
        "threshold": best_threshold,
        "model_name": best_name,
        "model_version": settings.model_version,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "prediction_horizon_minutes": 30,
        "test_metrics": final_test_metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    metrics_payload = {
        "best_model": best_name,
        "threshold": best_threshold,
        "test_metrics": final_test_metrics,
        "rows": int(len(df)),
        "positive_class_share": round(float(y.mean()), 4),
        "experiments_path": portable_path(experiments_path),
        "model_path": portable_path(model_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics_payload


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train AirGuard risk model.")
    parser.add_argument("--data", type=Path, default=settings.dataset_path, help="Training CSV path.")
    parser.add_argument("--model", type=Path, default=settings.model_artifact_path, help="Model artifact path.")
    parser.add_argument("--metrics", type=Path, default=settings.metrics_path, help="Metrics JSON path.")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=settings.experiments_path,
        help="Experiments CSV path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(
        data_path=args.data,
        model_path=args.model,
        metrics_path=args.metrics,
        experiments_path=args.experiments,
        random_state=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
