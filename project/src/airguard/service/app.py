from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from airguard.config import get_settings
from airguard.models.inference import AirGuardModel
from airguard.schemas import AirGuardRequest, AirGuardResponse
from airguard.service.metrics import MetricsRegistry


settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("airguard.service")
metrics = MetricsRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = AirGuardModel.load(settings.model_artifact_path)
        logger.info("Loaded model artifact from %s", settings.model_artifact_path)
    except FileNotFoundError:
        app.state.model = None
        logger.warning("Model artifact not found at %s", settings.model_artifact_path)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.model_version,
    description="Forecasts the risk of poor classroom air quality in the next 30 minutes.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency = time.perf_counter() - start
        metrics.record_request(request.method, request.url.path, status_code, latency)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": settings.model_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    model = getattr(request.app.state, "model", None)
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": str(settings.model_artifact_path),
        "model_version": settings.model_version,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint() -> str:
    return metrics.render_prometheus()


@app.post("/predict", response_model=AirGuardResponse)
def predict(payload: AirGuardRequest, request: Request) -> AirGuardResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Run airguard-train before using /predict.",
        )

    result = model.predict_one(payload.model_dump())
    metrics.record_prediction()
    logger.info(
        "Predicted room_id=%s risk=%s probability=%.4f",
        payload.room_id,
        result["risk_level"],
        result["risk_probability"],
    )
    return AirGuardResponse(room_id=payload.room_id, **result)
