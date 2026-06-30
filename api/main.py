from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated, Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import scorer
from api.schemas import BatchScoreRequest, BatchScoreResponse, ScoreResponse, TransactionInput
from api.scorer import ModelNotLoadedError, score as score_transaction


PROJECT_ROOT = scorer.PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

APP_STARTED_AT = time.time()
APP_NAME = "SentinelPay Fraud Detection API"
APP_VERSION = "1.1.0"
METRICS_WINDOW_SIZE = int(os.getenv("METRICS_WINDOW_SIZE", "10000"))
_metrics_lock = Lock()
_total_scored = 0
_fraud_count = 0
_latencies_ms: deque[float] = deque(maxlen=METRICS_WINDOW_SIZE)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if scorer.ensure_model_loaded():
        logger.info("Fraud model loaded from %s", scorer.MODEL_PATH)
    else:
        logger.warning("Fraud model is not loaded: %s", scorer.MODEL_LOAD_ERROR)
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Real-time anomaly scoring API for credit card transactions. "
        "Scores single transactions and batches with an Isolation Forest model."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _record_score(is_fraud: bool, latency_ms: float) -> None:
    global _total_scored, _fraud_count

    with _metrics_lock:
        _total_scored += 1
        if is_fraud:
            _fraud_count += 1
        _latencies_ms.append(float(latency_ms))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(values, percentile)), 4)


def reset_metrics() -> None:
    global _total_scored, _fraud_count

    with _metrics_lock:
        _total_scored = 0
        _fraud_count = 0
        _latencies_ms.clear()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "description": "Streaming-ready fraud anomaly detection with FastAPI and Kafka.",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
        "model_url": "/model",
    }


@app.post("/score", response_model=ScoreResponse)
def score_endpoint(
    transaction: TransactionInput,
    x_transaction_id: Annotated[str | None, Header(alias="X-Transaction-ID")] = None,
) -> ScoreResponse:
    payload: dict[str, Any] = transaction.model_dump()
    if x_transaction_id:
        payload["transaction_id"] = x_transaction_id

    try:
        result = score_transaction(payload)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _record_score(is_fraud=bool(result["is_fraud"]), latency_ms=float(result["latency_ms"]))
    return ScoreResponse(**result)


@app.post("/score/batch", response_model=BatchScoreResponse)
def batch_score_endpoint(batch: BatchScoreRequest) -> BatchScoreResponse:
    start = time.perf_counter()
    scores: list[ScoreResponse] = []

    try:
        for transaction in batch.transactions:
            result = score_transaction(transaction.model_dump())
            _record_score(is_fraud=bool(result["is_fraud"]), latency_ms=float(result["latency_ms"]))
            scores.append(ScoreResponse(**result))
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return BatchScoreResponse(
        transaction_count=len(scores),
        scores=scores,
        latency_ms=round((time.perf_counter() - start) * 1000.0, 4),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    status = scorer.model_status()
    return {
        "status": "ok" if status["model_loaded"] else "degraded",
        "name": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": status["model_loaded"],
        "model_path": status["model_path"],
        "model_load_error": status["model_load_error"],
        "uptime_seconds": round(time.time() - APP_STARTED_AT, 4),
    }


@app.get("/model")
def model() -> dict[str, Any]:
    return scorer.model_metadata()


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    with _metrics_lock:
        total = _total_scored
        fraud_count = _fraud_count
        latencies = list(_latencies_ms)

    avg_latency = round(float(np.mean(latencies)), 4) if latencies else 0.0
    return {
        "total_scored": total,
        "fraud_count": fraud_count,
        "legitimate_count": total - fraud_count,
        "fraud_rate": round((fraud_count / total) if total else 0.0, 4),
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": _percentile(latencies, 50),
        "p99_latency_ms": _percentile(latencies, 99),
        "latency_window_count": len(latencies),
        "metrics_window_size": METRICS_WINDOW_SIZE,
        "uptime_seconds": round(time.time() - APP_STARTED_AT, 4),
    }
