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
from fastapi.responses import HTMLResponse

from api.analyst import build_analyst_report
from api import scorer
from api.schemas import (
    AnalystScoreResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    ScoreResponse,
    TransactionInput,
)
from api.scorer import ModelNotLoadedError, score as score_transaction


PROJECT_ROOT = scorer.PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

APP_STARTED_AT = time.time()
APP_NAME = "SentinelPay Fraud Detection API"
APP_VERSION = "1.2.0"
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
        "analyst_score_url": "/analyst/score",
        "analyst_console_url": "/analyst/console",
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


@app.post("/analyst/score", response_model=AnalystScoreResponse)
def analyst_score_endpoint(
    transaction: TransactionInput,
    x_transaction_id: Annotated[str | None, Header(alias="X-Transaction-ID")] = None,
) -> AnalystScoreResponse:
    payload: dict[str, Any] = transaction.model_dump()
    if x_transaction_id:
        payload["transaction_id"] = x_transaction_id

    try:
        score_result = score_transaction(payload)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _record_score(
        is_fraud=bool(score_result["is_fraud"]),
        latency_ms=float(score_result["latency_ms"]),
    )
    return AnalystScoreResponse(**build_analyst_report(payload, score_result))


@app.get("/analyst/console", response_class=HTMLResponse)
def analyst_console() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SentinelPay Analyst Console</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #111827;
      --muted: #5b6472;
      --line: #d8dee8;
      --surface: #f7f8fb;
      --accent: #0f766e;
      --warn: #b45309;
      --danger: #b91c1c;
      --ok: #166534;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--surface);
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 0.9fr) minmax(360px, 1.1fr);
      gap: 1px;
      min-height: 100vh;
      background: var(--line);
    }
    section {
      background: #fff;
      padding: 24px;
      min-width: 0;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 16px;
      font-size: 16px;
      letter-spacing: 0;
    }
    p {
      color: var(--muted);
      margin: 0 0 18px;
      line-height: 1.45;
    }
    textarea {
      width: 100%;
      min-height: 520px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--ink);
    }
    button {
      margin-top: 12px;
      height: 40px;
      padding: 0 16px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .result {
      display: grid;
      gap: 14px;
    }
    .summary {
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      background: #f0fdfa;
      line-height: 1.45;
    }
    .facts {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .fact {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
    }
    .fact strong {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .fact span {
      font-size: 18px;
      font-weight: 800;
    }
    ul {
      margin: 0;
      padding-left: 20px;
      line-height: 1.55;
    }
    pre {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f172a;
      color: #e5e7eb;
      padding: 14px;
      max-height: 300px;
    }
    .LOW { color: var(--ok); }
    .ELEVATED { color: var(--warn); }
    .HIGH, .CRITICAL { color: var(--danger); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      textarea { min-height: 360px; }
      .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>SentinelPay Analyst Console</h1>
      <p>Maya turns model scores into a review queue, reason codes, and plain-language triage.</p>
      <textarea id="payload">{
  "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
  "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
  "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
  "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
  "V17": 0.2079, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
  "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
  "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
  "Amount": 149.62
}</textarea>
      <button id="score">Ask Maya</button>
    </section>
    <section>
      <h2>Analyst View</h2>
      <div id="result" class="result">
        <p>No transaction scored yet.</p>
      </div>
    </section>
  </main>
  <script>
    const payload = document.querySelector("#payload");
    const button = document.querySelector("#score");
    const result = document.querySelector("#result");

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }

    function list(items) {
      return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
    }

    function render(data) {
      result.innerHTML = `
        <div class="summary">${escapeHtml(data.analyst.summary)}</div>
        <div class="facts">
          <div class="fact"><strong>Risk</strong><span>${data.risk_score.toFixed(4)}</span></div>
          <div class="fact"><strong>Severity</strong><span class="${data.severity}">${data.severity}</span></div>
          <div class="fact"><strong>Queue</strong><span>${data.decision_queue}</span></div>
          <div class="fact"><strong>Label</strong><span>${data.label}</span></div>
        </div>
        <h2>Reason Codes</h2>
        ${list(data.reason_codes.map((reason) => `${reason.code}: ${reason.detail}`))}
        <h2>Recommended Actions</h2>
        ${list(data.recommended_actions)}
        <h2>Raw Response</h2>
        <pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>
      `;
    }

    button.addEventListener("click", async () => {
      button.disabled = true;
      result.innerHTML = "<p>Scoring...</p>";
      try {
        const response = await fetch("/analyst/score", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Transaction-ID": crypto.randomUUID() },
          body: payload.value
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Unable to score transaction.");
        }
        render(data);
      } catch (error) {
        result.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


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
