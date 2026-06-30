from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

FEATURE_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
PROJECT_NAME = "SentinelPay"
TRAINED_MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "model.pkl"
DEMO_MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "demo_model.pkl"
DEFAULT_MODEL_PATH = TRAINED_MODEL_PATH if TRAINED_MODEL_PATH.exists() else DEMO_MODEL_PATH
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH

FRAUD_RISK_THRESHOLD = float(os.getenv("FRAUD_RISK_THRESHOLD", "0.6"))
DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", "local")
SCORING_MODEL = os.getenv("SCORING_MODEL", "auto")

MODEL_BUNDLE: dict[str, Any] | None = None
MODEL_LOAD_ERROR: str | None = None
MODEL_LOADED_AT: float | None = None


class ModelNotLoadedError(RuntimeError):
    """Raised when scoring is attempted before a trained model is available."""


def _load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python model/train.py` first."
        )

    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise TypeError("Model artifact must be a dictionary.")
    if "model" not in bundle and "models" not in bundle:
        raise KeyError("Model artifact is missing `model` or `models`.")
    if "amount_scaler" not in bundle and "scaler" not in bundle:
        raise KeyError("Model artifact is missing `amount_scaler` or `scaler`.")

    bundle.setdefault("feature_columns", FEATURE_COLUMNS)
    bundle.setdefault("score_min", -0.5)
    bundle.setdefault("score_max", 0.5)
    if "models" in bundle and "model" not in bundle:
        default_model = str(bundle.get("default_scoring_model", "isolation_forest"))
        bundle["model"] = bundle["models"][default_model]
    return bundle


def reload_model(path: Path | None = None) -> bool:
    global MODEL_BUNDLE, MODEL_LOAD_ERROR, MODEL_LOADED_AT

    target_path = path or MODEL_PATH
    try:
        MODEL_BUNDLE = _load_model_bundle(target_path)
        MODEL_LOAD_ERROR = None
        MODEL_LOADED_AT = time.time()
        return True
    except Exception as exc:
        MODEL_BUNDLE = None
        MODEL_LOAD_ERROR = str(exc)
        MODEL_LOADED_AT = None
        return False


def ensure_model_loaded() -> bool:
    if MODEL_BUNDLE is not None:
        return True
    return reload_model()


def model_status() -> dict[str, Any]:
    return {
        "model_loaded": MODEL_BUNDLE is not None,
        "model_path": str(MODEL_PATH),
        "model_load_error": MODEL_LOAD_ERROR,
        "model_loaded_at": round(MODEL_LOADED_AT, 4) if MODEL_LOADED_AT else None,
        "project": PROJECT_NAME,
    }


def model_metadata() -> dict[str, Any]:
    if MODEL_BUNDLE is None:
        return {
            **model_status(),
            "model_type": None,
            "model_version": DEFAULT_MODEL_VERSION,
            "feature_count": len(FEATURE_COLUMNS),
            "risk_threshold": round(FRAUD_RISK_THRESHOLD, 4),
            "scoring_model": SCORING_MODEL,
        }

    feature_columns = list(MODEL_BUNDLE.get("feature_columns", FEATURE_COLUMNS))
    models = MODEL_BUNDLE.get("models")
    available_models = sorted(models.keys()) if isinstance(models, dict) else ["isolation_forest"]
    return {
        **model_status(),
        "model_type": type(_select_model(MODEL_BUNDLE)[1]).__name__,
        "model_version": str(MODEL_BUNDLE.get("model_version", DEFAULT_MODEL_VERSION)),
        "schema_version": str(MODEL_BUNDLE.get("schema_version", "1.0")),
        "scoring_model": _select_model(MODEL_BUNDLE)[0],
        "available_models": available_models,
        "default_scoring_model": MODEL_BUNDLE.get("default_scoring_model", "isolation_forest"),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "risk_threshold": round(FRAUD_RISK_THRESHOLD, 4),
        "raw_score_threshold": round(float(MODEL_BUNDLE.get("raw_score_threshold", -0.1)), 4),
        "score_min": round(float(MODEL_BUNDLE.get("score_min", -0.5)), 4),
        "score_max": round(float(MODEL_BUNDLE.get("score_max", 0.5)), 4),
        "training_metrics": MODEL_BUNDLE.get("training_metrics", {}),
        "score_distribution": MODEL_BUNDLE.get("score_distribution", {}),
        "created_at": (
            round(float(MODEL_BUNDLE["created_at"]), 4)
            if MODEL_BUNDLE.get("created_at") is not None
            else None
        ),
    }


def _select_model(bundle: Mapping[str, Any]) -> tuple[str, Any]:
    models = bundle.get("models")
    if isinstance(models, dict) and models:
        selected = SCORING_MODEL
        if selected == "auto":
            selected = str(bundle.get("default_scoring_model", "supervised_baseline"))
        if selected not in models:
            raise ValueError(
                f"Unknown SCORING_MODEL `{selected}`. Available models: {', '.join(sorted(models))}"
            )
        return selected, models[selected]
    return "isolation_forest", bundle["model"]


def _feature_frame(transaction: Mapping[str, Any], bundle: Mapping[str, Any]) -> pd.DataFrame:
    feature_columns = list(bundle.get("feature_columns", FEATURE_COLUMNS))
    missing = [column for column in feature_columns if column not in transaction]
    if missing:
        raise ValueError(f"Transaction is missing required features: {', '.join(missing)}")

    frame = pd.DataFrame(
        [{column: float(transaction[column]) for column in feature_columns}],
        columns=feature_columns,
    )
    scaler = bundle.get("amount_scaler") or bundle.get("scaler")
    frame.loc[:, "Amount"] = scaler.transform(frame[["Amount"]])[:, 0]
    return frame


def _risk_from_raw_score(raw_score: float, score_min: float, score_max: float) -> float:
    if score_max <= score_min:
        risk = 1.0 / (1.0 + float(np.exp(raw_score * 5.0)))
    else:
        risk = 1.0 - ((raw_score - score_min) / (score_max - score_min))
    return float(np.clip(risk, 0.0, 1.0))


def _score_model(model_name: str, model: Any, frame: pd.DataFrame) -> tuple[float, float]:
    if model_name == "supervised_baseline" and hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0, 1])
        return probability, float(np.clip(probability, 0.0, 1.0))

    if hasattr(model, "decision_function"):
        raw_score = float(model.decision_function(frame)[0])
        risk_score = _risk_from_raw_score(
            raw_score=raw_score,
            score_min=float(MODEL_BUNDLE.get("score_min", -0.5)) if MODEL_BUNDLE else -0.5,
            score_max=float(MODEL_BUNDLE.get("score_max", 0.5)) if MODEL_BUNDLE else 0.5,
        )
        return raw_score, risk_score

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0, 1])
        return probability, float(np.clip(probability, 0.0, 1.0))

    raise TypeError(f"Selected model `{model_name}` does not support scoring.")


def score(transaction: Mapping[str, Any]) -> dict[str, Any]:
    if MODEL_BUNDLE is None:
        raise ModelNotLoadedError(MODEL_LOAD_ERROR or "Model is not loaded.")

    start = time.perf_counter()
    frame = _feature_frame(transaction, MODEL_BUNDLE)
    model_name, model = _select_model(MODEL_BUNDLE)
    raw_score, risk_score = _score_model(model_name, model, frame)
    is_fraud = bool(risk_score > FRAUD_RISK_THRESHOLD)
    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "transaction_id": str(transaction.get("transaction_id") or uuid.uuid4()),
        "model_name": model_name,
        "raw_anomaly_score": round(raw_score, 4),
        "risk_score": round(risk_score, 4),
        "is_fraud": is_fraud,
        "label": "SUSPICIOUS" if is_fraud else "LEGITIMATE",
        "latency_ms": round(latency_ms, 4),
    }


reload_model()
