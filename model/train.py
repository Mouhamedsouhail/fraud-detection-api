from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

FEATURE_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
RAW_SCORE_THRESHOLD = -0.1
DEFAULT_RISK_THRESHOLD = 0.6
DEFAULT_TARGET_PRECISION = 0.8
PROJECT_NAME = "SentinelPay"
DATA_PATH = Path(os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "creditcard.csv")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "model" / "artifacts" / "model.pkl")))
REPORT_DIR = Path(os.getenv("REPORT_DIR", str(PROJECT_ROOT / "reports")))
if not DATA_PATH.is_absolute():
    DATA_PATH = PROJECT_ROOT / DATA_PATH
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH
if not REPORT_DIR.is_absolute():
    REPORT_DIR = PROJECT_ROOT / REPORT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentinelPay fraud models.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="Path to creditcard.csv.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Output model artifact path.")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR, help="Evaluation report directory.")
    parser.add_argument("--contamination", type=float, default=0.002, help="Isolation Forest contamination.")
    parser.add_argument("--estimators", type=int, default=200, help="Isolation Forest estimator count.")
    parser.add_argument("--threshold", type=float, default=RAW_SCORE_THRESHOLD, help="Raw score threshold.")
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=DEFAULT_RISK_THRESHOLD,
        help="Normalized risk threshold for confusion matrices.",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=DEFAULT_TARGET_PRECISION,
        help="Precision target used to report recall at fixed precision.",
    )
    parser.add_argument(
        "--default-scoring-model",
        choices=["auto", "supervised_baseline", "isolation_forest"],
        default="auto",
        help="Model selected by the API when SCORING_MODEL=auto.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv into the data directory."
        )

    df = pd.read_csv(path)
    required_columns = ["Time", *FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return df.drop(columns=["Time"])[[*FEATURE_COLUMNS, TARGET_COLUMN]]


def scale_amount(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()
    x_train_scaled.loc[:, "Amount"] = scaler.fit_transform(x_train[["Amount"]])[:, 0]
    x_test_scaled.loc[:, "Amount"] = scaler.transform(x_test[["Amount"]])[:, 0]
    return x_train_scaled, x_test_scaled, scaler


def resample_with_smote(
    x_train_scaled: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    class_counts = y_train.value_counts()
    minority_count = int(class_counts.min())
    if minority_count < 2:
        print("Skipping SMOTE because the minority class has fewer than 2 samples.")
        return x_train_scaled, y_train

    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    x_resampled, y_resampled = smote.fit_resample(x_train_scaled, y_train)
    return pd.DataFrame(x_resampled, columns=FEATURE_COLUMNS), pd.Series(y_resampled, name=TARGET_COLUMN)


def score_distribution(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": round(float(np.min(scores)), 4),
        "max": round(float(np.max(scores)), 4),
        "mean": round(float(np.mean(scores)), 4),
        "p01": round(float(np.percentile(scores, 1)), 4),
        "p05": round(float(np.percentile(scores, 5)), 4),
        "p50": round(float(np.percentile(scores, 50)), 4),
        "p95": round(float(np.percentile(scores, 95)), 4),
        "p99": round(float(np.percentile(scores, 99)), 4),
    }


def risk_from_iforest_scores(
    raw_scores: np.ndarray,
    score_min: float,
    score_max: float,
) -> np.ndarray:
    if score_max <= score_min:
        risk_scores = 1.0 / (1.0 + np.exp(raw_scores * 5.0))
    else:
        risk_scores = 1.0 - ((raw_scores - score_min) / (score_max - score_min))
    return np.clip(risk_scores, 0.0, 1.0)


def confusion_metrics(
    y_true: pd.Series,
    risk_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (risk_scores >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def recall_at_precision(
    y_true: pd.Series,
    risk_scores: np.ndarray,
    target_precision: float,
) -> dict[str, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, risk_scores)
    if len(thresholds) == 0:
        return {
            "target_precision": round(target_precision, 4),
            "precision": 0.0,
            "recall": 0.0,
            "threshold": 1.0,
        }

    valid_indices = np.where(precisions[:-1] >= target_precision)[0]
    if len(valid_indices) == 0:
        return {
            "target_precision": round(target_precision, 4),
            "precision": 0.0,
            "recall": 0.0,
            "threshold": 1.0,
        }

    best_index = int(valid_indices[np.argmax(recalls[valid_indices])])
    return {
        "target_precision": round(float(target_precision), 4),
        "recall": round(float(recalls[best_index]), 4),
        "threshold": round(float(thresholds[best_index]), 4),
        "precision": round(float(precisions[best_index]), 4),
    }


def evaluate_model(
    name: str,
    y_true: pd.Series,
    risk_scores: np.ndarray,
    risk_threshold: float,
    target_precision: float,
) -> dict[str, Any]:
    return {
        "model": name,
        "pr_auc": round(float(average_precision_score(y_true, risk_scores)), 4),
        "confusion_at_risk_threshold": confusion_metrics(y_true, risk_scores, risk_threshold),
        "recall_at_fixed_precision": recall_at_precision(y_true, risk_scores, target_precision),
        "risk_score_distribution": score_distribution(risk_scores),
    }


def write_reports(report_dir: Path, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "evaluation.json"
    markdown_path = report_dir / "evaluation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    model_rows = "\n".join(
        (
            f"| {metric['model']} | {metric['pr_auc']:.4f} | "
            f"{metric['recall_at_fixed_precision']['precision']:.4f} | "
            f"{metric['recall_at_fixed_precision']['recall']:.4f} | "
            f"{metric['recall_at_fixed_precision']['threshold']:.4f} |"
        )
        for metric in report["model_metrics"]
    )
    matrix_blocks = "\n\n".join(
        (
            f"### {metric['model']}\n\n"
            f"Threshold: `{metric['confusion_at_risk_threshold']['threshold']}`\n\n"
            "```text\n"
            f"{np.asarray(metric['confusion_at_risk_threshold']['confusion_matrix'])}\n"
            "```"
        )
        for metric in report["model_metrics"]
    )
    markdown_path.write_text(
        (
            "# SentinelPay Evaluation Report\n\n"
            f"- Dataset rows: `{report['dataset']['rows']}`\n"
            f"- Fraud rate: `{report['dataset']['fraud_rate']:.4f}`\n"
            f"- Target precision: `{report['target_precision']:.4f}`\n"
            f"- Risk threshold: `{report['risk_threshold']:.4f}`\n"
            f"- Recommended default model: `{report['recommended_default_model']}`\n\n"
            "## Model Comparison\n\n"
            "| Model | PR-AUC | Precision at fixed point | Recall at fixed precision | Threshold |\n"
            "| --- | ---: | ---: | ---: | ---: |\n"
            f"{model_rows}\n\n"
            "## Confusion Matrices\n\n"
            f"{matrix_blocks}\n"
        ),
        encoding="utf-8",
    )
    print(f"Wrote evaluation reports to {json_path} and {markdown_path}")


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    data_path = resolve_path(args.data_path)
    model_path = resolve_path(args.model_path)
    report_dir = resolve_path(args.report_dir)
    df = load_dataset(data_path)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    x_train_scaled, x_test_scaled, scaler = scale_amount(x_train, x_test)
    x_resampled, y_resampled = resample_with_smote(x_train_scaled, y_train)

    print("Class distribution before SMOTE:")
    print(y_train.value_counts().sort_index().to_string())
    print("Class distribution after SMOTE:")
    print(y_resampled.value_counts().sort_index().to_string())

    isolation_forest = IsolationForest(
        contamination=args.contamination,
        n_estimators=args.estimators,
        random_state=42,
        n_jobs=-1,
    )
    isolation_forest.fit(x_resampled[FEATURE_COLUMNS])

    supervised_baseline = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        solver="liblinear",
    )
    supervised_baseline.fit(x_resampled[FEATURE_COLUMNS], y_resampled)

    iforest_train_raw = isolation_forest.decision_function(x_resampled[FEATURE_COLUMNS])
    score_min = float(np.percentile(iforest_train_raw, 1))
    score_max = float(np.percentile(iforest_train_raw, 99))
    iforest_test_raw = isolation_forest.decision_function(x_test_scaled[FEATURE_COLUMNS])
    iforest_risk = risk_from_iforest_scores(iforest_test_raw, score_min, score_max)
    supervised_risk = supervised_baseline.predict_proba(x_test_scaled[FEATURE_COLUMNS])[:, 1]

    iforest_raw_predictions = (iforest_test_raw < args.threshold).astype(int)
    raw_iforest_metrics = {
        "threshold": round(float(args.threshold), 4),
        "precision": round(float(precision_score(y_test, iforest_raw_predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, iforest_raw_predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, iforest_raw_predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, iforest_raw_predictions).tolist(),
    }

    model_metrics = [
        evaluate_model("isolation_forest", y_test, iforest_risk, args.risk_threshold, args.target_precision),
        evaluate_model("supervised_baseline", y_test, supervised_risk, args.risk_threshold, args.target_precision),
    ]
    recommended_default_model = max(model_metrics, key=lambda item: item["pr_auc"])["model"]
    if args.default_scoring_model != "auto":
        recommended_default_model = args.default_scoring_model

    report = {
        "project": PROJECT_NAME,
        "created_at": round(time.time(), 4),
        "dataset": {
            "path": str(data_path),
            "rows": int(len(df)),
            "fraud_count": int(y.sum()),
            "fraud_rate": round(float(y.mean()), 4),
        },
        "target_precision": round(float(args.target_precision), 4),
        "risk_threshold": round(float(args.risk_threshold), 4),
        "raw_iforest_metrics": raw_iforest_metrics,
        "model_metrics": model_metrics,
        "recommended_default_model": recommended_default_model,
    }

    artifact = {
        "schema_version": "2.0",
        "project": PROJECT_NAME,
        "model": isolation_forest,
        "models": {
            "isolation_forest": isolation_forest,
            "supervised_baseline": supervised_baseline,
        },
        "default_scoring_model": recommended_default_model,
        "amount_scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "model_version": os.getenv("MODEL_VERSION", f"sentinelpay-{time.strftime('%Y%m%d%H%M%S')}"),
        "raw_score_threshold": args.threshold,
        "risk_threshold": args.risk_threshold,
        "target_precision": args.target_precision,
        "score_min": score_min,
        "score_max": score_max,
        "training_metrics": report,
        "score_distribution": {
            "isolation_forest_raw": score_distribution(iforest_test_raw),
            "isolation_forest_risk": score_distribution(iforest_risk),
            "supervised_baseline_risk": score_distribution(supervised_risk),
        },
        "created_at": time.time(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    write_reports(report_dir, report)

    for metric in model_metrics:
        fixed = metric["recall_at_fixed_precision"]
        print(
            f"{metric['model']}: PR-AUC={metric['pr_auc']:.4f}, "
            f"recall@precision>={fixed['target_precision']:.4f} is {fixed['recall']:.4f} "
            f"at threshold {fixed['threshold']:.4f}"
        )
        print("Confusion matrix at risk threshold:")
        print(np.asarray(metric["confusion_at_risk_threshold"]["confusion_matrix"]))

    elapsed_seconds = time.perf_counter() - started_at
    print(f"Recommended default model: {recommended_default_model}")
    print(f"Saved model artifact to {model_path}")
    print(f"Training time: {elapsed_seconds:.4f} seconds")


if __name__ == "__main__":
    main()
