from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

FEATURE_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
RAW_SCORE_THRESHOLD = -0.1
DATA_PATH = Path(os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "creditcard.csv")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "model" / "artifacts" / "model.pkl")))
if not DATA_PATH.is_absolute():
    DATA_PATH = PROJECT_ROOT / DATA_PATH
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH


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


def score_stats(scores: np.ndarray) -> str:
    stats = pd.Series(scores, name="decision_function").describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    )
    return stats.round(4).to_string()


def main() -> None:
    started_at = time.perf_counter()
    df = load_dataset(DATA_PATH)

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

    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x_train_scaled, y_train)
    print("Class distribution before SMOTE:")
    print(y_train.value_counts().sort_index().to_string())
    print("Class distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts().sort_index().to_string())

    model = IsolationForest(
        contamination=0.002,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_resampled[FEATURE_COLUMNS])

    test_scores = model.decision_function(x_test_scaled[FEATURE_COLUMNS])
    y_pred = (test_scores < RAW_SCORE_THRESHOLD).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Precision @ threshold {RAW_SCORE_THRESHOLD}: {precision:.4f}")
    print(f"Recall @ threshold {RAW_SCORE_THRESHOLD}: {recall:.4f}")
    print(f"F1 @ threshold {RAW_SCORE_THRESHOLD}: {f1:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print("Score distribution stats:")
    print(score_stats(test_scores))

    train_scores = model.decision_function(x_resampled[FEATURE_COLUMNS])
    artifact = {
        "model": model,
        "amount_scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "raw_score_threshold": RAW_SCORE_THRESHOLD,
        "score_min": float(np.percentile(train_scores, 1)),
        "score_max": float(np.percentile(train_scores, 99)),
        "created_at": time.time(),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    elapsed_seconds = time.perf_counter() - started_at
    print(f"Saved model artifact to {MODEL_PATH}")
    print(f"Training time: {elapsed_seconds:.4f} seconds")


if __name__ == "__main__":
    main()
