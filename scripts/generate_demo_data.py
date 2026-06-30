from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic creditcard.csv-compatible dataset for local demos. "
            "This is not a replacement for the Kaggle ULB dataset."
        )
    )
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows to generate.")
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.02,
        help="Synthetic fraud ratio. Use a higher demo rate than production so SMOTE has samples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    args = parser.parse_args()
    if args.rows < 100:
        parser.error("--rows must be at least 100.")
    if args.fraud_rate <= 0 or args.fraud_rate >= 0.5:
        parser.error("--fraud-rate must be greater than 0 and less than 0.5.")
    return args


def build_dataset(rows: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fraud_count = max(8, int(rows * fraud_rate))
    labels = np.zeros(rows, dtype=int)
    fraud_indices = rng.choice(rows, size=fraud_count, replace=False)
    labels[fraud_indices] = 1

    features = rng.normal(loc=0.0, scale=1.0, size=(rows, len(FEATURE_COLUMNS)))
    anomaly_columns = [2, 3, 9, 11, 13, 16, 26]
    features[np.ix_(fraud_indices, anomaly_columns)] += rng.normal(
        loc=-3.0,
        scale=0.75,
        size=(fraud_count, len(anomaly_columns)),
    )
    features[np.ix_(fraud_indices, [0, 6, 17])] += rng.normal(
        loc=2.5,
        scale=0.6,
        size=(fraud_count, 3),
    )

    amount = rng.lognormal(mean=3.2, sigma=1.0, size=rows)
    amount[fraud_indices] *= rng.uniform(1.5, 5.0, size=fraud_count)
    amount = np.clip(amount, 0.01, 25000.0)

    data = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    data.insert(0, "Time", np.sort(rng.integers(0, 172800, size=rows)))
    data["Amount"] = amount.round(2)
    data["Class"] = labels
    return data


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_dataset(rows=args.rows, fraud_rate=args.fraud_rate, seed=args.seed)
    data.to_csv(output_path, index=False)
    fraud_rate = round(float(data["Class"].mean()), 4)
    print(f"Wrote {len(data)} rows to {output_path}")
    print(f"Synthetic fraud rate: {fraud_rate}")
    print("Use this only for demos; train against the Kaggle ULB dataset for real evaluation.")


if __name__ == "__main__":
    main()
