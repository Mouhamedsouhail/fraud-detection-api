from __future__ import annotations

import numpy as np

from api import scorer


class FakeScaler:
    def transform(self, values: object) -> np.ndarray:
        return np.asarray(values, dtype=float)


class FakeModel:
    def decision_function(self, frame: object) -> np.ndarray:
        return np.asarray([-0.2], dtype=float)


def synthetic_transaction(amount: float = 25.0) -> dict[str, float]:
    return {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": amount}


def test_score_returns_bounded_risk_and_bool_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        scorer,
        "MODEL_BUNDLE",
        {
            "model": FakeModel(),
            "amount_scaler": FakeScaler(),
            "feature_columns": scorer.FEATURE_COLUMNS,
            "score_min": -0.5,
            "score_max": 0.5,
        },
    )

    result = scorer.score(synthetic_transaction())

    assert 0 <= result["risk_score"] <= 1
    assert isinstance(result["is_fraud"], bool)
    assert "raw_anomaly_score" in result
    assert result["label"] in {"SUSPICIOUS", "LEGITIMATE"}
