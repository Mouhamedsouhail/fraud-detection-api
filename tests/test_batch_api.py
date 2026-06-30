from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from api import main


def valid_payload(amount: float = 12.34) -> dict[str, float]:
    return {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": amount}


def fake_score(transaction: dict[str, Any]) -> dict[str, Any]:
    return {
        "transaction_id": str(transaction.get("transaction_id", "tx-batch")),
        "raw_anomaly_score": -0.2,
        "risk_score": 0.65,
        "is_fraud": True,
        "label": "SUSPICIOUS",
        "latency_ms": 0.5,
    }


def test_batch_score_endpoint(monkeypatch) -> None:
    main.reset_metrics()
    monkeypatch.setattr(main, "score_transaction", fake_score)

    with TestClient(main.app) as client:
        response = client.post(
            "/score/batch",
            json={"transactions": [valid_payload(10.0), valid_payload(20.0)]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_count"] == 2
    assert len(body["scores"]) == 2
    assert body["scores"][0]["label"] == "SUSPICIOUS"
    assert body["latency_ms"] >= 0
