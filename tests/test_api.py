from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from api import main


def valid_payload(amount: float = 12.34) -> dict[str, float]:
    return {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": amount}


def fake_score(transaction: dict[str, Any]) -> dict[str, Any]:
    return {
        "transaction_id": str(transaction.get("transaction_id", "tx-test")),
        "raw_anomaly_score": -0.2,
        "risk_score": 0.7,
        "is_fraud": True,
        "label": "SUSPICIOUS",
        "latency_ms": 1.2345,
    }


def test_score_endpoint_with_valid_input(monkeypatch) -> None:
    main.reset_metrics()
    monkeypatch.setattr(main, "score_transaction", fake_score)

    with TestClient(main.app) as client:
        response = client.post(
            "/score",
            json=valid_payload(),
            headers={"X-Transaction-ID": "tx-123"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "tx-123"
    assert 0 <= body["risk_score"] <= 1
    assert isinstance(body["is_fraud"], bool)


def test_health_endpoint() -> None:
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "model_loaded" in body
    assert "uptime_seconds" in body


def test_metrics_after_scores(monkeypatch) -> None:
    main.reset_metrics()
    monkeypatch.setattr(main, "score_transaction", fake_score)

    with TestClient(main.app) as client:
        for _ in range(3):
            score_response = client.post("/score", json=valid_payload())
            assert score_response.status_code == 200
        metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200
    body = metrics_response.json()
    assert body["total_scored"] == 3
    assert body["fraud_count"] == 3
    assert body["fraud_rate"] == 1.0
    assert body["avg_latency_ms"] == 1.2345
    assert body["p50_latency_ms"] == 1.2345
    assert body["p99_latency_ms"] == 1.2345
