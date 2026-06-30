from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from api import main
from api.analyst import build_analyst_report


def valid_payload(amount: float = 7500.0) -> dict[str, float]:
    values = {f"V{i}": 0.0 for i in range(1, 29)}
    values["V14"] = -4.2
    values["V17"] = 3.1
    values["Amount"] = amount
    return values


def fake_score(transaction: dict[str, Any]) -> dict[str, Any]:
    return {
        "transaction_id": str(transaction.get("transaction_id", "tx-human")),
        "raw_anomaly_score": -0.4,
        "risk_score": 0.88,
        "is_fraud": True,
        "label": "SUSPICIOUS",
        "latency_ms": 2.5,
    }


def test_build_analyst_report_contains_human_triage() -> None:
    report = build_analyst_report(valid_payload(), fake_score({"transaction_id": "tx-human"}))

    assert report["severity"] == "CRITICAL"
    assert report["decision_queue"] == "manual_review_urgent"
    assert report["analyst"]["name"] == "Maya"
    assert report["reason_codes"]
    assert report["recommended_actions"]


def test_analyst_score_endpoint(monkeypatch) -> None:
    main.reset_metrics()
    monkeypatch.setattr(main, "score_transaction", fake_score)

    with TestClient(main.app) as client:
        response = client.post(
            "/analyst/score",
            json=valid_payload(),
            headers={"X-Transaction-ID": "tx-human"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "tx-human"
    assert body["severity"] == "CRITICAL"
    assert body["analyst"]["summary"].startswith("Maya:")
    assert body["reason_codes"][0]["weight"] <= 1


def test_analyst_console_loads() -> None:
    with TestClient(main.app) as client:
        response = client.get("/analyst/console")

    assert response.status_code == 200
    assert "SentinelPay Analyst Console" in response.text
