from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from api import main


def review_payload(amount: float = 7500.0) -> dict[str, float]:
    values = {f"V{i}": 0.0 for i in range(1, 29)}
    values["V14"] = -4.2
    values["V17"] = 3.1
    values["Amount"] = amount
    return values


def fake_high_risk_score(transaction: dict[str, Any]) -> dict[str, Any]:
    return {
        "transaction_id": str(transaction.get("transaction_id", "tx-case")),
        "model_name": "supervised_baseline",
        "raw_anomaly_score": 0.91,
        "risk_score": 0.91,
        "is_fraud": True,
        "label": "SUSPICIOUS",
        "latency_ms": 3.0,
    }


def test_dashboard_and_prometheus_metrics_load() -> None:
    with TestClient(main.app) as client:
        dashboard = client.get("/dashboard")
        prometheus = client.get("/metrics/prometheus")

    assert dashboard.status_code == 200
    assert "SentinelPay Live Dashboard" in dashboard.text
    assert prometheus.status_code == 200
    assert "sentinelpay_scores_total" in prometheus.text


def test_case_workflow_and_retraining_candidates(monkeypatch) -> None:
    main.reset_metrics()
    monkeypatch.setattr(main, "score_transaction", fake_high_risk_score)

    with TestClient(main.app) as client:
        score_response = client.post(
            "/analyst/score",
            json=review_payload(),
            headers={"X-Transaction-ID": "tx-case"},
        )
        assert score_response.status_code == 200

        cases_response = client.get("/cases?status=OPEN")
        assert cases_response.status_code == 200
        cases = cases_response.json()
        assert len(cases) == 1
        case_id = cases[0]["case_id"]

        disposition_response = client.post(
            f"/cases/{case_id}/disposition",
            json={
                "disposition": "CONFIRMED_FRAUD",
                "analyst_id": "maya-reviewer",
                "notes": "Confirmed after review.",
            },
        )
        assert disposition_response.status_code == 200
        assert disposition_response.json()["status"] == "CLOSED"

        feedback_response = client.post(
            f"/cases/{case_id}/feedback",
            json={
                "analyst_id": "maya-reviewer",
                "useful": True,
                "notes": "Reason codes matched the review.",
                "corrected_label": "SUSPICIOUS",
            },
        )
        assert feedback_response.status_code == 200

        candidates_response = client.get("/retraining/candidates")
        assert candidates_response.status_code == 200
        assert candidates_response.json()[0]["case_id"] == case_id


def test_recent_events_after_analyst_score(monkeypatch) -> None:
    main.reset_metrics()
    monkeypatch.setattr(main, "score_transaction", fake_high_risk_score)

    with TestClient(main.app) as client:
        response = client.post(
            "/analyst/score",
            json=review_payload(),
            headers={"X-Transaction-ID": "tx-event"},
        )
        assert response.status_code == 200
        events_response = client.get("/events/recent?limit=5")

    assert events_response.status_code == 200
    events = events_response.json()
    assert events[0]["transaction_id"] == "tx-event"
    assert events[0]["analyst_summary"].startswith("Maya:")
