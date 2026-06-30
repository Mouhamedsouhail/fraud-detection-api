from __future__ import annotations

from fastapi.testclient import TestClient

from api import main
from api import scorer


def demo_payload() -> dict[str, float]:
    return {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 42.0}


def test_bundled_demo_model_loads_and_scores() -> None:
    assert scorer.ensure_model_loaded()

    result = scorer.score(demo_payload())

    assert result["model_name"] in {"supervised_baseline", "isolation_forest"}
    assert 0 <= result["risk_score"] <= 1
    assert isinstance(result["is_fraud"], bool)


def test_score_endpoint_works_with_bundled_demo_model() -> None:
    main.reset_metrics()

    with TestClient(main.app) as client:
        response = client.post(
            "/score",
            json=demo_payload(),
            headers={"X-Transaction-ID": "demo-ready"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "demo-ready"
    assert body["model_name"] in {"supervised_baseline", "isolation_forest"}
    assert 0 <= body["risk_score"] <= 1
