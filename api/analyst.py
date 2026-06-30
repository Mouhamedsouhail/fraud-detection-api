from __future__ import annotations

from typing import Any, Mapping


ANALYST_NAME = "Maya"
ANALYST_ROLE = "SentinelPay fraud analyst"

FEATURE_SIGNAL_LABELS: dict[str, str] = {
    "V1": "profile drift",
    "V2": "spend path deviation",
    "V3": "merchant-context anomaly",
    "V4": "authorization behavior shift",
    "V5": "account velocity signal",
    "V6": "network pattern deviation",
    "V7": "purchase shape anomaly",
    "V8": "session consistency shift",
    "V9": "timing pattern deviation",
    "V10": "counterparty risk signal",
    "V11": "authentication pattern shift",
    "V12": "historical behavior drift",
    "V13": "merchant cluster deviation",
    "V14": "fraud-like latent pattern",
    "V15": "customer context drift",
    "V16": "device/session anomaly",
    "V17": "authorization fingerprint shift",
    "V18": "behavioral distance signal",
    "V19": "settlement pattern deviation",
    "V20": "transaction profile shift",
    "V21": "edge-case latent signal",
    "V22": "routing pattern deviation",
    "V23": "amount-context interaction",
    "V24": "merchant timing signal",
    "V25": "account posture drift",
    "V26": "verification pattern shift",
    "V27": "rare-event latent signal",
    "V28": "residual profile anomaly",
}


def severity_from_risk(risk_score: float) -> str:
    if risk_score >= 0.85:
        return "CRITICAL"
    if risk_score >= 0.7:
        return "HIGH"
    if risk_score >= 0.6:
        return "ELEVATED"
    return "LOW"


def _feature_reason(feature: str, value: float) -> dict[str, Any]:
    direction = "above" if value > 0 else "below"
    return {
        "code": f"LATENT_{feature}_SHIFT",
        "signal": FEATURE_SIGNAL_LABELS.get(feature, "latent anomaly signal"),
        "detail": (
            f"{feature} is {direction} the local baseline with magnitude "
            f"{abs(value):.4f}. This is an anonymized PCA signal, not a named behavior."
        ),
        "weight": round(min(abs(value) / 6.0, 1.0), 4),
    }


def build_reason_codes(
    transaction: Mapping[str, Any],
    score_result: Mapping[str, Any],
    max_reasons: int = 4,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    amount = float(transaction.get("Amount", 0.0))
    risk_score = float(score_result["risk_score"])

    if amount >= 5000:
        reasons.append(
            {
                "code": "HIGH_AMOUNT",
                "signal": "large transaction amount",
                "detail": f"Amount is {amount:.4f}, which should receive additional review.",
                "weight": round(min(amount / 25000.0, 1.0), 4),
            }
        )
    elif amount <= 1:
        reasons.append(
            {
                "code": "MICRO_AMOUNT",
                "signal": "very small transaction amount",
                "detail": f"Amount is {amount:.4f}; small authorization probes can precede abuse.",
                "weight": 0.35,
            }
        )

    ranked_features = sorted(
        (
            (feature, float(transaction.get(feature, 0.0)))
            for feature in FEATURE_SIGNAL_LABELS
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    for feature, value in ranked_features:
        if len(reasons) >= max_reasons:
            break
        if abs(value) >= 2.5:
            reasons.append(_feature_reason(feature, value))

    if not reasons:
        reasons.append(
            {
                "code": "MODEL_CONSENSUS",
                "signal": "model risk posture",
                "detail": (
                    "No single transaction field stands out strongly; the decision is driven "
                    f"by the model's combined anomaly score of {risk_score:.4f}."
                ),
                "weight": round(risk_score, 4),
            }
        )

    return reasons[:max_reasons]


def recommended_actions(severity: str, is_fraud: bool) -> list[str]:
    if severity == "CRITICAL":
        return [
            "Temporarily hold the transaction.",
            "Require step-up verification before approval.",
            "Open an analyst case with the top reason codes attached.",
        ]
    if severity == "HIGH":
        return [
            "Route to manual review.",
            "Request additional verification if customer friction is acceptable.",
            "Monitor the next account events for velocity changes.",
        ]
    if severity == "ELEVATED" or is_fraud:
        return [
            "Queue for lightweight review.",
            "Allow only if business rules and customer context agree.",
            "Record feedback after analyst disposition.",
        ]
    return [
        "Approve automatically if no external rule blocks it.",
        "Keep the transaction in monitoring metrics.",
    ]


def build_summary(
    score_result: Mapping[str, Any],
    reasons: list[Mapping[str, Any]],
    severity: str,
) -> str:
    risk_score = float(score_result["risk_score"])
    label = str(score_result["label"]).lower()
    top_signal = reasons[0]["signal"] if reasons else "the combined anomaly pattern"
    if severity in {"CRITICAL", "HIGH"}:
        stance = "I would slow this down for review"
    elif severity == "ELEVATED":
        stance = "I would not block this automatically, but I would keep it in the review lane"
    else:
        stance = "I would let this pass unless another rule disagrees"

    return (
        f"{ANALYST_NAME}: This transaction looks {label} with risk {risk_score:.4f}. "
        f"The strongest signal is {top_signal}. {stance}."
    )


def decision_queue(severity: str, is_fraud: bool) -> str:
    if severity == "CRITICAL":
        return "manual_review_urgent"
    if severity == "HIGH":
        return "manual_review"
    if severity == "ELEVATED" or is_fraud:
        return "watchlist"
    return "auto_approve"


def build_analyst_report(
    transaction: Mapping[str, Any],
    score_result: Mapping[str, Any],
) -> dict[str, Any]:
    risk_score = float(score_result["risk_score"])
    severity = severity_from_risk(risk_score)
    reasons = build_reason_codes(transaction, score_result)

    return {
        "transaction_id": str(score_result["transaction_id"]),
        "risk_score": round(risk_score, 4),
        "is_fraud": bool(score_result["is_fraud"]),
        "label": str(score_result["label"]),
        "severity": severity,
        "decision_queue": decision_queue(severity, bool(score_result["is_fraud"])),
        "analyst": {
            "name": ANALYST_NAME,
            "role": ANALYST_ROLE,
            "summary": build_summary(score_result, reasons, severity),
        },
        "reason_codes": reasons,
        "recommended_actions": recommended_actions(severity, bool(score_result["is_fraud"])),
        "latency_ms": round(float(score_result["latency_ms"]), 4),
    }
