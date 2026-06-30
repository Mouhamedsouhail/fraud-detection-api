from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Mapping


RECENT_EVENTS_LIMIT = 250
_lock = Lock()
_recent_events: deque[dict[str, Any]] = deque(maxlen=RECENT_EVENTS_LIMIT)
_cases: dict[str, dict[str, Any]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def record_score_event(
    score_result: Mapping[str, Any],
    analyst_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "timestamp": now_iso(),
        "transaction_id": str(score_result["transaction_id"]),
        "model_name": str(score_result.get("model_name", "unknown")),
        "risk_score": round(float(score_result["risk_score"]), 4),
        "is_fraud": bool(score_result["is_fraud"]),
        "label": str(score_result["label"]),
        "latency_ms": round(float(score_result["latency_ms"]), 4),
        "severity": analyst_report.get("severity") if analyst_report else None,
        "decision_queue": analyst_report.get("decision_queue") if analyst_report else None,
        "analyst_summary": (
            analyst_report.get("analyst", {}).get("summary") if analyst_report else None
        ),
    }
    with _lock:
        _recent_events.appendleft(event)
    return deepcopy(event)


def recent_events(limit: int = 50) -> list[dict[str, Any]]:
    with _lock:
        return deepcopy(list(_recent_events)[: max(1, min(limit, RECENT_EVENTS_LIMIT))])


def maybe_open_case(
    transaction: Mapping[str, Any],
    analyst_report: Mapping[str, Any],
) -> dict[str, Any] | None:
    queue = str(analyst_report["decision_queue"])
    if queue == "auto_approve":
        return None

    transaction_id = str(analyst_report["transaction_id"])
    case_id = f"case-{transaction_id}"
    with _lock:
        existing = _cases.get(case_id)
        if existing:
            existing["updated_at"] = now_iso()
            existing["score"] = _round_floats(deepcopy(dict(analyst_report)))
            return deepcopy(existing)

        case = {
            "case_id": case_id,
            "transaction_id": transaction_id,
            "status": "OPEN",
            "queue": queue,
            "severity": str(analyst_report["severity"]),
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "transaction": _round_floats(deepcopy(dict(transaction))),
            "score": _round_floats(deepcopy(dict(analyst_report))),
            "disposition": None,
            "feedback": [],
            "retraining_candidate": False,
        }
        _cases[case_id] = case
        return deepcopy(case)


def list_cases(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    with _lock:
        cases = list(_cases.values())
    if status:
        cases = [case for case in cases if str(case["status"]).upper() == status.upper()]
    cases.sort(key=lambda case: str(case["updated_at"]), reverse=True)
    return deepcopy(cases[: max(1, min(limit, 500))])


def get_case(case_id: str) -> dict[str, Any] | None:
    with _lock:
        case = _cases.get(case_id)
        return deepcopy(case) if case else None


def add_disposition(
    case_id: str,
    disposition: str,
    analyst_id: str,
    notes: str,
) -> dict[str, Any] | None:
    with _lock:
        case = _cases.get(case_id)
        if not case:
            return None
        case["disposition"] = {
            "disposition": disposition,
            "analyst_id": analyst_id,
            "notes": notes,
            "created_at": now_iso(),
        }
        case["status"] = "ESCALATED" if disposition == "ESCALATED" else "CLOSED"
        case["updated_at"] = now_iso()
        case["retraining_candidate"] = disposition in {"CONFIRMED_FRAUD", "FALSE_POSITIVE"}
        return deepcopy(case)


def add_feedback(
    case_id: str,
    analyst_id: str,
    useful: bool,
    notes: str,
    corrected_label: str | None,
) -> dict[str, Any] | None:
    with _lock:
        case = _cases.get(case_id)
        if not case:
            return None
        case["feedback"].append(
            {
                "analyst_id": analyst_id,
                "useful": useful,
                "notes": notes,
                "corrected_label": corrected_label,
                "created_at": now_iso(),
            }
        )
        case["updated_at"] = now_iso()
        case["retraining_candidate"] = True
        return deepcopy(case)


def retraining_candidates() -> list[dict[str, Any]]:
    with _lock:
        candidates = [case for case in _cases.values() if case.get("retraining_candidate")]
    candidates.sort(key=lambda case: str(case["updated_at"]), reverse=True)
    return deepcopy(candidates)


def reset_cases() -> None:
    with _lock:
        _recent_events.clear()
        _cases.clear()
