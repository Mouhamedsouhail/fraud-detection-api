from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


RESERVED_FIELDS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in RESERVED_FIELDS and not key.startswith("_")
        }
        payload.update(extras)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    root = logging.getLogger()
    if getattr(root, "_sentinelpay_configured", False):
        return

    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    structured_logs = os.getenv("STRUCTURED_LOGS", "true").lower() in {"1", "true", "yes", "on"}
    if structured_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root.addHandler(handler)
    root.setLevel((level or os.getenv("LOG_LEVEL", "INFO")).upper())
    setattr(root, "_sentinelpay_configured", True)
