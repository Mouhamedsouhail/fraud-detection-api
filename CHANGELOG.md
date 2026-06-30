# Changelog

## 1.2.0

- Added Sentinel Analyst mode with a named analyst persona, severity, decision queues, reason codes, and recommended actions.
- Added `POST /analyst/score` for human-readable triage output.
- Added `GET /analyst/console` for a local browser-based analyst console.
- Added `API_SCORE_PATH` so the Kafka consumer can publish plain scores or analyst-enriched scores.

## 1.1.0

- Rebranded the project as SentinelPay.
- Added batch scoring via `POST /score/batch`.
- Added `GET /` and `GET /model` endpoints.
- Added synthetic demo data generation.
- Added Dockerfile, Makefile, CI workflow, license, security notes, and architecture docs.

## 1.0.0

- Initial FastAPI scoring service, Isolation Forest training, Kafka producer/consumer, Docker Compose, and tests.
