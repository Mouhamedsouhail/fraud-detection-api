# Changelog

## 2.0.0

- Added supervised logistic baseline training beside Isolation Forest.
- Added PR-AUC, recall at fixed precision, and confusion matrix reports.
- Added a bundled synthetic-data demo model for immediate API startup.
- Added Prometheus metrics at `/metrics/prometheus`.
- Added structured JSON logging.
- Added live dashboard at `/dashboard`.
- Added investigation cases, dispositions, analyst feedback, and retraining candidate endpoints.
- Expanded Docker Compose to run API, Kafka, producer, and consumer as a full stack.

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
