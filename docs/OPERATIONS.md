# Operations

## Full-Stack Demo

The Compose stack runs:

- Zookeeper
- Kafka
- topic initializer
- FastAPI service
- synthetic transaction producer
- analyst-enriched consumer

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000/dashboard
http://localhost:8000/analyst/console
http://localhost:8000/docs
```

The producer generates synthetic demo transactions in its own container, publishes to `transactions`, and the consumer calls `/analyst/score` before publishing to `fraud-results`.

## Prometheus Metrics

Prometheus text format is exposed at:

```text
GET /metrics/prometheus
```

Key metrics:

- `sentinelpay_scores_total`
- `sentinelpay_score_latency_seconds`
- `sentinelpay_fraud_rate`
- `sentinelpay_cases_total`

## Structured Logs

JSON logs are enabled by default:

```env
STRUCTURED_LOGS=true
LOG_LEVEL=INFO
```

Score and case workflow events include transaction IDs, risk scores, labels, severity, queue names, and analyst IDs where applicable.

## Case Workflow

```text
POST /analyst/score
GET /cases
GET /cases/{case_id}
POST /cases/{case_id}/disposition
POST /cases/{case_id}/feedback
GET /retraining/candidates
```

Cases are in-memory for the demo. A production deployment should replace this with a database and immutable audit trail.
