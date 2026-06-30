# SentinelPay Fraud Detection API

[![CI](https://github.com/Mouhamedsouhail/fraud-detection-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Mouhamedsouhail/fraud-detection-api/actions/workflows/ci.yml)

SentinelPay is a streaming-ready fraud anomaly detection service built with FastAPI, scikit-learn Isolation Forest, Kafka, pandas, numpy, and Docker Compose.

It scores credit card transaction vectors in real time, exposes operational and Prometheus metrics, supports batch scoring, includes Kafka replay tools, compares unsupervised and supervised models, and adds a human-readable analyst persona named **Maya** for fraud triage.

## Why This Project Stands Out

- **Real API, not just a notebook**: train a model, load it once, serve `/score`, `/score/batch`, `/health`, `/metrics`, and `/model`.
- **Streaming workflow included**: replay transactions into Kafka and publish enriched fraud results.
- **Humanoid analyst mode**: Maya converts anomaly scores into severity, review queues, reason codes, and recommended actions.
- **Actual model comparison**: training reports PR-AUC, recall at fixed precision, and confusion matrices for Isolation Forest and a supervised logistic baseline.
- **Ready-to-run API**: the repo includes a tiny synthetic-data demo model so `/score` works immediately after install.
- **Fraud operations workflow**: analyst cases, dispositions, feedback, and retraining candidates are built into the API.
- **Monitoring included**: structured JSON logs and Prometheus metrics are exposed without extra services.
- **Demo-friendly**: generate synthetic `creditcard.csv` data when you want to test the pipeline before downloading Kaggle data.
- **Public-repo ready**: CI, Dockerfile, Makefile, model card, architecture docs, security notes, and contribution guide.
- **Production-minded defaults**: no hardcoded secrets, environment-based config, ignored datasets/artifacts, and explicit risk-score caveats.

## Stack

- Python 3.11+
- FastAPI
- scikit-learn Isolation Forest
- imbalanced-learn SMOTE
- Kafka via `confluent-kafka-python`
- Docker Compose for Kafka and Zookeeper
- pandas, numpy, joblib
- pytest

## Project Structure

```text
fraud-detection-api/
├── api/                    # FastAPI app, schemas, model scoring
├── data/                   # creditcard.csv goes here, ignored by Git
├── docs/                   # architecture and model card
├── model/                  # training script and artifact directory
├── reports/                # generated model comparison reports
├── scripts/                # local demo utilities
├── streaming/              # Kafka producer and consumer
├── tests/                  # API and scorer tests
├── docker-compose.yml      # Kafka + Zookeeper + topic init
├── Dockerfile              # API container image
├── Makefile                # common developer commands
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/Mouhamedsouhail/fraud-detection-api.git
cd fraud-detection-api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

The API can run immediately with the bundled synthetic-data demo model:

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://localhost:8000/dashboard
http://localhost:8000/analyst/console
http://localhost:8000/docs
```

Start the full Kafka + API + producer + consumer stack:

```bash
docker compose up --build
```

## Option A: Demo Without Kaggle

Generate synthetic data that matches the Kaggle file shape and retrain the local model:

```bash
python scripts/generate_demo_data.py --rows 5000 --fraud-rate 0.02
python model/train.py
uvicorn api.main:app --reload
```

The synthetic generator is for local demos only. It is intentionally not a substitute for real model evaluation.

## Option B: Use the Kaggle ULB Dataset

Download the dataset from [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```text
data/creditcard.csv
```

Then train:

```bash
python model/train.py
```

The training script:

- drops `Time`
- keeps `V1` through `V28`, `Amount`, and `Class`
- scales `Amount` with `StandardScaler`
- applies SMOTE on the training split
- trains `IsolationForest(contamination=0.002, n_estimators=200, random_state=42)`
- trains a supervised `LogisticRegression(class_weight="balanced")` baseline
- prints and writes PR-AUC, recall at fixed precision, confusion matrices, training time, and score distribution stats
- saves `model/artifacts/model.pkl`
- writes `reports/evaluation.json` and `reports/evaluation.md`

## Run the API

```bash
uvicorn api.main:app --reload
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:8000/metrics/prometheus`
- Model metadata: `http://localhost:8000/model`
- Analyst mode: `http://localhost:8000/analyst/score`
- Analyst console: `http://localhost:8000/analyst/console`
- Live dashboard: `http://localhost:8000/dashboard`

## Score One Transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -H "X-Transaction-ID: demo-001" \
  -d '{
    "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0,
    "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0,
    "V9": 0.0, "V10": 0.0, "V11": 0.0, "V12": 0.0,
    "V13": 0.0, "V14": 0.0, "V15": 0.0, "V16": 0.0,
    "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
    "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0,
    "Amount": 42.0
  }'
```

Example response:

```json
{
  "transaction_id": "demo-001",
  "risk_score": 0.3821,
  "is_fraud": false,
  "label": "LEGITIMATE",
  "latency_ms": 2.1049
}
```

## Batch Score

```bash
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [
    {
      "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0,
      "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0,
      "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0,
      "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0,
      "Amount": 42
    }
  ]}'
```

## Analyst Mode

Maya is SentinelPay's deterministic fraud analyst layer. She does not use an LLM or any external API. She turns the model output into a human review object:

- `severity`: `LOW`, `ELEVATED`, `HIGH`, or `CRITICAL`
- `decision_queue`: `auto_approve`, `watchlist`, `manual_review`, or `manual_review_urgent`
- `analyst.summary`: plain-language triage
- `reason_codes`: top latent signals and amount signals
- `recommended_actions`: next best operational steps

```bash
curl -X POST http://localhost:8000/analyst/score \
  -H "Content-Type: application/json" \
  -H "X-Transaction-ID: tx-human" \
  -d '{
    "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0,
    "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0,
    "V9": 0.0, "V10": 0.0, "V11": 0.0, "V12": 0.0,
    "V13": 0.0, "V14": -4.2, "V15": 0.0, "V16": 0.0,
    "V17": 3.1, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
    "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0,
    "Amount": 7500.0
  }'
```

For a local browser view, open:

```text
http://localhost:8000/analyst/console
```

## Stream Transactions

Terminal 1:

```bash
python streaming/producer.py --rate 50
```

Terminal 2:

```bash
python streaming/consumer.py
```

The producer publishes rows to `transactions`. The consumer calls `/score`, then publishes the original transaction plus the score response to `fraud-results`.

To publish Maya's analyst-enriched response instead:

```bash
API_SCORE_PATH=/analyst/score python streaming/consumer.py
```

## Investigation Workflow

Maya automatically opens in-memory cases for transactions routed to `watchlist`, `manual_review`, or `manual_review_urgent`.

```bash
curl http://localhost:8000/cases
curl http://localhost:8000/events/recent
curl http://localhost:8000/retraining/candidates
```

Record disposition:

```bash
curl -X POST http://localhost:8000/cases/case-tx-human/disposition \
  -H "Content-Type: application/json" \
  -d '{"disposition":"CONFIRMED_FRAUD","analyst_id":"analyst-1","notes":"Confirmed after review."}'
```

Record analyst feedback:

```bash
curl -X POST http://localhost:8000/cases/case-tx-human/feedback \
  -H "Content-Type: application/json" \
  -d '{"analyst_id":"analyst-1","useful":true,"notes":"Reason codes were useful.","corrected_label":"SUSPICIOUS"}'
```

## Configuration

Environment variables are loaded from `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `KAFKA_BROKER` | `localhost:9092` | Kafka bootstrap server |
| `API_URL` | `http://localhost:8000` | FastAPI base URL used by the consumer |
| `API_SCORE_PATH` | `/score` | Score endpoint used by the consumer |
| `API_TIMEOUT_SECONDS` | `5` | HTTP timeout for the consumer |
| `MODEL_PATH` | `model/artifacts/demo_model.pkl` | Model artifact path |
| `MODEL_VERSION` | `local` | Optional version label stored in metadata |
| `SCORING_MODEL` | `auto` | `auto`, `supervised_baseline`, or `isolation_forest` |
| `FRAUD_RISK_THRESHOLD` | `0.6` | Fraud flag threshold for normalized risk |
| `METRICS_WINDOW_SIZE` | `10000` | Number of recent latencies retained |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated CORS origins |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `STRUCTURED_LOGS` | `true` | Emit JSON application logs |

## Risk Score Interpretation

The Kaggle ULB dataset has an expected fraud rate of about **0.17%**, so fraud alerts should be rare. SentinelPay maps the Isolation Forest decision score to a normalized `risk_score` from `0` to `1`, where higher means more anomalous. A transaction is labeled `SUSPICIOUS` when:

```text
risk_score > 0.6
```

This score is an anomaly signal, not proof of fraud. Tune thresholds against validation data and analyst review capacity before using it in production.

## Developer Commands

```bash
make install
make demo-data
make train
make test
make compile
make compose-up
make api
```

On Windows without `make`, run the equivalent commands shown in the Makefile.

## Tests

```bash
pytest
python -m compileall api model streaming scripts tests
docker compose config
```

The tests monkeypatch a fake model so they run before `creditcard.csv` is downloaded and before `model.pkl` exists.

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Model Card](docs/MODEL_CARD.md)
- [Analyst Mode](docs/ANALYST_MODE.md)
- [Threshold Tuning](docs/THRESHOLD_TUNING.md)
- [Operations](docs/OPERATIONS.md)
- [Security Notes](SECURITY.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT
