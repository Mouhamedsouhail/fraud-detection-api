# Fraud Detection API

Real-time fraud scoring API using FastAPI, scikit-learn Isolation Forest, Kafka, pandas, numpy, and joblib.

## Setup

Use Python 3.11 or newer.

```bash
cd fraud-detection-api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
```

Download the Kaggle ULB credit card fraud dataset and place the CSV at:

```text
data/creditcard.csv
```

Dataset link: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains anonymized PCA features `V1` through `V28`, `Time`, `Amount`, and `Class`.

## Train

```bash
python model/train.py
```

The training script:

- drops `Time`
- scales `Amount` with `StandardScaler`
- applies SMOTE to the training split
- trains `IsolationForest(contamination=0.002, n_estimators=200, random_state=42)`
- evaluates precision, recall, F1, and confusion matrix at raw score threshold `-0.1`
- saves `model/artifacts/model.pkl`

## Run API

```bash
uvicorn api.main:app --reload
```

Health and metrics:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Example scoring request:

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

## Stream

Run the producer in one terminal:

```bash
python streaming/producer.py --rate 50
```

Run the consumer in another terminal:

```bash
python streaming/consumer.py
```

The producer publishes transactions to `transactions`. The consumer calls `/score`, then publishes the original transaction plus the score response to `fraud-results`.

## Configuration

Environment variables are loaded from `.env` with `python-dotenv`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `KAFKA_BROKER` | `localhost:9092` | Kafka bootstrap server |
| `API_URL` | `http://localhost:8000` | FastAPI base URL used by the consumer |
| `MODEL_PATH` | `model/artifacts/model.pkl` | Model artifact path |
| `FRAUD_RISK_THRESHOLD` | `0.6` | Fraud flag threshold for normalized risk |
| `METRICS_WINDOW_SIZE` | `10000` | Number of recent latencies retained |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated CORS origins |

## Risk Scores

The original Kaggle dataset has an expected fraud rate of about 0.17 percent, so fraud alerts should be rare. The API maps the Isolation Forest decision score to a normalized `risk_score` from `0` to `1`, where higher means more anomalous. A transaction is labeled `SUSPICIOUS` when `risk_score > 0.6`; otherwise it is `LEGITIMATE`.

This score is an anomaly signal, not proof of fraud. Tune the threshold against validation data and operational review capacity before using it in production.

## Tests

```bash
pytest
```

The tests monkeypatch a fake model so they can run before `creditcard.csv` is downloaded and before `model.pkl` exists.
