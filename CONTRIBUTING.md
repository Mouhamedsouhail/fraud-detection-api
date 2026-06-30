# Contributing

Thanks for improving SentinelPay.

## Local Checks

```bash
python -m pip install -r requirements.txt
pytest
python -m compileall api model streaming scripts tests
docker compose config
```

## Development Flow

1. Create a focused branch.
2. Keep model artifacts and raw datasets out of Git.
3. Add or update tests for changed API behavior.
4. Include clear notes when changing thresholds, feature order, or Kafka topics.

## Data Policy

Do not commit real cardholder data, Kaggle CSV files, model artifacts, `.env` files, credentials, or logs containing customer identifiers.
