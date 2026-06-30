# Model Card

## Model

- Name: SentinelPay Isolation Forest
- Type: Unsupervised anomaly detection
- Library: scikit-learn `IsolationForest`
- Default contamination: `0.002`
- Default estimators: `200`
- Risk threshold: normalized `risk_score > 0.6`

## Intended Use

The model is intended to prioritize credit card transactions for fraud review. It should be used as an anomaly signal alongside human review, business rules, and downstream investigation tools. Analyst Mode adds human-readable triage, but it does not convert the model into a final fraud authority.

## Dataset

The intended dataset is the Kaggle ULB credit card fraud dataset, containing 284,807 European cardholder transactions across two days with 492 fraud cases. The repository does not include this dataset. Users must download it separately from Kaggle.

## Features

- `V1` through `V28`: anonymized PCA components
- `Amount`: scaled with `StandardScaler`
- `Time`: dropped during training
- `Class`: used for evaluation and SMOTE, never sent through the live scoring payload

## Limitations

- The model is not calibrated as a probability estimator.
- Fraud behavior changes over time, so thresholds should be monitored and retrained.
- The Kaggle dataset is anonymized and historical; it does not represent every merchant or payment network.
- The synthetic demo data generator is only for local smoke testing and presentations.
- Analyst reason codes for `V1` through `V28` describe anonymized latent signals, not literal customer behaviors.

## Production Checklist

- Retrain on current, permissioned data.
- Validate thresholds with fraud operations teams.
- Add authentication, authorization, rate limits, and audit logs.
- Monitor drift, alert volume, recall, precision, and analyst feedback.
- Store model artifacts in a controlled registry instead of local disk.
