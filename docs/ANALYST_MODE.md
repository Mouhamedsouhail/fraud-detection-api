# Analyst Mode

SentinelPay's humanoid layer is **Maya**, a deterministic fraud analyst persona.

Maya does not use an LLM and does not call external APIs. The analyst layer converts the model score and input transaction vector into:

- severity: `LOW`, `ELEVATED`, `HIGH`, or `CRITICAL`
- decision queue: `auto_approve`, `watchlist`, `manual_review`, or `manual_review_urgent`
- plain-language summary
- reason codes
- recommended actions

## Endpoints

```bash
POST /analyst/score
GET /analyst/console
```

The input body for `/analyst/score` is the same as `/score`.

## Kafka Consumer

By default, the consumer calls `/score`. To publish analyst-enriched fraud results instead:

```bash
API_SCORE_PATH=/analyst/score python streaming/consumer.py
```

## Reason Code Boundaries

The Kaggle ULB features `V1` through `V28` are anonymized PCA components. SentinelPay does not pretend these are literal customer behaviors. Analyst Mode labels them as latent or anonymized signals and explains their magnitude, not their real-world source.

## Example Output

```json
{
  "transaction_id": "tx-human",
  "risk_score": 0.88,
  "is_fraud": true,
  "label": "SUSPICIOUS",
  "severity": "CRITICAL",
  "decision_queue": "manual_review_urgent",
  "analyst": {
    "name": "Maya",
    "role": "SentinelPay fraud analyst",
    "summary": "Maya: This transaction looks suspicious with risk 0.8800..."
  },
  "reason_codes": [
    {
      "code": "HIGH_AMOUNT",
      "signal": "large transaction amount",
      "detail": "Amount is 7500.0000, which should receive additional review.",
      "weight": 0.3
    }
  ],
  "recommended_actions": [
    "Temporarily hold the transaction.",
    "Require step-up verification before approval."
  ]
}
```
