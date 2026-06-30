# Threshold Tuning

Fraud detection is usually constrained by review capacity. A threshold that maximizes recall can overwhelm analysts, while a threshold that maximizes precision can miss too much fraud.

SentinelPay reports:

- **PR-AUC**: precision-recall area under curve, better for rare fraud than ROC-AUC.
- **Recall at fixed precision**: how much fraud the model catches when precision is at or above a target such as `0.80`.
- **Confusion matrix at risk threshold**: operational false positives and false negatives for the chosen cutoff.

## Train With a Precision Target

```bash
python model/train.py --target-precision 0.8 --risk-threshold 0.6
```

The report is written to:

```text
reports/evaluation.json
reports/evaluation.md
```

For the bundled demo artifact, see:

```text
reports/demo_evaluation.json
reports/demo_evaluation.md
```

## Recommended Workflow

1. Pick an analyst review budget, such as 200 cases per day.
2. Train on historical data.
3. Inspect `recall_at_fixed_precision` for both Isolation Forest and the supervised baseline.
4. Choose a risk threshold that fits review capacity.
5. Route `CRITICAL` and `HIGH` cases to manual review.
6. Feed dispositions and analyst corrections into `/cases/{case_id}/feedback`.
7. Export `/retraining/candidates` before the next training run.

## Caveat

The default `0.6` threshold is a starting point. It should not be treated as production-ready without validation against current, permissioned transaction data.
