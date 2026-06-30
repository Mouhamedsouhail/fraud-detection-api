# SentinelPay Evaluation Report

- Dataset rows: `3000`
- Fraud rate: `0.0250`
- Target precision: `0.8000`
- Risk threshold: `0.6000`
- Recommended default model: `supervised_baseline`

## Model Comparison

| Model | PR-AUC | Precision at fixed point | Recall at fixed precision | Threshold |
| --- | ---: | ---: | ---: | ---: |
| isolation_forest | 0.0536 | 0.0000 | 0.0000 | 1.0000 |
| supervised_baseline | 1.0000 | 0.8333 | 1.0000 | 0.0833 |

## Confusion Matrices

### isolation_forest

Threshold: `0.6`

```text
[[461 124]
 [  8   7]]
```

### supervised_baseline

Threshold: `0.6`

```text
[[585   0]
 [  0  15]]
```
