from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "V1": -1.3598,
                "V2": -0.0728,
                "V3": 2.5363,
                "V4": 1.3782,
                "V5": -0.3383,
                "V6": 0.4624,
                "V7": 0.2396,
                "V8": 0.0987,
                "V9": 0.3638,
                "V10": 0.0908,
                "V11": -0.5516,
                "V12": -0.6178,
                "V13": -0.9914,
                "V14": -0.3112,
                "V15": 1.4682,
                "V16": -0.4704,
                "V17": 0.2079,
                "V18": 0.0258,
                "V19": 0.4040,
                "V20": 0.2514,
                "V21": -0.0183,
                "V22": 0.2778,
                "V23": -0.1105,
                "V24": 0.0669,
                "V25": 0.1285,
                "V26": -0.1891,
                "V27": 0.1336,
                "V28": -0.0211,
                "Amount": 149.62,
            }
        },
    )

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., ge=0, le=1)
    is_fraud: bool
    label: Literal["SUSPICIOUS", "LEGITIMATE"]
    latency_ms: float = Field(..., ge=0)


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionInput] = Field(..., min_length=1, max_length=1000)


class BatchScoreResponse(BaseModel):
    transaction_count: int = Field(..., ge=1)
    scores: list[ScoreResponse]
    latency_ms: float = Field(..., ge=0)
