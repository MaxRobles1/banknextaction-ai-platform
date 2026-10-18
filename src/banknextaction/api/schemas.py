"""API contract (Pydantic). Field names match the training DataFrame columns exactly.

Ranges are plausibility limits (wider than the observed data); categories are the ones the
model was trained on, recorded in models/modelo.json. Unknown fields are rejected
(extra="forbid"), so a typo never becomes a silently ignored input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Job = Literal[
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
Marital = Literal["divorced", "married", "single"]
Education = Literal["primary", "secondary", "tertiary", "unknown"]
YesNo = Literal["no", "yes"]
PreviousOutcome = Literal["failure", "other", "success", "unknown"]

EXAMPLE_CLIENT = {
    "age": 41,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


class ClientFeatures(BaseModel):
    """Information known about the client BEFORE the campaign contact."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CLIENT]})

    age: int = Field(ge=18, le=100, description="Age in years.")
    job: Job = Field(description="Occupation.")
    marital: Marital = Field(description="Marital status.")
    education: Education = Field(description="Education level.")
    default: YesNo = Field(description="Has credit in default.")
    balance: int = Field(
        ge=-100_000, le=1_000_000, description="Average yearly balance (original currency)."
    )
    housing: YesNo = Field(description="Has a housing loan.")
    loan: YesNo = Field(description="Has a personal loan.")
    pdays: int = Field(
        ge=-1, le=999, description="Days since a previous campaign contact; -1 = never contacted."
    )
    previous: int = Field(ge=0, le=300, description="Contacts before this campaign.")
    poutcome: PreviousOutcome = Field(description="Outcome of the previous campaign.")


class PredictionResponse(BaseModel):
    subscription_probability: float = Field(ge=0.0, le=1.0)
    non_subscription_probability: float = Field(ge=0.0, le=1.0)
    predicted_class: Literal[0, 1] = Field(
        description="1 when the probability reaches the decision threshold."
    )
    decision_threshold: float = Field(description="Chosen on validation (max F1), see /health.")
    model_name: str
    model_version: str
    inference_time_ms: float = Field(description="Measured time of the model call.")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_name: str
    model_version: str
    decision_threshold: float
    artifact_created_at: str
    receipt_generated_at: str
    artifacts: list[dict[str, str | int]]
