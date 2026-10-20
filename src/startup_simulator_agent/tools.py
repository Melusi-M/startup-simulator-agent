from typing import Literal

from crewai.tools import BaseTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class StrictToolModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class RiskScoreInput(StrictToolModel):
    probability: int = Field(
        ge=1,
        le=5,
        description=(
            "Likelihood from 1 (rare) to 5 "
            "(almost certain)."
        ),
    )
    impact: int = Field(
        ge=1,
        le=5,
        description=(
            "Impact from 1 (minimal) to 5 "
            "(critical)."
        ),
    )
    rationale: str = Field(
        min_length=5,
        max_length=500,
        description=(
            "Reason for the probability and impact values."
        ),
    )


class RiskScoreResult(StrictToolModel):
    probability: int
    impact: int
    score: int
    severity: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]
    rationale: str


def classify_risk(score: int) -> str:
    if not 1 <= score <= 25:
        raise ValueError(
            "Risk score must be between 1 and 25"
        )

    if score <= 4:
        return "low"

    if score <= 9:
        return "medium"

    if score <= 16:
        return "high"

    return "critical"


class RiskScoreTool(BaseTool):
    name: str = "calculate_startup_risk"
    description: str = (
        "Calculate a deterministic startup risk score. "
        "Use this when assessing a risk with explicit "
        "probability and impact values from 1 to 5. "
        "Do not invent evidence to justify the values."
    )
    args_schema: type[BaseModel] = RiskScoreInput

    def _run(
        self,
        probability: int,
        impact: int,
        rationale: str,
    ) -> str:
        validated = RiskScoreInput(
            probability=probability,
            impact=impact,
            rationale=rationale,
        )

        score = (
            validated.probability
            * validated.impact
        )

        result = RiskScoreResult(
            probability=validated.probability,
            impact=validated.impact,
            score=score,
            severity=classify_risk(score),
            rationale=validated.rationale,
        )

        return result.model_dump_json()


class AssumptionInput(StrictToolModel):
    claim: str = Field(
        min_length=5,
        max_length=500,
        description=(
            "The claim or assumption being classified."
        ),
    )
    evidence_status: Literal[
        "known",
        "inferred",
        "unvalidated",
    ] = Field(
        description=(
            "Whether the claim is supported, inferred, "
            "or still unvalidated."
        ),
    )
    evidence: str | None = Field(
        default=None,
        max_length=1_000,
        description=(
            "Evidence or reasoning supporting the claim."
        ),
    )
    validation_action: str | None = Field(
        default=None,
        max_length=1_000,
        description=(
            "Action required to validate an inference "
            "or unsupported assumption."
        ),
    )

    @model_validator(mode="after")
    def validate_supporting_information(
        self,
    ) -> "AssumptionInput":
        if (
            self.evidence_status == "known"
            and not self.evidence
        ):
            raise ValueError(
                "A known claim must include evidence"
            )

        if (
            self.evidence_status == "inferred"
            and (
                not self.evidence
                or not self.validation_action
            )
        ):
            raise ValueError(
                "An inferred claim must include evidence "
                "and a validation action"
            )

        if (
            self.evidence_status == "unvalidated"
            and not self.validation_action
        ):
            raise ValueError(
                "An unvalidated claim must include "
                "a validation action"
            )

        return self


class AssumptionRecord(StrictToolModel):
    claim: str
    evidence_status: Literal[
        "known",
        "inferred",
        "unvalidated",
    ]
    evidence: str | None = None
    validation_action: str | None = None


class AssumptionRegisterTool(BaseTool):
    name: str = "register_startup_assumption"
    description: str = (
        "Classify a startup claim as known, inferred, "
        "or unvalidated. Use this before presenting "
        "market, customer, pricing, regulatory, or "
        "technical claims as evidence."
    )
    args_schema: type[BaseModel] = AssumptionInput

    def _run(
        self,
        claim: str,
        evidence_status: str,
        evidence: str | None = None,
        validation_action: str | None = None,
    ) -> str:
        validated = AssumptionInput(
            claim=claim,
            evidence_status=evidence_status,
            evidence=evidence,
            validation_action=validation_action,
        )

        record = AssumptionRecord(
            claim=validated.claim,
            evidence_status=validated.evidence_status,
            evidence=validated.evidence,
            validation_action=(
                validated.validation_action
            ),
        )

        return record.model_dump_json()