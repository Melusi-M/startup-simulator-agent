from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


EXECUTIVE_SUMMARY_MAX_LENGTH = 2_500


class StrictModel(BaseModel):
    """Base model for strict agent inputs and outputs."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class AgentRole(StrEnum):
    CEO = "ceo"
    PRODUCT_STRATEGIST = "product_strategist"
    UX_DESIGNER = "ux_designer"
    TECHNICAL_LEAD = "technical_lead"
    CRITICAL_REVIEWER = "critical_reviewer"


class StartupBrief(StrictModel):
    name: str = Field(
        min_length=2,
        max_length=100,
    )
    description: str = Field(
        min_length=20,
        max_length=2_000,
    )
    target_customer: str = Field(
        min_length=3,
        max_length=500,
    )
    problem_statement: str = Field(
        min_length=10,
        max_length=1_000,
    )
    constraints: list[str] = Field(
        default_factory=list,
        max_length=20,
    )


class AgentAnalysis(StrictModel):
    agent_role: AgentRole
    summary: str = Field(
        min_length=10,
        max_length=2_000,
    )
    findings: list[str] = Field(
        min_length=1,
        max_length=20,
    )
    risks: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    recommendations: list[str] = Field(
        min_length=1,
        max_length=20,
    )
    confidence: float = Field(
        ge=0,
        le=1,
    )


class ReviewResult(StrictModel):
    approved: bool
    score: int = Field(
        ge=0,
        le=100,
    )
    strengths: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    concerns: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    revision_requests: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    @model_validator(mode="after")
    def validate_review_decision(
        self,
    ) -> "ReviewResult":
        if self.approved and self.score < 70:
            raise ValueError(
                "An approved review must score at least 70"
            )

        if (
            not self.approved
            and not self.revision_requests
        ):
            raise ValueError(
                "A rejected review must include "
                "revision requests"
            )

        return self


class PlannedWorkItem(StrictModel):
    agent_role: AgentRole
    objective: str = Field(
        min_length=10,
        max_length=500,
    )
    expected_deliverable: str = Field(
        min_length=10,
        max_length=500,
    )


class SimulationPlan(StrictModel):
    strategy: str = Field(
        min_length=20,
        max_length=1_500,
    )
    work_items: list[PlannedWorkItem] = Field(
        min_length=3,
        max_length=3,
    )
    success_criteria: list[str] = Field(
        min_length=2,
        max_length=10,
    )

    @model_validator(mode="after")
    def validate_specialist_coverage(
        self,
    ) -> "SimulationPlan":
        required_roles = {
            AgentRole.PRODUCT_STRATEGIST,
            AgentRole.UX_DESIGNER,
            AgentRole.TECHNICAL_LEAD,
        }

        assigned_roles = {
            item.agent_role
            for item in self.work_items
        }

        if assigned_roles != required_roles:
            raise ValueError(
                "The plan must assign exactly one work "
                "item to each specialist role"
            )

        return self


def normalize_executive_summary(
    value: object,
) -> object:
    """Bound generated summaries before field validation."""

    if not isinstance(value, str):
        return value

    normalized = value.strip()

    if len(normalized) <= EXECUTIVE_SUMMARY_MAX_LENGTH:
        return normalized

    candidate = normalized[
        :EXECUTIVE_SUMMARY_MAX_LENGTH
    ]

    sentence_endings = [
        candidate.rfind("."),
        candidate.rfind("!"),
        candidate.rfind("?"),
    ]
    final_sentence_end = max(sentence_endings)

    if final_sentence_end >= 1_500:
        return candidate[: final_sentence_end + 1]

    return candidate.rstrip()


ExecutiveSummary = Annotated[
    str,
    BeforeValidator(normalize_executive_summary),
    Field(
        min_length=1,
        max_length=EXECUTIVE_SUMMARY_MAX_LENGTH,
    ),
]


class FinalDecision(StrictModel):
    recommendation: Literal[
        "pursue",
        "revise",
        "reject",
    ]
    executive_summary: ExecutiveSummary


class StartupSimulationResult(StrictModel):
    brief: StartupBrief
    plan: SimulationPlan | None = None
    analyses: list[AgentAnalysis] = Field(
        min_length=3,
        max_length=10,
    )
    review: ReviewResult
    review_history: list[ReviewResult] = Field(
        default_factory=list,
        max_length=2,
    )
    revision_count: int = Field(
        default=0,
        ge=0,
        le=1,
    )
    recommendation: Literal[
        "pursue",
        "revise",
        "reject",
    ]
    executive_summary: ExecutiveSummary

    @model_validator(mode="after")
    def validate_reflection_history(
        self,
    ) -> "StartupSimulationResult":
        if not self.review_history:
            if self.revision_count != 0:
                raise ValueError(
                    "revision_count must be zero when "
                    "review_history is empty"
                )

            return self

        if self.review_history[-1] != self.review:
            raise ValueError(
                "The final review must be the last item "
                "in review_history"
            )

        expected_history_length = (
            self.revision_count + 1
        )

        if (
            len(self.review_history)
            != expected_history_length
        ):
            raise ValueError(
                "review_history length must equal "
                "revision_count plus one"
            )

        return self