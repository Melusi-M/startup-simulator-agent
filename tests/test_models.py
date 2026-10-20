import pytest
from pydantic import ValidationError

from startup_simulator_agent.models import (
    AgentAnalysis,
    AgentRole,
    FinalDecision,
    ReviewResult,
    StartupBrief,
    StartupSimulationResult,
)


def create_brief() -> StartupBrief:
    return StartupBrief(
        name="Compliance Copilot",
        description=(
            "An AI platform that helps estate agencies "
            "operate their compliance processes."
        ),
        target_customer=(
            "Small and medium South African estate agencies"
        ),
        problem_statement=(
            "Compliance work is manual, inconsistent, "
            "and difficult to audit."
        ),
        constraints=[
            "Small founding team",
            "South African market first",
        ],
    )


def create_analysis(
    role: AgentRole,
) -> AgentAnalysis:
    return AgentAnalysis(
        agent_role=role,
        summary=(
            "The idea addresses a meaningful operational "
            "problem for regulated businesses."
        ),
        findings=[
            "The workflow is currently document-heavy.",
        ],
        risks=[
            "Regulatory interpretation may change.",
        ],
        recommendations=[
            "Validate the workflow with five agencies.",
        ],
        confidence=0.8,
    )


def test_startup_brief_normalizes_text() -> None:
    brief = StartupBrief(
        name="  Compliance Copilot  ",
        description=(
            "  A platform that operationalises "
            "compliance work for agencies.  "
        ),
        target_customer="  Estate agencies  ",
        problem_statement=(
            "  Compliance processes are fragmented.  "
        ),
    )

    assert brief.name == "Compliance Copilot"
    assert brief.target_customer == "Estate agencies"


def test_startup_brief_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        StartupBrief(
            name="   ",
            description=(
                "A sufficiently detailed startup description."
            ),
            target_customer="Estate agencies",
            problem_statement=(
                "Compliance processes are fragmented."
            ),
        )


def test_startup_brief_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        StartupBrief(
            name="Compliance Copilot",
            description=(
                "A sufficiently detailed startup description."
            ),
            target_customer="Estate agencies",
            problem_statement=(
                "Compliance processes are fragmented."
            ),
            invented_field="not allowed",
        )


def test_agent_analysis_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        AgentAnalysis(
            agent_role=AgentRole.TECHNICAL_LEAD,
            summary="A valid technical analysis summary.",
            findings=["The architecture is feasible."],
            recommendations=["Build a narrow prototype."],
            confidence=1.5,
        )


def test_approved_review_requires_score_of_seventy() -> None:
    with pytest.raises(
        ValidationError,
        match="at least 70",
    ):
        ReviewResult(
            approved=True,
            score=60,
        )


def test_rejected_review_requires_revision_requests() -> None:
    with pytest.raises(
        ValidationError,
        match="revision requests",
    ):
        ReviewResult(
            approved=False,
            score=55,
        )


def test_complete_simulation_result_is_valid() -> None:
    brief = create_brief()

    analyses = [
        create_analysis(AgentRole.PRODUCT_STRATEGIST),
        create_analysis(AgentRole.UX_DESIGNER),
        create_analysis(AgentRole.TECHNICAL_LEAD),
    ]

    review = ReviewResult(
        approved=True,
        score=82,
        strengths=[
            "The target customer is clearly defined.",
        ],
    )

    result = StartupSimulationResult(
        brief=brief,
        analyses=analyses,
        review=review,
        recommendation="pursue",
        executive_summary=(
            "The startup addresses a credible compliance "
            "problem and should proceed to customer discovery."
        ),
    )

    assert result.recommendation == "pursue"
    assert len(result.analyses) == 3
    assert result.review.approved is True

def test_final_decision_accepts_summary_within_safety_limit() -> None:
    decision = FinalDecision(
        recommendation="revise",
        executive_summary="A" * 2_500,
    )

    assert len(decision.executive_summary) == 2_500


def test_final_decision_bounds_excessively_long_summary(
) -> None:
    decision = FinalDecision(
        recommendation="revise",
        executive_summary="A" * 6_001,
    )

    assert len(decision.executive_summary) == 2_500

def test_simulation_result_accepts_summary_within_safety_limit() -> None:
    result = StartupSimulationResult(
        brief=create_brief(),
        analyses=[
            create_analysis(AgentRole.PRODUCT_STRATEGIST),
            create_analysis(AgentRole.UX_DESIGNER),
            create_analysis(AgentRole.TECHNICAL_LEAD),
        ],
        review=ReviewResult(
            approved=False,
            score=60,
            concerns=["Customer evidence is insufficient."],
            revision_requests=[
                "Complete customer discovery before proceeding.",
            ],
        ),
        recommendation="revise",
        executive_summary="A" * 4_000,
    )

    assert len(result.executive_summary) == 2_500

def create_rejected_review() -> ReviewResult:
    return ReviewResult(
        approved=False,
        score=58,
        concerns=[
            "The specialist analyses conflict.",
        ],
        revision_requests=[
            "Resolve the conflicting recommendations.",
        ],
    )


def test_result_records_single_review_without_revision() -> None:
    review = create_rejected_review()

    result = StartupSimulationResult(
        brief=create_brief(),
        analyses=[
            create_analysis(AgentRole.PRODUCT_STRATEGIST),
            create_analysis(AgentRole.UX_DESIGNER),
            create_analysis(AgentRole.TECHNICAL_LEAD),
        ],
        review=review,
        review_history=[review],
        revision_count=0,
        recommendation="revise",
        executive_summary=(
            "The assessment requires additional validation."
        ),
    )

    assert result.review_history == [review]
    assert result.revision_count == 0


def test_result_records_one_bounded_revision() -> None:
    initial_review = create_rejected_review()

    final_review = ReviewResult(
        approved=True,
        score=78,
        strengths=[
            "The conflicts were resolved.",
        ],
    )

    result = StartupSimulationResult(
        brief=create_brief(),
        analyses=[
            create_analysis(AgentRole.PRODUCT_STRATEGIST),
            create_analysis(AgentRole.UX_DESIGNER),
            create_analysis(AgentRole.TECHNICAL_LEAD),
        ],
        review=final_review,
        review_history=[
            initial_review,
            final_review,
        ],
        revision_count=1,
        recommendation="pursue",
        executive_summary=(
            "The revised assessment is now coherent."
        ),
    )

    assert len(result.review_history) == 2
    assert result.review is final_review
    assert result.revision_count == 1


def test_result_rejects_inconsistent_review_history() -> None:
    initial_review = create_rejected_review()

    final_review = ReviewResult(
        approved=True,
        score=80,
        strengths=[
            "The revised assessment is coherent.",
        ],
    )

    with pytest.raises(
        ValidationError,
        match="final review",
    ):
        StartupSimulationResult(
            brief=create_brief(),
            analyses=[
                create_analysis(
                    AgentRole.PRODUCT_STRATEGIST
                ),
                create_analysis(
                    AgentRole.UX_DESIGNER
                ),
                create_analysis(
                    AgentRole.TECHNICAL_LEAD
                ),
            ],
            review=final_review,
            review_history=[initial_review],
            revision_count=0,
            recommendation="pursue",
            executive_summary=(
                "The result contains inconsistent history."
            ),
        )
def test_final_decision_bounds_summary_above_safety_limit(
) -> None:
    decision = FinalDecision(
        recommendation="revise",
        executive_summary="A" * 2_501,
    )

    assert len(decision.executive_summary) == 2_500

def test_final_decision_preserves_bounded_summary() -> None:
    summary = (
        "The startup should be revised before launch."
    )

    decision = FinalDecision(
        recommendation="revise",
        executive_summary=summary,
    )

    assert decision.executive_summary == summary

def test_final_decision_bounds_generated_summary() -> None:
    decision = FinalDecision(
        recommendation="revise",
        executive_summary=(
            ("A" * 1_600)
            + ". "
            + ("B" * 1_500)
        ),
    )

    assert len(decision.executive_summary) <= 2_500
    assert decision.executive_summary.endswith(".")

def test_final_decision_strips_summary_whitespace() -> None:
    decision = FinalDecision(
        recommendation="revise",
        executive_summary=(
            "  Revise the startup before launch.  "
        ),
    )

    assert decision.executive_summary == (
        "Revise the startup before launch."
    )

def test_simulation_result_bounds_generated_summary() -> None:
    review = ReviewResult(
        approved=False,
        score=60,
        concerns=[
            "Customer evidence is insufficient.",
        ],
        revision_requests=[
            "Complete customer discovery before proceeding.",
        ],
    )

    result = StartupSimulationResult(
        brief=create_brief(),
        analyses=[
            create_analysis(
                AgentRole.PRODUCT_STRATEGIST
            ),
            create_analysis(
                AgentRole.UX_DESIGNER
            ),
            create_analysis(
                AgentRole.TECHNICAL_LEAD
            ),
        ],
        review=review,
        recommendation="revise",
        executive_summary="A" * 4_000,
    )

    assert len(result.executive_summary) == 2_500
    assert result.review_history == []
    assert result.revision_count == 0


def test_final_decision_bounds_excessively_long_summary(
) -> None:
    decision = FinalDecision(
        recommendation="revise",
        executive_summary="A" * 6_001,
    )

    assert len(decision.executive_summary) == 2_500


def test_final_decision_bounds_summary_above_safety_limit(
) -> None:
    decision = FinalDecision(
        recommendation="revise",
        executive_summary="A" * 2_501,
    )

    assert len(decision.executive_summary) == 2_500


def test_simulation_result_bounds_generated_summary() -> None:
    review = ReviewResult(
        approved=False,
        score=60,
        concerns=[
            "Customer evidence is insufficient.",
        ],
        revision_requests=[
            "Complete customer discovery before proceeding.",
        ],
    )

    result = StartupSimulationResult(
        brief=create_brief(),
        analyses=[
            create_analysis(
                AgentRole.PRODUCT_STRATEGIST
            ),
            create_analysis(
                AgentRole.UX_DESIGNER
            ),
            create_analysis(
                AgentRole.TECHNICAL_LEAD
            ),
        ],
        review=review,
        recommendation="revise",
        executive_summary="A" * 4_000,
    )

    assert len(result.executive_summary) == 2_500
    assert result.review_history == []
    assert result.revision_count == 0