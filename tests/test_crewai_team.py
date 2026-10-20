from types import SimpleNamespace

import pytest

from startup_simulator_agent.crewai_team import (
    CrewAIStartupTeam,
)
from startup_simulator_agent.models import (
    AgentAnalysis,
    AgentRole,
    FinalDecision,
    PlannedWorkItem,
    ReviewResult,
    SimulationPlan,
    StartupBrief,
)


def create_brief() -> StartupBrief:
    return StartupBrief(
        name="Compliance Copilot",
        description=(
            "An AI platform that helps estate agencies "
            "operate their compliance processes."
        ),
        target_customer="Estate agencies",
        problem_statement=(
            "Compliance work is manual and difficult "
            "to audit."
        ),
        constraints=[
            "South African market first",
        ],
    )


def create_plan() -> SimulationPlan:
    return SimulationPlan(
        strategy=(
            "Evaluate the idea through product, user "
            "experience, and technical perspectives."
        ),
        work_items=[
            PlannedWorkItem(
                agent_role=(
                    AgentRole.PRODUCT_STRATEGIST
                ),
                objective=(
                    "Evaluate customer and market needs."
                ),
                expected_deliverable=(
                    "A product and market assessment."
                ),
            ),
            PlannedWorkItem(
                agent_role=AgentRole.UX_DESIGNER,
                objective=(
                    "Evaluate the primary user journey."
                ),
                expected_deliverable=(
                    "A focused user experience assessment."
                ),
            ),
            PlannedWorkItem(
                agent_role=AgentRole.TECHNICAL_LEAD,
                objective=(
                    "Evaluate technical feasibility."
                ),
                expected_deliverable=(
                    "A practical architecture assessment."
                ),
            ),
        ],
        success_criteria=[
            "Major assumptions are identified.",
            "Recommendations are actionable.",
        ],
    )


def create_analysis(
    role: AgentRole,
) -> AgentAnalysis:
    return AgentAnalysis(
        agent_role=role,
        summary=(
            "The specialist completed a focused "
            "assessment of the startup idea."
        ),
        findings=[
            "The problem is specific enough to validate.",
        ],
        risks=[
            "Customer demand remains unverified.",
        ],
        recommendations=[
            "Run focused customer interviews.",
        ],
        confidence=0.8,
    )

class FakeCrew:
    def __init__(
        self,
        outputs: list[object],
    ) -> None:
        self.outputs = outputs
        self.received_inputs = None
        self.call_count = 0

    def kickoff(
        self,
        inputs: dict[str, object],
    ):
        self.call_count += 1
        self.received_inputs = inputs

        return SimpleNamespace(
            tasks_output=[
                SimpleNamespace(pydantic=output)
                for output in self.outputs
            ]
        )

def create_rejected_initial_outputs() -> list[object]:
    outputs = create_successful_outputs()

    outputs[4] = ReviewResult(
        approved=False,
        score=58,
        concerns=[
            "The specialist recommendations conflict.",
        ],
        revision_requests=[
            "Resolve the integration-scope conflict.",
        ],
    )

    outputs[5] = FinalDecision(
        recommendation="revise",
        executive_summary=(
            "The specialist analyses require revision."
        ),
    )

    return outputs

def create_revision_outputs(
    approved: bool = True,
) -> list[object]:
    revised_product = create_analysis(
        AgentRole.PRODUCT_STRATEGIST
    )
    revised_ux = create_analysis(
        AgentRole.UX_DESIGNER
    )
    revised_technical = create_analysis(
        AgentRole.TECHNICAL_LEAD
    )

    if approved:
        review = ReviewResult(
            approved=True,
            score=78,
            strengths=[
                "The original conflicts were resolved.",
            ],
        )

        decision = FinalDecision(
            recommendation="pursue",
            executive_summary=(
                "The revised assessment is coherent enough "
                "to proceed to customer validation."
            ),
        )
    else:
        review = ReviewResult(
            approved=False,
            score=62,
            concerns=[
                "Important assumptions remain unresolved.",
            ],
            revision_requests=[
                "Validate the remaining assumptions "
                "outside this bounded simulation."
            ],
        )

        decision = FinalDecision(
            recommendation="revise",
            executive_summary=(
                "The bounded revision completed, but "
                "external validation is still required."
            ),
        )

    return [
        revised_product,
        revised_ux,
        revised_technical,
        review,
        decision,
    ]

def test_approved_initial_review_skips_revision_crew() -> None:
    initial_crew = FakeCrew(
        create_successful_outputs()
    )
    revision_crew = FakeCrew(
        create_revision_outputs()
    )

    team = CrewAIStartupTeam(
        crew=initial_crew,
        revision_crew=revision_crew,
    )

    result = team.run(create_brief())

    assert initial_crew.call_count == 1
    assert revision_crew.call_count == 0
    assert result.revision_count == 0
    assert result.review_history == [
        result.review
    ]


def test_rejected_review_runs_one_revision_cycle() -> None:
    initial_crew = FakeCrew(
        create_rejected_initial_outputs()
    )
    revision_crew = FakeCrew(
        create_revision_outputs(approved=True)
    )

    team = CrewAIStartupTeam(
        crew=initial_crew,
        revision_crew=revision_crew,
    )

    result = team.run(create_brief())

    assert initial_crew.call_count == 1
    assert revision_crew.call_count == 1
    assert result.revision_count == 1
    assert len(result.review_history) == 2
    assert result.review.approved is True
    assert result.recommendation == "pursue"


def test_revision_crew_receives_initial_assessment() -> None:
    initial_crew = FakeCrew(
        create_rejected_initial_outputs()
    )
    revision_crew = FakeCrew(
        create_revision_outputs()
    )

    team = CrewAIStartupTeam(
        crew=initial_crew,
        revision_crew=revision_crew,
    )

    team.run(create_brief())

    inputs = revision_crew.received_inputs

    assert inputs is not None
    assert "startup_brief" in inputs
    assert "initial_product_analysis" in inputs
    assert "initial_ux_analysis" in inputs
    assert "initial_technical_analysis" in inputs
    assert "initial_review" in inputs


def test_rejected_second_review_does_not_loop_again() -> None:
    initial_crew = FakeCrew(
        create_rejected_initial_outputs()
    )
    revision_crew = FakeCrew(
        create_revision_outputs(approved=False)
    )

    team = CrewAIStartupTeam(
        crew=initial_crew,
        revision_crew=revision_crew,
    )

    result = team.run(create_brief())

    assert revision_crew.call_count == 1
    assert result.revision_count == 1
    assert result.review.approved is False
    assert result.recommendation == "revise"

def create_successful_outputs() -> list[object]:
    return [
        create_plan(),
        create_analysis(
            AgentRole.PRODUCT_STRATEGIST
        ),
        create_analysis(
            AgentRole.UX_DESIGNER
        ),
        create_analysis(
            AgentRole.TECHNICAL_LEAD
        ),
        ReviewResult(
            approved=True,
            score=82,
            strengths=[
                "The assessment is coherent.",
            ],
        ),
        FinalDecision(
            recommendation="pursue",
            executive_summary=(
                "The startup should proceed to focused "
                "customer discovery before MVP delivery."
            ),
        ),
    ]

def test_rejected_second_review_cannot_produce_pursue() -> None:
    revision_outputs = create_revision_outputs(
        approved=False
    )

    revision_outputs[4] = FinalDecision(
        recommendation="pursue",
        executive_summary=(
            "This invalid decision contradicts "
            "the rejected review."
        ),
    )

    initial_crew = FakeCrew(
        create_rejected_initial_outputs()
    )
    revision_crew = FakeCrew(
        revision_outputs
    )

    team = CrewAIStartupTeam(
        crew=initial_crew,
        revision_crew=revision_crew,
    )

    with pytest.raises(
        RuntimeError,
        match="cannot produce",
    ):
        team.run(create_brief())

    assert initial_crew.call_count == 1
    assert revision_crew.call_count == 1

def test_crewai_team_assembles_structured_result() -> None:
    fake_crew = FakeCrew(
        create_successful_outputs()
    )
    team = CrewAIStartupTeam(fake_crew)
    brief = create_brief()

    result = team.run(brief)

    assert result.brief == brief
    assert result.plan == create_plan()
    assert len(result.analyses) == 3
    assert result.review.approved is True
    assert result.recommendation == "pursue"


def test_crewai_team_passes_brief_as_inputs() -> None:
    fake_crew = FakeCrew(
        create_successful_outputs()
    )
    team = CrewAIStartupTeam(fake_crew)

    team.run(create_brief())

    assert fake_crew.received_inputs == {
        "startup_name": "Compliance Copilot",
        "startup_description": (
            "An AI platform that helps estate agencies "
            "operate their compliance processes."
        ),
        "target_customer": "Estate agencies",
        "problem_statement": (
            "Compliance work is manual and difficult "
            "to audit."
        ),
        "constraints": (
            "- South African market first"
        ),
        "memory_context": (
            "No relevant previous simulations "
            "were found."
        ),
    }

def test_crewai_team_rejects_wrong_specialist_role() -> None:
    outputs = create_successful_outputs()

    outputs[1] = create_analysis(
        AgentRole.UX_DESIGNER
    )

    team = CrewAIStartupTeam(
        FakeCrew(outputs)
    )

    with pytest.raises(
        RuntimeError,
        match="wrong role",
    ):
        team.run(create_brief())


def test_crewai_team_rejects_missing_output() -> None:
    outputs = create_successful_outputs()[:-1]

    team = CrewAIStartupTeam(
        FakeCrew(outputs)
    )

    with pytest.raises(
        RuntimeError,
        match="Missing output",
    ):
        team.run(create_brief())


def test_rejected_review_cannot_produce_pursue() -> None:
    outputs = create_successful_outputs()

    outputs[4] = ReviewResult(
        approved=False,
        score=55,
        concerns=[
            "The evidence is insufficient.",
        ],
        revision_requests=[
            "Validate the customer demand.",
        ],
    )

    team = CrewAIStartupTeam(
        FakeCrew(outputs)
    )

    with pytest.raises(
        RuntimeError,
        match="cannot produce",
    ):
        team.run(create_brief())


def create_assessment_outputs(
    approved: bool = True,
) -> list[object]:
    outputs = create_successful_outputs()[:4]

    if approved:
        review = ReviewResult(
            approved=True,
            score=82,
            strengths=[
                "The assessment is coherent.",
            ],
        )
    else:
        review = ReviewResult(
            approved=False,
            score=58,
            concerns=[
                "The specialist recommendations conflict.",
            ],
            revision_requests=[
                "Resolve the integration-scope conflict.",
            ],
        )

    return [
        *outputs,
        review,
    ]


def create_decision_outputs() -> list[object]:
    return [
        FinalDecision(
            recommendation="pursue",
            executive_summary=(
                "The approved assessment supports focused "
                "customer validation."
            ),
        )
    ]

def test_approved_assessment_routes_to_decision_crew() -> None:
    assessment_crew = FakeCrew(
        create_assessment_outputs(approved=True)
    )
    decision_crew = FakeCrew(
        create_decision_outputs()
    )
    revision_crew = FakeCrew(
        create_revision_outputs()
    )

    team = CrewAIStartupTeam(
        crew=assessment_crew,
        decision_crew=decision_crew,
        revision_crew=revision_crew,
    )

    result = team.run(create_brief())

    assert assessment_crew.call_count == 1
    assert decision_crew.call_count == 1
    assert revision_crew.call_count == 0
    assert result.revision_count == 0
    assert result.recommendation == "pursue"


def test_rejected_assessment_skips_decision_crew() -> None:
    assessment_crew = FakeCrew(
        create_assessment_outputs(approved=False)
    )
    decision_crew = FakeCrew(
        create_decision_outputs()
    )
    revision_crew = FakeCrew(
        create_revision_outputs(approved=True)
    )

    team = CrewAIStartupTeam(
        crew=assessment_crew,
        decision_crew=decision_crew,
        revision_crew=revision_crew,
    )

    result = team.run(create_brief())

    assert assessment_crew.call_count == 1
    assert decision_crew.call_count == 0
    assert revision_crew.call_count == 1
    assert result.revision_count == 1

def test_crewai_team_passes_memory_to_initial_crew() -> None:
    assessment_crew = FakeCrew(
        create_successful_outputs()
    )

    team = CrewAIStartupTeam(
        crew=assessment_crew,
    )

    memory_context = (
        "Relevant historical simulations follow.\n"
        "Previous compliance startup required "
        "customer validation."
    )

    team.run(
        create_brief(),
        memory_context=memory_context,
    )

    assert assessment_crew.received_inputs is not None
    assert (
        assessment_crew
        .received_inputs["memory_context"]
        == memory_context
    )

def test_crewai_team_passes_memory_to_initial_crew() -> None:
    assessment_crew = FakeCrew(
        create_successful_outputs()
    )

    team = CrewAIStartupTeam(
        crew=assessment_crew,
        decision_crew=None,
        revision_crew=None,
    )

    memory_context = (
        "Relevant historical simulations follow.\n"
        "Previous compliance startup required "
        "customer validation."
    )

    team.run(
        create_brief(),
        memory_context=memory_context,
    )

    assert assessment_crew.received_inputs is not None
    assert (
        assessment_crew
        .received_inputs["memory_context"]
        == memory_context
    )