import pytest
from crewai import LLM
from pydantic import ValidationError

from startup_simulator_agent.agents import (
    create_startup_agents,
)
from startup_simulator_agent.models import (
    AgentAnalysis,
    AgentRole,
    FinalDecision,
    PlannedWorkItem,
    ReviewResult,
    SimulationPlan,
)
from startup_simulator_agent.tasks import (
    create_startup_tasks,
)


def create_test_tasks():
    llm = LLM(
        model=(
            "anthropic/"
            "claude-haiku-4-5-20251001"
        ),
        api_key="test-api-key",
        temperature=0,
    )

    agents = create_startup_agents(llm)

    return agents, create_startup_tasks(agents)


def test_startup_tasks_contains_six_tasks() -> None:
    _, tasks = create_test_tasks()

    assert len(tasks.as_list()) == 6


def test_tasks_are_assigned_to_correct_agents() -> None:
    agents, tasks = create_test_tasks()

    assert tasks.planning.agent is agents.ceo
    assert (
        tasks.product_analysis.agent
        is agents.product_strategist
    )
    assert (
        tasks.ux_analysis.agent
        is agents.ux_designer
    )
    assert (
        tasks.technical_analysis.agent
        is agents.technical_lead
    )
    assert (
        tasks.critical_review.agent
        is agents.critical_reviewer
    )
    assert tasks.final_synthesis.agent is agents.ceo


def test_tasks_use_structured_output_models() -> None:
    _, tasks = create_test_tasks()

    assert (
        tasks.planning.output_pydantic
        is SimulationPlan
    )
    assert (
        tasks.product_analysis.output_pydantic
        is AgentAnalysis
    )
    assert (
        tasks.ux_analysis.output_pydantic
        is AgentAnalysis
    )
    assert (
        tasks.technical_analysis.output_pydantic
        is AgentAnalysis
    )
    assert (
        tasks.critical_review.output_pydantic
        is ReviewResult
    )
    assert (
        tasks.final_synthesis.output_pydantic
        is FinalDecision
    )


def test_specialist_tasks_depend_on_plan() -> None:
    _, tasks = create_test_tasks()

    assert tasks.product_analysis.context == [
        tasks.planning
    ]
    assert tasks.ux_analysis.context == [
        tasks.planning
    ]
    assert tasks.technical_analysis.context == [
        tasks.planning
    ]


def test_review_depends_on_all_specialists() -> None:
    _, tasks = create_test_tasks()

    assert tasks.critical_review.context == [
        tasks.product_analysis,
        tasks.ux_analysis,
        tasks.technical_analysis,
    ]


def test_plan_requires_exact_specialist_coverage() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one work item",
    ):
        SimulationPlan(
            strategy=(
                "Evaluate the startup with focused "
                "specialist assessments."
            ),
            work_items=[
                PlannedWorkItem(
                    agent_role=(
                        AgentRole.PRODUCT_STRATEGIST
                    ),
                    objective=(
                        "Evaluate the customer problem."
                    ),
                    expected_deliverable=(
                        "A product strategy assessment."
                    ),
                ),
                PlannedWorkItem(
                    agent_role=AgentRole.UX_DESIGNER,
                    objective=(
                        "Evaluate the customer journey."
                    ),
                    expected_deliverable=(
                        "A user experience assessment."
                    ),
                ),
                PlannedWorkItem(
                    agent_role=AgentRole.CEO,
                    objective=(
                        "Repeat the product assessment."
                    ),
                    expected_deliverable=(
                        "Another product assessment."
                    ),
                ),
            ],
            success_criteria=[
                "All major assumptions are identified.",
                "Recommendations are actionable.",
            ],
        )
def test_synthesis_task_requests_concise_summary() -> None:
    _, tasks = create_test_tasks()

    description = " ".join(
        tasks.final_synthesis.description
        .lower()
        .split()
    )

    assert (
        "no more than 2,500 characters"
        in description
    )
    assert (
        "exactly two short paragraphs"
        in description
    )
    assert "do not use headings" in description
    assert (
        "stop writing immediately"
        in description
    )

def test_specialist_tasks_request_bounded_output() -> None:
    _, tasks = create_test_tasks()

    specialist_tasks = [
        tasks.product_analysis,
        tasks.ux_analysis,
        tasks.technical_analysis,
    ]

    for task in specialist_tasks:
        description = task.description.lower()

        assert "summary below 1,200 characters" in description
        assert "no more than 8 findings" in description
        assert "no more than 8 risks" in description
        assert "no more than 8 recommendations" in description
        assert "list item below 400 characters" in description

def test_assessment_list_excludes_final_synthesis() -> None:
    _, tasks = create_test_tasks()

    assert tasks.assessment_list() == [
        tasks.planning,
        tasks.product_analysis,
        tasks.ux_analysis,
        tasks.technical_analysis,
        tasks.critical_review,
    ]

    assert tasks.final_synthesis not in (
        tasks.assessment_list()
    )

def test_planning_task_accepts_memory_context() -> None:
    _, tasks = create_test_tasks()

    description = tasks.planning.description.lower()

    assert "{memory_context}" in description
    assert "reference material" in description
    assert "verified facts" in description

def test_critical_review_requests_bounded_output() -> None:
    _, tasks = create_test_tasks()

    description = " ".join(
        tasks.critical_review.description
        .lower()
        .split()
    )

    assert "no more than 8 strengths" in description
    assert "no more than 8 concerns" in description
    assert "no more than 8 revision requests" in description
    assert "combine related concerns" in description