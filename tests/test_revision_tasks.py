from crewai import LLM

from startup_simulator_agent.agents import (
    create_startup_agents,
)
from startup_simulator_agent.models import (
    AgentAnalysis,
    FinalDecision,
    ReviewResult,
)
from startup_simulator_agent.revision_tasks import (
    create_revision_tasks,
)

from startup_simulator_agent.tasks import (
    create_startup_tasks,
)


def create_test_revision_tasks():
    llm = LLM(
        model=(
            "anthropic/"
            "claude-haiku-4-5-20251001"
        ),
        api_key="test-api-key",
        temperature=0,
    )

    agents = create_startup_agents(llm)

    return agents, create_revision_tasks(agents)


def test_revision_team_contains_five_tasks() -> None:
    _, tasks = create_test_revision_tasks()

    assert len(tasks.as_list()) == 5


def test_revision_tasks_use_correct_agents() -> None:
    agents, tasks = create_test_revision_tasks()

    assert (
        tasks.product_revision.agent
        is agents.product_strategist
    )
    assert tasks.ux_revision.agent is agents.ux_designer
    assert (
        tasks.technical_revision.agent
        is agents.technical_lead
    )
    assert (
        tasks.second_review.agent
        is agents.critical_reviewer
    )
    assert tasks.final_synthesis.agent is agents.ceo


def test_revision_tasks_use_structured_outputs() -> None:
    _, tasks = create_test_revision_tasks()

    assert (
        tasks.product_revision.output_pydantic
        is AgentAnalysis
    )
    assert (
        tasks.ux_revision.output_pydantic
        is AgentAnalysis
    )
    assert (
        tasks.technical_revision.output_pydantic
        is AgentAnalysis
    )
    assert (
        tasks.second_review.output_pydantic
        is ReviewResult
    )
    assert (
        tasks.final_synthesis.output_pydantic
        is FinalDecision
    )


def test_second_review_and_synthesis_have_context() -> None:
    _, tasks = create_test_revision_tasks()

    assert tasks.second_review.context == [
        tasks.product_revision,
        tasks.ux_revision,
        tasks.technical_revision,
    ]

    assert tasks.final_synthesis.context == [
        tasks.product_revision,
        tasks.ux_revision,
        tasks.technical_revision,
        tasks.second_review,
    ]

def test_specialist_revisions_require_tools() -> None:
    _, tasks = create_test_revision_tasks()

    specialist_revisions = [
        tasks.product_revision,
        tasks.ux_revision,
        tasks.technical_revision,
    ]

    for task in specialist_revisions:
        description = task.description

        assert (
            "register_startup_assumption"
            in description
        )
        assert (
            "calculate_startup_risk"
            in description
        )
        assert "1-to-5 scale" in description
        assert (
            "deterministic calculations"
            in description
        )

def test_second_review_receives_inputs_and_requires_tools(
) -> None:
    _, tasks = create_test_revision_tasks()

    description = tasks.second_review.description
    normalized_description = " ".join(
        description.split()
    )

    assert "{startup_brief}" in description
    assert "{initial_review}" in description
    assert "register_startup_assumption" in description
    assert "calculate_startup_risk" in description
    assert (
        "Do not treat a tool-calculated risk score "
        "as proof"
        in normalized_description
    )
def test_final_revision_synthesis_does_not_require_tools(
) -> None:
    _, tasks = create_test_revision_tasks()

    description = tasks.final_synthesis.description

    assert "register_startup_assumption" not in description
    assert "calculate_startup_risk" not in description

def test_synthesis_tasks_require_strictly_bounded_output(
) -> None:
    agents, revision_tasks = (
        create_test_revision_tasks()
    )
    initial_tasks = create_startup_tasks(agents)

    for task in [
        initial_tasks.final_synthesis,
        revision_tasks.final_synthesis,
    ]:
        description = " ".join(
            task.description.lower().split()
        )

        assert (
         "no more than 2,500 characters" in description
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

def test_second_review_requests_bounded_output() -> None:
    _, tasks = create_test_revision_tasks()

    description = " ".join(
        tasks.second_review.description
        .lower()
        .split()
    )

    assert "no more than 8 strengths" in description
    assert "no more than 8 concerns" in description
    assert "no more than 8 revision requests" in description
    assert "combine related concerns" in description