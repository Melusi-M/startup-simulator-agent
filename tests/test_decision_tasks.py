from crewai import LLM

from startup_simulator_agent.agents import (
    create_startup_agents,
)
from startup_simulator_agent.decision_tasks import (
    create_decision_tasks,
)
from startup_simulator_agent.models import (
    FinalDecision,
)


def create_test_decision_tasks():
    llm = LLM(
        model=(
            "anthropic/"
            "claude-haiku-4-5-20251001"
        ),
        api_key="test-api-key",
        temperature=0,
    )

    agents = create_startup_agents(llm)

    return agents, create_decision_tasks(agents)


def test_decision_crew_contains_one_task() -> None:
    _, tasks = create_test_decision_tasks()

    assert len(tasks.as_list()) == 1


def test_decision_task_uses_ceo_and_structured_output() -> None:
    agents, tasks = create_test_decision_tasks()

    assert tasks.final_synthesis.agent is agents.ceo
    assert (
        tasks.final_synthesis.output_pydantic
        is FinalDecision
    )