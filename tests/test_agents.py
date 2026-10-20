import pytest
from crewai import LLM

from startup_simulator_agent.agents import (
    create_startup_agents,
)
from startup_simulator_agent.model_factory import (
    create_llm_from_environment,
)


TEST_MODEL = (
    "anthropic/claude-haiku-4-5-20251001"
)


def create_test_llm() -> LLM:
    return LLM(
        model=TEST_MODEL,
        api_key="test-api-key",
        temperature=0,
    )


def test_create_agents_returns_five_distinct_roles() -> None:
    agents = create_startup_agents(
        create_test_llm()
    )

    roles = {
        agent.role
        for agent in agents.as_list()
    }

    assert len(agents.as_list()) == 5
    assert len(roles) == 5


def test_agents_have_distinct_goals() -> None:
    agents = create_startup_agents(
        create_test_llm()
    )

    goals = {
        agent.goal
        for agent in agents.as_list()
    }

    assert len(goals) == 5


def test_agents_disable_implicit_delegation() -> None:
    agents = create_startup_agents(
        create_test_llm()
    )

    assert all(
        agent.allow_delegation is False
        for agent in agents.as_list()
    )


def test_agents_use_bounded_iterations() -> None:
    agents = create_startup_agents(
        create_test_llm()
    )

    assert all(
        agent.max_iter == 5
        for agent in agents.as_list()
    )


def test_model_factory_requires_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "STARTUP_SIMULATOR_MODEL",
        raising=False,
    )
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY",
        "test-api-key",
    )

    with pytest.raises(
        RuntimeError,
        match="STARTUP_SIMULATOR_MODEL",
    ):
        create_llm_from_environment()


def test_model_factory_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "STARTUP_SIMULATOR_MODEL",
        TEST_MODEL,
    )
    monkeypatch.delenv(
        "ANTHROPIC_API_KEY",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="ANTHROPIC_API_KEY",
    ):
        create_llm_from_environment()


def test_model_factory_rejects_wrong_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY",
        "test-api-key",
    )

    with pytest.raises(
        ValueError,
        match="anthropic/",
    ):
        create_llm_from_environment(
            model="openai/example-model"
        )

def test_model_factory_creates_anthropic_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY",
        "test-api-key",
    )

    llm = create_llm_from_environment(
        model=TEST_MODEL
    )

    assert llm.provider == "anthropic"
    assert llm.model == (
        "claude-haiku-4-5-20251001"
    )

def test_model_factory_uses_default_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY",
        "test-api-key",
    )
    monkeypatch.delenv(
        "STARTUP_SIMULATOR_MAX_TOKENS",
        raising=False,
    )

    llm = create_llm_from_environment(
        model=TEST_MODEL,
    )

    assert llm.max_tokens == 4_000


def test_model_factory_rejects_invalid_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY",
        "test-api-key",
    )
    monkeypatch.setenv(
        "STARTUP_SIMULATOR_MAX_TOKENS",
        "invalid",
    )

    with pytest.raises(
        RuntimeError,
        match="must be an integer",
    ):
        create_llm_from_environment(
            model=TEST_MODEL,
        )


def test_model_factory_rejects_non_positive_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY",
        "test-api-key",
    )
    monkeypatch.setenv(
        "STARTUP_SIMULATOR_MAX_TOKENS",
        "0",
    )

    with pytest.raises(
        RuntimeError,
        match="must be greater than zero",
    ):
        create_llm_from_environment(
            model=TEST_MODEL,
        )

def get_tool_names(agent) -> set[str]:
    return {
        tool.name
        for tool in agent.tools
    }


def test_ceo_has_assumption_tool_only() -> None:
    agents = create_startup_agents(
        create_test_llm()
    )

    assert get_tool_names(agents.ceo) == {
        "register_startup_assumption",
    }


@pytest.mark.parametrize(
    "agent_name",
    [
        "product_strategist",
        "ux_designer",
        "technical_lead",
        "critical_reviewer",
    ],
)
def test_specialists_have_analysis_tools(
    agent_name: str,
) -> None:
    agents = create_startup_agents(
        create_test_llm()
    )

    agent = getattr(agents, agent_name)

    assert get_tool_names(agent) == {
        "register_startup_assumption",
        "calculate_startup_risk",
    }


def test_agents_do_not_share_tool_instances() -> None:
    agents = create_startup_agents(
        create_test_llm()
    )

    product_tool_ids = {
        id(tool)
        for tool in agents.product_strategist.tools
    }
    technical_tool_ids = {
        id(tool)
        for tool in agents.technical_lead.tools
    }

    assert product_tool_ids.isdisjoint(
        technical_tool_ids
    )