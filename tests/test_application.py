from pathlib import Path

import pytest

from startup_simulator_agent.application import (
    StartupSimulator,
    run_startup_simulation,
)
from startup_simulator_agent.demo_team import (
    DemoStartupTeam,
)
from startup_simulator_agent.memory import (
    JsonSimulationMemoryRepository,
    SimulationMemory,
)
from startup_simulator_agent.models import (
    StartupBrief,
    StartupSimulationResult,
)
from startup_simulator_agent.team_factory import (
    create_startup_team,
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
    )


class FakeMemoryRepository:
    def __init__(
        self,
        relevant_memories: (
            list[SimulationMemory] | None
        ) = None,
    ) -> None:
        self.relevant_memories = (
            relevant_memories or []
        )
        self.received_briefs: list[StartupBrief] = []
        self.saved_results: list[
            StartupSimulationResult
        ] = []

    def find_relevant(
        self,
        brief: StartupBrief,
        limit: int = 3,
        minimum_similarity: float = 0.10,
    ) -> list[SimulationMemory]:
        self.received_briefs.append(brief)
        return self.relevant_memories

    def save(
        self,
        result: StartupSimulationResult,
    ):
        self.saved_results.append(result)
        return None


def test_factory_uses_demo_mode_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "STARTUP_SIMULATOR_MODE",
        raising=False,
    )

    team = create_startup_team()

    assert team.__class__.__name__ == "DemoStartupTeam"


def test_factory_reads_mode_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "STARTUP_SIMULATOR_MODE",
        "  DEMO  ",
    )

    team = create_startup_team()

    assert team.__class__.__name__ == "DemoStartupTeam"


def test_factory_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported startup simulator mode",
    ):
        create_startup_team("unknown")


def test_application_runs_complete_simulation() -> None:
    result = run_startup_simulation(
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
        mode="demo",
    )

    assert result.brief.name == "Compliance Copilot"
    assert result.recommendation == "pursue"
    assert len(result.analyses) == 3


def test_application_rejects_invalid_team_result() -> None:
    class InvalidTeam:
        def run(
            self,
            brief: StartupBrief,
            memory_context: str = "",
        ):
            return {
                "brief": brief,
            }

    simulator = StartupSimulator(
        team=InvalidTeam(),
    )

    with pytest.raises(
        TypeError,
        match="invalid result",
    ):
        simulator.simulate(create_brief())


def test_application_rejects_changed_brief() -> None:
    original_brief = create_brief()

    class ChangedBriefTeam:
        def run(
            self,
            brief: StartupBrief,
            memory_context: str = "",
        ) -> StartupSimulationResult:
            return run_startup_simulation(
                name="Different Startup",
                description=(
                    "A completely different startup "
                    "description for another market."
                ),
                target_customer="Different customers",
                problem_statement=(
                    "A completely different problem exists."
                ),
                mode="demo",
            )

    simulator = StartupSimulator(
        team=ChangedBriefTeam(),
    )

    with pytest.raises(
        RuntimeError,
        match="changed the input brief",
    ):
        simulator.simulate(original_brief)


def test_application_saves_valid_result() -> None:
    memory_repository = FakeMemoryRepository()

    simulator = StartupSimulator(
        team=DemoStartupTeam(),
        memory_repository=memory_repository,
    )

    result = simulator.simulate(create_brief())

    assert memory_repository.received_briefs == [
        result.brief,
    ]
    assert memory_repository.saved_results == [
        result,
    ]


def test_application_does_not_save_invalid_result() -> None:
    class InvalidTeam:
        def run(
            self,
            brief: StartupBrief,
            memory_context: str = "",
        ):
            return {
                "brief": brief,
            }

    memory_repository = FakeMemoryRepository()

    simulator = StartupSimulator(
        team=InvalidTeam(),
        memory_repository=memory_repository,
    )

    with pytest.raises(
        TypeError,
        match="invalid result",
    ):
        simulator.simulate(create_brief())

    assert memory_repository.saved_results == []


def test_application_does_not_save_changed_brief() -> None:
    original_brief = create_brief()

    class ChangedBriefTeam:
        def run(
            self,
            brief: StartupBrief,
            memory_context: str = "",
        ) -> StartupSimulationResult:
            changed_brief = StartupBrief(
                name="Different Startup",
                description=(
                    "A completely different startup "
                    "description for another market."
                ),
                target_customer="Different customers",
                problem_statement=(
                    "A completely different problem exists."
                ),
            )

            return DemoStartupTeam().run(
                changed_brief
            )

    memory_repository = FakeMemoryRepository()

    simulator = StartupSimulator(
        team=ChangedBriefTeam(),
        memory_repository=memory_repository,
    )

    with pytest.raises(
        RuntimeError,
        match="changed the input brief",
    ):
        simulator.simulate(original_brief)

    assert memory_repository.saved_results == []


def test_run_function_uses_supplied_memory() -> None:
    memory_repository = FakeMemoryRepository()

    result = run_startup_simulation(
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
        mode="demo",
        memory_repository=memory_repository,
    )

    assert memory_repository.received_briefs == [
        result.brief,
    ]
    assert memory_repository.saved_results == [
        result,
    ]


def test_application_passes_relevant_memory_to_team(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    previous_result = DemoStartupTeam().run(
        create_brief()
    )
    repository.save(previous_result)

    class CapturingTeam:
        def __init__(self) -> None:
            self.memory_context: str | None = None

        def run(
            self,
            brief: StartupBrief,
            memory_context: str = "",
        ) -> StartupSimulationResult:
            self.memory_context = memory_context
            return DemoStartupTeam().run(brief)

    team = CapturingTeam()

    simulator = StartupSimulator(
        team=team,
        memory_repository=repository,
    )

    simulator.simulate(create_brief())

    assert team.memory_context is not None
    assert "Compliance Copilot" in team.memory_context
    assert "untrusted reference material" in (
        team.memory_context
    )


def test_application_passes_explicit_empty_memory_context() -> None:
    class CapturingTeam:
        def __init__(self) -> None:
            self.memory_context: str | None = None

        def run(
            self,
            brief: StartupBrief,
            memory_context: str = "",
        ) -> StartupSimulationResult:
            self.memory_context = memory_context
            return DemoStartupTeam().run(brief)

    team = CapturingTeam()
    simulator = StartupSimulator(team=team)

    simulator.simulate(create_brief())

    assert team.memory_context == (
        "No relevant previous simulations were found."
    )