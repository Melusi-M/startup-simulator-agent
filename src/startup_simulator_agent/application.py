from startup_simulator_agent.memory import (
    SimulationMemoryRepository,
)
from startup_simulator_agent.memory_context import (
    build_memory_context,
)
from startup_simulator_agent.models import (
    StartupBrief,
    StartupSimulationResult,
)
from startup_simulator_agent.team import StartupTeam
from startup_simulator_agent.team_factory import (
    create_startup_team,
)


class StartupSimulator:
    """Application service for running simulations."""

    def __init__(
        self,
        team: StartupTeam,
        memory_repository: (
            SimulationMemoryRepository | None
        ) = None,
    ) -> None:
        self.team = team
        self.memory_repository = memory_repository

    def simulate(
        self,
        brief: StartupBrief,
    ) -> StartupSimulationResult:
        relevant_memories = (
            self.memory_repository.find_relevant(brief)
            if self.memory_repository is not None
            else []
        )

        memory_context = build_memory_context(
            relevant_memories
        )

        result = self.team.run(
            brief,
            memory_context=memory_context,
        )

        if not isinstance(
            result,
            StartupSimulationResult,
        ):
            raise TypeError(
                "Startup team returned an invalid result"
            )

        if result.brief != brief:
            raise RuntimeError(
                "Startup team changed the input brief"
            )

        if self.memory_repository is not None:
            self.memory_repository.save(result)

        return result


def run_startup_simulation(
    name: str,
    description: str,
    target_customer: str,
    problem_statement: str,
    constraints: list[str] | None = None,
    mode: str | None = None,
    verbose: bool = False,
    memory_repository: (
        SimulationMemoryRepository | None
    ) = None,
) -> StartupSimulationResult:
    brief = StartupBrief(
        name=name,
        description=description,
        target_customer=target_customer,
        problem_statement=problem_statement,
        constraints=constraints or [],
    )

    team = create_startup_team(
        mode=mode,
        verbose=verbose,
    )

    simulator = StartupSimulator(
        team=team,
        memory_repository=memory_repository,
    )

    return simulator.simulate(brief)