from typing import Protocol

from startup_simulator_agent.models import (
    StartupBrief,
    StartupSimulationResult,
)


class StartupTeam(Protocol):
    """Contract implemented by every startup team."""

    def run(
        self,
        brief: StartupBrief,
        memory_context: str = "",
    ) -> StartupSimulationResult:
        ...