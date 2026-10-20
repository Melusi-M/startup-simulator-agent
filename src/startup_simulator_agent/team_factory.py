import os

from startup_simulator_agent.demo_team import (
    DemoStartupTeam,
)
from startup_simulator_agent.team import StartupTeam


SUPPORTED_MODES = {
    "crewai",
    "demo",
}


def create_startup_team(
    mode: str | None = None,
    verbose: bool = False,
) -> StartupTeam:
    selected_mode = (
        mode
        or os.getenv(
            "STARTUP_SIMULATOR_MODE",
            "demo",
        )
    ).strip().lower()

    if selected_mode == "demo":
        return DemoStartupTeam()

    if selected_mode == "crewai":
        from startup_simulator_agent.crewai_team import (
            create_crewai_startup_team,
        )

        return create_crewai_startup_team(
            verbose=verbose,
        )

    supported_modes = ", ".join(
        sorted(SUPPORTED_MODES)
    )

    raise ValueError(
        f"Unsupported startup simulator mode: "
        f"{selected_mode}. "
        f"Supported modes: {supported_modes}"
    )