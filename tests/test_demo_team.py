from startup_simulator_agent.demo_team import (
    DemoStartupTeam,
)
from startup_simulator_agent.models import (
    AgentRole,
    StartupBrief,
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


def test_demo_team_returns_all_specialist_analyses() -> None:
    team = DemoStartupTeam()

    result = team.run(create_brief())

    roles = {
        analysis.agent_role
        for analysis in result.analyses
    }

    assert roles == {
        AgentRole.PRODUCT_STRATEGIST,
        AgentRole.UX_DESIGNER,
        AgentRole.TECHNICAL_LEAD,
    }


def test_demo_team_preserves_startup_brief() -> None:
    brief = create_brief()
    team = DemoStartupTeam()

    result = team.run(brief)

    assert result.brief == brief


def test_demo_team_returns_approved_review() -> None:
    team = DemoStartupTeam()

    result = team.run(create_brief())

    assert result.review.approved is True
    assert result.review.score == 82
    assert result.recommendation == "pursue"


def test_demo_team_is_deterministic() -> None:
    team = DemoStartupTeam()
    brief = create_brief()

    first_result = team.run(brief)
    second_result = team.run(brief)

    assert first_result == second_result


def test_demo_team_includes_constraints_in_analysis() -> None:
    team = DemoStartupTeam()

    result = team.run(create_brief())

    technical_analysis = next(
        analysis
        for analysis in result.analyses
        if (
            analysis.agent_role
            == AgentRole.TECHNICAL_LEAD
        )
    )

    combined_findings = " ".join(
        technical_analysis.findings
    )

    assert "Small founding team" in combined_findings
    assert (
        "South African market first"
        in combined_findings
    )