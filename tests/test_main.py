import json
from pathlib import Path

from startup_simulator_agent.application import (
    run_startup_simulation,
)
from startup_simulator_agent.main import (
    format_result,
    main,
)
from startup_simulator_agent.memory import (
    JsonSimulationMemoryRepository,
)
from startup_simulator_agent.models import (
    AgentRole,
    PlannedWorkItem,
    SimulationPlan,
)


BASE_INPUT_ARGUMENTS = [
    "--name",
    "Compliance Copilot",
    "--description",
    (
        "An AI platform that helps estate agencies "
        "operate their compliance processes."
    ),
    "--target-customer",
    "Estate agencies",
    "--problem",
    (
        "Compliance work is manual and difficult "
        "to audit."
    ),
    "--mode",
    "demo",
]

BASE_ARGUMENTS = [
    *BASE_INPUT_ARGUMENTS,
    "--no-memory",
]


def test_main_prints_readable_report(
    capsys,
) -> None:
    exit_code = main(BASE_ARGUMENTS)

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Startup Simulator" in captured.out
    assert "Compliance Copilot" in captured.out
    assert "Recommendation: PURSUE" in captured.out
    assert "Product Strategist" in captured.out
    assert "Critical review" in captured.out


def test_main_prints_json(
    capsys,
) -> None:
    exit_code = main(
        [
            *BASE_ARGUMENTS,
            "--json",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert exit_code == 0
    assert result["brief"]["name"] == (
        "Compliance Copilot"
    )
    assert result["recommendation"] == "pursue"
    assert len(result["analyses"]) == 3


def test_main_accepts_multiple_constraints(
    capsys,
) -> None:
    exit_code = main(
        [
            *BASE_ARGUMENTS,
            "--constraint",
            "Small founding team",
            "--constraint",
            "South African market first",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert exit_code == 0
    assert result["brief"]["constraints"] == [
        "Small founding team",
        "South African market first",
    ]


def test_format_result_displays_ceo_plan() -> None:
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
    )

    result.plan = SimulationPlan(
        strategy=(
            "Evaluate product, user experience, and "
            "technical feasibility before investment."
        ),
        work_items=[
            PlannedWorkItem(
                agent_role=(
                    AgentRole.PRODUCT_STRATEGIST
                ),
                objective=(
                    "Validate the customer problem."
                ),
                expected_deliverable=(
                    "A product and market assessment."
                ),
            ),
            PlannedWorkItem(
                agent_role=AgentRole.UX_DESIGNER,
                objective=(
                    "Validate the primary user journey."
                ),
                expected_deliverable=(
                    "A user experience assessment."
                ),
            ),
            PlannedWorkItem(
                agent_role=AgentRole.TECHNICAL_LEAD,
                objective=(
                    "Validate technical feasibility."
                ),
                expected_deliverable=(
                    "A technical architecture assessment."
                ),
            ),
        ],
        success_criteria=[
            "Major assumptions are identified.",
            "Recommendations are actionable.",
        ],
    )

    output = format_result(result)

    assert "CEO evaluation plan" in output
    assert "Product Strategist" in output
    assert "Validate the customer problem" in output
    assert "Major assumptions are identified" in output


def test_main_persists_completed_simulation(
    tmp_path: Path,
    capsys,
) -> None:
    memory_path = tmp_path / "simulation_memory.json"

    exit_code = main(
        [
            *BASE_INPUT_ARGUMENTS,
            "--memory-path",
            str(memory_path),
        ]
    )

    capsys.readouterr()

    repository = JsonSimulationMemoryRepository(
        memory_path
    )
    records = repository.list_all()

    assert exit_code == 0
    assert memory_path.exists()
    assert len(records) == 1
    assert records[0].result.brief.name == (
        "Compliance Copilot"
    )


def test_main_no_memory_does_not_create_file(
    tmp_path: Path,
    capsys,
) -> None:
    memory_path = tmp_path / "simulation_memory.json"

    exit_code = main(
        [
            *BASE_INPUT_ARGUMENTS,
            "--memory-path",
            str(memory_path),
            "--no-memory",
        ]
    )

    capsys.readouterr()

    assert exit_code == 0
    assert not memory_path.exists()


def test_configure_console_encoding(
    monkeypatch,
) -> None:
    import importlib

    main_module = importlib.import_module(
        "startup_simulator_agent.main"
    )

    received_calls = []

    class FakeStream:
        def reconfigure(self, **kwargs) -> None:
            received_calls.append(kwargs)

    monkeypatch.setattr(
        main_module.sys,
        "stdout",
        FakeStream(),
    )
    monkeypatch.setattr(
        main_module.sys,
        "stderr",
        FakeStream(),
    )

    main_module.configure_console_encoding()

    assert received_calls == [
        {
            "encoding": "utf-8",
            "errors": "replace",
        },
        {
            "encoding": "utf-8",
            "errors": "replace",
        },
    ]