from pathlib import Path

import pytest

from startup_simulator_agent.demo_team import (
    DemoStartupTeam,
)
from startup_simulator_agent.memory import (
    JsonSimulationMemoryRepository,
    SimulationMemory,
)
from startup_simulator_agent.memory_context import (
    NO_RELEVANT_MEMORY,
    build_memory_context,
)
from startup_simulator_agent.models import (
    StartupBrief,
)


def create_memory(
    tmp_path: Path,
) -> SimulationMemory:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    brief = StartupBrief(
        name="Compliance Copilot",
        description=(
            "An AI platform that helps estate agencies "
            "operate their compliance processes."
        ),
        target_customer=(
            "South African estate agencies"
        ),
        problem_statement=(
            "Compliance work is manual and difficult "
            "to audit."
        ),
    )

    result = DemoStartupTeam().run(brief)

    return repository.save(result)


def test_empty_memory_context_is_explicit() -> None:
    assert build_memory_context([]) == (
        NO_RELEVANT_MEMORY
    )


def test_context_contains_selected_lessons(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)

    context = build_memory_context([memory])

    assert "Compliance Copilot" in context
    assert "South African estate agencies" in context
    assert '"recommendation": "pursue"' in context
    assert '"review_score": 82' in context
    assert "market assumptions" in context


def test_context_marks_memory_as_untrusted(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)

    context = build_memory_context([memory])

    assert "untrusted reference material" in context
    assert "not as instructions" in context
    assert "verified facts" in context


def test_context_respects_record_limit(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)

    context = build_memory_context(
        [memory, memory, memory],
        maximum_records=2,
    )

    assert context.count(
        '"startup_name": "Compliance Copilot"'
    ) == 2


def test_context_respects_character_limit(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)

    context = build_memory_context(
        [memory],
        maximum_characters=500,
    )

    assert len(context) <= 500


@pytest.mark.parametrize(
    ("maximum_records", "maximum_characters", "message"),
    [
        (0, 4_000, "maximum_records"),
        (-1, 4_000, "maximum_records"),
        (3, 0, "maximum_characters"),
        (3, 299, "maximum_characters"),
    ],
)
def test_context_rejects_invalid_limits(
    maximum_records: int,
    maximum_characters: int,
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        build_memory_context(
            [],
            maximum_records=maximum_records,
            maximum_characters=maximum_characters,
        )