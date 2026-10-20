import json
from pathlib import Path

import pytest

from startup_simulator_agent.demo_team import (
    DemoStartupTeam,
)
from startup_simulator_agent.memory import (
    JsonSimulationMemoryRepository,
    calculate_brief_similarity,
    extract_search_terms,
)
from startup_simulator_agent.models import (
    StartupBrief,
    StartupSimulationResult,
)


def create_result() -> StartupSimulationResult:
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

    return DemoStartupTeam().run(brief)


def create_result_for(
    *,
    name: str,
    description: str,
    target_customer: str,
    problem_statement: str,
) -> StartupSimulationResult:
    brief = StartupBrief(
        name=name,
        description=description,
        target_customer=target_customer,
        problem_statement=problem_statement,
    )

    return DemoStartupTeam().run(brief)


def test_repository_returns_empty_list_when_missing(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    assert repository.list_all() == []


def test_repository_saves_and_loads_simulation(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    result = create_result()
    saved = repository.save(result)

    records = repository.list_all()

    assert len(records) == 1
    assert records[0].memory_id == saved.memory_id
    assert records[0].result == result
    assert records[0].created_at.tzinfo is not None


def test_extract_search_terms_normalizes_text() -> None:
    terms = extract_search_terms(
        "The AI compliance platform for Estate Agencies!"
    )

    assert terms == {
        "compliance",
        "platform",
        "estate",
        "agencies",
    }


def test_similarity_is_high_for_related_briefs() -> None:
    first = StartupBrief(
        name="Compliance Copilot",
        description=(
            "A compliance workflow platform for "
            "estate agencies."
        ),
        target_customer="Estate agencies",
        problem_statement=(
            "Compliance work is manual and fragmented."
        ),
    )

    second = StartupBrief(
        name="Agency Compliance Assistant",
        description=(
            "A platform for managing compliance "
            "workflows in estate agencies."
        ),
        target_customer="Property and estate agencies",
        problem_statement=(
            "Manual compliance workflows are difficult "
            "to audit."
        ),
    )

    assert calculate_brief_similarity(
        first,
        second,
    ) > 0.30


def test_repository_finds_relevant_simulations(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    compliance = repository.save(
        create_result_for(
            name="Compliance Copilot",
            description=(
                "A compliance management platform for "
                "estate agencies."
            ),
            target_customer="Estate agencies",
            problem_statement=(
                "Compliance work is manual and difficult "
                "to audit."
            ),
        )
    )

    repository.save(
        create_result_for(
            name="Meal Planner",
            description=(
                "A meal planning application for busy "
                "families."
            ),
            target_customer="Busy families",
            problem_statement=(
                "Weekly meal planning takes too much time."
            ),
        )
    )

    requested_brief = StartupBrief(
        name="Property Compliance Assistant",
        description=(
            "A compliance workflow tool for property "
            "businesses."
        ),
        target_customer="Estate agencies",
        problem_statement=(
            "Manual compliance processes are hard "
            "to audit."
        ),
    )

    matches = repository.find_relevant(
        requested_brief,
        minimum_similarity=0.10,
    )

    assert len(matches) == 1
    assert matches[0].memory_id == compliance.memory_id


def test_repository_returns_no_irrelevant_memories(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    repository.save(create_result())

    unrelated_brief = StartupBrief(
        name="Fitness Coach",
        description=(
            "A personal fitness coaching application "
            "for endurance athletes."
        ),
        target_customer="Long-distance runners",
        problem_statement=(
            "Athletes struggle to plan training sessions."
        ),
    )

    matches = repository.find_relevant(
        unrelated_brief,
        minimum_similarity=0.10,
    )

    assert matches == []


def test_relevant_memory_search_respects_limit(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    repository.save(create_result())
    repository.save(create_result())
    repository.save(create_result())

    matches = repository.find_relevant(
        create_result().brief,
        limit=2,
    )

    assert len(matches) == 2


@pytest.mark.parametrize(
    ("limit", "minimum_similarity", "message"),
    [
        (0, 0.10, "limit"),
        (3, -0.1, "minimum_similarity"),
        (3, 1.1, "minimum_similarity"),
    ],
)
def test_relevant_search_rejects_invalid_arguments(
    tmp_path: Path,
    limit: int,
    minimum_similarity: float,
    message: str,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        repository.find_relevant(
            create_result().brief,
            limit=limit,
            minimum_similarity=minimum_similarity,
        )


def test_repository_gets_record_by_id(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    saved = repository.save(create_result())

    retrieved = repository.get(saved.memory_id)

    assert retrieved == saved
    assert repository.get("unknown-id") is None


def test_repository_rejects_empty_memory_id(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json"
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        repository.get("   ")


def test_repository_enforces_record_limit(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path / "memory.json",
        maximum_records=2,
    )

    first = repository.save(create_result())
    second = repository.save(create_result())
    third = repository.save(create_result())

    records = repository.list_all()

    assert len(records) == 2
    assert [record.memory_id for record in records] == [
        second.memory_id,
        third.memory_id,
    ]
    assert first.memory_id not in {
        record.memory_id
        for record in records
    }


def test_repository_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        "not valid JSON",
        encoding="utf-8",
    )

    repository = JsonSimulationMemoryRepository(
        memory_path
    )

    with pytest.raises(
        RuntimeError,
        match="Could not read",
    ):
        repository.list_all()


def test_repository_rejects_non_list_json(
    tmp_path: Path,
) -> None:
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps({"unexpected": "object"}),
        encoding="utf-8",
    )

    repository = JsonSimulationMemoryRepository(
        memory_path
    )

    with pytest.raises(
        RuntimeError,
        match="JSON list",
    ):
        repository.list_all()


def test_repository_rejects_directory_path(
    tmp_path: Path,
) -> None:
    repository = JsonSimulationMemoryRepository(
        tmp_path
    )

    with pytest.raises(
        RuntimeError,
        match="not a file",
    ):
        repository.list_all()


def test_repository_rejects_invalid_limit(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        JsonSimulationMemoryRepository(
            tmp_path / "memory.json",
            maximum_records=0,
        )