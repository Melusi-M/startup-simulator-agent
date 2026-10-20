import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field

from startup_simulator_agent.models import (
    StartupBrief,
    StartupSimulationResult,
)


IGNORED_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


def extract_search_terms(text: str) -> set[str]:
    """Normalize text into meaningful retrieval terms."""

    terms = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    return {
        term
        for term in terms
        if len(term) >= 3
        and term not in IGNORED_WORDS
    }


def brief_search_terms(
    brief: StartupBrief,
) -> set[str]:
    searchable_text = " ".join(
        [
            brief.name,
            brief.description,
            brief.target_customer,
            brief.problem_statement,
            *brief.constraints,
        ]
    )

    return extract_search_terms(searchable_text)


def calculate_brief_similarity(
    requested: StartupBrief,
    stored: StartupBrief,
) -> float:
    """Calculate Jaccard similarity between two briefs."""

    requested_terms = brief_search_terms(requested)
    stored_terms = brief_search_terms(stored)

    if not requested_terms or not stored_terms:
        return 0.0

    shared_terms = requested_terms & stored_terms
    combined_terms = requested_terms | stored_terms

    return len(shared_terms) / len(combined_terms)

class SimulationMemory(BaseModel):
    """A completed simulation stored in long-term memory."""

    model_config = ConfigDict(
        extra="forbid",
    )

    memory_id: str = Field(
        min_length=1,
    )
    created_at: datetime
    result: StartupSimulationResult

class SimulationMemoryRepository(Protocol):
    """Storage contract used by the application layer."""

    def save(
        self,
        result: StartupSimulationResult,
    ) -> SimulationMemory:
        ...

    def find_relevant(
        self,
        brief: StartupBrief,
        limit: int = 3,
        minimum_similarity: float = 0.10,
    ) -> list[SimulationMemory]:
        ...


class JsonSimulationMemoryRepository:
    """Stores completed simulations in a local JSON file."""

    def find_relevant(
        self,
        brief: StartupBrief,
        limit: int = 3,
        minimum_similarity: float = 0.10,
    ) -> list[SimulationMemory]:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        if not 0 <= minimum_similarity <= 1:
            raise ValueError(
                "minimum_similarity must be between 0 and 1"
            )

        scored_records = [
            (
                record,
                calculate_brief_similarity(
                    requested=brief,
                    stored=record.result.brief,
                ),
            )
            for record in self.list_all()
        ]

        relevant_records = [
            (record, score)
            for record, score in scored_records
            if score >= minimum_similarity
        ]

        relevant_records.sort(
            key=lambda item: (
                item[1],
                item[0].created_at,
            ),
            reverse=True,
        )

        return [
            record
            for record, _score
            in relevant_records[:limit]
        ]

    def __init__(
        self,
        path: str | Path,
        maximum_records: int = 100,
    ) -> None:
        if maximum_records <= 0:
            raise ValueError(
                "maximum_records must be greater than zero"
            )

        self.path = Path(path)
        self.maximum_records = maximum_records

    def save(
        self,
        result: StartupSimulationResult,
    ) -> SimulationMemory:
        records = self.list_all()

        memory = SimulationMemory(
            memory_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            result=result,
        )

        records.append(memory)

        # Retain only the newest bounded set of records.
        records = records[-self.maximum_records :]

        self._write(records)

        return memory

    def list_all(self) -> list[SimulationMemory]:
        if not self.path.exists():
            return []

        if not self.path.is_file():
            raise RuntimeError(
                f"Memory path is not a file: {self.path}"
            )

        try:
            raw_content = self.path.read_text(
                encoding="utf-8"
            )
            raw_records = json.loads(raw_content)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Could not read simulation memory: {self.path}"
            ) from error

        if not isinstance(raw_records, list):
            raise RuntimeError(
                "Simulation memory must contain a JSON list"
            )

        try:
            return [
                SimulationMemory.model_validate(record)
                for record in raw_records
            ]
        except Exception as error:
            raise RuntimeError(
                "Simulation memory contains invalid records"
            ) from error

    def get(
        self,
        memory_id: str,
    ) -> SimulationMemory | None:
        normalized_id = memory_id.strip()

        if not normalized_id:
            raise ValueError(
                "memory_id cannot be empty"
            )

        return next(
            (
                record
                for record in self.list_all()
                if record.memory_id == normalized_id
            ),
            None,
        )

    def _write(
        self,
        records: list[SimulationMemory],
    ) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = self.path.with_suffix(
            f"{self.path.suffix}.tmp"
        )

        serialized_records = [
            record.model_dump(mode="json")
            for record in records
        ]

        try:
            temporary_path.write_text(
                json.dumps(
                    serialized_records,
                    indent=2,
                ),
                encoding="utf-8",
            )

            # Replace only after serialization and writing succeed.
            temporary_path.replace(self.path)
        except OSError as error:
            raise RuntimeError(
                f"Could not write simulation memory: {self.path}"
            ) from error