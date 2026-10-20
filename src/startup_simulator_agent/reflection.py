from dataclasses import dataclass

from startup_simulator_agent.models import (
    ReviewResult,
)


@dataclass(frozen=True)
class ReflectionPolicy:
    """Controls whether another revision cycle is allowed."""

    max_revisions: int = 1

    def __post_init__(self) -> None:
        if self.max_revisions < 0:
            raise ValueError(
                "max_revisions cannot be negative"
            )

    def should_revise(
        self,
        review: ReviewResult,
        revisions_completed: int,
    ) -> bool:
        if revisions_completed < 0:
            raise ValueError(
                "revisions_completed cannot be negative"
            )

        return (
            not review.approved
            and revisions_completed < self.max_revisions
        )