import pytest

from startup_simulator_agent.models import (
    ReviewResult,
)
from startup_simulator_agent.reflection import (
    ReflectionPolicy,
)


def create_rejected_review() -> ReviewResult:
    return ReviewResult(
        approved=False,
        score=58,
        concerns=[
            "Specialist recommendations conflict.",
        ],
        revision_requests=[
            "Resolve the integration-scope conflict.",
        ],
    )


def test_rejected_review_allows_first_revision() -> None:
    policy = ReflectionPolicy(max_revisions=1)

    assert policy.should_revise(
        create_rejected_review(),
        revisions_completed=0,
    )


def test_approved_review_does_not_trigger_revision() -> None:
    policy = ReflectionPolicy(max_revisions=1)

    review = ReviewResult(
        approved=True,
        score=80,
        strengths=[
            "The assessment is coherent.",
        ],
    )

    assert not policy.should_revise(
        review,
        revisions_completed=0,
    )


def test_policy_stops_after_revision_limit() -> None:
    policy = ReflectionPolicy(max_revisions=1)

    assert not policy.should_revise(
        create_rejected_review(),
        revisions_completed=1,
    )


def test_policy_rejects_negative_revision_limit() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        ReflectionPolicy(max_revisions=-1)


def test_policy_rejects_negative_completed_count() -> None:
    policy = ReflectionPolicy()

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        policy.should_revise(
            create_rejected_review(),
            revisions_completed=-1,
        )