import json

import pytest
from pydantic import ValidationError

from startup_simulator_agent.tools import (
    AssumptionRegisterTool,
    RiskScoreTool,
    classify_risk,
)


@pytest.mark.parametrize(
    ("score", "severity"),
    [
        (1, "low"),
        (4, "low"),
        (5, "medium"),
        (9, "medium"),
        (10, "high"),
        (16, "high"),
        (17, "critical"),
        (25, "critical"),
    ],
)
def test_classify_risk(
    score: int,
    severity: str,
) -> None:
    assert classify_risk(score) == severity


@pytest.mark.parametrize(
    "score",
    [0, 26],
)
def test_classify_risk_rejects_invalid_score(
    score: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 1 and 25",
    ):
        classify_risk(score)


def test_risk_tool_calculates_score() -> None:
    tool = RiskScoreTool()

    raw_result = tool._run(
        probability=4,
        impact=5,
        rationale=(
            "Regulatory interpretation has not "
            "been independently validated."
        ),
    )

    result = json.loads(raw_result)

    assert result["probability"] == 4
    assert result["impact"] == 5
    assert result["score"] == 20
    assert result["severity"] == "critical"


def test_risk_tool_rejects_invalid_probability() -> None:
    tool = RiskScoreTool()

    with pytest.raises(ValidationError):
        tool._run(
            probability=6,
            impact=3,
            rationale="Probability is outside the scale.",
        )


def test_assumption_tool_accepts_known_claim() -> None:
    tool = AssumptionRegisterTool()

    raw_result = tool._run(
        claim=(
            "The startup targets estate agencies."
        ),
        evidence_status="known",
        evidence=(
            "The target customer is explicitly stated "
            "in the startup brief."
        ),
    )

    result = json.loads(raw_result)

    assert result["evidence_status"] == "known"
    assert result["evidence"] is not None


def test_assumption_tool_requires_known_evidence() -> None:
    tool = AssumptionRegisterTool()

    with pytest.raises(
        ValidationError,
        match="must include evidence",
    ):
        tool._run(
            claim="Customers will pay for the product.",
            evidence_status="known",
        )


def test_assumption_tool_accepts_inference() -> None:
    tool = AssumptionRegisterTool()

    raw_result = tool._run(
        claim=(
            "Manual compliance work probably creates "
            "operational delays."
        ),
        evidence_status="inferred",
        evidence=(
            "The brief describes the process as manual."
        ),
        validation_action=(
            "Measure delays during customer interviews."
        ),
    )

    result = json.loads(raw_result)

    assert result["evidence_status"] == "inferred"
    assert result["validation_action"] is not None


def test_unvalidated_assumption_requires_action() -> None:
    tool = AssumptionRegisterTool()

    with pytest.raises(
        ValidationError,
        match="validation action",
    ):
        tool._run(
            claim=(
                "Five hundred agencies will buy "
                "the product."
            ),
            evidence_status="unvalidated",
        )


def test_unvalidated_assumption_is_recorded() -> None:
    tool = AssumptionRegisterTool()

    raw_result = tool._run(
        claim=(
            "Agencies are willing to pay R2,000 "
            "per month."
        ),
        evidence_status="unvalidated",
        validation_action=(
            "Test pricing with ten qualified buyers."
        ),
    )

    result = json.loads(raw_result)

    assert result == {
        "claim": (
            "Agencies are willing to pay R2,000 "
            "per month."
        ),
        "evidence_status": "unvalidated",
        "evidence": None,
        "validation_action": (
            "Test pricing with ten qualified buyers."
        ),
    }
