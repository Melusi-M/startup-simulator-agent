import sys
import argparse
import json
from collections.abc import Sequence

from startup_simulator_agent.application import (
    run_startup_simulation,
)
from startup_simulator_agent.models import (
    StartupSimulationResult,
)

from startup_simulator_agent.memory import (
    JsonSimulationMemoryRepository,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-agent startup simulation."
        )
    )


    parser.add_argument(
        "--memory-path",
        default="data/simulation_memory.json",
        help=(
            "Path used to store completed simulations. "
            "Defaults to data/simulation_memory.json."
        ),
    )

    parser.add_argument(
        "--no-memory",
        action="store_true",
        help=(
            "Run without reading or writing long-term "
            "simulation memory."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed agent and task activity",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Startup name",
    )
    parser.add_argument(
        "--description",
        required=True,
        help="Description of the startup idea",
    )
    parser.add_argument(
        "--target-customer",
        required=True,
        help="Primary target customer",
    )
    parser.add_argument(
        "--problem",
        required=True,
        help="Problem the startup solves",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help=(
            "Startup constraint. May be supplied "
            "more than once."
        ),
    )
    parser.add_argument(
        "--mode",
        default=None,
        help=(
            "Team implementation. Defaults to the "
            "STARTUP_SIMULATOR_MODE environment "
            "variable or demo."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured result as JSON",
    )

    return parser


def format_result(
    result: StartupSimulationResult,
) -> str:
    lines = [
        "Startup Simulator",
        "=" * 60,
        f"Startup: {result.brief.name}",
        (
            "Target customer: "
            f"{result.brief.target_customer}"
        ),
        (
            "Recommendation: "
            f"{result.recommendation.upper()}"
        ),
        f"Review score: {result.review.score}/100",
        (
            "Revision cycles: "
            f"{result.revision_count}"
        ),
        "",
        "Executive summary",
        "-" * 60,
        result.executive_summary,
    ]

    if result.plan is not None:
        lines.extend(
            [
                "",
                "CEO evaluation plan",
                "-" * 60,
                f"Strategy: {result.plan.strategy}",
                "",
                "Work assignments:",
            ]
        )

        for work_item in result.plan.work_items:
            role_name = (
                work_item.agent_role.value
                .replace("_", " ")
                .title()
            )

            lines.extend(
                [
                    f"- {role_name}",
                    (
                        "  Objective: "
                        f"{work_item.objective}"
                    ),
                    (
                        "  Deliverable: "
                        f"{work_item.expected_deliverable}"
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "Success criteria:",
            ]
        )

        lines.extend(
            f"- {criterion}"
            for criterion in result.plan.success_criteria
        )

    lines.extend(
        [
            "",
            "Specialist analyses",
            "-" * 60,
        ]
    )

    for analysis in result.analyses:
        role_name = (
            analysis.agent_role.value
            .replace("_", " ")
            .title()
        )

        lines.extend(
            [
                "",
                role_name,
                f"Confidence: {analysis.confidence:.0%}",
                f"Summary: {analysis.summary}",
                "Findings:",
            ]
        )

        lines.extend(
            f"- {finding}"
            for finding in analysis.findings
        )

        if analysis.risks:
            lines.append("Risks:")
            lines.extend(
                f"- {risk}"
                for risk in analysis.risks
            )

        lines.append("Recommendations:")
        lines.extend(
            f"- {recommendation}"
            for recommendation
            in analysis.recommendations
        )

    lines.extend(
        [
            "",
            "Critical review",
            "-" * 60,
            (
                "Approved: "
                f"{'Yes' if result.review.approved else 'No'}"
            ),
        ]
    )

    if result.review.strengths:
        lines.append("Strengths:")
        lines.extend(
            f"- {strength}"
            for strength in result.review.strengths
        )

    if result.review.concerns:
        lines.append("Concerns:")
        lines.extend(
            f"- {concern}"
            for concern in result.review.concerns
        )

    if result.review.revision_requests:
        lines.append("Revision requests:")
        lines.extend(
            f"- {request}"
            for request
            in result.review.revision_requests
        )

    if result.review_history:
        lines.extend(
            [
                "",
                "Review history",
                "-" * 60,
            ]
        )

        for number, review in enumerate(
            result.review_history,
            start=1,
        ):
            status = (
                "Approved"
                if review.approved
                else "Rejected"
            )

            lines.append(
                f"{number}. {status} "
                f"({review.score}/100)"
            )

    return "\n".join(lines)

def configure_console_encoding() -> None:
    """Configure CLI streams for CrewAI Unicode output."""

    for stream in (
        sys.stdout,
        sys.stderr,
    ):
        reconfigure = getattr(
            stream,
            "reconfigure",
            None,
        )

        if callable(reconfigure):
            reconfigure(
                encoding="utf-8",
                errors="replace",
            )

def main(
    arguments: Sequence[str] | None = None,
) -> int:
    # Do not reconfigure pytest's captured streams.
    if arguments is None:
        configure_console_encoding()

    parser = build_parser()
    parsed = parser.parse_args(arguments)

    memory_repository = (
        None
        if parsed.no_memory
        else JsonSimulationMemoryRepository(
            parsed.memory_path
        )
    )

    result = run_startup_simulation(
        name=parsed.name,
        description=parsed.description,
        target_customer=parsed.target_customer,
        problem_statement=parsed.problem,
        constraints=parsed.constraint,
        mode=parsed.mode,
        verbose=parsed.verbose,
        memory_repository=memory_repository,
    )

    if parsed.json:
        print(
            json.dumps(
                result.model_dump(mode="json"),
                indent=2,
            )
        )
    else:
        print(format_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())