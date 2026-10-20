from dataclasses import dataclass

from crewai import Task

from startup_simulator_agent.agents import (
    StartupAgents,
)
from startup_simulator_agent.models import (
    FinalDecision,
)


@dataclass(frozen=True)
class StartupDecisionTasks:
    final_synthesis: Task

    def as_list(self) -> list[Task]:
        return [
            self.final_synthesis,
        ]


def create_decision_tasks(
    agents: StartupAgents,
) -> StartupDecisionTasks:
    final_synthesis = Task(
        description="""
Make the final startup decision.

Original startup brief:
{startup_brief}

CEO evaluation plan:
{evaluation_plan}

Specialist analyses:
{specialist_analyses}

Critical review:
{final_review}

Return only:
- recommendation: pursue, revise, or reject
- a concise executive summary

Decision rules:
- pursue only when the review is approved
- revise when important validation work remains but the
  startup may still be viable
- reject when the assessment indicates that the startup
  should not proceed

Executive-summary requirements:
- Maximum 2,500 characters.
- Use no more than five short paragraphs.
- Do not reproduce complete specialist analyses.
- Mention the review conclusion and immediate next action.
""".strip(),
        expected_output=(
            "A FinalDecision containing one recommendation "
            "and a concise executive summary."
        ),
        agent=agents.ceo,
        output_pydantic=FinalDecision,
    )

    return StartupDecisionTasks(
        final_synthesis=final_synthesis,
    )