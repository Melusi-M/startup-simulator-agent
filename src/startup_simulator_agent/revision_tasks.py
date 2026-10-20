from dataclasses import dataclass

from crewai import Task

from startup_simulator_agent.agents import (
    StartupAgents,
)
from startup_simulator_agent.models import (
    AgentAnalysis,
    FinalDecision,
    ReviewResult,
)
from startup_simulator_agent.tasks import (
    ANALYSIS_TOOL_REQUIREMENTS,
    REVIEW_OUTPUT_CONSTRAINTS,
    REVIEW_TOOL_REQUIREMENTS,
    SPECIALIST_OUTPUT_CONSTRAINTS,
)


@dataclass(frozen=True)
class StartupRevisionTasks:
    product_revision: Task
    ux_revision: Task
    technical_revision: Task
    second_review: Task
    final_synthesis: Task

    def as_list(self) -> list[Task]:
        return [
            self.product_revision,
            self.ux_revision,
            self.technical_revision,
            self.second_review,
            self.final_synthesis,
        ]


def create_revision_tasks(
    agents: StartupAgents,
) -> StartupRevisionTasks:
    product_revision = Task(
        description=(
            """
Revise the product and market analysis.

Original startup brief:
{startup_brief}

Original product analysis:
{initial_product_analysis}

Critical review:
{initial_review}

Address every applicable revision request.
Correct unsupported claims and clearly identify any
questions that cannot be answered without external evidence.

Your agent_role must remain exactly:
product_strategist

Do not invent customer or market evidence.
""".strip()
            + "\n\n"
            + ANALYSIS_TOOL_REQUIREMENTS
            + "\n\n"
            + SPECIALIST_OUTPUT_CONSTRAINTS
        ),
        expected_output=(
            "A revised AgentAnalysis addressing the critical "
            "review without inventing evidence."
        ),
        agent=agents.product_strategist,
        output_pydantic=AgentAnalysis,
    )

    ux_revision = Task(
        description=(
            """
Revise the UX and product-design analysis.

Original startup brief:
{startup_brief}

Original UX analysis:
{initial_ux_analysis}

Critical review:
{initial_review}

Address every applicable revision request.
Resolve scope conflicts with the technical assessment and
separate validated facts from design assumptions.

Your agent_role must remain exactly:
ux_designer
""".strip()
            + "\n\n"
            + ANALYSIS_TOOL_REQUIREMENTS
            + "\n\n"
            + SPECIALIST_OUTPUT_CONSTRAINTS
        ),
        expected_output=(
            "A revised AgentAnalysis addressing the critical "
            "review and resolving UX scope conflicts."
        ),
        agent=agents.ux_designer,
        output_pydantic=AgentAnalysis,
    )

    technical_revision = Task(
        description=(
            """
Revise the software and AI architecture analysis.

Original startup brief:
{startup_brief}

Original technical analysis:
{initial_technical_analysis}

Critical review:
{initial_review}

Address every applicable revision request.
Remove unsupported vendor, competitor, regulatory, cost,
certification, and integration claims.
Align the MVP scope with the revised product and UX position.

Your agent_role must remain exactly:
technical_lead
""".strip()
            + "\n\n"
            + ANALYSIS_TOOL_REQUIREMENTS
            + "\n\n"
            + SPECIALIST_OUTPUT_CONSTRAINTS
        ),
        expected_output=(
            "A revised AgentAnalysis with evidence-aware, "
            "appropriately scoped technical recommendations."
        ),
        agent=agents.technical_lead,
        output_pydantic=AgentAnalysis,
    )

    second_review = Task(
        description=(
            """
Re-evaluate the three revised specialist analyses.

Original startup brief:
{startup_brief}

Initial critical review:
{initial_review}

Compare the revised analyses against:
- the original startup brief
- the initial critical review
- every initial revision request

Check whether:
- contradictions were resolved
- unsupported claims were removed
- uncertainty is stated honestly
- recommendations now form a coherent plan
- specialists stayed within their areas of responsibility

Do not approve merely because revisions were attempted.
Approve only if the revised assessment is coherent and
actionable.

A rejected review must include revision requests.
An approved review must score at least 70.
""".strip()
            + "\n\n"
            + REVIEW_TOOL_REQUIREMENTS
            + "\n\n"
            + REVIEW_OUTPUT_CONSTRAINTS
        ),
        expected_output=(
            "A second structured ReviewResult with no more "
            "than 8 strengths, 8 concerns, and 8 revision "
            "requests."
        ),
        agent=agents.critical_reviewer,
        context=[
            product_revision,
            ux_revision,
            technical_revision,
        ],
        output_pydantic=ReviewResult,
    )

    final_synthesis = Task(
        description="""
Make the final startup decision using the revised specialist
analyses and second critical review.

The system has reached its maximum revision count.
Do not request another agent revision cycle.

Return only:
- recommendation: pursue, revise, or reject
- a concise executive summary

Decision rules:
- pursue only when the final review is approved
- revise when important validation work remains but the idea
  remains potentially viable
- reject when evidence indicates the idea should not proceed

Executive-summary requirements:
- The executive_summary value must contain no more than
  2,500 characters.
- Aim for 1,500 to 2,000 characters.
- Use exactly two short paragraphs.
- Do not use headings, bullet points, numbered lists,
  or markdown.
- Do not reproduce specialist findings or revision requests.
- Summarize only the decision, decisive evidence, major
  unresolved risk, review conclusion, and immediate next
  action.
- Stop writing immediately after the immediate next action.
""".strip(),
        expected_output=(
            "A bounded final decision with a concise executive "
            "summary based on the revised assessment."
        ),
        agent=agents.ceo,
        context=[
            product_revision,
            ux_revision,
            technical_revision,
            second_review,
        ],
        output_pydantic=FinalDecision,
    )

    return StartupRevisionTasks(
        product_revision=product_revision,
        ux_revision=ux_revision,
        technical_revision=technical_revision,
        second_review=second_review,
        final_synthesis=final_synthesis,
    )