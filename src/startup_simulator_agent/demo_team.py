from startup_simulator_agent.models import (
    AgentAnalysis,
    AgentRole,
    ReviewResult,
    StartupBrief,
    StartupSimulationResult,
)


class DemoStartupTeam:
    """Deterministic multi-agent team for tests and demos."""

    def run(
        self,
        brief: StartupBrief,
        memory_context: str = "",
    ) -> StartupSimulationResult:
        # Demo mode remains deterministic and does not
        # alter its output based on historical memory.
        _ = memory_context
        analyses = [
            self._create_product_analysis(brief),
            self._create_ux_analysis(brief),
            self._create_technical_analysis(brief),
        ]

        review = self._review_analyses(
            brief=brief,
            analyses=analyses,
        )

        recommendation = (
            "pursue"
            if review.approved
            else "revise"
        )

        executive_summary = (
            f"{brief.name} addresses the problem: "
            f"{brief.problem_statement} "
            "The specialist team recommends validating "
            "the highest-risk assumptions with target "
            "customers before expanding the product."
        )

        return StartupSimulationResult(
            brief=brief,
            analyses=analyses,
            review=review,
            review_history=[review],
            revision_count=0,
            recommendation=recommendation,
            executive_summary=executive_summary,
        )
    def _create_product_analysis(
        self,
        brief: StartupBrief,
    ) -> AgentAnalysis:
        return AgentAnalysis(
            agent_role=AgentRole.PRODUCT_STRATEGIST,
            summary=(
                f"{brief.name} has a clearly identifiable "
                f"customer segment: {brief.target_customer}."
            ),
            findings=[
                (
                    "The problem statement describes a "
                    "specific operational pain."
                ),
                (
                    "The target customer is narrow enough "
                    "for initial customer discovery."
                ),
            ],
            risks=[
                (
                    "The willingness of customers to pay "
                    "has not yet been validated."
                ),
            ],
            recommendations=[
                (
                    "Interview at least five target "
                    "customers before building the MVP."
                ),
            ],
            confidence=0.82,
        )

    def _create_ux_analysis(
        self,
        brief: StartupBrief,
    ) -> AgentAnalysis:
        return AgentAnalysis(
            agent_role=AgentRole.UX_DESIGNER,
            summary=(
                "The MVP should guide the customer from "
                "their current problem to one measurable "
                "successful outcome."
            ),
            findings=[
                (
                    "The first user journey should focus "
                    "on one high-frequency workflow."
                ),
                (
                    "Users need visible progress and a "
                    "clear completion state."
                ),
            ],
            risks=[
                (
                    "Supporting too many workflows in the "
                    "first release could confuse users."
                ),
            ],
            recommendations=[
                (
                    "Prototype the primary journey and "
                    "test it with representative users."
                ),
            ],
            confidence=0.78,
        )

    def _create_technical_analysis(
        self,
        brief: StartupBrief,
    ) -> AgentAnalysis:
        constraint_summary = (
            ", ".join(brief.constraints)
            if brief.constraints
            else "No explicit constraints supplied"
        )

        return AgentAnalysis(
            agent_role=AgentRole.TECHNICAL_LEAD,
            summary=(
                "The product should begin as a modular "
                "monolith with clearly separated domain "
                "and integration boundaries."
            ),
            findings=[
                (
                    "A narrow vertical slice can validate "
                    "the architecture before scaling."
                ),
                f"Known constraints: {constraint_summary}.",
            ],
            risks=[
                (
                    "Premature distributed architecture "
                    "would increase delivery complexity."
                ),
            ],
            recommendations=[
                (
                    "Build one end-to-end workflow with "
                    "observability and automated tests."
                ),
            ],
            confidence=0.85,
        )

    def _review_analyses(
        self,
        brief: StartupBrief,
        analyses: list[AgentAnalysis],
    ) -> ReviewResult:
        required_roles = {
            AgentRole.PRODUCT_STRATEGIST,
            AgentRole.UX_DESIGNER,
            AgentRole.TECHNICAL_LEAD,
        }

        supplied_roles = {
            analysis.agent_role
            for analysis in analyses
        }

        missing_roles = required_roles - supplied_roles

        if missing_roles:
            role_names = ", ".join(
                sorted(role.value for role in missing_roles)
            )

            return ReviewResult(
                approved=False,
                score=55,
                concerns=[
                    "The assessment is incomplete.",
                ],
                revision_requests=[
                    (
                        "Add analyses for the missing "
                        f"roles: {role_names}."
                    ),
                ],
            )

        return ReviewResult(
            approved=True,
            score=82,
            strengths=[
                (
                    "The problem, user experience, and "
                    "technical feasibility were reviewed."
                ),
                (
                    f"The assessment remains focused on "
                    f"{brief.target_customer}."
                ),
            ],
            concerns=[
                (
                    "The market assumptions still require "
                    "real customer evidence."
                ),
            ],
        )