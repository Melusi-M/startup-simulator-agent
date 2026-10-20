from typing import Any, Protocol, TypeVar

import json

from startup_simulator_agent.decision_tasks import (
    create_decision_tasks,
)

from crewai import Crew, Process
from pydantic import BaseModel

from startup_simulator_agent.reflection import (
    ReflectionPolicy,
)
from startup_simulator_agent.revision_tasks import (
    create_revision_tasks,
)

from startup_simulator_agent.agents import (
    create_startup_agents,
)
from startup_simulator_agent.model_factory import (
    create_llm_from_environment,
)
from startup_simulator_agent.models import (
    AgentAnalysis,
    AgentRole,
    FinalDecision,
    ReviewResult,
    SimulationPlan,
    StartupBrief,
    StartupSimulationResult,
)
from startup_simulator_agent.tasks import (
    create_startup_tasks,
)


ModelType = TypeVar(
    "ModelType",
    bound=BaseModel,
)


class CrewRunner(Protocol):
    def kickoff(
        self,
        inputs: dict[str, Any],
    ) -> Any:
        ...


def require_structured_output(
    task_outputs: list[Any],
    index: int,
    expected_type: type[ModelType],
    task_name: str,
) -> ModelType:
    try:
        task_output = task_outputs[index]
    except IndexError as error:
        raise RuntimeError(
            f"Missing output for task: {task_name}"
        ) from error

    value = getattr(
        task_output,
        "pydantic",
        None,
    )

    if not isinstance(value, expected_type):
        raise RuntimeError(
            f"Task {task_name} did not return "
            f"{expected_type.__name__}"
        )

    return value


class CrewAIStartupTeam:
    """Claude-powered team with bounded reflection."""

    def __init__(
        self,
        crew: CrewRunner,
        revision_crew: CrewRunner | None = None,
        decision_crew: CrewRunner | None = None,
        reflection_policy: ReflectionPolicy | None = None,
    ) -> None:
        self.crew = crew
        self.revision_crew = revision_crew
        self.decision_crew = decision_crew
        self.reflection_policy = (
            reflection_policy
            or ReflectionPolicy(max_revisions=1)
        )

    def run(
        self,
        brief: StartupBrief,
        memory_context: str = "",
    ) -> StartupSimulationResult:
        initial_output = self.crew.kickoff(
            inputs=self._create_initial_inputs(
                brief=brief,
                memory_context=memory_context,
            )
        )

        initial_tasks = self._get_task_outputs(
            initial_output
        )

        plan = require_structured_output(
            initial_tasks,
            index=0,
            expected_type=SimulationPlan,
            task_name="planning",
        )
        product_analysis = require_structured_output(
            initial_tasks,
            index=1,
            expected_type=AgentAnalysis,
            task_name="product_analysis",
        )
        ux_analysis = require_structured_output(
            initial_tasks,
            index=2,
            expected_type=AgentAnalysis,
            task_name="ux_analysis",
        )
        technical_analysis = require_structured_output(
            initial_tasks,
            index=3,
            expected_type=AgentAnalysis,
            task_name="technical_analysis",
        )
        initial_review = require_structured_output(
            initial_tasks,
            index=4,
            expected_type=ReviewResult,
            task_name="critical_review",
        )
        initial_decision: FinalDecision | None = None

        # Temporary compatibility for callers still supplying
        # the legacy six-task initial crew.
        if self.decision_crew is None:
            initial_decision = require_structured_output(
                initial_tasks,
                index=5,
                expected_type=FinalDecision,
                task_name="final_synthesis",
            )

        initial_analyses = [
            product_analysis,
            ux_analysis,
            technical_analysis,
        ]

        self._validate_specialist_roles(
            initial_analyses
        )
        if initial_decision is not None:
            self._validate_decision(
            initial_review,
            initial_decision,
        )

        should_revise = (
            self.revision_crew is not None
            and self.reflection_policy.should_revise(
                initial_review,
                revisions_completed=0,
            )
        )

        if should_revise:
            return self._run_revision(
                brief=brief,
                plan=plan,
                initial_analyses=initial_analyses,
                initial_review=initial_review,
            )

        if self.decision_crew is not None:
            return self._run_decision(
                brief=brief,
                plan=plan,
                analyses=initial_analyses,
                review=initial_review,
            )

        if initial_decision is None:
            raise RuntimeError(
                "Decision crew is not configured and the "
                "initial crew did not return a decision"
            )

        return StartupSimulationResult(
            brief=brief,
            plan=plan,
            analyses=initial_analyses,
            review=initial_review,
            review_history=[initial_review],
            revision_count=0,
            recommendation=(
                initial_decision.recommendation
            ),
            executive_summary=(
                initial_decision.executive_summary
            ),
        )

    def _run_decision(
        self,
        brief: StartupBrief,
        plan: SimulationPlan,
        analyses: list[AgentAnalysis],
        review: ReviewResult,
    ) -> StartupSimulationResult:
        if self.decision_crew is None:
            raise RuntimeError(
                "Decision crew is not configured"
            )

        decision_output = self.decision_crew.kickoff(
            inputs={
                "startup_brief": (
                    brief.model_dump_json(indent=2)
                ),
                "evaluation_plan": (
                    plan.model_dump_json(indent=2)
                ),
                "specialist_analyses": json.dumps(
                    [
                        analysis.model_dump(mode="json")
                        for analysis in analyses
                    ],
                    indent=2,
                ),
                "final_review": (
                    review.model_dump_json(indent=2)
                ),
            }
        )

        decision_tasks = self._get_task_outputs(
            decision_output
        )

        decision = require_structured_output(
            decision_tasks,
            index=0,
            expected_type=FinalDecision,
            task_name="final_synthesis",
        )

        self._validate_decision(
            review,
            decision,
        )

        return StartupSimulationResult(
            brief=brief,
            plan=plan,
            analyses=analyses,
            review=review,
            review_history=[review],
            revision_count=0,
            recommendation=decision.recommendation,
            executive_summary=(
                decision.executive_summary
            ),
        )

    def _run_revision(
        self,
        brief: StartupBrief,
        plan: SimulationPlan,
        initial_analyses: list[AgentAnalysis],
        initial_review: ReviewResult,
    ) -> StartupSimulationResult:
        if self.revision_crew is None:
            raise RuntimeError(
                "Revision crew is not configured"
            )

        revision_output = self.revision_crew.kickoff(
            inputs={
                "startup_brief": (
                    brief.model_dump_json(indent=2)
                ),
                "initial_product_analysis": (
                    initial_analyses[0].model_dump_json(
                        indent=2
                    )
                ),
                "initial_ux_analysis": (
                    initial_analyses[1].model_dump_json(
                        indent=2
                    )
                ),
                "initial_technical_analysis": (
                    initial_analyses[2].model_dump_json(
                        indent=2
                    )
                ),
                "initial_review": (
                    initial_review.model_dump_json(
                        indent=2
                    )
                ),
            }
        )

        revision_tasks = self._get_task_outputs(
            revision_output
        )

        product_analysis = require_structured_output(
            revision_tasks,
            index=0,
            expected_type=AgentAnalysis,
            task_name="product_revision",
        )
        ux_analysis = require_structured_output(
            revision_tasks,
            index=1,
            expected_type=AgentAnalysis,
            task_name="ux_revision",
        )
        technical_analysis = require_structured_output(
            revision_tasks,
            index=2,
            expected_type=AgentAnalysis,
            task_name="technical_revision",
        )
        final_review = require_structured_output(
            revision_tasks,
            index=3,
            expected_type=ReviewResult,
            task_name="second_review",
        )
        final_decision = require_structured_output(
            revision_tasks,
            index=4,
            expected_type=FinalDecision,
            task_name="revision_final_synthesis",
        )

        revised_analyses = [
            product_analysis,
            ux_analysis,
            technical_analysis,
        ]

        self._validate_specialist_roles(
            revised_analyses
        )
        self._validate_decision(
            final_review,
            final_decision,
        )

        return StartupSimulationResult(
            brief=brief,
            plan=plan,
            analyses=revised_analyses,
            review=final_review,
            review_history=[
                initial_review,
                final_review,
            ],
            revision_count=1,
            recommendation=(
                final_decision.recommendation
            ),
            executive_summary=(
                final_decision.executive_summary
            ),
        )

    @staticmethod
    def _create_initial_inputs(
        brief: StartupBrief,
        memory_context: str = "",
    ) -> dict[str, str]:
        constraints = (
            "\n".join(
                f"- {constraint}"
                for constraint in brief.constraints
            )
            if brief.constraints
            else "No explicit constraints supplied."
        )

        normalized_memory_context = (
            memory_context.strip()
            or (
                "No relevant previous simulations "
                "were found."
            )
        )

        return {
            "startup_name": brief.name,
            "startup_description": (
                brief.description
            ),
            "target_customer": (
                brief.target_customer
            ),
            "problem_statement": (
                brief.problem_statement
            ),
            "constraints": constraints,
            "memory_context": (
                normalized_memory_context
            ),
        }
    @staticmethod
    def _get_task_outputs(
        crew_output: Any,
    ) -> list[Any]:
        task_outputs = getattr(
            crew_output,
            "tasks_output",
            None,
        )

        if not isinstance(task_outputs, list):
            raise RuntimeError(
                "CrewAI did not return task outputs"
            )

        return task_outputs

    @classmethod
    def _validate_specialist_roles(
        cls,
        analyses: list[AgentAnalysis],
    ) -> None:
        expected_roles = [
            AgentRole.PRODUCT_STRATEGIST,
            AgentRole.UX_DESIGNER,
            AgentRole.TECHNICAL_LEAD,
        ]

        for analysis, expected_role in zip(
            analyses,
            expected_roles,
            strict=True,
        ):
            cls._validate_specialist_role(
                analysis,
                expected_role,
            )

    @staticmethod
    def _validate_decision(
        review: ReviewResult,
        decision: FinalDecision,
    ) -> None:
        if (
            not review.approved
            and decision.recommendation == "pursue"
        ):
            raise RuntimeError(
                "A rejected review cannot produce "
                "a pursue recommendation"
            )

    @staticmethod
    def _validate_specialist_role(
        analysis: AgentAnalysis,
        expected_role: AgentRole,
    ) -> None:
        if analysis.agent_role != expected_role:
            raise RuntimeError(
                "Specialist returned the wrong role: "
                f"expected {expected_role.value}, "
                f"received {analysis.agent_role.value}"
            )

def create_crewai_startup_team(
    verbose: bool = False,
) -> CrewAIStartupTeam:
    llm = create_llm_from_environment()

    agents = create_startup_agents(
        llm=llm,
        verbose=verbose,
    )

    assessment_tasks = create_startup_tasks(
        agents
    )
    decision_tasks = create_decision_tasks(
        agents
    )
    revision_tasks = create_revision_tasks(
        agents
    )

    assessment_crew = Crew(
        agents=agents.as_list(),
        tasks=assessment_tasks.assessment_list(),
        process=Process.sequential,
        verbose=verbose,
        memory=False,
        planning=False,
    )

    decision_crew = Crew(
        agents=agents.as_list(),
        tasks=decision_tasks.as_list(),
        process=Process.sequential,
        verbose=verbose,
        memory=False,
        planning=False,
    )

    revision_crew = Crew(
        agents=agents.as_list(),
        tasks=revision_tasks.as_list(),
        process=Process.sequential,
        verbose=verbose,
        memory=False,
        planning=False,
    )

    return CrewAIStartupTeam(
        crew=assessment_crew,
        decision_crew=decision_crew,
        revision_crew=revision_crew,
        reflection_policy=ReflectionPolicy(
            max_revisions=1,
        ),
    )