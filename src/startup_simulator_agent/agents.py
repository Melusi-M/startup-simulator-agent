from dataclasses import dataclass

from startup_simulator_agent.tools import (
    AssumptionRegisterTool,
    RiskScoreTool,
)
from crewai import Agent, LLM


@dataclass(frozen=True)
class StartupAgents:
    ceo: Agent
    product_strategist: Agent
    ux_designer: Agent
    technical_lead: Agent
    critical_reviewer: Agent

    def as_list(self) -> list[Agent]:
        return [
            self.ceo,
            self.product_strategist,
            self.ux_designer,
            self.technical_lead,
            self.critical_reviewer,
        ]


def create_startup_agents(
    llm: LLM,
    verbose: bool = False,
) -> StartupAgents:
    common_options = {
        "llm": llm,
        "verbose": verbose,
        "allow_delegation": False,
        "max_iter": 5,
        "max_retry_limit": 2,
    }

    ceo = Agent(
        role="Startup CEO and Coordinator",
        goal=(
            "Create a focused evaluation plan, reconcile "
            "specialist disagreements, and produce a clear "
            "startup recommendation without inventing evidence."
        ),
        backstory=(
            "You are an experienced startup operator. You "
            "separate facts from assumptions, keep teams "
            "focused on the target customer, and make explicit "
            "decisions under uncertainty."
        ),
        tools=create_assumption_tools(),
        **common_options,
    )

    product_strategist = Agent(
        role="Product and Market Strategist",
        goal=(
            "Evaluate the customer problem, ideal customer "
            "profile, value proposition, positioning, market "
            "assumptions, and validation priorities."
        ),
        backstory=(
            "You specialise in customer discovery and product "
            "strategy. You challenge vague customer segments "
            "and unsupported willingness-to-pay assumptions."
        ),
        tools=create_analysis_tools(),
        **common_options,
    )

    ux_designer = Agent(
        role="Senior UX and Product Designer",
        goal=(
            "Design the smallest coherent user journey that "
            "solves the primary customer problem and can be "
            "tested with representative users."
        ),
        backstory=(
            "You design low-friction products around observable "
            "user outcomes. You resist feature overload and "
            "identify trust, usability, and adoption risks."
        ),
        tools=create_analysis_tools(),
        **common_options,
    )

    technical_lead = Agent(
        role="Principal Software and AI Architect",
        goal=(
            "Assess technical feasibility and propose a secure, "
            "testable MVP architecture with explicit trade-offs, "
            "risks, dependencies, and delivery phases."
        ),
        backstory=(
            "You build reliable software and AI systems. You "
            "avoid premature distribution, hidden operational "
            "costs, and architecture that exceeds the team's "
            "current needs."
        ),
        tools=create_analysis_tools(),
        **common_options,
    )

    critical_reviewer = Agent(
        role="Critical Startup Investment Reviewer",
        goal=(
            "Find contradictions, unsupported claims, missing "
            "risks, weak recommendations, and disagreements "
            "between specialist analyses."
        ),
        backstory=(
            "You are an evidence-driven reviewer. You do not "
            "approve work because it sounds polished. You "
            "require clear reasoning, realistic assumptions, "
            "and actionable revision requests."
        ),
        tools=create_analysis_tools(),
        **common_options,
    )

    return StartupAgents(
        ceo=ceo,
        product_strategist=product_strategist,
        ux_designer=ux_designer,
        technical_lead=technical_lead,
        critical_reviewer=critical_reviewer,
    )

def create_assumption_tools() -> list:
    return [
        AssumptionRegisterTool(),
    ]


def create_analysis_tools() -> list:
    return [
        AssumptionRegisterTool(),
        RiskScoreTool(),
    ]