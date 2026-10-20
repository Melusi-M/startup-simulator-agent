import os

from crewai import LLM
from dotenv import load_dotenv


load_dotenv()


def create_llm_from_environment(
    model: str | None = None,
) -> LLM:
    selected_model = (
        model
        or os.getenv("STARTUP_SIMULATOR_MODEL")
        or ""
    ).strip()

    if not selected_model:
        raise RuntimeError(
            "STARTUP_SIMULATOR_MODEL is not configured"
        )

    if not selected_model.startswith("anthropic/"):
        raise ValueError(
            "STARTUP_SIMULATOR_MODEL must use the "
            "anthropic/ provider prefix"
        )

    api_key = (
        os.getenv("ANTHROPIC_API_KEY")
        or ""
    ).strip()

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured"
        )

    max_tokens_text = (
        os.getenv(
            "STARTUP_SIMULATOR_MAX_TOKENS",
            "4000",
        )
        or ""
    ).strip()

    try:
        max_tokens = int(max_tokens_text)
    except ValueError as error:
        raise RuntimeError(
            "STARTUP_SIMULATOR_MAX_TOKENS "
            "must be an integer"
        ) from error

    if max_tokens <= 0:
        raise RuntimeError(
            "STARTUP_SIMULATOR_MAX_TOKENS "
            "must be greater than zero"
        )

    return LLM(
        model=selected_model,
        api_key=api_key,
        temperature=0.2,
        max_tokens=max_tokens,
    )