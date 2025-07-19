from smolagents import CodeAgent
from typing import Literal
from .tools.fetch import fetch_crisiswatch_data, init_db
from .tools.search import search_reports_rag
from .tools.summarize import summarize_reports


def create_agent(model: Literal["openai", "smollm"] = "openai") -> CodeAgent:
    """
    Creates the CrisisWatchAgent with the appropriate model backend.

    Parameters
    ----------
    model : {'openai', 'smollm'}
        The model backend to use for the agent.

    Returns
    -------
    CodeAgent
        The configured agent.
    """
    if model == "smollm":
        from smolagents import TransformersModel
        backend = TransformersModel("HuggingFaceTB/SmolLM-360M-Instruct")
        return CodeAgent(
            name="CrisisWatchAgent",
            instructions=(
                "You are a geopolitical analyst agent. "
                "Use `fetch_crisiswatch_data` to load data, `search_reports_rag` to retrieve relevant info, "
                "and `summarize_reports` to give users a regional trend summary."
            ),
            model=backend,
            tools=[fetch_crisiswatch_data, search_reports_rag, summarize_reports],
        )
    else:
        backend = None

        return CodeAgent(
            name="CrisisWatchAgent",
            instructions=(
                "You are a geopolitical analyst agent. "
                "Use `fetch_crisiswatch_data` to load data, `search_reports_rag` to retrieve relevant info, "
                "and `summarize_reports` to give users a regional trend summary."
            ),
            tools=[fetch_crisiswatch_data, search_reports_rag, summarize_reports],
            backend=backend,
        )
