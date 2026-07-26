"""The single-node LangGraph workflow."""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from .config import Settings, get_settings

SYSTEM_PROMPT = """You are a concise technical explainer.

The user will ask "What is <topic>?"

Respond with exactly one short paragraph (2–4 sentences, under 100 words). Explain the
topic in simple, accurate language, starting with a clear definition followed by its
primary purpose or significance. Avoid lists, headings, examples, history, opinions,
or unnecessary details. If the topic is ambiguous, ask a brief clarification question
instead of guessing."""


def build_graph(settings: Settings | None = None):
    """Build and compile the one-node agent graph."""

    runtime_settings = settings or get_settings()
    if runtime_settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required")

    model = ChatOpenAI(
        api_key=runtime_settings.openai_api_key,
        base_url=runtime_settings.openai_base_url,
        model=runtime_settings.openai_model,
        reasoning_effort=runtime_settings.openai_reasoning_effort,
        temperature=runtime_settings.openai_temperature,
        timeout=runtime_settings.openai_timeout_seconds,
        max_retries=runtime_settings.openai_max_retries,
    )

    def call_model(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = model.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile()
