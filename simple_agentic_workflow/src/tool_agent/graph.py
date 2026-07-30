"""Single-agent LangGraph workflow with a small tool loop."""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import Settings, get_settings
from .tools import TOOLS

SYSTEM_PROMPT = """You are a helpful assistant with access to a few precise tools.

Use a tool whenever the user asks you to calculate a sum or count words or characters.
Do not do those operations yourself when a tool can do them. You can answer directly
for general questions. After a tool returns, explain the result briefly and clearly.
"""


def build_graph(settings: Settings | None = None):
    """Build and compile the agent -> tools -> agent workflow."""

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
    ).bind_tools(TOOLS)

    def call_model(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()
