"""A small wrapper that turns a graph run into an evaluation-friendly result."""

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from .config import Settings
from .graph import build_graph
from .observability import (
    create_langfuse_handler,
    flush_langfuse,
    trace_agent_run,
)


@dataclass
class ToolCallRecord:
    """One tool call made during an agent run."""

    name: str
    input_parameters: dict[str, Any]
    output: Any = None
    tool_call_id: str | None = None

    def as_deepeval_tool_call(self) -> Any:
        """Convert this record to DeepEval's ToolCall type when needed."""

        from deepeval.test_case import ToolCall

        return ToolCall(
            name=self.name,
            input_parameters=self.input_parameters,
            output=self.output,
        )


@dataclass
class AgentResult:
    """The useful pieces of one agent execution."""

    input: str
    final_output: str
    tools_called: list[ToolCallRecord]
    errors: list[str]
    agent_steps: int
    tool_steps: int
    trace_id: str | None = None


def _content_to_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def run_agent(
    user_input: str,
    settings: Settings | None = None,
    *,
    trace_name: str = "simple-tool-agent",
    tags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> AgentResult:
    """Run the workflow and return output plus simple execution details.

    Langfuse tracing is enabled only when its environment variables are set.
    The callback observes the LangGraph agent, model calls, and ToolNode calls.
    """

    graph = build_graph(settings)
    handler = create_langfuse_handler()
    invoke_config: dict[str, Any] = {
        "run_name": trace_name,
        "metadata": metadata or {},
    }
    if handler is not None:
        invoke_config["callbacks"] = [handler]

    with trace_agent_run(
        handler=handler,
        trace_name=trace_name,
        tags=tags,
        metadata=metadata,
    ) as (_, trace_id):
        graph_input = {"messages": [HumanMessage(content=user_input)]}
        if handler is None:
            result = graph.invoke(graph_input)
        else:
            result = graph.invoke(graph_input, config=invoke_config)
    flush_langfuse()
    messages = result["messages"]

    tool_calls: list[ToolCallRecord] = []
    calls_by_id: dict[str, ToolCallRecord] = {}
    errors: list[str] = []

    for message in messages:
        for call in getattr(message, "tool_calls", []):
            record = ToolCallRecord(
                name=call["name"],
                input_parameters=dict(call.get("args") or {}),
                tool_call_id=call.get("id"),
            )
            tool_calls.append(record)
            if record.tool_call_id:
                calls_by_id[record.tool_call_id] = record

        if getattr(message, "type", None) == "tool":
            record = calls_by_id.get(getattr(message, "tool_call_id", None))
            if record is not None:
                record.output = message.content
            if getattr(message, "status", None) == "error":
                errors.append(_content_to_text(message.content))

    return AgentResult(
        input=user_input,
        final_output=_content_to_text(messages[-1].content),
        tools_called=tool_calls,
        errors=errors,
        agent_steps=sum(getattr(message, "type", None) == "ai" for message in messages),
        tool_steps=sum(getattr(message, "type", None) == "tool" for message in messages),
        trace_id=trace_id,
    )
