"""LLM-backed tool-selection evaluations.

Run with:

    deepeval test run evals/test_tool_selection.py
"""

from dataclasses import dataclass

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric, ToolPermissionMetric, ToolUseMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall, Turn

from tool_agent.config import get_settings
from tool_agent.observability import record_score
from tool_agent.runner import run_agent

pytestmark = pytest.mark.agent_eval


@dataclass(frozen=True)
class ToolSelectionCase:
    name: str
    user_input: str
    expected_tool_names: tuple[str, ...]
    allowed_tool_names: tuple[str, ...]


CASES = [
    ToolSelectionCase(
        name="addition",
        user_input="What is 12 plus 8?",
        expected_tool_names=("add_numbers",),
        allowed_tool_names=("add_numbers",),
    ),
    ToolSelectionCase(
        name="word_count",
        user_input="How many words are in: LangGraph makes tool loops easy?",
        expected_tool_names=("count_words",),
        allowed_tool_names=("count_words",),
    ),
    ToolSelectionCase(
        name="character_count",
        user_input="Count the characters excluding spaces in: hello world",
        expected_tool_names=("count_characters",),
        allowed_tool_names=("count_characters",),
    ),
    ToolSelectionCase(
        name="no_tool_needed",
        user_input="What is LangGraph?",
        expected_tool_names=(),
        allowed_tool_names=(),
    ),
]

AVAILABLE_TOOLS = [
    ToolCall(name="add_numbers", description="Add two numbers."),
    ToolCall(name="count_words", description="Count whitespace-separated words."),
    ToolCall(name="count_characters", description="Count characters with or without spaces."),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_agent_uses_the_expected_tools(case: ToolSelectionCase) -> None:
    result = run_agent(
        case.user_input,
        trace_name=f"deepeval-tool-selection-{case.name}",
        tags=("deepeval", "component-eval", "tool-selection"),
        metadata={
            "eval_case": case.name,
            "expected_tools": list(case.expected_tool_names),
        },
    )
    actual_tools = [call.as_deepeval_tool_call() for call in result.tools_called]
    expected_tools = [ToolCall(name=name) for name in case.expected_tool_names]

    test_case = LLMTestCase(
        input=case.user_input,
        actual_output=result.final_output,
        tools_called=actual_tools,
        expected_tools=expected_tools,
    )

    settings = get_settings()
    metrics = [
        ToolCorrectnessMetric(
            threshold=1.0,
            async_mode=False,
            should_exact_match=True,
        ),
        ToolPermissionMetric(
            allowed_tools=list(case.allowed_tool_names),
            threshold=1.0,
        ),
    ]
    try:
        assert_test(test_case=test_case, metrics=metrics)
    finally:
        for metric_name, metric in (
            ("tool_correctness", metrics[0]),
            ("tool_permission", metrics[1]),
        ):
            if metric.score is not None:
                record_score(
                    trace_id=result.trace_id,
                    name=metric_name,
                    value=metric.score,
                    comment=metric.reason,
                )

    if not case.expected_tool_names:
        assert result.tools_called == [], "This case should not call any tools."
        return

    tool_use_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content=case.user_input),
            Turn(role="assistant", content=result.final_output, tools_called=actual_tools),
        ]
    )
    tool_use_metric = ToolUseMetric(
        available_tools=AVAILABLE_TOOLS,
        model=settings.openai_model,
        threshold=0.7,
        async_mode=False,
    )
    tool_use_metric.measure(tool_use_case)
    if tool_use_metric.score is not None:
        record_score(
            trace_id=result.trace_id,
            name="tool_use",
            value=tool_use_metric.score,
            comment=tool_use_metric.reason,
        )
    assert tool_use_metric.is_successful(), (
        f"ToolUseMetric failed with score={tool_use_metric.score}: "
        f"{tool_use_metric.reason}"
    )
