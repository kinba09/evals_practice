"""Tests for the structured agent result wrapper."""

from langchain_core.messages import AIMessage, ToolMessage

from tool_agent import runner


class FakeGraph:
    def invoke(self, _state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "add_numbers",
                            "args": {"first": 2, "second": 3},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="5", tool_call_id="call-1", name="add_numbers"),
                AIMessage(content="The answer is 5."),
            ]
        }


def test_run_agent_collects_tool_calls_and_steps(monkeypatch) -> None:
    monkeypatch.setattr(runner, "build_graph", lambda _settings: FakeGraph())

    result = runner.run_agent("What is 2 plus 3?")

    assert result.final_output == "The answer is 5."
    assert result.tools_called[0].name == "add_numbers"
    assert result.tools_called[0].input_parameters == {"first": 2, "second": 3}
    assert result.tools_called[0].output == "5"
    assert result.errors == []
    assert result.agent_steps == 2
    assert result.tool_steps == 1
