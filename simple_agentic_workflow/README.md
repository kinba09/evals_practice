# Simple Agentic Workflow

A small single-agent LangGraph workflow for practicing component-level evals with
DeepEval. The agent uses the same model configuration as `simple_langgraph_agent`,
but now it can decide when to call deterministic tools.

## Tools

- `add_numbers(first, second)` adds two numbers.
- `count_words(text)` counts whitespace-separated words.
- `count_characters(text, include_spaces)` counts characters, optionally excluding whitespace.

The graph has three parts: an agent node, a tool node, and a loop back to the agent
so it can turn tool results into a concise response.

## Setup

```bash
cd simple_agentic_workflow
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# Set OPENAI_API_KEY in .env
```

The example uses the same ASU gateway and `qwen3-30b-a3b-instruct-2507` model as the
existing simple agent. The gateway URL must include `/v1`.

## Run

```bash
python -m tool_agent.cli "What is 17 plus 25?"
python -m tool_agent.cli "How many words are in: LangGraph makes tool loops easy?"
python -m tool_agent.cli "Count the characters excluding spaces in: hello world"
```

## Use as a library

```python
from langchain_core.messages import HumanMessage

from tool_agent.config import get_settings
from tool_agent.graph import build_graph

graph = build_graph(get_settings())
result = graph.invoke({"messages": [HumanMessage(content="Add 8.5 and 2.5.")]})
print(result["messages"][-1].content)
```

The tool functions are intentionally standalone and deterministic, which makes them
easy to test directly before adding DeepEval component-level tests.
