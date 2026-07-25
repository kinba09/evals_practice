# Simple LangGraph Agent

A small single-agent baseline for practicing evals.

The agent has one graph node: it receives a user message, adds a system instruction, and calls an OpenAI chat model through LangChain. There are no tools, memory, or extra application layers yet.

## Requirements

- Python 3.11+
- An API key from your model gateway

## Setup

```bash
cd simple_langgraph_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# Edit .env and set your university-issued OPENAI_API_KEY
```

The included example points to the ASU gateway and uses `gpt-oss-120b`. The gateway
URL must include `/v1`.

## Run

```bash
python -m simple_agent.cli "Explain what an API is in one sentence."
```

## Use as a library

```python
from langchain_core.messages import HumanMessage

from simple_agent.config import get_settings
from simple_agent.graph import build_graph

graph = build_graph(get_settings())
result = graph.invoke({"messages": [HumanMessage(content="What is a database index?")]})
print(result["messages"][-1].content)
```

## Project layout

```text
src/simple_agent/
  config.py       Environment-backed settings
  graph.py        The one-node LangGraph workflow
  cli.py          Minimal CLI
```
