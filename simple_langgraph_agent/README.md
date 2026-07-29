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

The included example points to the ASU gateway and uses
`qwen3-30b-a3b-instruct-2507`. The gateway URL must include `/v1`.

## Run

```bash
python -m simple_agent.cli "Explain what an API is in one sentence."
```

## Run the DeepEval practice eval

Generate synthetic single-turn goldens using the configured gateway:

```bash
python -m evals.generate_dataset --num-goldens 10
```

Run the agent against the synthetic and manually reviewed goldens:

```bash
python -m evals.run_eval
```

The synthetic run checks format, relevance, definition and purpose, and prompt
alignment. The manually reviewed API case adds a correctness check against its
expected output.

Choose the format checker in `.env`:

```env
FORMAT_CHECKER=deterministic
```

Available values are `deterministic`, `llm`, and `both`. The deterministic checker
uses direct rules. The LLM checker uses a `GEval` judge. All metrics print scores and
reasons to help explain failures.

The evals run sequentially by default and wait one second between gateway requests.
Adjust `DEEPEVAL_MAX_CONCURRENT`, `DEEPEVAL_THROTTLE_SECONDS`, and
`DEEPEVAL_REQUEST_DELAY_SECONDS` in `.env` if your gateway has different limits.

Synthetic goldens are saved under `evals/data/` and are distributed across these
categories: basic concepts, difficult technical concepts, ambiguous questions, very
short inputs, long inputs, topics with multiple meanings, clarification questions,
and prompt-injection attempts. Each golden stores its category in
`additional_metadata`. Review them before treating them as a regression dataset.
The manually reviewed correctness case is stored in `evals/data/manual_goldens.json`.

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
evals/
  generate_dataset.py   Synthetic DeepEval golden generation
  run_eval.py           End-to-end DeepEval run
  metrics/              DeepEval metrics
```
