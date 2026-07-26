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

Run the agent against those goldens with three metrics:

```bash
python -m evals.run_eval
```

The deterministic format metric is enabled by default. To disable it for a later
LLM-judge format metric, set this in `.env`:

```env
ENABLE_DETERMINISTIC_FORMAT=false
```

The evals run sequentially by default and wait one second between gateway requests.
Adjust `DEEPEVAL_MAX_CONCURRENT`, `DEEPEVAL_THROTTLE_SECONDS`, and
`DEEPEVAL_REQUEST_DELAY_SECONDS` in `.env` if your gateway has different limits.

Synthetic goldens are saved under `evals/data/`. Review them before treating them as
a regression dataset. Each saved golden contains only `input`, `actual_output`, and
`expected_output`; the latter two are `null` because the agent output is created during
the evaluation run.

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
