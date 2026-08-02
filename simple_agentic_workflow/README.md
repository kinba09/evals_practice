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

From the repository root, enter this project:

```bash
cd /Users/abnikahilasamy/Personal_coding/evals_practice/simple_agentic_workflow
```

Create the virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If virtual-environment creation was interrupted, remove the incomplete environment
first and recreate it:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Create the environment file:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. The example uses the same ASU gateway and
`qwen3-30b-a3b-instruct-2507` model as the existing simple agent:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://openai.rc.asu.edu/v1
OPENAI_MODEL=qwen3-30b-a3b-instruct-2507
```

The gateway URL must include `/v1`.

## Add local Langfuse tracing

This project keeps DeepEval as the scoring layer and uses Langfuse for traces,
tool-call visibility, and score history. Langfuse is optional: without the
`LANGFUSE_*` variables, the agent runs normally and does not send telemetry.

Start the local self-hosted Langfuse stack:

```bash
docker compose -f docker-compose.langfuse.yml up -d
```

Open `http://localhost:3000`. The compose file initializes a local learning
project with these credentials:

```text
email:    dev@example.com
password: change-me-local
public:   pk-lf-local
secret:   sk-lf-local
```

These credentials are for local learning only. Add these values to `.env` after
starting Langfuse:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-local
LANGFUSE_SECRET_KEY=sk-lf-local
LANGFUSE_BASE_URL=http://localhost:3000
```

Then run the agent:

```bash
python -m tool_agent.cli "What is 17 plus 25?"
```

The Langfuse callback traces the LangGraph run, model calls, and ToolNode tool
calls. Find the trace in the dashboard under `simple-tool-agent`.

Stop the stack when finished:

```bash
docker compose -f docker-compose.langfuse.yml down
```

Use `down -v` only when you intentionally want to delete the local Langfuse
database and trace history.

## Run

```bash
python -m tool_agent.cli "What is 17 plus 25?"
python -m tool_agent.cli "How many words are in: LangGraph makes tool loops easy?"
python -m tool_agent.cli "Count the characters excluding spaces in: hello world"
```

## Run the deterministic tests

These tests do not call the model or DeepEval judges. They test the tools and the
structured `run_agent()` result wrapper:

```bash
python -m pytest tests/ -q
```

Expected result:

```text
12 passed
```

If pytest reports a socket or `pytest-rerunfailures` plugin error on macOS, run with
third-party pytest plugin discovery disabled:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

## Use as a library

```python
from tool_agent.config import get_settings
from tool_agent.runner import run_agent

result = run_agent("Add 8.5 and 2.5.", settings=get_settings())
print(result.final_output)
print(result.tools_called)
```

The tool functions are intentionally standalone and deterministic, which makes them
easy to test directly before running the LLM-backed evaluations.

## First evaluation exercises

Run the tool-selection evaluations against the configured model and DeepEval metrics:

```bash
deepeval test run evals/test_tool_selection.py -v
```

The tool-selection cases check expected tools with `ToolCorrectnessMetric`, allowed
tools with `ToolPermissionMetric`, and tool choice plus arguments with `ToolUseMetric`.
The `run_agent()` wrapper returns the final answer, tool calls, tool outputs, errors,
and step counts so later component-level tracing can use the same result shape.

`ToolCorrectnessMetric` and `ToolPermissionMetric` run through `assert_test()`. Because
`ToolUseMetric` is a conversational metric in the installed DeepEval version, it runs
separately with a `ConversationalTestCase` for tool-required cases. The no-tool case
uses exact tool-correctness and permission checks because this DeepEval version's
`ToolUseMetric` scores a no-tool interaction as zero even when no tool was called.

When Langfuse is enabled, each evaluation case creates a trace tagged
`deepeval`, `component-eval`, and `tool-selection`. The evaluation records these
DeepEval scores on the matching trace:

- `tool_correctness`
- `tool_permission`
- `tool_use` for tool-required cases

This lets you inspect the full agent/tool trace and its scores together in
Langfuse. Run the same evaluation after each agent update and use the
`eval_case` metadata and `LANGFUSE_TRACING_RELEASE` value to compare runs.

## Save evaluation results locally

To save each evaluation run as a timestamped result file:

```bash
export DEEPEVAL_RESULTS_FOLDER=./evals/results

deepeval test run evals/test_tool_selection.py \
  -v \
  -id "local-$(date +%Y%m%d-%H%M%S)"
```

Results are saved under `evals/results/`. DeepEval also keeps its local cache under
`.deepeval/`; these generated files should not be committed.

For repeated evaluations, the cache can be used to avoid recalculating unchanged
metric results:

```bash
deepeval test run evals/test_tool_selection.py -c
```

## Complete local workflow

After changing the agent, run:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
deepeval test run evals/test_tool_selection.py -v
```

With local Langfuse enabled, the complete learning loop is:

```bash
docker compose -f docker-compose.langfuse.yml up -d
source .venv/bin/activate
python -m pytest tests/ -q
deepeval test run evals/test_tool_selection.py -v
```

Then open `http://localhost:3000` and compare the new traces and scores with
earlier runs. DeepEval still decides whether the evaluation passes; Langfuse
stores and displays the execution details and score history.
