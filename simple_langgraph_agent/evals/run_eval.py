"""Run the simple agent against synthetic DeepEval goldens."""

from __future__ import annotations

import os
from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from langchain_core.messages import HumanMessage

from simple_agent.config import get_settings
from simple_agent.graph import build_graph

from .metrics import build_metrics

DATASET_PATH = Path(__file__).parent / "data" / "synthetic_goldens.json"


def run_eval() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run "
            "python -m evals.generate_dataset first."
        )

    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(DATASET_PATH))
    graph = build_graph(get_settings())
    test_cases = []

    for golden in dataset.goldens:
        result = graph.invoke({"messages": [HumanMessage(content=golden.input)]})
        actual_output = result["messages"][-1].content
        test_cases.append(
            LLMTestCase(
                name=golden.name,
                input=golden.input,
                actual_output=actual_output,
                expected_output=golden.expected_output,
            )
        )

    evaluate(
        test_cases=test_cases,
        metrics=build_metrics(),
        identifier="simple-langgraph-agent",
        async_config=AsyncConfig(
            run_async=False,
            throttle_value=float(os.getenv("DEEPEVAL_THROTTLE_SECONDS", "1")),
            max_concurrent=int(os.getenv("DEEPEVAL_MAX_CONCURRENT", "1")),
        ),
    )


if __name__ == "__main__":
    run_eval()
