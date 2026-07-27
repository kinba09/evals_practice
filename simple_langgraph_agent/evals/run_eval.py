"""Run the simple agent against synthetic and manually reviewed goldens."""

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

DATA_DIR = Path(__file__).parent / "data"
DATASET_PATH = DATA_DIR / "synthetic_goldens.json"
REVIEWED_DATASET_PATH = DATA_DIR / "manual_goldens.json"


def _load_test_cases(dataset_path: Path) -> list[LLMTestCase]:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(dataset_path))
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
    return test_cases


def _run_dataset(
    dataset_path: Path,
    identifier: str,
    include_correctness: bool,
) -> None:
    if not dataset_path.exists():
        message = f"Dataset not found at {dataset_path}."
        if dataset_path == DATASET_PATH:
            message += " Run python -m evals.generate_dataset first."
        raise FileNotFoundError(message)

    evaluate(
        test_cases=_load_test_cases(dataset_path),
        metrics=build_metrics(include_correctness=include_correctness),
        identifier=identifier,
        async_config=AsyncConfig(
            run_async=False,
            throttle_value=float(os.getenv("DEEPEVAL_THROTTLE_SECONDS", "1")),
            max_concurrent=int(os.getenv("DEEPEVAL_MAX_CONCURRENT", "1")),
        ),
    )


def run_eval() -> None:
    _run_dataset(
        DATASET_PATH,
        identifier="simple-langgraph-agent-synthetic",
        include_correctness=False,
    )
    _run_dataset(
        REVIEWED_DATASET_PATH,
        identifier="simple-langgraph-agent-reviewed",
        include_correctness=True,
    )


if __name__ == "__main__":
    run_eval()
