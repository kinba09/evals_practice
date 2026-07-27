"""Build the small, switchable metric set for the agent."""

from __future__ import annotations

import os

from deepeval.metrics import AnswerRelevancyMetric, GEval, PromptAlignmentMetric
from deepeval.test_case import SingleTurnParams

from ..models import GatewayEvaluationModel
from .deterministic_format import DeterministicFormatMetric


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_metrics(
    include_deterministic_format: bool | None = None,
    include_correctness: bool = True,
):
    """Return the enabled DeepEval metrics for one evaluation run."""

    if include_deterministic_format is None:
        include_deterministic_format = _env_bool("ENABLE_DETERMINISTIC_FORMAT", True)

    judge = GatewayEvaluationModel()
    metrics = []
    if include_deterministic_format:
        metrics.append(DeterministicFormatMetric())

    metrics.extend(
        [
            AnswerRelevancyMetric(
                threshold=0.8,
                model=judge,
                async_mode=False,
            ),
            GEval(
                name="Definition and Purpose",
                evaluation_steps=[
                    "Check that the response gives a clear and accurate definition of the topic.",
                    "Check that the response explains the topic's primary purpose or significance.",
                    "Check that the explanation uses simple language appropriate for a "
                    "general audience.",
                    "Penalize unsupported claims, irrelevant details, and failure to "
                    "address the topic.",
                ],
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                ],
                threshold=0.8,
                model=judge,
                async_mode=False,
            ),
            PromptAlignmentMetric(
                prompt_instructions=[
                    "Respond in 2–4 sentences.",
                    "Use one paragraph.",
                    "Stay under 100 words.",
                    "Put a clear definition first.",
                    "Explain the topic's primary purpose or significance.",
                    "Do not use lists.",
                    "Do not use headings.",
                    "Do not include unnecessary details.",
                ],
                threshold=0.8,
                model=judge,
                async_mode=False,
            ),
        ]
    )

    if include_correctness:
        metrics.append(
            GEval(
                name="Correctness",
                evaluation_steps=[
                    "Compare the actual output with the expected output.",
                    "Reward answers that preserve the expected output's factual meaning, "
                    "even when the wording differs.",
                    "Penalize factual errors, contradictions, missing core meaning, and "
                    "unsupported claims.",
                ],
                evaluation_params=[
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                threshold=0.8,
                model=judge,
                async_mode=False,
            )
        )

    return metrics
