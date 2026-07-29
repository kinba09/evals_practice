"""Build the small, switchable metric set for the agent."""

from __future__ import annotations

import os

from deepeval.metrics import AnswerRelevancyMetric, GEval, PromptAlignmentMetric
from deepeval.test_case import SingleTurnParams

from ..models import GatewayEvaluationModel
from .deterministic_format import DeterministicFormatMetric


def _format_checker(value: str | None) -> str:
    checker = (value or os.getenv("FORMAT_CHECKER", "deterministic")).strip().lower()
    if checker not in {"deterministic", "llm", "both"}:
        raise ValueError("FORMAT_CHECKER must be deterministic, llm, or both")
    return checker


def _llm_format_metric(judge: GatewayEvaluationModel) -> GEval:
    return GEval(
        name="LLM Format",
        evaluation_steps=[
            "Check that the response contains exactly one paragraph.",
            "Check that the response has 2 to 4 sentences.",
            "Check that the response is under 100 words.",
            "Check that the response avoids lists and headings.",
        ],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=judge,
        async_mode=False,
        verbose_mode=True,
    )


def build_metrics(
    format_checker: str | None = None,
    include_correctness: bool = True,
):
    """Return the enabled DeepEval metrics for one evaluation run."""

    checker = _format_checker(format_checker)
    judge = GatewayEvaluationModel()
    metrics = []
    if checker in {"deterministic", "both"}:
        metrics.append(DeterministicFormatMetric())
    if checker in {"llm", "both"}:
        metrics.append(_llm_format_metric(judge))

    metrics.extend(
        [
            AnswerRelevancyMetric(
                threshold=0.8,
                model=judge,
                include_reason=True,
                async_mode=False,
                verbose_mode=True,
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
                verbose_mode=True,
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
                include_reason=True,
                async_mode=False,
                verbose_mode=True,
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
                verbose_mode=True,
            )
        )

    return metrics
