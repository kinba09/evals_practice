"""Build the small, switchable metric set for the agent."""

from __future__ import annotations

import os

from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import SingleTurnParams

from ..models import GatewayEvaluationModel
from .deterministic_format import DeterministicFormatMetric


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_metrics(include_deterministic_format: bool | None = None):
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
        ]
    )
    return metrics
