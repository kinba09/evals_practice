"""Deterministic format checks implemented as a DeepEval custom metric."""

from __future__ import annotations

import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams


class DeterministicFormatMetric(BaseMetric):
    """Check the output shape required by the agent system prompt."""

    _required_params = [SingleTurnParams.ACTUAL_OUTPUT]

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True
        self.verbose_mode = False
        self.score = None
        self.reason = None
        self.success = None

    @property
    def __name__(self) -> str:
        return "Deterministic Format"

    @staticmethod
    def _sentence_count(output: str) -> int:
        return len(re.findall(r"[.!?](?=\s|$)", output))

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        output = (test_case.actual_output or "").strip()
        paragraphs = [part for part in re.split(r"\n\s*\n", output) if part.strip()]

        checks = {
            "non-empty": bool(output),
            "one paragraph": len(paragraphs) == 1,
            "2-4 sentences": 2 <= self._sentence_count(output) <= 4,
            "under 100 words": len(re.findall(r"\b[\w'-]+\b", output)) < 100,
            "no headings or lists": not re.search(
                r"(?m)^\s*(?:#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)", output
            ),
        }

        passed = [name for name, result in checks.items() if result]
        failed = [name for name, result in checks.items() if not result]
        self.score = len(passed) / len(checks)
        self.success = self.score >= self.threshold
        self.reason = (
            "All format checks passed."
            if not failed
            else f"Passed: {', '.join(passed) or 'none'}. Failed: {', '.join(failed)}."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)
