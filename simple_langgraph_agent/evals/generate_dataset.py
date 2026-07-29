"""Generate synthetic single-turn DeepEval goldens for the agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, StylingConfig

from simple_agent.config import get_settings

from .models import GatewayEvaluationModel

DATA_DIR = Path(__file__).parent / "data"

DATASET_CATEGORIES = [
    (
        "Basic concepts",
        "common foundational topics such as APIs, databases, caching, or encryption",
    ),
    (
        "Difficult technical concepts",
        "advanced topics such as distributed systems, compilers, or quantum computing",
    ),
    (
        "Ambiguous questions",
        "topics with more than one reasonable interpretation, such as Python or Java",
    ),
    (
        "Very short inputs",
        "short prompts such as 'API?' or 'AI?'",
    ),
    (
        "Long inputs",
        "long questions that include extra context before asking about a concept",
    ),
    (
        "Topics with multiple meanings",
        "terms with different meanings in computing and other fields, such as token",
    ),
    (
        "Questions that should trigger clarification",
        "underspecified questions where the assistant should ask what the user means",
    ),
    (
        "Prompt-injection attempts",
        "questions that try to override the system prompt or force a list",
    ),
]


def _styling_config(category: str, focus: str) -> StylingConfig:
    return StylingConfig(
        scenario=(
            "A general-audience user asks for a concise explanation of a technical "
            f"concept. This example belongs to the '{category}' category: {focus}."
        ),
        task=(
            "Generate single-turn questions that fit this category. Keep the input "
            "realistic and make the category's edge case clear."
        ),
        input_format="A natural-language user question or prompt.",
        expected_output_format=(
            "A concise explanation with a definition and the topic's primary purpose "
            "or significance in simple language."
        ),
    )


def _category_counts(num_goldens: int) -> list[tuple[str, str, int]]:
    if num_goldens < 1:
        raise ValueError("num_goldens must be at least 1")

    base_count, remainder = divmod(num_goldens, len(DATASET_CATEGORIES))
    return [
        (category, focus, base_count + (index < remainder))
        for index, (category, focus) in enumerate(DATASET_CATEGORIES)
        if base_count + (index < remainder) > 0
    ]


def generate_dataset(num_goldens: int) -> str:
    settings = get_settings()
    model = GatewayEvaluationModel(settings)
    generated_goldens = []

    for category, focus, category_count in _category_counts(num_goldens):
        synthesizer = Synthesizer(
            model=model,
            styling_config=_styling_config(category, focus),
            evolution_config=EvolutionConfig(num_evolutions=0),
            async_mode=False,
            max_concurrent=int(os.getenv("DEEPEVAL_MAX_CONCURRENT", "1")),
        )
        category_goldens = synthesizer.generate_goldens_from_scratch(
            num_goldens=category_count
        )
        category_slug = category.lower().replace(" ", "-")
        for index, golden in enumerate(category_goldens, start=1):
            generated_goldens.append(
                {
                    "name": f"{category_slug}-{index}",
                    "input": golden.input,
                    "actual_output": golden.actual_output,
                    "expected_output": golden.expected_output,
                    "additional_metadata": {"category": category},
                }
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "synthetic_goldens.json"
    output_path.write_text(json.dumps(generated_goldens, indent=2) + "\n", encoding="utf-8")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DeepEval goldens.")
    parser.add_argument("--num-goldens", type=int, default=10)
    args = parser.parse_args()
    print(generate_dataset(args.num_goldens))


if __name__ == "__main__":
    main()
