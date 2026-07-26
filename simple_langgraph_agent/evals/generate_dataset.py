"""Generate synthetic single-turn DeepEval goldens for the agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, StylingConfig

from simple_agent.config import get_settings

from .models import GatewayEvaluationModel

DATA_DIR = Path(__file__).parent / "data"


def generate_dataset(num_goldens: int) -> str:
    settings = get_settings()
    styling_config = StylingConfig(
        scenario=(
            "A general-audience user asks for a concise explanation of a technical "
            "concept and may sometimes ask about an ambiguous topic."
        ),
        task=(
            "Generate single-turn questions about technical concepts using the form "
            "What is <topic>? Include a mix of common concepts, less common concepts, "
            "and a few topics that could reasonably need clarification."
        ),
        input_format="A single natural-language question beginning with 'What is'.",
        expected_output_format=(
            "A concise explanation with a definition and the topic's primary purpose "
            "or significance in simple language."
        ),
    )
    synthesizer = Synthesizer(
        model=GatewayEvaluationModel(settings),
        styling_config=styling_config,
        evolution_config=EvolutionConfig(num_evolutions=0),
        async_mode=False,
        max_concurrent=int(os.getenv("DEEPEVAL_MAX_CONCURRENT", "1")),
    )
    goldens = synthesizer.generate_goldens_from_scratch(num_goldens=num_goldens)
    dataset = EvaluationDataset(goldens=goldens)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "synthetic_goldens.json"
    minimal_goldens = [
        {
            "input": golden.input,
            "actual_output": golden.actual_output,
            "expected_output": golden.expected_output,
        }
        for golden in dataset.goldens
    ]
    output_path.write_text(json.dumps(minimal_goldens, indent=2) + "\n", encoding="utf-8")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DeepEval goldens.")
    parser.add_argument("--num-goldens", type=int, default=10)
    args = parser.parse_args()
    print(generate_dataset(args.num_goldens))


if __name__ == "__main__":
    main()
