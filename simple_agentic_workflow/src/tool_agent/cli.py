"""Command-line entry point for a single agent invocation."""

import argparse

from .config import get_settings
from .runner import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simple tool-using LangGraph agent.")
    parser.add_argument("message", help="The message to send to the agent")
    args = parser.parse_args()

    result = run_agent(args.message, settings=get_settings())
    print(result.final_output)


if __name__ == "__main__":
    main()
