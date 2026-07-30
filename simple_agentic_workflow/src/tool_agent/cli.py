"""Command-line entry point for a single agent invocation."""

import argparse

from langchain_core.messages import HumanMessage

from .config import get_settings
from .graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simple tool-using LangGraph agent.")
    parser.add_argument("message", help="The message to send to the agent")
    args = parser.parse_args()

    graph = build_graph(get_settings())
    result = graph.invoke({"messages": [HumanMessage(content=args.message)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
