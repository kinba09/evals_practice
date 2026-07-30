"""Small deterministic tools used by the agent."""

from langchain_core.tools import tool


@tool
def add_numbers(first: float, second: float) -> float:
    """Add two numbers and return the result."""

    return first + second


@tool
def count_words(text: str) -> int:
    """Count the whitespace-separated words in a piece of text."""

    return len(text.split())


@tool
def count_characters(text: str, include_spaces: bool = True) -> int:
    """Count characters in text, optionally excluding whitespace characters."""

    if include_spaces:
        return len(text)
    return sum(not character.isspace() for character in text)


TOOLS = [add_numbers, count_words, count_characters]
