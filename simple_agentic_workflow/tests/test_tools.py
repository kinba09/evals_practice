"""Direct tests for the deterministic tools."""

import pytest

from tool_agent.tools import add_numbers, count_characters, count_words


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (2, 3, 5),
        (2.5, 0.5, 3),
        (-4, 10, 6),
        (0, 0, 0),
    ],
)
def test_add_numbers(first: float, second: float, expected: float) -> None:
    assert add_numbers.invoke({"first": first, "second": second}) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("one two three", 3),
        ("one,\ntwo\tthree!", 3),
        ("", 0),
        ("   ", 0),
    ],
)
def test_count_words(text: str, expected: int) -> None:
    assert count_words.invoke({"text": text}) == expected


def test_count_characters_with_and_without_spaces() -> None:
    text = "hello world"
    assert count_characters.invoke({"text": text, "include_spaces": True}) == 11
    assert count_characters.invoke({"text": text, "include_spaces": False}) == 10


def test_tools_handle_large_text() -> None:
    text = "word " * 10_000
    assert count_words.invoke({"text": text}) == 10_000


def test_add_numbers_requires_both_inputs() -> None:
    with pytest.raises((TypeError, ValueError)):
        add_numbers.invoke({"first": 1})
