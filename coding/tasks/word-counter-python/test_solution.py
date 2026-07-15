import sys

sys.path.insert(0, "/workspace")

from solution import count_lines, count_words


def test_counts_words_and_lines() -> None:
    assert count_words("one two three") == 3
    assert count_words("   ") == 0
    assert count_words("") == 0
    assert count_lines("a\nb\nc") == 3
    assert count_lines("only") == 1
