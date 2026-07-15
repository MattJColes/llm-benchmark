import sys

sys.path.insert(0, "/workspace")

from solution import is_palindrome


def test_palindrome_detection() -> None:
    assert is_palindrome("racecar") is True
    assert is_palindrome("RaceCar") is True
    assert is_palindrome("hello") is False
    assert is_palindrome("") is True
