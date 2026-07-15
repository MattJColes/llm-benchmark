import sys

sys.path.insert(0, "/workspace")

from solution import to_roman


def test_roman_conversion() -> None:
    assert to_roman(1) == "I"
    assert to_roman(4) == "IV"
    assert to_roman(9) == "IX"
    assert to_roman(58) == "LVIII"
    assert to_roman(1994) == "MCMXCIV"
