import sys

sys.path.insert(0, "/workspace")

from solution import sum_column


def test_column_sum() -> None:
    assert sum_column([["a", "1"], ["b", "2"], ["c", "3"]], 1) == 6.0
    assert sum_column([], 0) == 0.0
    assert sum_column([["x", "y"], ["1", "2"]], 0) == 1.0
