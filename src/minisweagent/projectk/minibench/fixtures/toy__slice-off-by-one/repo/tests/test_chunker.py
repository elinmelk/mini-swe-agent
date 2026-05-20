import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chunker import chunks


def test_even_split():
    assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_tail_dropped():
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4]]


def test_zero_items():
    assert chunks([], 3) == []


def test_shorter_than_n():
    assert chunks([1, 2], 3) == []
