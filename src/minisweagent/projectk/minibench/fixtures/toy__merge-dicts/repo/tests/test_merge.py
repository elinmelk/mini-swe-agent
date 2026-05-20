import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from merge import merge_counts


def test_overlap():
    assert merge_counts({"a": 1, "b": 2}, {"a": 3}) == {"a": 4, "b": 2}


def test_b_only_keys():
    assert merge_counts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_empty():
    assert merge_counts({}, {}) == {}


def test_all_unique():
    assert merge_counts({"x": 1}, {"y": 2}) == {"x": 1, "y": 2}
