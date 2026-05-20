import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bag import add_item


def test_explicit_basket():
    b = []
    assert add_item("apple", b) == ["apple"]
    assert b == ["apple"]


def test_independent_baskets():
    a = add_item("x")
    b = add_item("y")
    assert a == ["x"]
    assert b == ["y"]
