import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calendar_utils import is_leap_year


def test_divisible_by_4_not_100():
    assert is_leap_year(2024) is True
    assert is_leap_year(2020) is True


def test_century_not_400():
    assert is_leap_year(1900) is False
    assert is_leap_year(2100) is False


def test_divisible_by_400():
    assert is_leap_year(2000) is True
    assert is_leap_year(1600) is True


def test_odd():
    assert is_leap_year(2023) is False
