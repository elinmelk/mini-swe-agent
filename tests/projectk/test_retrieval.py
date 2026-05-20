"""Unit tests for retrieval (SymbolIndex + identifier extraction)."""

from __future__ import annotations

from pathlib import Path

from minisweagent.agents.projectk.retrieval_agent import (
    _candidate_identifiers,
    _normalize_token,
)
from minisweagent.projectk.retrieval import SymbolIndex


def test_normalize_token_splits_paths_and_dots() -> None:
    out = _normalize_token("mathy/ops.py")
    assert "mathy/ops.py" in out
    assert "mathy" in out
    assert "ops.py" in out
    assert "ops" in out


def test_normalize_token_strips_py_suffix() -> None:
    out = _normalize_token("calendar_utils.py")
    assert "calendar_utils" in out
    assert "calendar_utils.py" in out


def test_normalize_token_dotted_qualname() -> None:
    out = _normalize_token("calendar_utils.is_leap_year")
    assert "calendar_utils" in out
    assert "is_leap_year" in out
    assert "calendar_utils.is_leap_year" in out


def test_candidate_identifiers_picks_backtick_words() -> None:
    task = "The `add` function in `mathy/ops.py` is broken."
    ids = _candidate_identifiers(task)
    # Backtick-fenced single-word identifiers like `add` should be admitted
    assert "add" in ids
    assert "mathy/ops.py" in ids
    assert "mathy" in ids


def test_candidate_identifiers_rejects_stopwords() -> None:
    task = "the the the the function function"
    ids = _candidate_identifiers(task)
    # All English stopwords; nothing useful to pull out
    assert ids == []


def test_candidate_identifiers_dedupes_and_orders_backticks_first() -> None:
    task = "Fix `add` in `mathy/ops.py`. Also see calendar_utils.is_leap_year."
    ids = _candidate_identifiers(task)
    # `add` comes from backticks (first pass) before `calendar_utils.is_leap_year` (second pass)
    assert ids.index("add") < ids.index("calendar_utils.is_leap_year")


def test_symbol_index_finds_class_and_function(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "ops.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "class Calculator:\n"
        "    def multiply(self, a, b):\n"
        "        return a * b\n"
    )

    idx = SymbolIndex(tmp_path)
    names = {s.name for s in idx.symbols}
    assert "add" in names
    assert "Calculator" in names
    assert "multiply" in names

    hits = idx.lookup("add")
    assert any(s.name == "add" and s.kind == "function" for s in hits)

    method_hits = idx.lookup("multiply")
    assert any(s.kind == "method" and s.qualname == "Calculator.multiply" for s in method_hits)


def test_symbol_index_lookup_ranks_exact_first(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def add(): pass\ndef adder(): pass\ndef readd(): pass\n")
    idx = SymbolIndex(tmp_path)
    ranked = idx.lookup("add")
    # exact "add" should come before "adder" (prefix) which comes before "readd" (substring)
    names = [s.name for s in ranked]
    assert names.index("add") < names.index("adder")
    assert names.index("adder") < names.index("readd")
