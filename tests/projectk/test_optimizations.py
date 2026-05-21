"""Tests for the four optimization techniques: Best-of-N, Reflexion, hybrid
retrieval (symbol + BM25 + RRF), and dynamic re-planning trigger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------- Best-of-N
def test_bestofn_counts_failing_tests():
    from minisweagent.projectk.bestofn import _count_failing
    assert _count_failing("1 failed, 2 passed in 0.01s") == 1
    assert _count_failing("3 failed, 5 passed in 0.02s") == 3
    assert _count_failing("4 passed in 0.01s") == 0
    # error during collection should count as huge failure
    assert _count_failing("ERROR during collection") == 999


def test_bestofn_rank_candidates_prefers_passing(tmp_path: Path):
    from minisweagent.projectk.bestofn import CandidateScore
    # Tie-break: pass beats fail, then fewer failures wins, then shorter patch wins
    a = CandidateScore(tmp_path/"a", patch="diff a", tests_pass=False, failing_tests=5, patch_size=200)
    b = CandidateScore(tmp_path/"b", patch="diff b", tests_pass=True,  failing_tests=0, patch_size=400)
    c = CandidateScore(tmp_path/"c", patch="diff c", tests_pass=False, failing_tests=2, patch_size=300)
    ranked = sorted([a, b, c], key=lambda s: (not s.tests_pass, s.failing_tests, s.patch_size))
    assert ranked[0] is b
    assert ranked[1] is c
    assert ranked[2] is a


def test_bestofn_rank_skips_missing_traj(tmp_path: Path):
    from minisweagent.projectk.bestofn import rank_candidates
    from minisweagent.projectk.minibench.runner import Instance
    # No trajectory files in tmp_path -> rank should return None
    inst = Instance(instance_id="x", problem_statement="p", test_command="pytest -q",
                    repo_dir=tmp_path, fixture_dir=tmp_path, setup_commands=[])
    assert rank_candidates(inst, [tmp_path/"missing"], {}) is None


# ---------------------------------------------------------------- Reflexion
def test_reflexion_agent_class_registered():
    from minisweagent.agents import get_agent_class
    cls = get_agent_class("reflexion")
    assert cls.__name__ == "ReflexionAgent"


def test_reflexion_attempt_succeeded_logic():
    from minisweagent.agents.projectk.reflexion import ReflexionAgent
    # We can't easily instantiate without a model/env; test the logic statically
    # by inspecting the method against synthetic info dicts.
    method = ReflexionAgent._attempt_succeeded
    # Need a self stub; the method only reads info
    class Stub:
        pass
    assert method(Stub(), {"exit_status": "Submitted", "submission": "diff --git a/x b/x\n"}) is True
    assert method(Stub(), {"exit_status": "Submitted", "submission": ""}) is False
    assert method(Stub(), {"exit_status": "LimitsExceeded", "submission": "diff --git a/x b/x\n"}) is False


# ---------------------------------------------------------------- BM25 retrieval
def test_bm25_index_finds_by_body_keywords(tmp_path: Path):
    """BM25 needs a non-degenerate corpus (>2 docs) for meaningful IDF."""
    from minisweagent.projectk.retrieval import BM25Index
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    # 1 target + 5 distractors so IDF is informative for the discriminating tokens
    (pkg / "ops.py").write_text(
        '"""math operations."""\n'
        'def flatten(xs):\n'
        '    """Flatten a nested list into a single list of elements."""\n'
        '    result = []\n'
        '    for x in xs:\n'
        '        result.extend(x)\n'
        '    return result\n'
        '\n'
        'def add_numbers(a, b):\n'
        '    """Return the sum."""\n'
        '    return a + b\n'
        '\n'
        'def open_file(path):\n'
        '    """Open and read a file."""\n'
        '    return open(path).read()\n'
        '\n'
        'def parse_date(s):\n'
        '    """Parse an ISO date string."""\n'
        '    from datetime import date\n'
        '    return date.fromisoformat(s)\n'
        '\n'
        'def compute_hash(blob):\n'
        '    """Compute SHA-256 hash of a bytes blob."""\n'
        '    import hashlib\n'
        '    return hashlib.sha256(blob).hexdigest()\n'
        '\n'
        'def render_template(tpl, ctx):\n'
        '    """Render a Jinja template with the given context."""\n'
        '    return tpl.render(**ctx)\n'
    )
    idx = BM25Index(tmp_path)
    hits = idx.search("nested list flatten into single list of elements")
    assert hits, "BM25 returned no hits for a clearly matching query"
    assert hits[0].qualname == "flatten"


def test_bm25_index_empty_repo(tmp_path: Path):
    from minisweagent.projectk.retrieval import BM25Index
    idx = BM25Index(tmp_path)
    assert idx.search("anything") == []


# ---------------------------------------------------------------- Reciprocal-rank fusion
def test_rrf_combines_two_rankings():
    from minisweagent.projectk.retrieval import reciprocal_rank_fusion
    a = [("x", 1), ("y", 2), ("z", 3)]
    b = [("y", 1), ("x", 2), ("w", 3)]
    fused = reciprocal_rank_fusion([a, b], k=10)
    # x and y both appear at rank 1 once + rank 2 once -> tie at top
    assert {fused[0][0], fused[1][0]} == {"x", "y"}
    # z and w each appear once -> tied at the bottom
    assert {fused[2][0], fused[3][0]} == {"z", "w"}


def test_rrf_handles_empty_rankings():
    from minisweagent.projectk.retrieval import reciprocal_rank_fusion
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[]], k=60) == []


# ---------------------------------------------------------------- Hybrid retrieval via RetrievalAgent
def test_hybrid_retrieval_picks_correct_file(tmp_path: Path, monkeypatch):
    """Construct a repo with two files; only one matches by content. Hybrid should rank it first."""
    from minisweagent.projectk.retrieval import BM25Index, SymbolIndex, reciprocal_rank_fusion
    from minisweagent.agents.projectk.retrieval_agent import _candidate_identifiers

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "merge.py").write_text(
        'def merge_counts(a, b):\n'
        '    """Return a new dict whose keys are union of a and b with values summed."""\n'
        '    out = {}\n'
        '    for k, v in a.items():\n'
        '        out[k] = v + b.get(k, 0)\n'
        '    return out\n'
    )
    (repo / "unrelated.py").write_text(
        'def helper():\n'
        '    """Some helper that does nothing relevant."""\n'
        '    return None\n'
    )

    issue = ("`merge.merge_counts(a, b)` drops keys that only appear in `b`. "
             "Fix `merge.py`.")
    sym = SymbolIndex(repo)
    bm = BM25Index(repo)

    # Symbol ranking from keyword extractor
    keywords = _candidate_identifiers(issue)
    sym_ranked: list[tuple[str, int]] = []
    seen = set()
    for kw in keywords:
        for s in sym.lookup(kw, limit=4):
            key = f"{s.qualname}@{s.filepath}:{s.line}"
            if key not in seen:
                seen.add(key)
                sym_ranked.append((key, len(sym_ranked) + 1))

    bm25_hits = bm.search(issue)
    bm_ranked = [(f"{h.qualname}@{h.filepath}:{h.line}", i + 1) for i, h in enumerate(bm25_hits)]
    fused = reciprocal_rank_fusion([sym_ranked, bm_ranked])
    assert fused, "RRF returned no hits"
    top_key = fused[0][0]
    assert top_key.startswith("merge_counts@merge.py"), f"Expected merge_counts to rank first, got {top_key}"


# ---------------------------------------------------------------- Dynamic replanning trigger
def test_planner_consecutive_failure_counter():
    """The on-demand replan trigger should count consecutive non-zero returncodes."""
    from minisweagent.agents.projectk.planner_executor import PlannerExecutorAgent

    # Build a fake instance with just .messages — we don't need a full agent.
    class Fake(PlannerExecutorAgent):
        def __init__(self): pass  # bypass real __init__

    f = Fake()
    f.messages = [
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>ok</output>"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "<returncode>1</returncode>\n<output>err</output>"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "<returncode>2</returncode>\n<output>err</output>"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "<returncode>1</returncode>\n<output>err</output>"},
    ]
    assert f._consecutive_failure_count() == 3  # noqa: SLF001


def test_planner_consecutive_failure_counter_resets_on_success():
    from minisweagent.agents.projectk.planner_executor import PlannerExecutorAgent

    class Fake(PlannerExecutorAgent):
        def __init__(self): pass

    f = Fake()
    f.messages = [
        {"role": "user", "content": "<returncode>1</returncode>"},
        {"role": "user", "content": "<returncode>1</returncode>"},
        {"role": "user", "content": "<returncode>0</returncode>"},  # reset point
        {"role": "user", "content": "<returncode>1</returncode>"},
    ]
    assert f._consecutive_failure_count() == 1
