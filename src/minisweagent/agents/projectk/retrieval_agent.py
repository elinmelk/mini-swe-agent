"""Retrieval-augmented agent.

Before the executor loop starts, this agent inspects the repository (the
environment's cwd) and runs one of two retrievers over the task text:

  - "symbol":    AST-based name/qualname matching via projectk.retrieval.SymbolIndex
  - "embedding": code-chunk vector similarity via projectk.retrieval.EmbeddingIndex
  - "none":      no retrieval (baseline)

The top-k results are formatted into a string and exposed to the prompt
template as `{{retrieved_context}}`. The agent class itself doesn't change the
control loop — it just augments the first user message.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.projectk.retrieval import (
    BM25Index,
    EmbeddingIndex,
    SymbolIndex,
    reciprocal_rank_fusion,
)

logger = logging.getLogger("agent.retrieval")


# Pull out identifier-like tokens from a problem statement.
# We accept snake_case, CamelCase, dotted.qualified.names, slash/file/paths,
# and anything inside backticks — these are the things most likely to match a
# function/class/file/module in the repo.
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\b")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_STOPWORDS = {
    "the", "a", "an", "is", "in", "of", "to", "for", "and", "or", "not", "but",
    "test", "tests", "function", "method", "class", "module", "file", "code",
    "fix", "bug", "issue", "should", "must", "will", "can", "this", "that",
    "with", "without", "do", "does", "did", "be", "been", "being", "are", "was",
    "were", "have", "has", "had", "it", "its", "if", "then", "else", "when",
    "while", "as", "by", "on", "at", "from", "into", "out", "over", "under",
    "please", "ensure", "expected", "actual", "correctly", "incorrectly",
    "return", "returns", "value", "values",
}


def _normalize_token(tok: str) -> list[str]:
    """Turn a raw token into one or more lookup candidates.

    "calendar_utils.py"            -> ["calendar_utils.py", "calendar_utils"]
    "mathy/ops.py"                 -> ["mathy/ops.py", "mathy", "ops", "ops.py"]
    "calendar_utils.is_leap_year"  -> ["calendar_utils.is_leap_year", "calendar_utils", "is_leap_year"]
    "add(a, b)"                    -> ["add"]
    """
    tok = tok.strip().strip("(){}[],.;:")
    if not tok:
        return []
    out = [tok]
    # Split paths and dotted names
    for sep in ("/", "."):
        if sep in tok:
            for part in tok.split(sep):
                part = part.strip()
                if part:
                    out.append(part)
    # Strip .py suffix on each candidate
    extra: list[str] = []
    for c in out:
        if c.endswith(".py"):
            extra.append(c[:-3])
    out.extend(extra)
    # Deduplicate while preserving order
    seen, deduped = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _candidate_identifiers(task: str, limit: int = 16) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def consider(raw: str, *, is_backtick: bool = False) -> None:
        for cand in _normalize_token(raw):
            low = cand.lower()
            if low in _STOPWORDS or cand in seen:
                continue
            # Backtick-fenced things are explicit user picks — always include them.
            # Otherwise require some "interesting" structure to avoid English noise.
            if is_backtick or "_" in cand or "." in cand or "/" in cand or any(
                c.isupper() for c in cand[1:]
            ):
                seen.add(cand)
                out.append(cand)
                if len(out) >= limit:
                    return

    # First pass: backtick-quoted picks (highest signal).
    for match in _BACKTICK_RE.findall(task):
        consider(match, is_backtick=True)
        if len(out) >= limit:
            return out
    # Second pass: structured identifiers anywhere in the task.
    for match in _IDENT_RE.findall(task):
        consider(match)
        if len(out) >= limit:
            return out
    return out


class RetrievalAgentConfig(AgentConfig):
    retrieval_mode: Literal["none", "symbol", "bm25", "hybrid", "embedding"] = "symbol"
    retrieval_top_k: int = 8
    retrieval_embedding_model: str = "nvidia_nim/nvidia/nv-embedcode-7b-v1"


class RetrievalAgent(DefaultAgent):
    """Augments the first user message with repo-level retrieval results."""

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=RetrievalAgentConfig, **kwargs)

    # ------------------------------------------------------------------ build
    def _repo_root(self) -> Path | None:
        cwd = getattr(self.env.config, "cwd", "") or ""
        if cwd and Path(cwd).exists():
            return Path(cwd)
        return None

    def _format_symbols(self, idx: SymbolIndex, task: str) -> str:
        keywords = _candidate_identifiers(task)
        seen: set[tuple[str, int, str]] = set()
        rows: list[str] = []
        for kw in keywords:
            for sym in idx.lookup(kw, limit=4):
                key = (sym.filepath, sym.line, sym.qualname)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(f"- {sym.kind:8s} {sym.qualname:30s} {sym.filepath}:{sym.line}")
                if len(rows) >= self.config.retrieval_top_k:
                    break
            if len(rows) >= self.config.retrieval_top_k:
                break
        if not rows:
            return ""
        header = (
            f"# Symbol-index retrieval (top-{len(rows)} matches for keywords "
            f"{', '.join(keywords[:6])}{'…' if len(keywords) > 6 else ''})\n"
        )
        return header + "\n".join(rows)

    def _format_bm25(self, idx: BM25Index, task: str) -> str:
        hits = idx.search(task, limit=self.config.retrieval_top_k)
        if not hits:
            return ""
        rows = [
            f"- {h.kind:8s} {h.qualname:30s} {h.filepath}:{h.line}  (bm25={h.score:.2f})"
            for h in hits
        ]
        return f"# BM25 content retrieval (top-{len(rows)})\n" + "\n".join(rows)

    def _format_hybrid(self, root: Path, task: str) -> str:
        """Symbol exact-match + BM25 body match, fused via reciprocal-rank fusion."""
        symbol_idx = SymbolIndex(root)
        bm25_idx = BM25Index(root)
        keywords = _candidate_identifiers(task)

        # Build ranked lists keyed by "qualname@file:line"
        sym_ranked: list[tuple[str, int]] = []
        seen: set[str] = set()
        for kw in keywords:
            for sym in symbol_idx.lookup(kw, limit=6):
                key = f"{sym.qualname}@{sym.filepath}:{sym.line}"
                if key in seen:
                    continue
                seen.add(key)
                sym_ranked.append((key, len(sym_ranked) + 1))

        bm25_hits = bm25_idx.search(task, limit=10)
        bm25_ranked = [
            (f"{h.qualname}@{h.filepath}:{h.line}", i + 1)
            for i, h in enumerate(bm25_hits)
        ]

        fused = reciprocal_rank_fusion([sym_ranked, bm25_ranked])
        if not fused:
            return ""

        # Decode keys back into rows
        meta: dict[str, str] = {}
        for sym in symbol_idx.symbols:
            meta[f"{sym.qualname}@{sym.filepath}:{sym.line}"] = f"{sym.kind:8s} {sym.qualname:30s} {sym.filepath}:{sym.line}"
        for h in bm25_hits:
            meta.setdefault(f"{h.qualname}@{h.filepath}:{h.line}",
                            f"{h.kind:8s} {h.qualname:30s} {h.filepath}:{h.line}")

        rows: list[str] = []
        for key, score in fused[: self.config.retrieval_top_k]:
            if key in meta:
                rows.append(f"- {meta[key]}  (rrf={score:.3f})")
        if not rows:
            return ""
        return (
            f"# Hybrid retrieval (top-{len(rows)} after reciprocal-rank fusion;\n"
            f"#   symbol-index keywords: {', '.join(keywords[:6])}{'…' if len(keywords) > 6 else ''})\n"
            + "\n".join(rows)
        )

    def _format_embedding(self, idx: EmbeddingIndex, task: str) -> str:
        hits = idx.search(task, limit=self.config.retrieval_top_k)
        if not hits:
            return ""
        rows = []
        for h in hits:
            preview = (h["text"] or "").splitlines()
            preview = "\n    ".join(preview[:8])
            rows.append(f"- {h['filepath']}:{h['line']} (score={h['score']:.2f})\n    {preview}")
        return f"# Embedding-index retrieval (top-{len(rows)} matches)\n" + "\n".join(rows)

    def _build_context(self, task: str) -> str:
        mode = self.config.retrieval_mode
        if mode == "none":
            return ""
        root = self._repo_root()
        if root is None:
            logger.warning("retrieval: env has no usable cwd; skipping")
            return ""
        try:
            if mode == "symbol":
                idx = SymbolIndex(root)
                logger.info(f"retrieval[symbol]: indexed {len(idx.symbols)} symbols under {root}")
                return self._format_symbols(idx, task)
            if mode == "bm25":
                idx = BM25Index(root)
                logger.info(f"retrieval[bm25]: indexed {len(idx._meta)} bodies under {root}")  # noqa: SLF001
                return self._format_bm25(idx, task)
            if mode == "hybrid":
                logger.info(f"retrieval[hybrid]: building symbol+bm25 indices under {root}")
                return self._format_hybrid(root, task)
            if mode == "embedding":
                idx = EmbeddingIndex(root, embedding_model=self.config.retrieval_embedding_model)
                logger.info(f"retrieval[embedding]: indexed {len(idx.entries)} chunks under {root}")
                return self._format_embedding(idx, task)
        except Exception as e:
            logger.warning(f"retrieval[{mode}] failed: {e}")
            return f"(retrieval failed: {e})"
        return ""

    # ------------------------------------------------------------------- run
    def run(self, task: str = "", **kwargs) -> dict:
        # Set retrieved_context BEFORE super().run() renders the instance template.
        ctx = self._build_context(task)
        self.extra_template_vars["retrieved_context"] = ctx
        return super().run(task=task, **kwargs)
