"""Repo-level retrieval helpers.

Three indices, all cheap to build for SWE-Bench-Lite-scale repos:

  1. Symbol index — AST-based name/qualname lookup over every function, method,
     and class. Lookups by name return exact > prefix > substring matches.

  2. BM25 content index — BM25 scoring over function/class *body text*
     (signature + docstring + body), so semantic descriptions in the issue
     ("function that drops keys only present in b") match the right symbol
     even when the symbol's name isn't in the issue.

  3. Embedding index — code-chunk vector similarity via litellm.embedding.
     Disabled by default because it requires a working embedding endpoint.

The hybrid retriever combines (1) and (2) with reciprocal-rank fusion (RRF):
the final score for a candidate s is sum over queries q of 1/(k + rank_q(s)),
which weights agreement across retrievers more than the score scale of any
one retriever.
"""

from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Symbol:
    filepath: str
    line: int
    kind: str  # "function" | "class" | "method"
    name: str
    qualname: str


def _iter_python_files(root: Path, excludes: Iterable[str] = ()) -> Iterable[Path]:
    excludes = tuple(excludes)
    for p in root.rglob("*.py"):
        rel = str(p.relative_to(root))
        if rel.startswith((".git/", "build/", "dist/")):
            continue
        if any(e in rel for e in excludes):
            continue
        yield p


class SymbolIndex:
    def __init__(self, root: str | Path, *, excludes: Iterable[str] = ()):
        self.root = Path(root).resolve()
        self.symbols: list[Symbol] = []
        self._build(excludes)

    def _build(self, excludes: Iterable[str]) -> None:
        for path in _iter_python_files(self.root, excludes):
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except SyntaxError:
                continue
            rel = str(path.relative_to(self.root))
            for node in tree.body:
                self._collect(node, rel, parent="")
            # Also walk classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.symbols.append(
                                Symbol(
                                    filepath=rel,
                                    line=item.lineno,
                                    kind="method",
                                    name=item.name,
                                    qualname=f"{node.name}.{item.name}",
                                )
                            )

    def _collect(self, node: ast.AST, rel: str, parent: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.symbols.append(
                Symbol(filepath=rel, line=node.lineno, kind="function", name=node.name, qualname=node.name)
            )
        elif isinstance(node, ast.ClassDef):
            self.symbols.append(
                Symbol(filepath=rel, line=node.lineno, kind="class", name=node.name, qualname=node.name)
            )

    def lookup(self, query: str, *, limit: int = 20) -> list[Symbol]:
        q = query.lower()
        exact = [s for s in self.symbols if s.name == query]
        prefix = [s for s in self.symbols if s.name.lower().startswith(q) and s not in exact]
        substring = [
            s for s in self.symbols
            if q in s.name.lower() and s not in exact and s not in prefix
        ]
        return (exact + prefix + substring)[:limit]


# ---------------------------------------------------------------------------
# BM25 content index
# ---------------------------------------------------------------------------


_BM25_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _tokenize_for_bm25(text: str) -> list[str]:
    """Identifier-aware tokenization, splitting CamelCase and snake_case."""
    raw = _BM25_TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in raw:
        out.append(tok)
        # Split snake_case
        if "_" in tok:
            out.extend(p for p in tok.split("_") if p)
    return out


@dataclass(frozen=True)
class BM25Hit:
    filepath: str
    line: int
    qualname: str
    kind: str  # "function" | "method" | "class"
    score: float


class BM25Index:
    """BM25 over function / class bodies (signature + docstring + body text).

    Each symbol becomes one document. At query time we tokenize the query the
    same way and return the top-k scoring symbols.
    """

    def __init__(self, root: str | Path, *, excludes: tuple[str, ...] = ()):
        from rank_bm25 import BM25Okapi  # lazy import

        self.root = Path(root).resolve()
        self._meta: list[dict[str, Any]] = []
        corpus: list[list[str]] = []

        for path in _iter_python_files(self.root, excludes):
            try:
                source = path.read_text(errors="replace")
                tree = ast.parse(source)
            except SyntaxError:
                continue
            rel = str(path.relative_to(self.root))
            source_lines = source.splitlines()

            def _collect(node: ast.AST, qual_prefix: str = "") -> None:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qual = f"{qual_prefix}{node.name}" if not qual_prefix else f"{qual_prefix}.{node.name}"
                    kind = "class" if isinstance(node, ast.ClassDef) else (
                        "method" if qual_prefix else "function"
                    )
                    end_line = getattr(node, "end_lineno", node.lineno) or node.lineno
                    body_text = "\n".join(source_lines[node.lineno - 1 : end_line])
                    self._meta.append({
                        "filepath": rel, "line": node.lineno,
                        "qualname": qual, "kind": kind, "body": body_text,
                    })
                    corpus.append(_tokenize_for_bm25(body_text))
                    if isinstance(node, ast.ClassDef):
                        for child in node.body:
                            _collect(child, qual_prefix=qual)

            for top in tree.body:
                _collect(top)

        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, *, limit: int = 8) -> list[BM25Hit]:
        if self._bm25 is None:
            return []
        q_tokens = _tokenize_for_bm25(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits: list[BM25Hit] = []
        for i in ranked[:limit]:
            if scores[i] <= 0:
                break
            m = self._meta[i]
            hits.append(BM25Hit(
                filepath=m["filepath"], line=m["line"],
                qualname=m["qualname"], kind=m["kind"],
                score=float(scores[i]),
            ))
        return hits


def reciprocal_rank_fusion(rankings: list[list[tuple[str, int]]], k: int = 60) -> list[tuple[str, float]]:
    """Combine multiple ranked lists into one ranking via RRF.

    Each ranking is a list of (item_id, rank) tuples (rank starts at 1).
    The fused score for an item is sum_r 1/(k + rank_r(item)).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for item_id, rank in ranking:
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------


def _chunk_file(path: Path, chunk_size: int = 1500, overlap: int = 200) -> list[tuple[str, int]]:
    text = path.read_text(errors="replace")
    if len(text) <= chunk_size:
        return [(text, 1)]
    chunks: list[tuple[str, int]] = []
    i = 0
    while i < len(text):
        block = text[i : i + chunk_size]
        line_start = text[:i].count("\n") + 1
        chunks.append((block, line_start))
        i += chunk_size - overlap
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


class EmbeddingIndex:
    """Tiny code-embedding index, backed by litellm.embedding."""

    def __init__(
        self,
        root: str | Path,
        *,
        embedding_model: str = "nvidia_nim/nvidia/nv-embedcode-7b-v1",
        excludes: Iterable[str] = (),
        chunk_size: int = 1500,
    ):
        self.root = Path(root).resolve()
        self.embedding_model = embedding_model
        self.entries: list[dict[str, Any]] = []
        self._chunk_size = chunk_size
        self._build(excludes)

    def _build(self, excludes: Iterable[str]) -> None:
        import litellm  # local import; heavy module

        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for path in _iter_python_files(self.root, excludes):
            rel = str(path.relative_to(self.root))
            for block, line in _chunk_file(path, self._chunk_size):
                texts.append(block)
                metas.append({"filepath": rel, "line": line, "text": block})
        if not texts:
            return
        # Batch embed in small windows to be polite to the API.
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            resp = litellm.embedding(model=self.embedding_model, input=batch)
            embeddings.extend(item["embedding"] for item in resp["data"])
        for meta, emb in zip(metas, embeddings):
            meta["embedding"] = emb
            self.entries.append(meta)

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        import litellm

        if not self.entries:
            return []
        qresp = litellm.embedding(model=self.embedding_model, input=[query])
        qvec = qresp["data"][0]["embedding"]
        scored = [
            (_cosine(qvec, entry["embedding"]), entry)
            for entry in self.entries
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": score, "filepath": e["filepath"], "line": e["line"], "text": e["text"][:600]}
            for score, e in scored[:limit]
        ]
