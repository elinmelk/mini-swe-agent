"""Repo-level retrieval helpers.

Two indices, both cheap to build for the small SWE-Bench-Lite repos:

  1. Symbol index — uses Python's `ast` module to enumerate every top-level and
     class-scope function/method/class. Lookups by name or substring return a
     ranked list of `(filepath, line, kind, name)` tuples.

  2. Embedding index — embeds the first N characters of every code chunk through
     LiteLLM (so any provider works) and stores them in a tiny in-memory FAISS-
     like dict, with cosine similarity ranking. We avoid hard-depending on FAISS
     so the project still imports on a fresh machine.

Both helpers are intentionally side-effect-free: they return ranked lists, and
it's the caller's job to format them into a prompt fragment.
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
