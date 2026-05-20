"""Curated mini-benchmark selection.

For Project K we evaluate on 20–30 issues. Two sources are supported:

  1. A slice of SWE-Bench-Lite (princeton-nlp/SWE-Bench_Lite, split=dev or test).
     This is the standard option and reuses the existing Docker images.

  2. A hand-curated JSONL file of `{instance_id, problem_statement, repo, base_commit,
     test_patch, patch}` records, for tiny repos where we don't want to pull a 1GB
     Docker image. The JSONL format is compatible with the SWE-Bench schema.

The selection here gives a deterministic, small slice that's roughly balanced
across repositories so a 20-instance run isn't dominated by one project.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def curated_lite_slice(instances: list[dict], n: int = 20, seed: int = 42) -> list[dict]:
    """Pick ~n instances from SWE-Bench-Lite, balanced across repos.

    Deterministic given (instances, n, seed). Sorts inputs first so order of the
    upstream dataset doesn't change the slice.
    """
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for inst in sorted(instances, key=lambda x: x["instance_id"]):
        repo = inst.get("repo", "unknown")
        by_repo[repo].append(inst)
    rng = random.Random(seed)
    repos = sorted(by_repo)
    rng.shuffle(repos)
    out: list[dict] = []
    i = 0
    while len(out) < n and repos:
        repo = repos[i % len(repos)]
        bucket = by_repo[repo]
        if bucket:
            out.append(bucket.pop(0))
        else:
            repos.remove(repo)
            continue
        i += 1
    return out[:n]


def load_jsonl_instances(path: str | Path) -> list[dict]:
    """Load a JSONL benchmark file (one instance per line)."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
