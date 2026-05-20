"""Project K — Mini Coding Agent (SWE-Bench-Lite style).

Adds:
  * A bash-surface tool suite (file-read, directory-list, grep, edit, run-tests).
  * A mini-benchmark + evaluation harness with resolve rate, tool-calls per attempt,
    and total token / latency cost.
  * Failure-analysis script.
  * Planner-Executor decomposition agent (stretch).
  * Persistent scratchpad agent (stretch).
  * Multi-model comparison runner (stretch).
  * Repo-level retrieval helpers (symbol + embedding index, stretch).

Everything else (model, environment, base agent loop) is reused from mini-swe-agent.
"""
