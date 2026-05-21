# Project K — Mini Coding Agent (SWE-Bench-Lite style)

A small agentic coder that, given a GitHub-issue-style bug description and a
Python repository, produces a patch that fixes the bug. The agent operates in
a bash-only tool loop (read file, grep, edit, run tests). Built on top of
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), with additional
scaffold variants, optimization techniques, a Docker-free mini-benchmark, an
8-bucket failure taxonomy, and a reproducible ablation harness.

> **Headline result.** Three independent scaffold mechanisms — static
> planner–executor decomposition, Best-of-N sampling with a test-based
> verifier, and Reflexion retry-on-failure — all reach **100% resolve rate**
> on a 3-fixture mini-benchmark with **Qwen2.5-Coder-14B** (open-weight,
> running locally on Ollama), versus 33% for the unmodified baseline.

---

## What this fork adds on top of `mini-swe-agent`

| Component | Location |
| --- | --- |
| **Docker-free mini-benchmark** (5 toy fixtures, ground-truth via `pytest -q`) | [`src/minisweagent/projectk/minibench/`](src/minisweagent/projectk/minibench/) |
| **Metrics** (resolve rate / steps / tokens / latency / cost) | [`projectk/metrics.py`](src/minisweagent/projectk/metrics.py) |
| **8-bucket failure taxonomy** | [`projectk/failure_analysis.py`](src/minisweagent/projectk/failure_analysis.py) |
| **Scaffold variants**: planner-executor, persistent scratchpad, symbol-graph retrieval | [`agents/projectk/`](src/minisweagent/agents/projectk/) |
| **Optimizations**: Best-of-N + test verifier, Reflexion retry, hybrid symbol+BM25 retrieval, dynamic re-planning | [`projectk/bestofn.py`](src/minisweagent/projectk/bestofn.py), [`agents/projectk/reflexion.py`](src/minisweagent/agents/projectk/reflexion.py), [`projectk/retrieval.py`](src/minisweagent/projectk/retrieval.py) |
| **Throttled litellm wrapper** with multi-key round-robin (free-tier-safe) | [`projectk/throttled_model.py`](src/minisweagent/projectk/throttled_model.py) |
| **CLI tools**: `projectk-mini`, `projectk-mini-compare`, `projectk-mini-bestofn`, `projectk-report`, `projectk-run` | [`projectk/cli_*.py`](src/minisweagent/projectk/) |
| **Configs** for Ollama / Groq / NVIDIA NIM + per-scaffold variants | [`config/projectk/`](src/minisweagent/config/projectk/) |
| **41 pytest tests** for every Project K module | [`tests/projectk/`](tests/projectk/) |
| **Makefile** (11 targets) for one-command reproduction | [`Makefile`](Makefile) |
| **Full write-up** (motivation, methods, results, discussion, limitations) | [`REPORT.md`](REPORT.md) |
| **Standalone results PDF** | [`PROJECT_K_RESULTS.pdf`](PROJECT_K_RESULTS.pdf) |

---

## Quickstart

```bash
# One-time setup
make install        # create venv + install package + pytest
make ollama         # brew install + start service + pull qwen2.5-coder:14b

# Verify
make test           # 41 tests, ~1 second

# Single-fixture demo (~1 minute)
make demo

# All 5 fixtures (~10 minutes)
make mini

# Full 4-condition × 3-fixture ablation grid (~1 hour)
make ablation
```

`make help` lists all targets.

---

## Project goals — how each was achieved

The brief lists five goals. Mapping to where each is addressed in the code
and the write-up:

| Goal | How it's met |
| --- | --- |
| **Tool suite**: file-read, dir-list, grep/search, edit, run-tests | All five surfaced as bash idioms in every system prompt (`cat`, `ls`/`find`, `grep -RIn`, Python file writes, `pytest -q`). See [`config/projectk/ollama.yaml`](src/minisweagent/config/projectk/ollama.yaml) and [REPORT.md §2.1](REPORT.md). |
| **Agent loop on an open-weight / free-tier LLM** | Inherited linear-history bash loop ([`agents/default.py`](src/minisweagent/agents/default.py)) running **Qwen2.5-Coder-14B locally on Ollama**. Also tested: Qwen-7B, Llama-3.3-70B on Groq, Llama-3.1-8B on NVIDIA NIM. All open-weight. |
| **Select 20–30 issues from SWE-Bench-Lite or curate a mini set** | 5 hand-curated toy fixtures bundled in the repo, each isolating a different bug class (sign error, off-by-one, mutable default, multi-condition logic, dict iteration). A deterministic 20-instance SWE-Bench-Lite selector also exists in [`projectk/dataset.py`](src/minisweagent/projectk/dataset.py), runnable via `projectk-run` when Docker is available. |
| **Evaluate resolve rate, tool-calls per attempt, total token / latency / cost** | All five metrics + submission rate + exit-status histogram aggregated by [`projectk/metrics.py`](src/minisweagent/projectk/metrics.py) and displayed by `projectk-report`. Full numbers in [REPORT.md §5](REPORT.md). |
| **Failure analysis**: wrong file edited, wrong edit, broken tests, exceeded budget, infinite loop, etc. | 8-bucket taxonomy (`RESOLVED`, `BUDGET_EXCEEDED`, `CRASH`, `INFINITE_LOOP`, `NO_PATCH`, `WRONG_FILE_EDITED`, `TESTS_BROKEN`, `PROBABLY_WRONG_EDIT`) evaluated in priority order. Implementation in [`failure_analysis.py`](src/minisweagent/projectk/failure_analysis.py); every bucket unit-tested in [`tests/projectk/test_failure_analysis.py`](tests/projectk/test_failure_analysis.py). |

### Stretch goals — all four implemented

| Stretch goal | Implementation | Run via |
| --- | --- | --- |
| **Plan–execute decomposition** | [`PlannerExecutorAgent`](src/minisweagent/agents/projectk/planner_executor.py) | `projectk-mini -c .../ollama_planner.yaml` |
| **Persistent scratchpad / memory** | [`ScratchpadAgent`](src/minisweagent/agents/projectk/scratchpad.py) | `projectk-mini -c .../ollama_scratchpad.yaml` |
| **≥2 base-model comparison** (general vs code-specialised) | [`cli_mini_compare.py`](src/minisweagent/projectk/cli_mini_compare.py) | `make compare-providers` |
| **Repo-level retrieval** (symbol graph + optional BM25 / embedding) | [`projectk/retrieval.py`](src/minisweagent/projectk/retrieval.py) + [`RetrievalAgent`](src/minisweagent/agents/projectk/retrieval_agent.py) | `projectk-mini -c .../ollama_retrieval.yaml` |

### Extra optimisation techniques

| Technique | Inspired by | Implementation |
| --- | --- | --- |
| Best-of-N=3 with test-based verifier | [Agentless](https://arxiv.org/abs/2407.01489), [SWE-Gym](https://arxiv.org/abs/2412.21139) | [`projectk/bestofn.py`](src/minisweagent/projectk/bestofn.py) |
| Reflexion retry-on-failure | [Shinn et al., 2023](https://arxiv.org/abs/2303.11366) | [`agents/projectk/reflexion.py`](src/minisweagent/agents/projectk/reflexion.py) |
| Hybrid symbol+BM25 retrieval with reciprocal-rank fusion | — | extends [`projectk/retrieval.py`](src/minisweagent/projectk/retrieval.py) |
| Dynamic re-planning on consecutive failures | — | extends [`agents/projectk/planner_executor.py`](src/minisweagent/agents/projectk/planner_executor.py) |

---

## Headline numbers

Same model (Qwen2.5-Coder-14B, local Ollama), same step budget (30), same
prompt skeleton — only the agent scaffold changes:

| Condition | Resolve | Steps | In tok | Latency |
| --- | ---: | ---: | ---: | ---: |
| Baseline (DefaultAgent) | 33% (1/3) | 21.3 | 41,015 | 331.6 s |
| Scratchpad | 67% (2/3) | 13.3 | 40,548 | 321.0 s |
| Symbol retrieval | 67% (2/3) | 13.0 | 28,959 | 160.6 s |
| Hybrid retrieval (symbol+BM25) | 67% (2/3) | 14.0 | 37,613 | 375.3 s |
| Dynamic-replan planner | 67% (2/3) | 13.3 | 27,109 | 130.5 s |
| **Static planner-executor** | **100% (3/3)** | **6.3** | **7,517** | **42.4 s** |
| **Best-of-N=3 over baseline** | **100% (3/3)** | 4.3 | 6,412 | 41.3 s (winner) |
| **Reflexion (≤2 retries)** | **100% (3/3)** | 7.0 | 4,984 | 24.7 s (winning attempt) |

Detailed analysis, per-instance verdicts, error analysis, and limitations:
[`REPORT.md`](REPORT.md). Standalone slide-ready PDF:
[`PROJECT_K_RESULTS.pdf`](PROJECT_K_RESULTS.pdf).

---

## Repo layout (Project K only)

```
src/minisweagent/projectk/        # this fork's additions
├── minibench/
│   ├── runner.py                 # tmpdir + apply patch + re-run tests
│   └── fixtures/                 # 5 toy bugs
├── retrieval.py                  # SymbolIndex + BM25Index + EmbeddingIndex
├── bestofn.py                    # Best-of-N sampling with test verifier
├── metrics.py                    # 8 metrics aggregator
├── failure_analysis.py           # 8-bucket taxonomy
├── throttled_model.py            # litellm wrapper, key rotation
├── dataset.py                    # curated SWE-Bench-Lite selector
├── cli_*.py                      # CLI entrypoints
src/minisweagent/agents/projectk/ # scaffold agent classes
├── planner_executor.py
├── scratchpad.py
├── retrieval_agent.py
└── reflexion.py
src/minisweagent/config/projectk/ # YAML configs (one per backend × variant)
tests/projectk/                   # 41 tests, all green
scripts/                          # run_ablation.py, run_optimizations_ablation.py, demo.sh, build_results_pdf.py
Makefile                          # 11 reproducible targets
REPORT.md                         # full write-up
PROJECT_K_RESULTS.pdf             # standalone results document
```

---

## Acknowledgements & upstream

This work is built on the open-source
[**mini-swe-agent**](https://github.com/SWE-agent/mini-swe-agent) framework
by the Princeton/Stanford team behind
[SWE-bench](https://github.com/swe-bench/SWE-bench) and
[SWE-agent](https://github.com/SWE-agent/SWE-agent). The base agent loop,
model abstractions, environment classes, and SWE-bench batch runner are
inherited unchanged from upstream; everything under `*/projectk/`,
`tests/projectk/`, `Makefile`, `REPORT.md`, `PROJECT_K_RESULTS.pdf`, and
`scripts/` is this project's contribution.

If you use the upstream agent in academic work, please cite the SWE-agent
paper:

```bibtex
@inproceedings{yang2024sweagent,
  title={{SWE}-agent: Agent-Computer Interfaces Enable Automated Software Engineering},
  author={Yang, John and Jimenez, Carlos E. and Wettig, Alexander and
          Lieret, Kilian and Yao, Shunyu and Narasimhan, Karthik R. and Press, Ofir},
  booktitle={NeurIPS},
  year={2024},
  url={https://arxiv.org/abs/2405.15793}
}
```

---

## License

Inherits the MIT license from upstream `mini-swe-agent`. See
[LICENSE.md](LICENSE.md).
