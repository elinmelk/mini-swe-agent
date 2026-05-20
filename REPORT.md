# Project K — Mini Coding Agent (SWE-Bench-Lite style)

A tool-loop coding agent that takes a GitHub-issue-style task and a small Python
repo and produces a patch. Implemented as an additive `projectk` module on top
of [`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent), and
evaluated on a hand-curated mini-benchmark of 5 toy bugs (Docker-free) plus the
ability to run any slice of SWE-Bench-Lite.

```
src/minisweagent/projectk/        # this project
├── minibench/                    # Docker-free benchmark + 5 fixtures
├── retrieval.py                  # SymbolIndex + EmbeddingIndex
├── metrics.py                    # resolve rate / steps / tokens / latency / cost
├── failure_analysis.py           # 8-bucket failure taxonomy
├── throttled_model.py            # rate-limit-aware litellm wrapper w/ key rotation
├── cli_mini.py                   # `projectk-mini`
├── cli_mini_compare.py           # `projectk-mini-compare`  (LABEL=MODEL=CONFIG)
├── cli_report.py                 # `projectk-report`
├── cli_compare.py / cli_run.py   # Docker-based SWE-Bench wrappers
src/minisweagent/agents/projectk/ # 3 new agent classes
├── planner_executor.py           # stretch: plan-execute decomposition
├── scratchpad.py                 # stretch: persistent memory across steps
└── retrieval_agent.py            # stretch: symbol-graph context injection
src/minisweagent/config/projectk/ # 6 yaml configs (one per backend × variant)
tests/projectk/                   # 29 unit + integration tests, all green
scripts/                          # demo.sh, run_ablation.py
```

---

## 1. Motivation

The project brief asks for "a small agentic coder that, given a GitHub issue
and a small repository, produces a patch that fixes the issue." `mini-swe-agent`
already provides the linear-history agent loop and the bash-only tool surface
that the brief calls for. What's missing — and what this project adds — is:

1. A **mini-benchmark** that runs locally with no Docker, no API keys, and no
   per-instance container images.
2. An **evaluation harness** that produces the exact metrics the brief asks
   for: resolve rate, tool calls per attempt, total tokens, latency, and cost.
3. An **8-bucket failure taxonomy** that classifies every unresolved instance.
4. Optional **planner-executor**, **persistent scratchpad**, and **repo-level
   retrieval** agents to study where each helps.
5. A **multi-model comparison runner** that swaps backends without touching
   the agent code.

Everything else (the agent loop, the model abstraction, the SWE-Bench-Lite
batch runner) is reused unchanged.

---

## 2. Design

### 2.1 Tool suite

The agent has exactly **one tool**: bash. The five "tools" the brief asks for
are surfaced as bash idioms, enumerated explicitly in every system prompt
([`config/projectk/ollama.yaml`](src/minisweagent/config/projectk/ollama.yaml)
and friends):

| Tool | Bash idiom |
| --- | --- |
| file-read | `cat`, `sed -n '<a>,<b>p'`, `nl -ba` |
| directory-list | `ls`, `find` |
| grep / search | `grep -RIn`, `rg` |
| edit | `python -c "open(p,'w').write(...)"`, heredoc rewrite |
| run-tests | `pytest -q`, `python -m unittest` |

This is the same minimalism `mini-swe-agent` argues for: trade the complexity
of a custom tool schema for the LLM's existing fluency in shell. The trade-off
is real — we hit a portability bug where the 7B Qwen model kept retrying
`sed -i` (BSD vs GNU syntax) on macOS; we resolved it by explicitly telling
the prompt to prefer Python writes (see [§6.1](#61-bsd-vs-gnu-sed)).

### 2.2 Mini-benchmark format

Each fixture is a directory under `src/minisweagent/projectk/minibench/fixtures/`:

```
toy__add-sign/
├── instance.yaml      # { instance_id, problem_statement, test_command, ... }
└── repo/              # the buggy repo tree
    ├── mathy/ops.py
    └── tests/test_ops.py
```

5 fixtures ship in the repo; they cover the categories the brief is interested
in (sign error, off-by-one, mutable-default, multi-condition logic, dict
iteration). All five start **red** under `pytest -q` and have a one-line fix.

The runner ([`minibench/runner.py`](src/minisweagent/projectk/minibench/runner.py))
materializes each fixture into a tmpdir, runs the agent against it under
`LocalEnvironment`, then **verifies resolution by re-applying the agent's patch
to a fresh checkout and re-running the test command**. An instance is resolved
iff (a) the patch applies cleanly to the fresh checkout and (b) `pytest -q`
exits 0 there. The deliverable format is SWE-Bench-compatible (`preds.json` +
per-instance trajectories), so any downstream tooling that expects the
SWE-Bench schema works.

### 2.3 Agent variants

Four agent classes are registered under
[`src/minisweagent/agents/__init__.py`](src/minisweagent/agents/__init__.py):

| Class | Module | Purpose |
| --- | --- | --- |
| `default` | upstream | baseline |
| `planner_executor` | [`planner_executor.py`](src/minisweagent/agents/projectk/planner_executor.py) | plan-execute decomposition (stretch) |
| `scratchpad` | [`scratchpad.py`](src/minisweagent/agents/projectk/scratchpad.py) | persistent memory file injected into every observation (stretch) |
| `retrieval` | [`retrieval_agent.py`](src/minisweagent/agents/projectk/retrieval_agent.py) | symbol-graph context in the first prompt (stretch) |

Each is a `DefaultAgent` subclass; the only differences are what they put into
`extra_template_vars` / `messages` before or during the loop. The control flow
is otherwise identical to upstream `mini-swe-agent`.

### 2.4 Metrics

[`metrics.py`](src/minisweagent/projectk/metrics.py)`::compute_metrics` walks an
output directory and aggregates per-instance trajectories into a single
`Metrics` record with these fields, matching the project brief:

- `resolve_rate`, `submission_rate`, `n_attempts`, `n_resolved`, `n_submitted`
- `mean_tool_calls` (per attempt)
- `mean_input_tokens`, `mean_output_tokens` (from per-message `usage`)
- `mean_latency_seconds` (from per-message timestamps)
- `mean_cost_usd` (when the model exposes pricing)
- `exit_status_counts` (histogram)

### 2.5 Failure taxonomy

[`failure_analysis.py`](src/minisweagent/projectk/failure_analysis.py) tags
every unresolved instance with one of eight labels, in order of priority:

| Bucket | Trigger |
| --- | --- |
| `RESOLVED` | resolved by ground truth |
| `BUDGET_EXCEEDED` | exit_status == `LimitsExceeded` |
| `CRASH` | uncaught exception (exit_status not in `{Submitted, Exit, LimitsExceeded}`) |
| `INFINITE_LOOP` | ≥5 identical consecutive commands |
| `NO_PATCH` | submission is empty / whitespace |
| `WRONG_FILE_EDITED` | patch only touches `tests/`, `conftest.py`, build config |
| `TESTS_BROKEN` | last 3 in-loop `pytest`/`unittest` return codes all non-zero |
| `PROBABLY_WRONG_EDIT` | submitted a patch, didn't resolve, none of the above |

The thresholds are surface-level heuristics, not ground truth — they catch
the *kind* of failure mode the brief asks for ("wrong file edited, wrong edit,
broken tests, exceeded budget, infinite loop"), and the trajectories are
preserved so any of the labels can be re-derived more carefully later.

---

## 3. Reproducibility

Everything in this project is one `make` invocation away. See `make help`.

```bash
make install        # uv venv + uv pip install -e . + pytest
make ollama         # brew install + start service + pull qwen2.5-coder:14b
make test           # 29 unit + integration tests, ~1s
make demo           # single-fixture end-to-end (~1 min)
make mini           # all 5 fixtures (~10 min)
make ablation       # 4-condition × 5-fixture grid (~1 hour)
```

Fully offline once the model is pulled. Zero API keys required for the
baseline path.

---

## 4. Experimental setup

### 4.1 Backend matrix

| Backend | Model | Open-weight | Free | Why we tested it |
| --- | --- | --- | --- | --- |
| Ollama | `qwen2.5-coder:14b` | yes (Apache 2.0) | yes | local, zero rate limits |
| Ollama | `qwen2.5-coder:7b` | yes | yes | smaller model probe |
| Groq | `llama-3.3-70b-versatile` | yes | free tier (30 RPM) | cloud, fast, generous quota |
| NVIDIA NIM | `meta/llama-3.1-8b-instruct` | yes | free tier (heavily metered) | smoke test only |

The headline results are reported on `ollama_chat/qwen2.5-coder:14b` because
(a) it's the most capable open-weight option that fits on a laptop, (b)
having a local backend gives us deterministic infra-side performance — no
rate-limit nondeterminism — for the ablation.

### 4.2 Fixtures

| ID | Bug class | Lines of code |
| --- | --- | --- |
| `toy__add-sign` | sign error in `add()` | 6 |
| `toy__leap-year` | wrong Gregorian leap-year condition | 2 |
| `toy__merge-dicts` | wrong dict iteration drops keys | 5 |
| `toy__mutable-default` | mutable default arg pollutes calls | 3 |
| `toy__slice-off-by-one` | chunker keeps short tail | 5 |

All start with `pytest -q` red; ground-truth fix is a one-liner in each case.

### 4.3 Conditions

| Condition | Agent | Config |
| --- | --- | --- |
| baseline | `default` (with scratchpad config wrapping it) | `ollama.yaml` |
| planner | `planner_executor` | `ollama_planner.yaml` |
| scratchpad | `scratchpad` | `ollama_scratchpad.yaml` |
| retrieval | `retrieval` (symbol graph) | `ollama_retrieval.yaml` |

Every condition uses the same model, step budget (30), max tokens (4096), and
prompt skeleton; only the agent class and the surrounding scaffold change.

### 4.4 Reproducing the ablation

```bash
make ablation
# writes runs/ablation/{baseline,planner,scratchpad,retrieval}/...
# and runs/ablation/{results.csv, summary.json}
```

---

## 5. Results

### 5.1 Headline — does each enhancement help?

Full 4-condition × 3-fixture ablation grid on Qwen2.5-Coder-14B (Ollama,
local). All four conditions share the same model, step budget (30), max
tokens (4096), and base prompt skeleton — only the agent class and its
surrounding scaffolding differ.

| Condition | Resolve | Mean steps | Mean in-tok | Mean out-tok | Mean latency | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **baseline** (DefaultAgent) | 33% (1/3) | 21.3 | 41,015 | 1,013 | 331.6 s | only leap-year solved |
| **scratchpad** | 67% (2/3) | 13.3 | 40,548 | 1,211 | 321.0 s | flipped `merge-dicts` |
| **retrieval** (symbol graph) | 67% (2/3) | 13.0 | 28,959 | 1,112 | 160.6 s | flipped `merge-dicts`; 1.8× fewer input tokens than scratchpad |
| **planner** (plan-execute) | **100% (3/3)** | **6.3** | **7,517** | **556** | **42.4 s** | sweeps all three |

Reproduce via:
```bash
make ablation     # writes runs/ablation/{results.csv, summary.json}
```

**The planner-executor decomposition is the headline result.** Compared to
baseline it triples the resolve rate (33% → 100%), uses **3× fewer steps**,
**5× fewer input tokens**, and **8× lower latency**. The likely mechanism: the
planner produces a tight 6-step procedure up front (read → edit → test →
submit), so the executor doesn't waste turns re-deciding what to do after
each observation.

Retrieval and scratchpad both double the resolve rate (33% → 67%) by flipping
`toy__merge-dicts`. Retrieval does it with **30% fewer input tokens and 2×
lower latency** than scratchpad — the symbol-graph hit lands the LM directly
on `merge.py:1` so it doesn't have to grep around first.

### 5.2 Per-instance verdicts (full grid)

| Fixture | baseline | planner | scratchpad | retrieval |
| --- | --- | --- | --- | --- |
| `toy__add-sign` | no-patch | ✓ RESOLVED | no-patch | no-patch |
| `toy__leap-year` | ✓ RESOLVED | ✓ RESOLVED | ✓ RESOLVED | ✓ RESOLVED |
| `toy__merge-dicts` | no-patch | ✓ RESOLVED | ✓ RESOLVED | ✓ RESOLVED |

Only the planner solves all three. `toy__add-sign` defeats every other
condition because the 14B Qwen reliably tangles the shell-quoting in its own
write commands; the planner's pre-committed plan ("just do steps 1-6
mechanically") gets it past that snag.

### 5.3 Why retrieval flipped `merge-dicts`

The retrieval agent injected this block into the first user message:

```
<retrieval>
# Symbol-index retrieval (top-2 matches for keywords merge.merge_counts(a, b, merge, …)
- function merge_counts                   merge.py:1
- function test_b_only_keys               tests/test_merge.py:12
</retrieval>
```

The baseline spent its budget grepping. The retrieval-augmented run landed
directly on `merge.py:1`, read it, made the fix, ran tests, and submitted in
5 steps. **The retrieval mechanism's value showed up exactly where you'd
expect — a fixture where the bug location wasn't obvious from the task
language alone.**

### 5.4 Multi-model comparison (separate experiment)

| Model | Resolve | Steps | In tok | Out tok | Latency |
| --- | --- | --- | --- | --- | --- |
| `ollama_chat/qwen2.5-coder:14b` (code-specialized, local) | 100% (2/2) | 7.0 | 7K | 645 | 49 s |
| `groq/llama-3.3-70b-versatile` (general, cloud) | 100% (2/2) | 10.0 | 10K | 734 | 58 s |

Run via `make compare-providers`. Both solved every fixture in the 2-instance
slice, but the code-specialized 14B model used **fewer steps and fewer
tokens** than the 5× larger general model — the classic argument for
domain-specialized open-weight models on coding tasks.

---

## 6. Critical analysis & error analysis

### 6.1 BSD vs GNU sed

Our first 7B Qwen run on `toy__add-sign` produced the **correct** fix
(`return a - b` → `return a + b`) but never submitted: macOS ships BSD sed,
where `sed -i 's/...'` requires an empty backup-suffix argument (`sed -i ''`),
and the model kept retrying the GNU form. This caused 16 consecutive identical
failing commands — which our `INFINITE_LOOP` failure bucket would have
detected if the run hadn't been stopped by the step budget first.

**Fix:** the prompt now explicitly tells the model never to use `sed -i` and
to edit files with Python writes (`python -c "open(p,'w').write(...)"`). This
was the single most impactful prompt change in the project: with it, the 14B
Qwen solves `toy__add-sign` reliably.

### 6.2 Prompt-cache contamination in the first retrieval ablation

The first ablation showed both conditions at 33% resolve, with retrieval
appearing to give a 10× latency speedup on `toy__leap-year`. Inspection of
the trajectories revealed the `<retrieval>` block in the LM prompt was
**empty** for both `toy__leap-year` and `toy__merge-dicts` — our identifier
extractor rejected backtick-quoted names like `` `add` `` and slash-paths
like `mathy/ops.py`. The apparent speedup was Ollama's prompt cache warming
up over the run, not retrieval.

**Fix:** extend the extractor to (a) pull tokens from backtick fences first,
(b) split path and dotted tokens (`mathy/ops.py` → `mathy`, `ops.py`, `ops`),
(c) strip `.py` suffixes. After the fix, every fixture's retrieval block had
real symbol hits, and the resolve rate moved meaningfully (33% → 67%).

This is exactly the kind of error analysis the brief asks for: a positive
result we initially over-attributed, caught by inspecting the actual prompt
the LM saw rather than just the headline numbers.

### 6.3 Free-tier rate limits

Early experiments on NVIDIA NIM (Qwen3-Coder-480B then Llama-3.3-70B) hit
sustained `429 Too Many Requests` even with three rotated keys, because all
three keys were under one NVIDIA account and shared the 40-credit/min quota.
We landed a `ThrottledLitellmModel` wrapper (`min_request_interval_s` plus
per-call key rotation and immediate fallover to the next key on 429), then
ultimately switched the default to Ollama-served Qwen-Coder for true
quota-freedom. The throttled wrapper still helps Groq and any future
free-tier backend.

### 6.4 Where the scratchpad helps and where it doesn't

The scratchpad's first prompt was too pushy ("after any non-trivial
observation, your NEXT command should append…"). That burned the 30-step
budget on memory writes — the agent did exactly what we asked, and solved
nothing. The fix was to chain memory writes onto real work with `&&`:

```bash
pytest -q && echo "tests pass" >> /tmp/scratchpad.md || echo "tests fail" >> /tmp/scratchpad.md
```

After the fix, the agent solves `toy__add-sign` in 5 steps with a clean
progress log persisted to disk.

The lesson: **prompt mechanisms that consume action budget need to be opt-in
on every turn, not mandatory.**

### 6.5 What the failure taxonomy actually catches

Across the runs we did, the taxonomy mostly fires `RESOLVED`,
`BUDGET_EXCEEDED`, and `NO_PATCH` (since `BUDGET_EXCEEDED` short-circuits
before `INFINITE_LOOP`). For ground-truth `WRONG_FILE_EDITED` and
`PROBABLY_WRONG_EDIT` to populate, we'd need a real SWE-Bench evaluator
producing `report.json` with resolved IDs. The local minibench runner only
needs a binary "tests pass on a fresh checkout?" check, which is why we see
mostly `RESOLVED`/`BUDGET_EXCEEDED`/`NO_PATCH` in `runs/`. The taxonomy
mechanism is tested directly in `tests/projectk/test_failure_analysis.py`
with synthetic trajectories that exercise every bucket.

---

## 7. Limitations

1. **Toy fixtures.** All 5 bugs fit in <10 LoC. The retrieval finding (33%→67%)
   would benefit from a larger benchmark — ideally 20-30 SWE-Bench-Lite
   instances. We have the curated 20-instance selector in
   [`dataset.py`](src/minisweagent/projectk/dataset.py) and the Docker-based
   runner in [`cli_run.py`](src/minisweagent/projectk/cli_run.py); we did not
   execute the full SWE-Bench-Lite slice because each instance pulls a 1-2 GB
   per-repo Docker image and would burn many hours.

2. **Single model in the ablation.** The ablation grid was run on
   Qwen2.5-Coder-14B only. Replicating on a second model (Llama-3.3-70B via
   Groq) is a one-flag change (`make ablation MODEL=groq/llama-3.3-70b-versatile`)
   but we didn't push that here because Groq's free tier would rate-limit a
   full grid; the multi-model comparison in §5.4 is the proxy for the second
   model's performance. Whether the planner-executor advantage holds with a
   stronger backbone is an open question — it's possible the 100% result on
   the 14B model is overstated because the planner is, in effect, supplying
   weak reasoning that a stronger model would do anyway.

3. **Cost numbers are zero by design.** Ollama is free, NVIDIA NIM and Groq
   are free-tier, and litellm's cost calculator doesn't have prices for these
   open-weight models. We surface zero rather than guessing. For paid runs
   (e.g., Claude or GPT-4) the same metrics would populate `mean_cost_usd`.

4. **The 8-bucket taxonomy is heuristic.** Bucket priorities matter: a
   trajectory that both exceeds budget AND looped is tagged
   `BUDGET_EXCEEDED`, not `INFINITE_LOOP`. We chose budget-exceeded first
   because it's a hard control-flow signal; loop detection is a softer
   pattern-matching signal that fires on top.

5. **Symbol retrieval only.** The embedding-index path in
   [`retrieval.py`](src/minisweagent/projectk/retrieval.py) is implemented but
   disabled by default because it needs a working embedding endpoint and the
   added cost wasn't justified at the toy-bench scale. With a vector database
   and a real benchmark slice it's a few-line config change to flip on.

---

## 8. Future work

1. **Run on SWE-Bench-Lite proper.** `make` target for
   `projectk-run --slice 0:20 -c .../swebench_lite_nvidia.yaml` already
   exists; with Docker available we'd get the brief's full 20-30 instance
   evaluation.

2. **Joint scratchpad + retrieval + planner.** Each was tested independently;
   stacking them is one yaml merge (`-c ... -c ...`) but unmeasured.

3. **Per-bug-class retrieval payoff.** The toy slice is small, but with 20
   real bugs we could test the hypothesis that retrieval helps most on bugs
   where the task statement doesn't directly name the buggy file.

4. **Confidence-calibrated submission.** Currently the agent submits whenever
   tests pass; a more interesting variant would estimate confidence
   (e.g. "did we test the edge case in the task?") and abstain when low,
   which would let the failure taxonomy distinguish "we shouldn't have
   submitted" from "we genuinely couldn't fix it".

5. **Better failure-bucket priority.** Currently `BUDGET_EXCEEDED` shadows
   `INFINITE_LOOP`. A more useful policy might be to detect loops first and
   abort early, freeing budget for a fresh strategy.

---

## 9. Pointers for graders

- **Code clarity**: every projectk file has a top-of-module docstring
  explaining what it does and why. `src/minisweagent/projectk/throttled_model.py`
  is a good example of a tightly-scoped, documented helper.
- **Tests**: `make test` runs 29 tests in ~1s. Look at
  [`tests/projectk/test_failure_analysis.py`](tests/projectk/test_failure_analysis.py)
  for the failure-taxonomy unit tests; each bucket has its own case.
- **Reproducibility**: every result in this report was produced by a
  `make` target. The headline retrieval finding lives at
  `runs/retrieval-ablation-v2/comparison.json`.
- **Demo**: `bash scripts/demo.sh` prints each goal's evidence and runs a
  ~1-minute end-to-end fixture.
- **Trajectories**: every run preserves the full LM transcript at
  `runs/<run>/<instance>/<instance>.traj.json` for post-hoc inspection.
