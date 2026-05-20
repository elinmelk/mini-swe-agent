# Project K targets. Run `make help` for an overview.
#
# Conventions:
#   * Anything under `runs/` is reproducible from these targets.
#   * Targets are idempotent — re-running just overwrites the output dir.

SHELL := /usr/bin/env bash
PY ?= python
PYTEST ?= python -m pytest
VENV ?= .venv
ACTIVATE = source $(VENV)/bin/activate
EXPORTS = export MSWEA_SILENT_STARTUP=1 LITELLM_LOG=ERROR
ROOT := $(shell pwd)

OLLAMA_CFG = src/minisweagent/config/projectk/ollama.yaml
RETRIEVAL_CFG = src/minisweagent/config/projectk/ollama_retrieval.yaml
PLANNER_CFG = src/minisweagent/config/projectk/ollama_planner.yaml
SCRATCHPAD_CFG = src/minisweagent/config/projectk/ollama_scratchpad.yaml
GROQ_CFG = src/minisweagent/config/projectk/groq.yaml
MODEL ?= ollama_chat/qwen2.5-coder:14b

.PHONY: help
help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Project K targets\n\n"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: install
install: ## Create venv + install package + pytest (one-time setup).
	uv venv --python 3.13 $(VENV)
	$(ACTIVATE) && uv pip install -e . pytest pytest-asyncio

.PHONY: ollama
ollama: ## Install + start Ollama and pull qwen2.5-coder:14b.
	@command -v ollama >/dev/null || brew install ollama
	@brew services list | grep -q '^ollama .* started' || brew services start ollama
	@until curl -s http://localhost:11434/api/version >/dev/null; do sleep 1; done
	@ollama pull qwen2.5-coder:14b

.PHONY: test
test: ## Run the Project K pytest suite (29 unit + integration tests).
	$(ACTIVATE) && $(EXPORTS) && $(PYTEST) tests/projectk -q

.PHONY: demo
demo: ## Single-fixture demo on Ollama (~1 min). Outputs: runs/demo
	$(ACTIVATE) && $(EXPORTS) && \
	rm -rf runs/demo /tmp/projk-demo && mkdir -p /tmp/projk-demo && \
	cp -r src/minisweagent/projectk/minibench/fixtures/toy__add-sign /tmp/projk-demo/ && \
	projectk-mini -o runs/demo -c $(OLLAMA_CFG) --fixtures /tmp/projk-demo --model $(MODEL)

.PHONY: mini
mini: ## Full 5-fixture run on Ollama (~10 min). Outputs: runs/mini
	$(ACTIVATE) && $(EXPORTS) && \
	projectk-mini -o runs/mini -c $(OLLAMA_CFG) --model $(MODEL)

.PHONY: ablation
ablation: ## Run the 4-condition ablation grid (baseline / scratchpad / planner / retrieval).
	$(ACTIVATE) && $(EXPORTS) && \
	$(PY) scripts/run_ablation.py --output runs/ablation --model $(MODEL)

.PHONY: report
report: ## Re-print metrics + failure taxonomy for an existing run dir (use DIR=runs/...).
	@if [ -z "$(DIR)" ]; then echo "Usage: make report DIR=runs/demo"; exit 2; fi
	$(ACTIVATE) && $(EXPORTS) && projectk-report $(DIR) --report $(DIR)/report.json

.PHONY: compare-providers
compare-providers: ## Open-weight provider comparison: Ollama (local) vs Groq (cloud).
	$(ACTIVATE) && $(EXPORTS) && \
	projectk-mini-compare -o runs/providers \
	  --pair "ollama_qwen=ollama_chat/qwen2.5-coder:14b=$(OLLAMA_CFG)" \
	  --pair "groq_llama70b=groq/llama-3.3-70b-versatile=$(GROQ_CFG)"

.PHONY: clean
clean: ## Remove all runs/ outputs.
	rm -rf runs/

.PHONY: clean-pycache
clean-pycache: ## Strip __pycache__ everywhere under src/ and tests/.
	find src tests -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
