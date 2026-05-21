"""ReflexionAgent — retry on failure with a self-critique of the prior attempt.

Following Shinn et al. (2023), we keep the outer agent loop intact and add an
outer retry loop. After each attempt:

  * If the agent submitted a patch that resolves the bug, we are done.
  * Otherwise, ask the LM to produce a short self-critique of what went wrong.
  * Restart the agent with that critique injected into the system prompt as
    a "previous attempts" block, up to `max_retries` times.

This is the "verbal RL" half of Reflexion. We do not implement policy updates;
the critique is a one-shot context-injection signal.
"""

from __future__ import annotations

import logging
from typing import Any

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded

logger = logging.getLogger("agent.reflexion")


_DEFAULT_CRITIC_PROMPT = """You are reviewing a failed software engineering attempt.

The original task was:
<task>
{task}
</task>

The agent's recent commands and outputs were:
<trajectory>
{tail}
</trajectory>

Final exit status: {exit_status}
Final submission length: {submission_length} chars

In 4-6 short bullet points, explain what went wrong and what a retry should do
differently. Be specific about file paths, function names, and any non-obvious
gotcha (e.g. quoting, OS portability, wrong fixture). Do not write any code.
End your response with the literal line: DONE_CRITIQUE
"""


class ReflexionAgentConfig(AgentConfig):
    max_retries: int = 2
    """Maximum number of retries after a failed attempt (so total tries = 1 + max_retries)."""
    critic_prompt_template: str = _DEFAULT_CRITIC_PROMPT
    tail_chars: int = 3000
    """Characters of trajectory tail to feed the critic."""


class ReflexionAgent(DefaultAgent):
    """DefaultAgent with an outer retry-on-failure loop driven by a self-critic."""

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=ReflexionAgentConfig, **kwargs)
        self.attempts: list[dict[str, Any]] = []
        self.critiques: list[str] = []

    # ---------------------------------------------------------------- critic
    def _critic_complete(self, prompt: str) -> str:
        """Free-form completion (bypass action parsing) so the critic can write prose."""
        import litellm

        cfg = self.model.config
        kwargs = dict(getattr(cfg, "model_kwargs", {}) or {})
        kwargs.setdefault("drop_params", True)
        response = litellm.completion(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _build_critique(self, task: str, info: dict[str, Any]) -> str:
        # Render trajectory tail
        chunks: list[str] = []
        for msg in self.messages[-12:]:
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            label = "AGENT" if role == "assistant" else "OBS"
            chunks.append(f"[{label}]\n{content[:600]}")
        tail = "\n\n".join(chunks)[-self.config.tail_chars:]

        prompt = self.config.critic_prompt_template.format(
            task=task,
            tail=tail,
            exit_status=info.get("exit_status", ""),
            submission_length=len(info.get("submission") or ""),
        )
        raw = self._critic_complete(prompt)
        return raw.split("DONE_CRITIQUE")[0].strip()

    # ----------------------------------------------------------------- run
    def _attempt_succeeded(self, info: dict[str, Any]) -> bool:
        # We treat "submitted a non-empty patch" as success at the agent level.
        # External verification (does the patch actually resolve?) is handled by
        # the minibench runner / projectk-report against ground truth.
        return bool((info.get("submission") or "").strip()) and info.get("exit_status") in {"Submitted", "Exit"}

    def run(self, task: str = "", **kwargs) -> dict:
        self.extra_template_vars |= {"task": task, **kwargs}
        last_info: dict[str, Any] = {}

        for attempt in range(1 + self.config.max_retries):
            # Reset per-attempt state; preserve cumulative attempts log.
            self.messages = []
            self.n_calls = 0
            # Inject prior critiques (if any) so the agent sees its own failure history.
            if self.critiques:
                joined = "\n\n".join(
                    f"<previous_attempt id=\"{i+1}\">\n{c}\n</previous_attempt>"
                    for i, c in enumerate(self.critiques)
                )
                self.extra_template_vars["reflexion_history"] = joined
            else:
                self.extra_template_vars["reflexion_history"] = ""

            logger.info(f"reflexion: attempt {attempt+1}/{1+self.config.max_retries}")
            try:
                info = super().run(task=task, **kwargs)
            except (InterruptAgentFlow, LimitsExceeded) as e:
                info = {"exit_status": type(e).__name__, "submission": ""}
            last_info = info
            self.attempts.append({"attempt": attempt + 1, **info})

            if self._attempt_succeeded(info):
                return info

            # Generate critique only if we have retries left
            if attempt < self.config.max_retries:
                try:
                    critique = self._build_critique(task, info)
                    logger.info(f"reflexion critique:\n{critique[:400]}")
                    self.critiques.append(critique)
                except Exception as e:
                    logger.warning(f"reflexion critic failed: {e}")
                    self.critiques.append(f"(critic failed: {e})")

        return last_info
