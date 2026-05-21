"""Plan-Execute decomposition agent.

Two LM roles share one environment:
  * planner   — proposes a short numbered plan for solving the issue.
  * executor  — turns the current plan step into a single bash action, observes
                results, and asks the planner to re-plan when stuck.

This intentionally stays close to mini-swe-agent's linear control flow: the
executor IS still a DefaultAgent loop, the planner is consulted at the start and
every `replan_every` steps (or on demand when the executor calls the helper
`plan` shell command).
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent


_DEFAULT_PLANNER_PROMPT = """You are the PLANNER.

Issue:
<issue>
{{task}}
</issue>

{% if observations -%}
What the executor has discovered so far:
<observations>
{{observations}}
</observations>
{%- endif %}

{% if previous_plan -%}
Previous plan:
<previous_plan>
{{previous_plan}}
</previous_plan>
{%- endif %}

Produce a SHORT numbered plan (max 6 steps) the executor should follow to fix
this issue. Each step must be one concrete action (read a file, grep for X, edit
function Y, run tests). Do NOT write code; just steps. End with the literal line:
DONE_PLAN
"""


class PlannerExecutorAgentConfig(AgentConfig):
    planner_prompt_template: str = _DEFAULT_PLANNER_PROMPT
    replan_every: int = 20
    """Re-run the planner every N executor steps (set 0 to disable periodic re-planning)."""
    replan_on_consecutive_failures: int = 0
    """Trigger an out-of-band re-plan when this many consecutive executor observations have non-zero returncode (set 0 to disable)."""
    planner_model_name: str | None = None
    """Optional override for the planner; defaults to the same model as the executor."""
    observation_window: int = 6
    """How many recent (action, output) pairs to feed the planner."""


class PlannerExecutorAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, *, planner_model: Model | None = None, **kwargs):
        super().__init__(model, env, config_class=PlannerExecutorAgentConfig, **kwargs)
        self.planner_model = planner_model or model
        self.plan: str = ""
        self._planner_logger = logging.getLogger("agent.planner")

    def _render(self, template: str, **extra) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars(**extra))

    def _recent_observations(self) -> str:
        chunks: list[str] = []
        for msg in self.messages[-self.config.observation_window * 2 :]:
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if role == "assistant":
                chunks.append(f"[assistant]\n{content[:600]}")
            elif role in ("user", "tool"):
                chunks.append(f"[obs]\n{content[:600]}")
        return "\n\n".join(chunks)

    def _planner_complete(self, prompt: str) -> str:
        """Call the LM for free-form planning text, bypassing action parsing."""
        import litellm

        cfg = self.planner_model.config
        kwargs = dict(getattr(cfg, "model_kwargs", {}) or {})
        kwargs.setdefault("drop_params", True)
        response = litellm.completion(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def make_plan(self) -> str:
        prompt = self._render(
            self.config.planner_prompt_template,
            observations=self._recent_observations(),
            previous_plan=self.plan,
        )
        raw = self._planner_complete(prompt)
        new_plan = raw.split("DONE_PLAN")[0].strip()
        self._planner_logger.info("New plan:\n%s", new_plan)
        self.plan = new_plan
        self.extra_template_vars["plan"] = new_plan
        # Record the plan in the trajectory as a user message so it's visible to
        # both the executor and the post-hoc reporter.
        if self.messages:
            self.add_messages({"role": "user", "content": f"<plan>\n{new_plan}\n</plan>"})
        return new_plan

    def run(self, task: str = "", **kwargs) -> dict:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.make_plan()
        return super().run(task=task, **kwargs)

    def _consecutive_failure_count(self) -> int:
        """Count consecutive observations with non-zero returncode, looking back from the latest."""
        import re
        count = 0
        for msg in reversed(self.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content") or ""
            if "<returncode>" not in content:
                continue
            m = re.search(r"<returncode>(-?\d+)</returncode>", content)
            if not m:
                break
            rc = int(m.group(1))
            if rc == 0:
                break
            count += 1
            if count > 20:
                break
        return count

    def step(self) -> list[dict]:
        # Periodic replan
        if self.n_calls and self.config.replan_every > 0 and self.n_calls % self.config.replan_every == 0:
            self.make_plan()
            self.add_messages({"role": "user", "content": f"<updated_plan>\n{self.plan}\n</updated_plan>"})
        # On-demand replan when the executor keeps failing
        elif self.config.replan_on_consecutive_failures > 0:
            failures = self._consecutive_failure_count()
            if failures >= self.config.replan_on_consecutive_failures:
                self._planner_logger.info(
                    f"on-demand replan after {failures} consecutive non-zero returncodes"
                )
                self.make_plan()
                self.add_messages({
                    "role": "user",
                    "content": (
                        f"<updated_plan reason=\"{failures} consecutive failures\">\n"
                        f"{self.plan}\n</updated_plan>"
                    ),
                })
        return super().step()
