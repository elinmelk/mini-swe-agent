"""Agent variant with a persistent on-disk scratchpad surfaced to the LM each step.

The scratchpad is a single text file the model can append to with shell commands.
After every step, its current contents are appended to the observation message so
the planner has a stable place to "remember" hypotheses, files-of-interest, and
failed attempts across many steps.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from pydantic import BaseModel

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent


class ScratchpadAgentConfig(AgentConfig):
    scratchpad_path: str = "/tmp/scratchpad.md"
    scratchpad_max_chars: int = 2000


class _Config(ScratchpadAgentConfig, BaseModel):
    pass


class ScratchpadAgent(DefaultAgent):
    """DefaultAgent that re-reads a small persistent scratchpad after each step.

    The scratchpad lives inside the agent's `env` (so for Docker runs it lives in
    the container at `scratchpad_path`). Its contents are summarized back to the
    LM at the end of each turn so the planner can rely on it as durable memory.
    """

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=ScratchpadAgentConfig, **kwargs)
        self.extra_template_vars["scratchpad_path"] = self.config.scratchpad_path
        self.extra_template_vars["scratchpad_run_id"] = uuid.uuid4().hex[:8]

    def _read_scratchpad(self) -> str:
        path = self.config.scratchpad_path
        out = self.env.execute({"command": f"test -f {path} && tail -c {self.config.scratchpad_max_chars} {path} || true"})
        return out.get("output", "").strip()

    def execute_actions(self, message: dict) -> list[dict]:
        messages = super().execute_actions(message)
        contents = self._read_scratchpad()
        if not contents or not messages:
            return messages
        rendered = (
            f"\n\n<scratchpad path=\"{self.config.scratchpad_path}\">\n"
            f"{contents}\n</scratchpad>"
        )
        last = messages[-1]
        if isinstance(last.get("content"), str):
            last["content"] = last["content"] + rendered
        return messages
