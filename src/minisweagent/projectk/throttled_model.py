"""LitellmModel variants that:

  1. Respect a configurable minimum interval between requests, so we don't
     hammer a rate-limited endpoint (e.g. NVIDIA NIM free tier).
  2. Optionally round-robin across a pool of API keys, so two free-tier
     accounts give roughly twice the throughput.

Two flavors mirror upstream:
  ThrottledLitellmTextbasedModel   — for our default projectk yaml configs
  ThrottledLitellmModel            — for tool-call style configs

All instances share a single process-global gate (keyed by model name) and a
single key-rotation counter, so concurrent workers don't stampede the endpoint.

Tune via env vars (overridable per-instance via the model config):
  MSWEA_MIN_REQUEST_INTERVAL_S   default: 0
  NVIDIA_NIM_API_KEY,
  NVIDIA_NIM_API_KEY_2..N        rotated round-robin if more than one is set.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from collections import defaultdict

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel, LitellmTextbasedModelConfig

logger = logging.getLogger("projectk.throttle")

_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)
_LAST_CALL: dict[str, float] = {}


def _gate(model_name: str, interval: float) -> None:
    if interval <= 0:
        return
    lock = _LOCKS[model_name]
    with lock:
        last = _LAST_CALL.get(model_name, 0.0)
        wait = interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[model_name] = time.monotonic()


def _provider_prefix(model_name: str) -> str:
    return model_name.split("/", 1)[0] if "/" in model_name else ""


def _collect_keys(env_prefix: str) -> list[str]:
    """Return [VAR, VAR_2, VAR_3, ...] keys present in the environment."""
    keys: list[str] = []
    primary = os.environ.get(env_prefix)
    if primary:
        keys.append(primary)
    i = 2
    while True:
        val = os.environ.get(f"{env_prefix}_{i}")
        if not val:
            break
        keys.append(val)
        i += 1
    return keys


_PROVIDER_KEY_ENV = {
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class _KeyPool:
    """Process-global round-robin key pool, keyed by model name."""

    _pools: dict[str, "_KeyPool"] = {}
    _pool_lock = threading.Lock()

    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self._cycle = itertools.cycle(self.keys) if self.keys else None
        self._lock = threading.Lock()

    @classmethod
    def for_model(cls, model_name: str) -> "_KeyPool | None":
        provider = _provider_prefix(model_name)
        env_var = _PROVIDER_KEY_ENV.get(provider)
        if env_var is None:
            return None
        with cls._pool_lock:
            pool = cls._pools.get(env_var)
            if pool is None:
                keys = _collect_keys(env_var)
                pool = cls(keys)
                cls._pools[env_var] = pool
                if len(keys) > 1:
                    logger.info(
                        "Rotating %d %s API keys for model %s", len(keys), provider, model_name
                    )
            return pool

    def next(self) -> str | None:
        if not self._cycle:
            return None
        with self._lock:
            return next(self._cycle)


class _ThrottleConfigMixin:
    min_request_interval_s: float = float(os.getenv("MSWEA_MIN_REQUEST_INTERVAL_S", "0"))
    rotate_api_keys: bool = True


class ThrottledLitellmModelConfig(_ThrottleConfigMixin, LitellmModelConfig):
    pass


class ThrottledLitellmTextbasedModelConfig(_ThrottleConfigMixin, LitellmTextbasedModelConfig):
    pass


def _kwargs_with_rotated_key(model_name: str, kwargs: dict) -> dict:
    pool = _KeyPool.for_model(model_name)
    if pool is None or len(pool.keys) < 2:
        return kwargs
    next_key = pool.next()
    if next_key is None:
        return kwargs
    new_kwargs = dict(kwargs)
    new_kwargs.setdefault("api_key", next_key)
    new_kwargs["api_key"] = next_key
    return new_kwargs


def _is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "429" in str(exc)


def _query_with_key_rotation(_super_query, model_name: str, rotate: bool, messages, kwargs):
    """Run _super_query once per key in the pool; on 429 immediately try the next key.

    If the pool only has 1 key (or rotation is disabled), just calls once and
    lets the outer tenacity retry handle backoff.
    """
    pool = _KeyPool.for_model(model_name) if rotate else None
    keys = pool.keys if pool else []
    if len(keys) < 2:
        return _super_query(messages, **kwargs)

    last_exc: Exception | None = None
    # try each key at most once per outer-retry attempt
    for _ in range(len(keys)):
        next_key = pool.next()
        if next_key is None:
            break
        attempt_kwargs = dict(kwargs)
        attempt_kwargs["api_key"] = next_key
        try:
            return _super_query(messages, **attempt_kwargs)
        except Exception as e:
            if not _is_rate_limit(e):
                raise
            last_exc = e
            logger.warning("429 on key ending ...%s, trying next key", next_key[-6:])
    assert last_exc is not None
    raise last_exc


class ThrottledLitellmModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=ThrottledLitellmModelConfig, **kwargs)

    def _query(self, messages, **kwargs):
        _gate(self.config.model_name, self.config.min_request_interval_s)
        return _query_with_key_rotation(
            super()._query, self.config.model_name, self.config.rotate_api_keys, messages, kwargs
        )


class ThrottledLitellmTextbasedModel(LitellmTextbasedModel):
    def __init__(self, **kwargs):
        from minisweagent.models.litellm_model import LitellmModel as _Base

        _Base.__init__(self, config_class=ThrottledLitellmTextbasedModelConfig, **kwargs)

    def _query(self, messages, **kwargs):
        _gate(self.config.model_name, self.config.min_request_interval_s)
        return _query_with_key_rotation(
            super()._query, self.config.model_name, self.config.rotate_api_keys, messages, kwargs
        )
