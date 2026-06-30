"""Centralized LLM configuration for the entire project.

Switching the local engine, model, or provider only requires editing this file
or setting environment variables. Provider selection is:

* ``LLM_PROVIDER=openai``                -> always cloud OpenAI
* ``LLM_PROVIDER=mlx`` / ``local`` / ``ollama`` -> always local MLX endpoint
* unset                                   -> cloud iff a real ``OPENAI_API_KEY`` is present

All clients are created with an explicit request timeout and bounded retries so a
stalled server fails a node cleanly instead of hanging the whole pipeline.
"""

from __future__ import annotations

import os
import threading

from langchain_openai import ChatOpenAI

# --- Engine configuration (env-overridable) ---
MLX_BASE_URL = os.getenv("MLX_BASE_URL", "http://localhost:8000/v1")
MLX_MODEL = os.getenv("MLX_MODEL", "Qwen3.6-35B-A3B-UD-MLX-4bit")
MLX_API_KEY = os.getenv("MLX_API_KEY", "EMPTY")  # MLX does not require authentication

OPENAI_CLOUD_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_REASONING_MODEL = os.getenv("OPENAI_REASONING_MODEL", "o3-mini")

LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))


def _use_cloud() -> bool:
    """Decide whether to use cloud OpenAI vs the local MLX endpoint."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "openai":
        return True
    if provider in ("mlx", "local", "ollama"):
        return False
    key = os.environ.get("OPENAI_API_KEY")
    return bool(key) and key != "your_openai_api_key_here"


def get_llm(
    temperature: float = 0.7,
    thinking: bool = False,
    max_tokens: int = 8192,
) -> ChatOpenAI:
    """Return an LLM configured for the current environment.

    Args:
        temperature: 0.0-1.0. Use 0.3 for precise tasks (QA, Reviewer),
                     0.7 for generation (Developer, Designer),
                     1.0 for creativity (PM, Architect).
        thinking: If True, activates reasoning mode (cloud: o-series model;
                  local: ``enable_thinking`` extra body flag).
        max_tokens: Token limit in the response (ignored for o-series reasoning
                    models, which manage their own budget).
    """
    if _use_cloud():
        openai_key = os.environ["OPENAI_API_KEY"]
        if thinking:
            # o-series reasoning models require temperature=1.0 and manage tokens
            # internally; passing max_tokens here can be rejected by the API.
            return ChatOpenAI(
                api_key=openai_key,
                model=OPENAI_REASONING_MODEL,
                temperature=1.0,
                timeout=LLM_REQUEST_TIMEOUT,
                max_retries=LLM_MAX_RETRIES,
            )
        return ChatOpenAI(
            api_key=openai_key,
            model=OPENAI_CLOUD_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=LLM_REQUEST_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )

    # Local MLX endpoint.
    return ChatOpenAI(
        base_url=MLX_BASE_URL,
        api_key=MLX_API_KEY,
        model=MLX_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=LLM_REQUEST_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
        extra_body={"enable_thinking": thinking},
    )


# --- Per-agent instances ---
# Each agent role has a preset. Instances are cached per-thread (threading.local)
# so we avoid rebuilding a ChatOpenAI + HTTP client on every attribute access,
# while still giving each LangGraph worker thread its own client (which is what
# the Pydantic concurrency constraint actually requires).

_AGENT_PRESETS: dict[str, dict] = {
    "llm_pm": dict(temperature=1.0, thinking=False),
    "llm_architect": dict(temperature=1.0, thinking=False),
    "llm_designer": dict(temperature=0.7, thinking=False),
    "llm_developer": dict(temperature=0.7, thinking=False),
    "llm_qa": dict(temperature=0.3, thinking=False),
    "llm_reviewer": dict(temperature=0.3, thinking=True, max_tokens=16384),
}

_thread_local = threading.local()

# Type-only declarations so callers/IDEs see these attributes.
llm_pm: ChatOpenAI
llm_architect: ChatOpenAI
llm_designer: ChatOpenAI
llm_developer: ChatOpenAI
llm_qa: ChatOpenAI
llm_reviewer: ChatOpenAI


def __getattr__(name: str) -> ChatOpenAI:
    preset = _AGENT_PRESETS.get(name)
    if preset is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    cache = getattr(_thread_local, "cache", None)
    if cache is None:
        cache = {}
        _thread_local.cache = cache
    if name not in cache:
        cache[name] = get_llm(**preset)
    return cache[name]
