"""Bring-your-own-key access to a language model. Thin on purpose.

The arena does not ship a model and does not proxy anyone's traffic. You point
it at a provider you already pay for — or at something running on your own
machine — and it sends one prompt per decision.

Anything LiteLLM understands works, because that is all this is: a prefixed
model name, a key from the environment, and a retry. Some examples:

    openai/gpt-5.2                  OPENAI_API_KEY
    anthropic/claude-sonnet-5       ANTHROPIC_API_KEY
    deepseek/deepseek-v4-flash      DEEPSEEK_API_KEY
    ollama/llama4                   nothing — set ARENA_LLM_API_BASE

For a local server that speaks the OpenAI protocol (Ollama, LM Studio, vLLM),
set ``ARENA_LLM_API_BASE`` to its URL; no key is needed and nothing leaves the
machine.
"""
from __future__ import annotations

import os
import time

# How long to wait between retries. Three attempts is enough to ride out a
# rate limit without keeping a human at the table waiting for half a minute.
BACKOFF_S = (0.5, 2.0)
DEFAULT_TIMEOUT_S = 20.0


class LlmUnavailable(RuntimeError):
    """The model could not be reached, or is not configured at all."""


def _litellm():
    try:
        import litellm
    except ImportError as exc:                    # pragma: no cover - install hint
        raise LlmUnavailable(
            "LLM seats need the optional dependency: pip install 'pokerarena[llm]'"
        ) from exc
    return litellm


def api_base() -> str | None:
    """Custom endpoint, if one is configured. Local models live here."""
    return os.environ.get("ARENA_LLM_API_BASE") or None


def complete(model: str, system: str, user: str, *,
             temperature: float = 0.2,
             timeout_s: float = DEFAULT_TIMEOUT_S,
             max_tokens: int = 400) -> str:
    """One completion, with retries. Returns the raw text the model produced.

    Errors are raised rather than swallowed: a seat that silently folds every
    hand because a key expired looks like a bad player, not a broken setup.
    """
    litellm = _litellm()
    kwargs: dict[str, object] = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "timeout": timeout_s,
        "max_tokens": max_tokens,
    }
    base = api_base()
    if base:
        # A local OpenAI-compatible server needs the protocol named explicitly;
        # the prefix in `model` is then just a label for us.
        kwargs["api_base"] = base
        kwargs["custom_llm_provider"] = "openai"
        kwargs.setdefault("api_key", os.environ.get("ARENA_LLM_API_KEY", "local"))

    last: Exception | None = None
    for attempt, pause in enumerate((*BACKOFF_S, None)):
        try:
            response = litellm.completion(**kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:                  # noqa: BLE001 — provider-agnostic
            last = exc
            if pause is None:
                break
            time.sleep(pause)
    raise LlmUnavailable(f"{model} did not answer: {last}") from last
