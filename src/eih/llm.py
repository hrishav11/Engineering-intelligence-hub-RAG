"""Centralized Claude client with automatic fallback.

Primary: configured Haiku model.
Fallback: Sonnet on retryable errors (overload, 5xx, rate limits).

Everywhere in the codebase that called `Anthropic().messages.create(...)` directly
should route through `call_with_fallback()` instead, so a Haiku outage doesn't
take the whole system down."""
from __future__ import annotations

import time
from typing import Any

import anthropic

from .config import cfg

FALLBACK_MODEL = "claude-sonnet-4-6"


_client_cache: anthropic.Anthropic | None = None


def _client() -> anthropic.Anthropic:
    global _client_cache
    if _client_cache is None:
        _client_cache = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    return _client_cache


def call_with_fallback(
    *,
    max_tokens: int,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    primary_model: str | None = None,
    fallback_model: str = FALLBACK_MODEL,
    max_retries: int = 2,
) -> Any:
    """Call Claude with primary model, falling back to a stronger model on retryable errors.

    Returns the Anthropic Message object. Re-raises non-retryable errors (bad request,
    auth, etc.) without falling back."""
    primary = primary_model or cfg.anthropic_model
    client = _client()

    def _do(model: str):
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools is not None:
            kwargs["tools"] = tools
        return client.messages.create(**kwargs)

    # Try primary with retries on transient errors
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return _do(primary)
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            last_exc = e
            # Don't retry on 4xx that aren't rate limits
            if isinstance(e, anthropic.APIStatusError) and 400 <= e.status_code < 500 \
               and not isinstance(e, anthropic.RateLimitError):
                break
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

    # Primary exhausted. Fall back if the error is one Sonnet might handle.
    if isinstance(last_exc, (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError)):
        if isinstance(last_exc, anthropic.APIStatusError) and 400 <= last_exc.status_code < 500 \
           and not isinstance(last_exc, anthropic.RateLimitError):
            # Hard 4xx — fallback won't help, re-raise
            raise last_exc
        try:
            return _do(fallback_model)
        except Exception as e:
            # Fallback also failed — raise the fallback's error (more informative)
            raise e

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("call_with_fallback exited without result or exception (unreachable)")
