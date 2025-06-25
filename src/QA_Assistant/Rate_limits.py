"""
rate_limits.py
~~~~~~~~~~~~~~
Centralised async rate‑limiting for

• OpenAI GPT‑4.1 (50 requests/min, 50 000 tokens/min)
• Cohere *rerank* endpoint (10 requests/min)

Install once:

    pip install aiolimiter tiktoken
"""
from __future__ import annotations
from typing import Callable, Awaitable, Any

from QA_Assistant.token_bucket import AsyncTokenBucket
from Assistant import SYSTEM_PROMPT, TOOLS

import tiktoken

# ───────────────────────────────────────── config ────────────────────────────
WINDOW                 = 60.0                      # seconds
OPENAI_REQ_CAP         = 50                        # requests / 60 s
OPENAI_TOKEN_CAP       = 50_000                   # tokens   / 60 s
COHERE_RERANK_CAP      = 20                        # requests / 60 s
DEFAULT_MAX_COMPLETION = 3_500                     # safety upper‑bound

# ───────────────────────────────────── limiters ──────────────────────────────
openai_req_limiter   = AsyncTokenBucket(OPENAI_REQ_CAP,    WINDOW)
openai_tok_limiter   = AsyncTokenBucket(OPENAI_TOKEN_CAP,  WINDOW)
cohere_rerank_limiter = AsyncTokenBucket(COHERE_RERANK_CAP, WINDOW)
ENCODER = tiktoken.get_encoding("cl100k_base")

# ───────────────────────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────────────────────
def count_tokens(text: str | None) -> int:
    """Fast GPT‑4/GPT‑4o token estimator."""
    if not text:
        return 0
    return len(ENCODER.encode(text))


# ───────────────────────────────────────────────────────────────────────────────
#  Public wrappers
# ───────────────────────────────────────────────────────────────────────────────
async def gated_openai_call(
    send_fn: Callable[..., Awaitable[Any]],
    *,
    prompt: str = "",
    max_tokens: int = 5_000,
    **kwargs,
):
    """
    Throttle *any* OpenAI SDK coroutine so the whole app stays within

      • ≤ 50   requests per 60 s
      • ≤ 50 000 tokens  per 60 s  (prompt + completion)

    Parameters
    ----------
    send_fn
        The coroutine you would normally `await`.
    prompt
        The text you just supplied to the model (user + system messages etc.).
        Only used for token bookkeeping.
    max_tokens
        Your `max_tokens=` hint for the call (upper bound on the reply).
    **kwargs
        Passed straight through to `send_fn`.
    """
    prompt_tokens = count_tokens(SYSTEM_PROMPT + TOOLS + prompt)
    total_tokens  = prompt_tokens + (max_tokens or DEFAULT_MAX_COMPLETION)

    if total_tokens > OPENAI_TOKEN_CAP:
        raise ValueError(
            f"Single call would consume {total_tokens:,} tokens "
            f"(limit per 60 s is {OPENAI_TOKEN_CAP:,})."
        )

    # Reserve *one* request + *total_tokens* tokens
    async with openai_req_limiter.acquire(1):
        async with openai_tok_limiter.acquire(total_tokens):
            return await send_fn(max_tokens=max_tokens, **kwargs)


async def gated_cohere_rerank_call(
    send_fn: Callable[..., Awaitable[Any]],
    **kwargs,
):
    """
    Throttle Cohere’s `/rerank` so we issue ≤ 20 calls per minute.

    Example
    -------
    .. code-block:: python

        from src.rate_limits import gated_cohere_rerank_call

        result = await gated_cohere_rerank_call(
            cohere_client.rerank,
            query=query,
            documents=docs,
        )
    """
    async with cohere_rerank_limiter.acquire(1):
        return await send_fn(**kwargs)

# rate_limits.py  (append right below gated_openai_call)
from contextlib import asynccontextmanager

@asynccontextmanager
async def gated_openai_stream(
    send_fn: Callable[..., Awaitable[Any]],
    *,
    prompt: str = "",
    max_tokens: int = 5_000,
    **kwargs,
):
    """Exactly like `gated_openai_call` but yields an SSE stream."""
    prompt_tokens = count_tokens(SYSTEM_PROMPT + TOOLS + prompt)
    reserve = prompt_tokens + (max_tokens or DEFAULT_MAX_COMPLETION)

    if reserve > OPENAI_TOKEN_CAP:
        raise ValueError(f"Single call would cost {reserve:,} tokens (limit {OPENAI_TOKEN_CAP:,}).")

    async with openai_req_limiter.acquire(1):
        async with openai_tok_limiter.acquire(reserve):
            stream = await send_fn(max_tokens=max_tokens, **kwargs)
            try:
                yield stream, reserve          # pass back what we reserved
            finally:
                # nothing to release – reservation auto‑expires after 60 s
                pass
