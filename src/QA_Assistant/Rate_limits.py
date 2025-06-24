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

import asyncio
from typing import Callable, Awaitable, Any

from aiolimiter import AsyncLimiter
import tiktoken
from Assistant import SYSTEM_PROMPT, TOOLS


# ───────────────────────────────────────────────────────────────────────────────
#  Global limiters
# ───────────────────────────────────────────────────────────────────────────────
WINDOW                  = 60                     # seconds

# OpenAI
OPENAI_REQ_LIMIT        = 50                     # requests / WINDOW
OPENAI_TOKEN_LIMIT      = 50_000                 # tokens   / WINDOW

openai_req_limiter      = AsyncLimiter(OPENAI_REQ_LIMIT,   WINDOW)
openai_tok_limiter      = AsyncLimiter(OPENAI_TOKEN_LIMIT, WINDOW)

# Cohere
COHERE_RERANK_LIMIT     = 20                     # rerank calls / WINDOW
cohere_rerank_limiter   = AsyncLimiter(COHERE_RERANK_LIMIT, WINDOW)

# Token encoder (same rules GPT‑4 uses)
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
    max_tokens: int = 2500,
    **kwargs,
):
    """
    Throttle *any* OpenAI SDK coroutine so the whole app stays within
    • ≤ 50 requests per 60 s
    • ≤ 50 000 tokens  per 60 s (prompt + completion)

    Parameters
    ----------
    send_fn
        The SDK coroutine you would normally `await`.
    prompt
        The text you just supplied to the model (user + system messages etc.).
        Used only for token counting.
    max_tokens
        Your `max_tokens=` setting for the call (upper bound on the model’s reply).
    **kwargs
        Passed straight through to *send_fn*.
    """ 
    prompt_tokens = count_tokens(SYSTEM_PROMPT + TOOLS + prompt)
    total_cost    = prompt_tokens + (max_tokens or 3500)

    if total_cost > OPENAI_TOKEN_LIMIT:
        raise ValueError(
            f"Single call would consume {total_cost} tokens, "
            f"exceeding the per‑minute allowance of {OPENAI_TOKEN_LIMIT}."
        )

    # one request, *total_cost* tokens
    async with openai_req_limiter:                    # request budget
        async with openai_tok_limiter.acquire(total_cost):  # token budget
            return await send_fn(**kwargs)


async def gated_cohere_rerank_call(
    send_fn: Callable[..., Awaitable[Any]],
    **kwargs,
):
    """
    Throttle Cohere’s `/rerank` so we issue ≤ 10 calls per minute.

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
    async with cohere_rerank_limiter:
        return await send_fn(**kwargs)
