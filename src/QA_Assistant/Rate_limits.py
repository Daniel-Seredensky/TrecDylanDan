"""
rate_limits.py
~~~~~~~~~~~~~~
Centralised async rate‑limiting for

• OpenAI GPT‑4.x (≤ 50 requests/min shared, ≤ 50 000 tokens/min shared,
  and ≤ 10 000 tokens/min per Assistant)
• Cohere *rerank* endpoint (≤ 20 requests/min)

Install once:
    pip install aiolimiter tiktoken
"""
from __future__ import annotations

import json
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict

import tiktoken

from src.QA_Assistant.Assistant import SYSTEM_PROMPT, TOOLS
from src.QA_Assistant.token_bucket import AsyncTokenBucket

# ───────────────────────────────────────── config ────────────────────────────
WINDOW: float = 60.0  # seconds
# Global organisation‑wide OpenAI limits
OPENAI_REQ_CAP: int = 50        # requests / minute (global)
OPENAI_TOKEN_CAP: int = 50_000  # tokens   / minute (global)
# Per‑assistant token budget (soft fairness / runaway protection)
PERSONAL_TOKEN_CAP: int = 10_000  # tokens / minute (per Assistant)

COHERE_RERANK_CAP: int = 20     # requests / minute
DEFAULT_MAX_COMPLETION: int = 3_500  # safety upper‑bound
PROMPT_BUFFER: int = 250 # safey bumper for hidden prompt tokens

TOOLS_STR = "\n".join(json.dumps(tool) for tool in TOOLS)
ENCODER = tiktoken.get_encoding("o200k_base")
SYSTEM_TOKS = len(ENCODER.encode(TOOLS_STR)) + len(ENCODER.encode(SYSTEM_PROMPT))


# ───────────────────────────────────── limiters ──────────────────────────────
# Requests – *global only* (all Assistants share this)
openai_req_limiter = AsyncTokenBucket(OPENAI_REQ_CAP, WINDOW)
# Tokens – global bucket + lazy‑initialised per‑assistant buckets
_global_tok_limiter = AsyncTokenBucket(OPENAI_TOKEN_CAP, WINDOW)
_assistant_tok_limiters: Dict[str, AsyncTokenBucket] = defaultdict(
    lambda: AsyncTokenBucket(PERSONAL_TOKEN_CAP, WINDOW)
)

# Cohere limiter (unchanged)
cohere_rerank_limiter = AsyncTokenBucket(COHERE_RERANK_CAP, WINDOW)

# ───────────────────────────────────────── helpers ───────────────────────────

def _count_tokens(text: str | None) -> int:
    """Fast GPT‑4/GPT‑4o token estimator."""
    return 0 if not text else len(ENCODER.encode(text))


def _get_token_buckets(assistant_id: str):
    """Return (personal_bucket, global_bucket) for the assistant."""
    return _assistant_tok_limiters[assistant_id], _global_tok_limiter

# ─────────────────────────────────── public wrappers ─────────────────────────

async def gated_openai_call(
    assistant_id: str,
    send_fn: Callable[..., Awaitable[Any]],
    *,
    prompt: str = "",
    max_tokens: int = DEFAULT_MAX_COMPLETION,
    **kwargs,
):
    """Throttle *any* OpenAI SDK coroutine with hierarchical token buckets.

    Enforced budgets
    ----------------
    • ≤ 50 requests / 60 s **(global)**
    • ≤ 50 000 tokens  / 60 s **(global)**
    • ≤ 10 000 tokens  / 60 s **(per‑assistant)**
    """

    personal_tok, global_tok = _get_token_buckets(assistant_id)

    prompt_tokens = _count_tokens(prompt) 
    print(f"Estimated prompt: {prompt_tokens}")
    print(f"Estimated completion tokes: {max_tokens}")
    total_tokens = prompt_tokens + max_tokens

    if total_tokens > PERSONAL_TOKEN_CAP:
        raise ValueError(
            f"Single call would consume {total_tokens:,} tokens which exceeds "
            f"the per‑assistant cap of {PERSONAL_TOKEN_CAP:,} tokens/min."
        )

    # 1️⃣ Reserve request globally; 2️⃣ reserve tokens (personal → global)
    async with openai_req_limiter.acquire(1):
        async with personal_tok.acquire(total_tokens):
            async with global_tok.acquire(total_tokens):
                return await send_fn(max_tokens=max_tokens, **kwargs)


async def gated_cohere_rerank_call(
    send_fn: Callable[..., Awaitable[Any]],
    **kwargs,
):
    """Throttle Cohere’s `/rerank` so we issue ≤ 20 calls per minute."""
    async with cohere_rerank_limiter.acquire(1):
        return await send_fn(**kwargs)


# ───────────────────────────── streaming convenience API ─────────────────────

@asynccontextmanager
async def gated_openai_stream(
    assistant_id: str,
    send_fn: Callable[..., Awaitable[Any]],
    *,
    max_tokens: int = 1000,
    context_tokens: str = "",
    old_reserve: int|None = None,
    **kwargs,
):
    """Exactly like `gated_openai_call` but yields an SSE/stream instead."""

    personal_tok, global_tok = _get_token_buckets(assistant_id)

    prompt_tokens = _count_tokens(context_tokens) + SYSTEM_TOKS + PROMPT_BUFFER
    print(f"Estimated prompt: {prompt_tokens}")
    print(f"Estimated completion tokes: {max_tokens}")
    reserve = prompt_tokens + max_tokens

    if reserve > PERSONAL_TOKEN_CAP:
        raise ValueError(
            f"Single stream would reserve {reserve:,} tokens (> per‑assistant cap "
            f"{PERSONAL_TOKEN_CAP:,})."
        )

    should_send_tokens = send_fn.__name__ != "submit_tool_outputs"
    modifier = "additional" if old_reserve is not None else ""
    print(f"Requesting {modifier} {reserve} tokens")
    print(f"Current total for run: {old_reserve + reserve if old_reserve is not None else 0 + reserve}")

    async with openai_req_limiter.acquire(1):
        async with personal_tok.acquire(reserve):
            async with global_tok.acquire(reserve):
                if should_send_tokens:
                    stream = await send_fn(max_completion_tokens=max_tokens,assistant_id = assistant_id, **kwargs)
                else:
                    stream = await send_fn(**kwargs)
                try:
                    if old_reserve is not None: reserve += old_reserve
                    yield stream, reserve  # caller can later refund if needed
                finally:
                    # No explicit release – tokens naturally expire after WINDOW s.
                    pass

# ───────────────────────────────── token refund helper ───────────────────────

async def refund_tokens(assistant_id: str, used_tokens: int, reserved: int):
    """Optional: call when you know the *exact* tokens used were < reserved."""
    diff = reserved - used_tokens
    print(f"Attempting to refund {diff} tokens")
    if diff <= 0:
        return
    personal_tok, global_tok = _get_token_buckets(assistant_id)
    # Return surplus to both buckets.
    await personal_tok.credit(diff)
    await global_tok.credit(diff)
