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
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict
from enum import Enum
from functools import lru_cache
import traceback
from dotenv import load_dotenv
import os, time

from openai import AsyncAzureOpenAI
from openai.types.responses import Response, ResponseUsage
import tiktoken

from src.IR_Ensemble.QA_Assistant.token_bucket import AsyncTokenBucket
from src.IR_Ensemble.QA_Assistant.answer_contracts import GLOBAL_FORMAT

load_dotenv()

class LoopStage(Enum):
    """
    Enum for the different stages of the loop
    each stage corresponds to a different set of responses parameters and a absolute max reservation size
    there is also a unique identifier for each stage at the end of its list so that LoopStages will never equal each other
    eg 
    python: 
    >>> LoopStage.PLAN_CALL.value
    [{"max_output_tokens":3_500,"model": "gpt-4.1","previous_response_id": None}, 50_000,1]
    """
    # Previous response id should be None for PLAN call always
    SEARCH_CALL = [{"max_output_tokens":3_000,
                  "model": "gpt-4.1",
                  "temperature":0.4,
                  "top_p":0.95}, 75_000, 1]
    SELECT_CALL = [{"max_output_tokens":3_000,
                  "model": "gpt-4.1-mini",
                  "temperature":0.2,
                  "top_p":0.9}, 100_000,2]
    UPDATE_CALL = [{"max_output_tokens":6_000,
                    "model": "gpt-4.1-mini",
                    "temperature":0.25,
                    "top_p":0.9}, 150_000,3]
    FINAL_CALL = [{"max_output_tokens":1_500,
                   "model": "gpt-4.1-mini",
                   "temperature":0.4,
                   "top_p":0.95}, 100_000,4]

# Assumes three assistants at work
# ───────────────────────────────────────── config ────────────────────────────
WINDOW: float = 62.0  # seconds
# Global OpenAI limits
PLAN_REQ_CAP: int = 50  # requests / minute (global)
PLAN_TOK_CAP: int = 50_000     # tokens   / minute (global)
GLOBAL_REQ_CAP: int = 200      # requests / minute (global)
GLOBAL_TOK_CAP: int = 200_000     # tokens   / minute (global)

# Per‑assistant token budget (soft fairness / runaway protection)
PERSONAL_TOK_CAP: int = 100_000  # tokens / minute 

# Global Cohere limits
COHERE_RERANK_CAP: int = 500     # requests / minute (global)

PROMPT_BUFFER = lambda max_out: int(max_out * 0.025) #safety buffer for tokens

@lru_cache
def ENCODER():
    return tiktoken.get_encoding("o200k_base")

# ───────────────────────────────────── limiters ──────────────────────────────
# Requests – *global only* (all Assistants share this)
plan_req_limiter = AsyncTokenBucket(PLAN_REQ_CAP, WINDOW)
plan_tok_limiter = AsyncTokenBucket(PLAN_TOK_CAP, WINDOW)
global_tok_limiter = AsyncTokenBucket(GLOBAL_TOK_CAP, WINDOW)
global_req_limiter = AsyncTokenBucket(GLOBAL_REQ_CAP, WINDOW)

# Tokens – lazy‑initialised per‑assistant buckets
assistant_tok_limiters: Dict[str, AsyncTokenBucket] = defaultdict(
    lambda: AsyncTokenBucket(PERSONAL_TOK_CAP, WINDOW)
)

# Cohere limiters
cohere_bucket = [AsyncTokenBucket(COHERE_RERANK_CAP, WINDOW), os.getenv("COHERE_API_KEY")]


# ───────────────────────────────────────── helpers ───────────────────────────

def _count_tokens(text: str | None) -> int:
    """Fast GPT‑4/GPT‑4o token estimator."""
    return 0 if not text else len(ENCODER().encode(text))

def _get_token_buckets(assistant_id: str):
    """Return (personal_bucket, global_bucket,global_req_bucket) for the assistant."""
    return assistant_tok_limiters[assistant_id], global_tok_limiter, global_req_limiter

# ─────────────────────────────────── public wrappers ─────────────────────────
async def gated_response(
    *,
    assistant_id: str,
    client: AsyncAzureOpenAI,
    prompt: str = "",
    stage: LoopStage,
    context: str = "",
    prev_id: str | None = None,
) -> Response:
    """
    Throttle an OpenAI Responses API call with hierarchical token buckets
    (per‑assistant + global).  Now uses *event IDs* so refunds target the
    exact reservation that was made.
    """
    personal_tok, global_tok, global_req = _get_token_buckets(assistant_id)

    m = {"role": "user", "content": prompt}
    prompt_tokens = _count_tokens(
        GLOBAL_FORMAT + context + f"<|{m['role']}|>\n{m['content']}\n"
    )
    reserve = (
        prompt_tokens
        + stage.value[0]["max_output_tokens"]
        + PROMPT_BUFFER(prompt_tokens + stage.value[0]["max_output_tokens"])
    )

    if reserve > stage.value[1]:
        raise ValueError(
            f"Single call would consume {reserve:,} tokens "
            f"which exceeds the cap of {stage.value[1]:,}/min."
        )

    result: Response | None = None
    ids: dict[str, str] = {}
    is_search_call = stage == LoopStage.SEARCH_CALL

    try:
        if is_search_call:
            # Planner uses its own global bucket 
            # Note: Planner is deprecated, the search query generation now uses the plan token bucket
            async with plan_tok_limiter.acquire(reserve) as plan_id, \
                       plan_req_limiter.acquire(1):
                ids["plan"] = plan_id
                result = await client.responses.create(
                    input=prompt,
                    instructions=GLOBAL_FORMAT,
                    **stage.value[0],
                    previous_response_id=prev_id,
                )
        else:
            # Ordinary call – personal + global buckets
            async with global_tok.acquire(reserve) as global_id, \
                       personal_tok.acquire(reserve) as personal_id, \
                       global_req.acquire(1):
                ids["global"] = global_id
                ids["personal"] = personal_id
                result = await client.responses.create(
                    input=prompt,
                    instructions=GLOBAL_FORMAT,
                    **stage.value[0],
                    previous_response_id=prev_id,
                )
    except Exception:
        # Let the caller handle/log – reservation stays until it ages out
        raise

    if result is None:
        raise ValueError("No response from OpenAI")

    # Refund any surplus
    await refund_tokens(
        assistant_id=assistant_id,
        used_tokens=result.usage.total_tokens,
        reserved=reserve,
        is_plan_call=is_search_call,
        ids=ids,
    )
    return result
                

async def gated_cohere_rerank_call(
    send_fn: Callable[..., Awaitable[Any]],
    **kwargs,
):
    bucket,key = cohere_bucket          
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json"
    }
    """Throttle Cohere’s `/rerank` so we issue ≤ 20 calls per minute."""
    async with bucket.acquire(1):
        return await send_fn(headers = headers,**kwargs)

# ───────────────────────────────── token refund helper ───────────────────────

async def refund_tokens(
    *,
    assistant_id: str,
    used_tokens: int,
    reserved: int,
    is_plan_call: bool,
    ids: dict[str, str],
) -> None:
    """
    Return **(reserved − used)** tokens to the *specific* reservation event(s).
    If the event has already expired from the sliding window, credit_by_id is
    a no‑op, so we never over‑refund.
    """
    diff = reserved - used_tokens
    if diff <= 0:
        return  # nothing to refund

    if is_plan_call:
        # Planner bucket only
        await plan_tok_limiter.credit_by_id(ids["plan"], diff)
        return

    personal_tok, global_tok, _ = _get_token_buckets(assistant_id)

    # Refund exactly to the caller's own events
    await personal_tok.credit_by_id(ids["personal"], diff)
    await global_tok.credit_by_id(ids["global"], diff)
