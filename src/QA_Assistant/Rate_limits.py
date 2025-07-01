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

from openai import AsyncAzureOpenAI
from openai.types.responses import Response, ResponseUsage
import tiktoken

from src.QA_Assistant.token_bucket import AsyncTokenBucket
from src.QA_Assistant.answer_contracts import GLOBAL_FORMAT

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
    PLAN_CALL = [{"max_output_tokens":3_500,
                  "model": "gpt-4.1",
                  "previous_response_id": None,
                  "temperature":0.4,
                  "top_p":0.95}, 50_000, 1]
    TOOL_CALL = [{"max_output_tokens":2_000,
                  "model": "gpt-4.1-mini",
                  "previous_response_id": None,
                  "temperature":0.2,
                  "top_p":0.9}, 100_000,2]
    UPDATE_CALL = [{"max_output_tokens":5_000,
                    "model": "gpt-4.1-mini",
                    "previous_response_id": None,
                    "temperature":0.25,
                    "top_p":0.9}, 100_000,3]
    FINAL_CALL = [{"max_output_tokens":1_000,
                   "model": "gpt-4.1-mini",
                   "previous_response_id": None,
                   "temperature":0.4,
                   "top_p":0.95}, 100_000,4]

# Assumes three assistants at work
# ───────────────────────────────────────── config ────────────────────────────
WINDOW: float = 60.0  # seconds
# Global OpenAI limits
PLAN_REQ_CAP: int = 50  # requests / minute (global)
PLAN_TOK_CAP: int = 50_000     # tokens   / minute (global)
GLOBAL_REQ_CAP: int = 200      # requests / minute (global)
GLOBAL_TOK_CAP: int = 200_000     # tokens   / minute (global)

# Per‑assistant token budget (soft fairness / runaway protection)
PERSONAL_TOK_CAP: int = 100_000  # tokens / minute 

# Global Cohere limits
COHERE_RERANK_CAP: int = 20     # requests / minute (global)

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
_assistant_tok_limiters: Dict[str, AsyncTokenBucket] = defaultdict(
    lambda: AsyncTokenBucket(PERSONAL_TOK_CAP, WINDOW)
)

# Cohere limiter 
cohere_rerank_limiter = AsyncTokenBucket(COHERE_RERANK_CAP, WINDOW)

# ───────────────────────────────────────── helpers ───────────────────────────

def _count_tokens(text: str | None) -> int:
    """Fast GPT‑4/GPT‑4o token estimator."""
    return 0 if not text else len(ENCODER().encode(text))

def _get_token_buckets(assistant_id: str):
    """Return (personal_bucket, global_bucket,global_req_bucket) for the assistant."""
    return _assistant_tok_limiters[assistant_id], global_tok_limiter, global_req_limiter

# ─────────────────────────────────── public wrappers ─────────────────────────
async def gated_response(
    *,
    assistant_id: str,
    client: AsyncAzureOpenAI,
    prompt: str = "",
    stage: LoopStage,
    context: str = "",
) -> Response:
    """Throttle *any* OpenAI SDK coroutine with hierarchical token buckets.

    Enforced budgets
    ----------------
    • ≤ 50 requests / 60 s **(global)**
    • ≤ 50 000 tokens  / 60 s **(global)**
    • ≤ 10 000 tokens  / 60 s **(per‑assistant)**
    """
    # reformat, get buckets, tally token usage
    personal_tok, global_tok, global_req = _get_token_buckets(assistant_id)
    m = {"role":"user","content":prompt}
    prompt_tokens = _count_tokens(GLOBAL_FORMAT + context + "".join(f"<|{m['role']}|>\n{m['content']}\n"))
    reserve = prompt_tokens + stage.value[0]['max_output_tokens'] + PROMPT_BUFFER(prompt_tokens + stage.value[0]['max_output_tokens'])

    if reserve > stage.value[1]: 
        raise ValueError(
            f"Single call would consume {reserve:,} tokens which exceeds "
            f"the per‑assistant cap of {stage.value[1]:,} tokens/min."
        )
    
    result = None
    is_plan_call = stage == LoopStage.PLAN_CALL
    try:
        if is_plan_call:
            # One planner model, planning is outside jurisdiction of per assistant limits
            async with plan_tok_limiter.acquire(reserve):
                async with plan_req_limiter.acquire(1):
                    result: Response =  await client.responses.create(
                        input = prompt,
                        **stage.value[0], # stage specific hyperparameters
                        instructions = GLOBAL_FORMAT
                    )
        else:
            # Reserve request globally; reserve tokens (personal → global)
            async with global_tok.acquire(reserve):
                async with personal_tok.acquire(reserve):
                    async with global_req.acquire(1):
                        result: Response =  await client.responses.create(
                            input = prompt,
                            **stage.value[0], # stage specific hyperparameters
                            instructions = GLOBAL_FORMAT
                        )
    except Exception as e: 
        traceback.print_exc()

    if result is None:
        raise ValueError("No response from OpenAI")
    
    usage: ResponseUsage = result.usage
    print(f"This is {"a plan call--crediting to global 4.1 bucket" if is_plan_call else "not a plan call--crediting to 4.1-mini buckets"}")
    print(f"Reserved tokens: {reserve}\n Actual tokens: {usage.total_tokens}")
    print(f"Attempting to refund {reserve-usage.total_tokens} tokens, with {personal_tok._in_window if not is_plan_call else plan_tok_limiter._in_window} tokens in bucket")
    await refund_tokens(assistant_id = assistant_id,
                        used_tokens = usage.total_tokens,
                        reserved = reserve,
                        is_plan_call = is_plan_call)
    return result
                


async def gated_cohere_rerank_call(
    send_fn: Callable[..., Awaitable[Any]],
    **kwargs,
):
    """Throttle Cohere’s `/rerank` so we issue ≤ 20 calls per minute."""
    async with cohere_rerank_limiter.acquire(1):
        return await send_fn(**kwargs)

# ───────────────────────────────── token refund helper ───────────────────────

async def refund_tokens(assistant_id: str, used_tokens: int, reserved: int, is_plan_call: bool):
    """Optional: call when you know the *exact* tokens used were < reserved."""
    diff = reserved - used_tokens
    if diff <= 0:
        return
    personal_tok, global_tok,_ = _get_token_buckets(assistant_id)
    dif_personal = min(diff, personal_tok._in_window) if not is_plan_call else min(diff, plan_tok_limiter._in_window)
    dif_global = min(diff, global_tok._in_window) if not is_plan_call else 0
    # Return surplus to both buckets.
    await personal_tok.credit(dif_personal) if not is_plan_call else await plan_tok_limiter.credit(dif_personal)
    await global_tok.credit(dif_global)
