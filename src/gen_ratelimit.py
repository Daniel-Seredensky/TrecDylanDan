from src.IR_Ensemble.QA_Assistant.token_bucket import AsyncTokenBucket
from src.IR_Ensemble.QA_Assistant.rate_limits import _count_tokens, PROMPT_BUFFER
from openai import AsyncAzureOpenAI
# GEN RATE LIMITS
RPM = 50          # requests per minute
TPM = 50_000      # tokens per minute

REQ_BUCKET = AsyncTokenBucket(RPM)
TOK_BUCKET = AsyncTokenBucket(TPM)

MAX_OUT = 5_000

async def gated_call_gen(prompt: str,
                         client: AsyncAzureOpenAI,
                         temperature: float):
    """
    Throttle a call to the report generator with a token bucket.
    """
    toks = _count_tokens(prompt)
    toks = toks + PROMPT_BUFFER(toks) + MAX_OUT 
    if toks > TOK_BUCKET.capacity:
        raise ValueError(f"Prompt is too large: {toks} tokens, max is {TOK_BUCKET.capacity}.")
    async with REQ_BUCKET.acquire(1) as _, \
        TOK_BUCKET.acquire(toks) as tok_id:
            try:
                response = await client.responses.create(
                    model="gpt-4.1",
                    max_output_tokens= MAX_OUT,
                    temperature=temperature,
                    input=prompt,
                )
                return response
            except Exception as e:
                # If the call fails attempt a recall
                try:
                    response = await client.responses.create(
                        model="gpt-4.1",
                        max_output_tokens= MAX_OUT,
                        temperature=temperature,
                        input=prompt,
                    )
                    return response
                except Exception as e:
                    print(f"Failed to call generator: {e}")
                    raise 
            finally:
                await refund_tokens(tok_id, response.usage.total_tokens, toks)


async def refund_tokens(tok_id: str, used: int, reserved: int) -> None:
    """
    Refund tokens to the token bucket.
    """
    if reserved > used:
        surplus = reserved - used
        await TOK_BUCKET.credit_by_id(tok_id, surplus)
    
    