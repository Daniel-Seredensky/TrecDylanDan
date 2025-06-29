"""
QuestionEval.py   •   Azure OpenAI Responses-API version
─────────────────────────────────────────────────────────
Core agent loop preserved:
  seed → tool rounds → update answer status / summary
  → continue or finalise.

Key differences vs. Assistants-API version
-----------------------------------------
•  No threads / runs: we hold a single `conversation_id`
   (= `previous_response_id`) and keep calling
   `client.responses.create`.
•  Tool calls arrive as `response.function_call` events
   (or as items in `response.output` if you don’t stream).
   We echo their outputs back via
   `client.responses.create(..., input=tool_outputs,
                            previous_response_id=...)`.
•  Token accounting uses `response.usage.total_tokens`.
•  `token_bucket`, `rate_limits`, and `self._record()` wired
   exactly as before – only the wrapper
   `gated_openai_stream()` needed a MINOR tweak:
      it now accepts **any** callable and simply forwards
      `*args/**kwargs`.  No other change required.
"""

from __future__ import annotations

import os, json, re, asyncio, aiofiles, asyncinit
from enum import Enum
from uuid import uuid4
from typing import Any, Dict, List, Optional, Callable

from openai import AsyncAzureOpenAI
from openai.types import StreamEvent  # ← preview SDK
# If your installed SDK exposes event classes differently,
# tweak the import above.

from src.QA_Assistant.Searcher import search
from src.QA_Assistant.DocSelect import select_documents
from src.QA_Assistant.rate_limits import (
    gated_openai_stream, refund_tokens, _get_token_buckets
)

# ───────────────────────────────────────────────────────────
#  Constants & regexen
# ───────────────────────────────────────────────────────────
MODEL = os.environ["AZURE_OPENAI_DEPLOYMENT"]
TOOLS = [
    {
        "type": "function",
        "name": "search",
        "description": "Hybrid lexical + semantic search over the MARCO index.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k":     {"type": "integer", "default": 20},
                "agentId": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "select_documents",
        "description": "Return top fragments for 1–2 docIds.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 2
                }
            },
            "required": ["doc_ids"]
        }
    },
]

NOTEPAD_RE = re.compile(r"<notepad>(.*?)</notepad>", re.S)
SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.S)
ANSWER_RE  = re.compile(r"<answer>(.*?)</answer>",  re.S)

# ───────────────────────────────────────────────────────────
class QAStatus(str, Enum):
    NO_ANSWER = "no_answer"
    PARTIAL   = "partial_answer"
    FINISHED  = "finished"

# ───────────────────────────────────────────────────────────
class BaseAgent:
    """
    M I G R A T E D   F O R   R E S P O N S E S   A P I
    --------------------------------------------------- 
    • async / await throughout
    • self._record() called on EVERY outbound or inbound
      message so token usage mirrors conversation faithfully
    • gated_openai_stream() still handles reservation /
      refund; only its inner callable changed
    """

    MAX_TOOL_ROUNDS = 5
    FINAL_PROMPT = (
        "Max iterations reached.  Compose ONE final JSON answer then stop.\n"
        "Briefly explain what you tried and what extra evidence would help."
    )

    # ────────────────────────────────────────────────────
    @asyncinit
    async def __init__(self,*, question: str, client: AsyncAzureOpenAI):
        self.question = question
        self.client   = client

        # The new conversation lives only as a chain of
        # response-ids (no separate thread object)
        self.prev_resp_id: Optional[str] = None

        # Local mirrors for history & agent bookkeeping
        self.history: List[Dict[str, str]] = []
        self.agent_id = str(uuid4())
        self.status   = QAStatus.NO_ANSWER
        self.answer_json: Optional[str] = None
        self.summary:   Optional[str] = None

        # Building artefact dirs
        self.results_path: str = os.path.join(os.getenv("BM25_RESULTS_PATH"), f"{self.agent_id}")
        os.makedirs(self.results_path, exist_ok=True)

        self.convo_path = os.path.join(self.results_path, "Convo.txt")
        self.tools_path = os.path.join(self.results_path, "Tools.txt")
        self.thread_path = os.path.join(self.results_path,"Thread.txt")
        # touch them so they always exist
        async with aiofiles.open(self.convo_path, "w"): pass
        async with aiofiles.open(self.tools_path, "w"): pass

    # ────────────────────────────────────────────────────
    async def run(self) -> Dict[str, Any]:
        """
        Main loop:
          • 1st turn – seed with the question
          • then:   while not finished → handle tool calls
          • summarise / continue if "finished": false
        """
        tool_rounds = 0
        first_turn  = True
        awaiting_tools: List[Dict[str, Any]] = []

        while True:
            # ------------------------------------------------------
            # Build INPUT for this Responses-API call
            # ------------------------------------------------------
            if first_turn:
                # Initial user question
                user_msg = (
                    "The questions you must answer:\n"
                    f"{self.question}\n\n"
                    "Use the first round to create a plan "
                    "inside <summary> tags then mark as complete."
                )
                self._record("user", user_msg)
                inputs = [{"role": "user", "content": user_msg}]
            elif awaiting_tools:
                # We have tool outputs to feed back
                inputs = awaiting_tools
            else:
                # Need a continuation because model said finished:false
                cont_prompt = (
                    "# Continue refining your answer.\n"
                    f"Seed:{self.summary or ''}\n"
                    f"Previous answer:{self.answer_json or ''}"
                )
                self._record("user", cont_prompt)
                inputs = [{"role": "user", "content": cont_prompt}]

            # Reserve tokens & stream the assistant’s turn
            stream_cm = gated_openai_stream(
                "responses",                      # ← bucket-key
                self.client.responses.create,     # ← callable
                model      = MODEL,
                tools      = TOOLS,
                input      = inputs,
                # Carry context if not first
                previous_response_id = self.prev_resp_id,
                stream     = True,
                max_tokens = 1500,
                context_tokens = self._serialise_history(),
            )

            assistant_text   = ""
            function_calls   : List[Dict[str, Any]] = []
            usage_total      = 0
            new_resp_id      = None

            async with stream_cm as (stream, reserved):
                async for ev in stream:
                    t = ev.type

                    # Text delta
                    if t == "response.output_text.delta":
                        assistant_text += ev.delta
                    # Function call request
                    elif t == "response.function_call":
                        function_calls.append({
                            "name":      ev.name,
                            "arguments": ev.arguments,
                            "call_id":   ev.call_id,
                        })
                    # Completed
                    elif t == "response.done":
                        usage_total  = ev.usage.total_tokens
                        new_resp_id  = ev.response_id
                        break
                    # TODO: adjust for exact event names if SDK changes

            # Record assistant message for token bookkeeping
            self._record("assistant", assistant_text)

            # Refund unused tokens
            bucket, _ = _get_token_buckets("responses")
            await refund_tokens("responses",
                                usage_total,
                                max(reserved, bucket._in_window))

            # --------------------------------------------------
            # Parse assistant answer JSON & status
            # --------------------------------------------------
            self._update_status(assistant_text)

            # --------------------------------------------------
            # Handle FUNCTION CALLS (tool invocation)
            # --------------------------------------------------
            if function_calls:
                if tool_rounds >= self.MAX_TOOL_ROUNDS:
                    # Too many – inject final prompt next loop
                    awaiting_tools = []
                    self.summary = self.summary or "(no summary)"
                    first_turn = False
                    tool_rounds += 1
                    continue

                tool_rounds += 1
                awaiting_tools = await asyncio.gather(
                    *[self._dispatch_tool(fc) for fc in function_calls]
                )
                # Record tool outputs for token mirror
                self._record("tool",
                             "\n".join(json.dumps(t) for t in awaiting_tools))
                # Continue loop – feed these outputs back
                self.prev_resp_id = new_resp_id
                first_turn = False
                continue

            # --------------------------------------------------
            # No tool calls – check if we're FINISHED
            # --------------------------------------------------
            if self.status == QAStatus.FINISHED:
                return {"status": "finished", "content": self.answer_json}

            # If not finished after max rounds, force FINAL prompt
            if tool_rounds >= self.MAX_TOOL_ROUNDS:
                awaiting_tools = []
                forced_msg = self.FINAL_PROMPT
                self._record("user", forced_msg)
                inputs = [{"role": "user", "content": forced_msg}]
                first_turn = False
                self.prev_resp_id = new_resp_id
                continue

            # Otherwise: loop again asking model to continue
            awaiting_tools = []
            first_turn = False
            self.prev_resp_id = new_resp_id

    # ────────────────────────────────────────────────────
    async def _dispatch_tool(self, fc: Dict[str, Any]) -> Dict[str, str]:
        """
        Execute local Python function and build function_call_output
        dict expected by Responses-API.
        """
        name  = fc["name"]
        args  = json.loads(fc["arguments"] or "{}")
        if name == "search":
            args["agentId"] = self.agent_id
        try:
            result = await {
                "search":           search,
                "select_documents": select_documents,
            }[name](**args)
            payload = json.dumps(result)[:4_000]  # truncate
        except Exception as e:
            payload = json.dumps({"error": str(e)})

        return {
            "type":    "function_call_output",
            "call_id": fc["call_id"],
            "output":  payload,
        }

    # ────────────────────────────────────────────────────
    def _update_status(self, assistant_txt: str) -> None:
        # Grab summary
        m_notepad = NOTEPAD_RE.search(assistant_txt)
        if m_notepad:
            m_sum = SUMMARY_RE.search(m_notepad.group(1))
            if m_sum:
                self.summary = m_sum.group(1).strip()

        # Grab answer JSON envelope
        m_ans = ANSWER_RE.search(assistant_txt)
        if m_ans:
            payload = m_ans.group(1).strip()
            if payload:
                self.answer_json = payload
                if '"finished": true' in payload.lower():
                    self.status = QAStatus.FINISHED
                else:
                    self.status = QAStatus.PARTIAL
            else:
                self.status = QAStatus.NO_ANSWER

    # ────────────────────────────────────────────────────
    #   h i s t o r y   &   t o k e n   a c c o u n t i n g
    # ────────────────────────────────────────────────────
    def _record(self, role: str, content: str) -> None:
        """Append to local history mirror for token estimation."""
        self.history.append({"role": role, "content": content})

    def _serialise_history(self) -> str:
        """Return condensed text used by token estimator."""
        return "".join(f"<|{m['role']}|>\n{m['content']}\n"
                       for m in self.history) + "<|assistant|>"