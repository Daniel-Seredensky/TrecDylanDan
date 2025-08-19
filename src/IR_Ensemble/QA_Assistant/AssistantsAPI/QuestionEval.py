import os
import json
import re
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from uuid import uuid4
from openai.types.beta.threads import Message, MessageContent


from asyncinit import asyncinit
import aiofiles
from openai import AsyncAzureOpenAI

from src.QA_Assistant.Searcher import search
from QA_Assistant.AssistantsAPI.DocSelect import select_documents
from src.QA_Assistant.rate_limits import gated_openai_stream, refund_tokens,_get_token_buckets


# ────────────────────────────────────────────────────────────────────────────────
# Constants & enums
# ────────────────────────────────────────────────────────────────────────────────
ASSISTANT_ID_FILE = "src/DerivedData/Assistant/AssistantId.txt"

class QAStatus(Enum):
    NO_ANSWER = "no_answer"
    PARTIAL_ANSWER = "partial_answer"
    FINISHED   = "finished"

# ────────────────────────────────────────────────────────────────────────────────
# Question‑assessment helper – Assistants v2 compliant (streaming)
# ────────────────────────────────────────────────────────────────────────────────
@asyncinit
class QuestionAssessmentAgent:
    """Single‑question assessment agent leveraging local search tools."""

    MAX_TOOL_ROUNDS: int = 15  # guard against infinite tool chains
    NOTEPAD_RE   = re.compile(r"<notepad>(.*?)</notepad>", re.S)
    SUMMARY_RE   = re.compile(r"<summary>(.*?)</summary>", re.S)
    FINAL_PROMPT = "".join([
                "Max iterations or an unexpected error stopped the run.\n\n",
                "Compose **one final assistant message** and then **stop**.\n\n",
                "• Briefly (1–3 bullets) explain what you tried and why it was insufficient.  \n",
                "• Note what additional evidence or queries would likely resolve the question.\n",
                "Provide your answer with the same jsonic format as the other answers placing your description in the answer field. Make sure to put citations in the citations field if there are any."
            ])

    async def __init__(
        self,
        question: str,
        *,
        client: AsyncAzureOpenAI,
        assistant_id: Optional[str] = None,
    ) -> None:
        # Basic fields
        self.question = question
        self.status: QAStatus = QAStatus.NO_ANSWER
        self.full_answer_status: str | None = None
        self.history: list[dict[str, str]] = []   # running chat log

        # Each agent run gets its own temp directory for search artefacts
        self.agent_id = uuid4()
        self.results_path: str = os.path.join(os.getenv("BM25_RESULTS_PATH"), f"{self.agent_id}")
        os.makedirs(self.results_path, exist_ok=True)

        self.convo_path = os.path.join(self.results_path, "Convo.txt")
        self.tools_path = os.path.join(self.results_path, "Tools.txt")
        self.thread_path = os.path.join(self.results_path,"Thread.txt")
        # touch them so they always exist
        async with aiofiles.open(self.convo_path, "w"): pass
        async with aiofiles.open(self.tools_path, "w"): pass

        # Async client (already configured for Azure endpoint + v2 header)
        self.client = client

        # ── Assistant ID comes from cache unless explicitly provided ─────────
        if assistant_id is None:
            async with aiofiles.open(ASSISTANT_ID_FILE) as f:
                assistant_id = (await f.read()).strip()
                if not assistant_id:
                    raise ValueError("No assistant id, create first using get_or_create_assistant()")

        self.assistant_id = assistant_id

        # Thread per question
        self.thread = await self.client.beta.threads.create()
        async with aiofiles.open(self.thread_path,"w") as w:
            await w.write(self.thread.id)

        # Local function‑tools wired into the assistant manifest
        self.LOCAL_FUNCTIONS: Dict[str, Callable[..., Any]] = {
            "search": search,
            "select_documents": select_documents,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ────────────────────────────────────────────────────────────────────────
    async def run(self) -> Dict[str, Any]:
        """
        Same control‑flow, but `context_tokens` now tracks the entire
        message history that is resident in the thread at each step.
        """

        # 0️⃣  Initialise thread and token mirror ---------------------------------
        await self._init_thread()
        print("Thread seeded")

        tool_calls_so_far   = 0
        first_run           = True
        final_prompt_sent   = False

        while True:
            # Exit conditions -----------------------------------------------------
            if self.status == QAStatus.FINISHED or final_prompt_sent and not first_run:
                break
            if tool_calls_so_far >= self.MAX_TOOL_ROUNDS and not final_prompt_sent:
                # Post the fall‑back prompt → add to context, then loop again
                await self._force_final_prompt()
                final_prompt_sent = True
                first_run = False
                continue

            # Spin a streamed run -------------------------------------------------
            print("Starting new run")
            stream_cm = gated_openai_stream(
                self.assistant_id,
                self.client.beta.threads.runs.create,
                thread_id=self.thread.id,
                max_tokens=1500,
                context_tokens=self._serialise_history(),
                stream=True,
            )
            first_run = False
            while stream_cm:
                async with stream_cm as (stream, reserved_tokens):
                    async for event in stream:
                        ev = getattr(event, "event", None)
                        if getattr(event.data, "id", None):
                            self.run_id = event.data.id

                        # Assistant message finished ---------------------------------
                        if ev == "thread.message.completed" and event.data.role == "assistant":
                            print("Single message complete")
                            content = self._as_text(event.data.content)
                            async with aiofiles.open(self.convo_path, "a") as f:
                                await f.write(content + "\n---\n")
                            self._update_status(event.data)
                            # assistant message is already stored by the service,
                            # so add it to our mirror
                            self._record("assistant",content)

                        # Tool calls --------------------------------------------------
                        elif ev == "thread.run.requires_action":
                            print("Assistant requires tools")
                            # ── 2. dispatch tool calls ──
                            calls   = event.data.required_action.submit_tool_outputs.tool_calls
                            for c in calls:
                                # The platform serialises each call as JSON with name & arguments:
                                fn_call_msg = json.dumps(
                                    {"name": c.function.name, "arguments": json.loads(c.function.arguments)}
                                )
                                self._record("assistant", fn_call_msg)
                            outputs = await asyncio.gather(*(self._dispatch_tool(c) for c in calls))
                            tool_calls_so_far += len(calls)

                            # append a clipped stringified blob of the tool outputs to the
                            # running `context_tokens` estimate
                            outputs_blob     = "\n".join(json.dumps(o) for o in outputs)
                            self._record ("tool",outputs_blob)

                            # ── 3. chain into a submit_tool_outputs stream ──
                            
                            stream_cm = gated_openai_stream(
                                self.assistant_id,
                                self.client.beta.threads.runs.submit_tool_outputs,
                                run_id=event.data.id,
                                thread_id=self.thread.id,
                                tool_outputs=outputs,
                                max_tokens=0,
                                context_tokens=self._serialise_history(),
                                old_reserve=reserved_tokens,
                                stream=True,
                            )
                            break  # re‑enter with the new stream

                        # Run completed normally --------------------------------------
                        elif ev == "thread.run.completed":
                            usage  = getattr(event.data, "usage", None)
                            actual = getattr(usage, "total_tokens", None) or 0
                            bucket, _ = _get_token_buckets(self.assistant_id)
                            await refund_tokens(self.assistant_id, actual, max(reserved_tokens,bucket._in_window))
                            print("-------------------------------------")
                            print(f"Total tokens used in the run: {usage.total_tokens}")
                            print(f"Prompt tokens: {usage.prompt_tokens}")
                            print(f"Completion tokens: {usage.completion_tokens}")
                            print(f"total tokens reserved for run: {reserved_tokens}")
                            print("-------------------------------------")

                            # Compact thread → new summary replaces everything else
                            if not final_prompt_sent:
                                await self._reset_thread_with_summary(
                                    seed=self.summary or ""
                                )
                            stream_cm = None
                            break

                        # Aborted run --------------------------------------------------
                        elif ev.startswith("thread.run.") and getattr(event.data, "status", "") in {
                            "failed", "cancelled", "expired", "incomplete"
                        }:
                            await refund_tokens(self.assistant_id, 0, reserved_tokens)
                            return {
                                "status": self.status.value,
                                "content": self.full_answer_status,
                            }

        return {"status": self.status.value, "content": self.full_answer_status}

    # ────────────────────────────────────────────────────────────────────────
    # Tool dispatch helper
    # ────────────────────────────────────────────────────────────────────────
    async def _dispatch_tool(self, call) -> Dict[str, Any]:
        """Execute a local function and return payload capped at 5 kB."""
        fn_name = call.function.name
        args = json.loads(call.function.arguments or "{}")
        if fn_name == "search":
            args["agentId"] = str(self.agent_id)  # pass search‑scoped ID
        try: 
            result = await self.LOCAL_FUNCTIONS[fn_name](**args)
            payload = json.dumps(result)[:2500]
            async with aiofiles.open(self.tools_path, "a") as f:
                await f.write(
                    json.dumps(
                        {
                            "tool": fn_name,
                            "args": args,
                            "result": payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return {"tool_call_id": call.id, "output": payload}
        except:
            async with aiofiles.open(self.tools_path, "a") as f:
                await f.write(
                    json.dumps(
                        {
                            "tool": fn_name,
                            "args": args,
                            "result": "tool called failed",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return {"tool_call_id": call.id, "output": "tool call failed"}

    # ────────────────────────────────────────────────────────────────────────
    # Helper utilities
    # ────────────────────────────────────────────────────────────────────────
    async def _force_final_prompt(self) -> None:
        """If the streamed run ends abruptly, ask the assistant to summarise."""
        await self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=self.FINAL_PROMPT,
        )
        self.record("user",self.FINAL_PROMPT)

    def _update_status(self, assistant_msg: Message) -> None:
        """Extract JSON inside <answer>…</answer> and update status flags."""
        content = self._as_text(assistant_msg.content)    
        self.summary = self._harvest_summary(content)      
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            payload = match.group(1).strip()
            self.full_answer_status = payload if payload != None else self.full_answer_status
            self.status = (QAStatus.FINISHED
                        if '"finished": true' in payload
                        else QAStatus.PARTIAL_ANSWER)

    @staticmethod
    def _as_text(blocks: List[MessageContent]) -> str:
        """
        v2 `Message.content` may be a string (rare) or a list of MessageContent objects.
        Concatenate **only the text blocks** into one string.
        """
        if isinstance(blocks, str):
            return blocks

        parts: List[str] = []
        for b in blocks:
            if b.type == "text":
                # b.text.value holds the actual text
                parts.append(b.text.value)
        return "\n".join(parts)
    
    async def cancel_run(self):
        await self.client.beta.threads.runs.cancel(
                        thread_id=self.thread.id, run_id=self.run_id
                    )

    async def _reset_thread_with_summary(self, seed: str) -> None:
        """
        Blow away every message in `thread_id`, then add a single user message
        containing `seed`.
        """
        # 1) Collect *all* message ids (pagination‑safe).
        cursor: Optional[str] = None
        msg_ids = []
        content = f"# Continue your work \n Seed:{seed} \n previous answer {self.full_answer_status}"
        keep = await self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=content
        )
        keep_id = keep.id

        while True:
            page = await self.client.beta.threads.messages.list(
                thread_id=self.thread.id,
                order="desc",
                after=cursor,      # ← first loop: None, later: previous .after
                limit=50,
            )
            msg_ids.extend(m.id for m in page.data if m.id != keep_id)

            # Use the paginator helpers that *do* exist
            if not page.has_next_page():
                break
            cursor = page.after 

        # 2) Delete them in parallel (or serially if you prefer).
        await asyncio.gather(*(
            self.client.beta.threads.messages.delete(
                thread_id=self.thread.id,
                message_id=m_id
            ) for m_id in msg_ids if m_id != keep_id
        ))

        await self._wait_until_compacted()
        self.history = []
        self._record("user",content)

        print (f"Deleted {len(msg_ids) -1 if keep_id in msg_ids else len(msg_ids)} messages")

    def _harvest_summary(self, assistant_msg):
        """Pull the <summary>…</summary> text out of the latest notepad."""
        m_notepad = self.NOTEPAD_RE.search(assistant_msg)
        if not m_notepad:
            return None

        m_sum = self.SUMMARY_RE.search(m_notepad.group(1))
        return m_sum.group(1).strip() if m_sum else None   
    
    async def _wait_until_compacted(self, *, poll=0.3, timeout=5.0) -> None:
        end = asyncio.get_event_loop().time() + timeout
        while True:
            page = await self.client.beta.threads.messages.list(
                thread_id=self.thread.id,
                order="desc",     # newest first
                limit=1           # ask for exactly one msg
            )

            # If the backend says there’s no “next page”, you’re down to ≤ 1 message.
            if not page.has_next_page():
                return            # ✅ compacted (the current page holds your keep)

            if asyncio.get_event_loop().time() >= end:
                raise TimeoutError("Thread compaction didn’t finish in time")

            await asyncio.sleep(poll)
        
    # Clean‑up – remove temp search directory ---------------------------------
    def close(self) -> None:
        try:
            os.rmdir(self.results_path)
        except OSError:
            # directory non‑empty or other I/O error – ignore for now
            pass

    async def _init_thread(self):
        seed = ("\n\nThe questions you must answer:\n"
                + self.question
                + "\n\nWork efficiently and accurately. "
                + "Use the first round to create a plan + selective memory put it in summary wrappers, then mark as complete"
        )
        seed_msg = await self.client.beta.threads.messages.create(
            thread_id=self.thread.id, role="user", content=seed
        )
        self._record("user",seed)
        
    
    def _record(self, role: str, content: str) -> None:
        """Append a message to the local history buffer."""
        self.history.append({"role": role, "content": content})

    def _serialise_history(self) -> str:
        """
        Convert history to the exact tokenised form:
          <|role|>\n<content>\n … <|assistant|>
        API automatically adds the trailing assistant tag, so we include it
        here for an accurate prompt‑token mirror.
        """
        return "".join(
            f"<|{m['role']}|>\n{m['content']}\n" for m in self.history
        ) + "<|assistant|>"

# ────────────────────────────────────────────────────────────────────────────────
# Convenience functional wrapper
# ────────────────────────────────────────────────────────────────────────────────
async def assess_question(
    *,
    question: str,
    client: AsyncAzureOpenAI,
    assistant_id: str,
) -> Dict[str, Any]:
    if not assistant_id:
        raise ValueError ("Make an assistant with `get_or_create_assistant` first")
    """One‑shot helper for callers that don’t want to manage the class."""
    agent = await QuestionAssessmentAgent(
        question=question,
        client=client,
        assistant_id=assistant_id,
    )
    try: 
        result = await agent.run()
        return result
    except Exception as e:
        print(e.with_traceback())
        if agent.run_id: await agent.cancel_run()
    
