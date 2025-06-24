import os
import json
import re
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Union
from uuid import uuid4
from openai.types.beta.threads import Message, MessageContent
import traceback


from asyncinit import asyncinit
import aiofiles
from openai import AsyncAzureOpenAI

from src.QA_Assistant.Searcher import search
from src.QA_Assistant.DocSelect import select_documents

# ────────────────────────────────────────────────────────────────────────────────
# Constants & enums
# ────────────────────────────────────────────────────────────────────────────────
ASSISTANT_ID_FILE = "DerivedData/Assistant/AssistantId.txt"

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

    async def __init__(
        self,
        question: str,
        document: Dict[str, Any],
        *,
        client: AsyncAzureOpenAI,
        assistant_id: Optional[str] = None,
    ) -> None:
        # Basic fields
        self.question = question
        self.document = document
        self.status: QAStatus = QAStatus.NO_ANSWER
        self.full_answer_status: str | None = None

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
        Keep launching new runs on the same thread until

        • the assistant sets "finished": true   → QAStatus.FINISHED
        • or we exceed MAX_TOOL_ROUNDS          → fallback summary
        """

        # 0.  Seed message (sent exactly once) ---------------------------------
        seed = (
            "This is the reference document the questions are based upon:\n"
            + json.dumps(self.document, ensure_ascii=False) + "\n\n"
            "The questions you must answer:\n" + self.question + "\n\n"
            "Work efficiently and accurately. Call `search` and "
            "`document_selection` as needed; stop only when you are fully confident "
            'and set `"finished": true` in your <answer> block.'
        )
        await self.client.beta.threads.messages.create(
            thread_id=self.thread.id, role="user", content=seed
        )

        tool_calls_so_far = 0

        # 1.  MAIN LOOP – one Assistant *run* per turn -------------------------
        try:
            while self.status != QAStatus.FINISHED:

                if tool_calls_so_far >= self.MAX_TOOL_ROUNDS:
                    await self._force_final_prompt()
                    break  # exit while loop

                # ── 1a. start a streaming run
                stream = await self.client.beta.threads.runs.create(
                    thread_id=self.thread.id,
                    assistant_id=self.assistant_id,
                    stream=True,
                )
                self.run_id = None

                # ── 1b. consume SSE events for this run
                while True:
                    async for event in stream:
                        e = getattr(event, "event", None)

                        # remember run‑id once
                        if self.run_id is None and hasattr(event.data, "id"):
                            self.run_id = event.data.id

                        # assistant message completed
                        if e == "thread.message.completed" and event.data.role == "assistant":
                            content = self._as_text(event.data.content)
                            async with aiofiles.open(self.convo_path, "a") as f:
                                await f.write(content + "\n---\n")
                            self._update_status(event.data)

                        # tool call(s) required
                        if e == "thread.run.requires_action":
                            calls   = event.data.required_action.submit_tool_outputs.tool_calls
                            outputs = await asyncio.gather(*[self._dispatch_tool(c) for c in calls])

                            tool_calls_so_far += len(calls)

                            # continue streaming after tool outputs
                            stream = await self.client.beta.threads.runs.submit_tool_outputs(
                                thread_id=self.thread.id,
                                run_id=event.data.id,
                                tool_outputs=outputs,
                                stream=True,
                            )
                            break  # restart inner async‑for with new iterator

                        # run finished normally
                        if e == "thread.run.completed":
                            break  # leave inner async‑for

                        # run ended abnormally
                        if e.startswith("thread.run.") and getattr(event.data, "status", "") in {
                            "failed", "cancelled", "expired", "incomplete"
                        }:
                            return {
                                "status": self.status.value,
                                "content": self.full_answer_status,
                            }
                    else:
                        break  # iterator exhausted unexpectedly

                    # inner while restart logic
                    if self.status == QAStatus.FINISHED:
                        break

                # 1c. if not finished, poke assistant to continue
                if self.status != QAStatus.FINISHED and tool_calls_so_far < self.MAX_TOOL_ROUNDS:
                    await self.client.beta.threads.messages.create(
                        thread_id=self.thread.id,
                        role="user",
                        content="Continue working.",
                    )

        except Exception:
            traceback.print_exc()
            return {
                "status": self.status.value,
                "content": self.full_answer_status,
            }

        # 2.  DONE -------------------------------------------------------------
        return {
            "status": self.status.value,
            "content": self.full_answer_status,
        }

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
            payload = json.dumps(result)[:5120]
            async with aiofiles.open(self.tools_path, "a") as f:
                await f.write(
                    json.dumps(
                        {
                            "tool": fn_name,
                            "args": args,
                            "result": result,
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
            content=(
                "Max iterations or an unexpected error stopped the run.\n\n"
                "Compose **one final assistant message** and then **stop**.\n\n"
                "• Briefly (1–3 bullets) explain what you tried and why it was insufficient.  \n"
                "• Note what additional evidence or queries would likely resolve the question.\n"
                "Provide your answer with the same jsonic format as the other answers placing your description in the answer field. Make sure to put citations in the citations field if there are any."
            ),
        )

    def _update_status(self, assistant_msg: Message) -> None:
        """Extract JSON inside <answer>…</answer> and update status flags."""
        content = self._as_text(assistant_msg.content)          
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            payload = match.group(1).strip()
            self.full_answer_status = payload if payload != None else self.full_answer_status
            self.status = (QAStatus.FINISHED
                        if '"finished": true' in payload
                        else QAStatus.PARTIAL_ANSWER)

    def _as_text(self,blocks: List[MessageContent]) -> str:
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
        

    # Clean‑up – remove temp search directory ---------------------------------
    def close(self) -> None:
        try:
            os.rmdir(self.results_path)
        except OSError:
            # directory non‑empty or other I/O error – ignore for now
            pass

# ────────────────────────────────────────────────────────────────────────────────
# Convenience functional wrapper
# ────────────────────────────────────────────────────────────────────────────────
async def assess_question(
    *,
    question: str,
    document: Dict[str, Any],
    client: AsyncAzureOpenAI,
    assistant_id: str,
) -> Dict[str, Any]:
    if not assistant_id:
        raise ValueError ("Make an assistant with `get_or_create_assistant` first")
    """One‑shot helper for callers that don’t want to manage the class."""
    agent = await QuestionAssessmentAgent(
        question=question,
        document=document,
        client=client,
        assistant_id=assistant_id,
    )
    try: 
        result = await agent.run()
        return result
    except Exception as e:
        agent.cancel_run()
        traceback.print_exc()
    
