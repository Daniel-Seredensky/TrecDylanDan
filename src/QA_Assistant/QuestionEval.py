import os
import json
import re
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Union
from uuid import uuid4
from openai.types.beta.threads import Message, MessageContent


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
    HAS_ANSWER = "has_answer"
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
        self.results_path = os.path.join(os.getenv("BM25_RESULTS_PATH"), f"{self.agent_id}")
        os.makedirs(self.results_path, exist_ok=True)

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

        # Local function‑tools wired into the assistant manifest
        self.LOCAL_FUNCTIONS: Dict[str, Callable[..., Any]] = {
            "search": search,
            "document_selection": select_documents,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ────────────────────────────────────────────────────────────────────────
    async def run(self) -> Dict[str, Any]:
        """Kick off a streaming run and return final / partial answer JSON."""

        # 1.  Seed user message ------------------------------------------------
        seed = (
            "This is the reference document the questions are based upon:\n" +
            json.dumps(self.document, ensure_ascii=False) + "\n\n" +
            "The questions you must answer:\n" + self.question + "\n\n" +
            "Work efficiently and accurately."
        )
        await self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=seed,
        )

        # 2.  Start assistant run – **streaming** -----------------------------
        stream = await self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id,
            stream=True,
        )

        # 3.  Event‑driven loop over the server‑sent events -------------------
        async for event in stream:
            e_type = getattr(event, "event", None)

            # ── Assistant message complete ---------------------------------
            if e_type == "thread.message.completed" and event.data.role == "assistant":
                self._update_status(event.data)

            # ── Tool calls ---------------------------------------------------
            if e_type == "thread.run.requires_action":
                calls = event.data.required_action.submit_tool_outputs.tool_calls
                # Evaluate tool calls concurrently
                outputs = await asyncio.gather(*[self._dispatch_tool(c) for c in calls])

                # Resume the run – still in streaming mode
                stream = await self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=event.data.id,  # run‑id now comes from data.id
                    tool_outputs=outputs,
                    stream=True,
                )
                continue  # next events come from the new stream generator

            # ── Terminal states ---------------------------------------------
            if e_type == "thread.run.completed":
                return {"status": self.status.value, "content": self.full_answer_status}

            if e_type.startswith("thread.run.") and getattr(event.data, "status", "") in {
                "failed", "cancelled", "expired", "incomplete"
            }:
                return {
                    "status": self.status.value,
                    "content": self.full_answer_status,
                    "run_state": event.data.status,
                }

        # 4.  Fallback – stream exhausted unexpectedly ------------------------
        await self._force_final_prompt()
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
        result = await self.LOCAL_FUNCTIONS[fn_name](**args)
        payload = json.dumps(result)[:5120]
        return {"tool_call_id": call.id, "output": payload}

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
                "<notepad>\n"
                "• Briefly (1–3 bullets) explain what you tried and why it was insufficient.  \n"
                "• Note what additional evidence or queries would likely resolve the question.\n"
                "</notepad>\n\n"
                "<answer>{\n"
                f"\"question\": \"{self.question}\",\n"
                "\"answer\": \"Summarise what you know so far and state clearly why you are not yet fully confident.\",\n"
                "\"citations\": [],\n"
                "\"finished\": false\n"
                "}</answer>"
            ),
        )

    def _update_status(self, assistant_msg: Message) -> None:
        """Extract JSON inside <answer>…</answer> and update status flags."""
        content = self._as_text(assistant_msg.content)          
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            payload = match.group(1).strip()
            self.full_answer_status = payload
            self.status = (QAStatus.FINISHED
                        if '"finished": true' in payload
                        else QAStatus.HAS_ANSWER)
        else:
            self.full_answer_status = ""
            self.status = QAStatus.NO_ANSWER

    def _as_text(blocks: Union[str, List[MessageContent]]) -> str:
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
    result = await agent.run()
    agent.close()
    return result
