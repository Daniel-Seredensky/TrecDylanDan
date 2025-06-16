import os
import json
from enum import Enum
import re
from typing import Dict, Any, Optional, Callable
from uuid import uuid4

from asyncinit import asyncinit
import asyncio
import aiofiles

from InfoRetrieval.Search.Searcher import search
from InfoRetrieval.Utils import document_selection

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Enum to track the lifecycle of the agent's work on a single question.
# ---------------------------------------------------------------------------
ASSISTANT_ID_FILE = "DerivedData/Assistant/AssistantId.txt"

class QAStatus(Enum):
    NO_ANSWER = "no_answer"
    HAS_ANSWER = "has_answer"
    FINISHED = "finished"

@asyncinit
class QuestionAssessmentAgent:
    MAX_ITERATIONS: int = 15
    POLL_INTERVAL_SECONDS: float = 0.5

    async def __init__(
        self,
        question: str,
        document: Dict[str, Any],
        client: Optional[AsyncOpenAI] = None,
        assistant_id: Optional[str] = None,
    ):
        """
        Initialize the agent with a question and a document.
        """
        self.question = question
        self.document = document
        self.status = QAStatus.NO_ANSWER
        self.full_answer_status: Optional[str] = None
        self.agent_id = uuid4()
        self.results_path = os.getenv("BM25_RESULTS_PATH") + f"/{self.agent_id}"
        os.mkdir(self.results_path)

        # ─── Async client setup ────────────────────────────────────────
        if client is None:
            client = AsyncOpenAI(
                azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version="2024-02-15-preview",
            )
        self.client = client

        # Assistant ID now comes from Assistant.py’s cache
        if assistant_id is None:
            async with aiofiles.open(ASSISTANT_ID_FILE) as f:
                assistant_id = await f.read()
                asisstant_id = assistant_id.strip()
        self.assistant_id = assistant_id

        if self.assistant_id is None: raise ValueError("Assistant ID not found, create one first")

        # One thread per question
        self.thread = await self.client.beta.threads.create()

        self.LOCAL_FUNCTIONS: Dict[str, Callable[..., Any]] = {
            "search": search,
            "document_selection": document_selection,
        }

    # ------------------------------------------------------------------
    # Public API – run the agent end‑to‑end on the current question
    # ------------------------------------------------------------------
    async def run(self) -> Dict[str, Any]:
        # Seed user question 
        seed = "This is the reference document the questions are based upon:" + \
                f"{self.document}\n\n" + \
                "The questions you must answer" + \
                f"{self.question}\n]n" + \
                """
                Work efficiently and accurately. 
                """

        await self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=seed,
        )

        # Kick‑off the assistant run (async)
        run = await self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id,
        ) 

        # All non‑recoverable terminal states
        _ABORT_STATES = {"failed", "cancelled", "expired", "incomplete"}

        # Manual polling loop with tool dispatch (async)
        for _ in range(self.MAX_ITERATIONS):
            # Poll until run.status moves out of "queued"/"in_progress"
            while True:
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id, run_id=run.id
                )
                if run.status in {"queued", "in_progress"}:
                    await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
                    continue
                break

            # always capture the latest assistant message
            try:
                msg = await self._latest_assistant_message()
                self._update_status(msg)
            except IndexError:
                # No assistant messages yet (e.g., first tool call)
                pass

            # Handle run status
            if run.status == "completed":
                return {"status": self.status.value, "content": self.full_answer_status}

            if run.status in _ABORT_STATES:
                # Gracefully return whatever partial answer we have
                return {
                    "status": self.status.value,          # NO_ANSWER or HAS_ANSWER
                    "content": self.full_answer_status,   # may be ""
                    "run_state": run.status,
                }

            if run.status != "requires_action":
                # Unexpected state (shouldn’t happen, but keep original safeguard)
                raise RuntimeError(f"Run ended in unexpected state: {run.status}")

            # Dispatch tools in parallel
            calls = run.required_action.submit_tool_outputs.tool_calls  # type: ignore[attr-defined]
            tool_outputs = await asyncio.gather(
                *[self._dispatch_tool(call) for call in calls]
            )

            run = await self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread.id, run_id=run.id, tool_outputs=tool_outputs
            )

            # Update status after the assistant incorporates tool results
            msg = await self._latest_assistant_message()
            self._update_status(msg)

        # Fallback: max iterations reached
        await self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content= \
                """
                Max iterations or an unexpected error stopped the run.

                Compose **one final assistant message** and then **stop**.

                <notepad>
                • Briefly (1–3 bullets) explain what you tried and why it was insufficient.  
                • Note what additional evidence or queries would likely resolve the question.
                </notepad>

                <answer>{
                """ + \
                f"\"question\": \"{self.question}\"," + \
                """
                "answer": "Summarise what you know so far and state clearly why you are not yet fully confident.",
                "citations": [],
                "finished": false
                }</answer>
                """
        )

        # Short follow‑up run (3 rounds max)
        run = await self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id,
            max_rounds=3,
        )

        if run.status == "completed":
            msg = await self._latest_assistant_message()
            self._update_status(msg)
            return {"status": self.status.value, "content": self.full_answer_status}

        # Final graceful exit if still not completed
        return {"status": self.status.value, "content": self.full_answer_status}

    # ────────────────────────────────────────────────────────────────────────────
    # Dispatching a single tool call
    # ────────────────────────────────────────────────────────────────────────────
    async def _dispatch_tool(self, call) -> Dict[str, Any]:
        """Invoke a blocking local function on a thread pool to avoid blocking the event loop."""
        fn_name = call.function.name  # type: ignore[attr-defined]
        args = json.loads(call.function.arguments or "{}")  # type: ignore[attr-defined]
        if fn_name == "search":
            args["agentId"] =  self.agent_id# append agent search id for path creation
        # OpenAI/Azure caps tool output at 5 kB
        result = await self.LOCAL_FUNCTIONS[fn_name](**args)
        payload = json.dumps(result)[:5120]
        return {"tool_call_id": call.id, "output": payload}
    
    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    async def _latest_assistant_message(self):
        page = await self.client.beta.threads.messages.list(
            thread_id=self.thread.id, order="desc", limit=1
        )
        return page.data[0]

    def _update_status(self, assistant_msg):
        """Extract JSON inside <answer>...</answer> and update status."""
        content = assistant_msg.content or ""
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            json_payload = answer_match.group(1).strip()
            self.full_answer_status = json_payload  # store only JSON
            if "\"finished\": true" in json_payload:
                self.status = QAStatus.FINISHED
            else:
                self.status = QAStatus.HAS_ANSWER
        else:
            self.full_answer_status = ""  # reset if no answer present
            self.status = QAStatus.NO_ANSWER
    
    # Small util, removes scratch file 
    def close (self):
        os.rmdir(self.results_path)


# ---------------------------------------------------------------------------
# Convenience procedural API
# ---------------------------------------------------------------------------

async def assess_question(question: str, document: Dict[str, Any], client: AsyncOpenAI, assistant_id: str|None = None) -> Dict[str, Any]:
    """Simple functional entry point for callers outside the class world."""
    agent = await QuestionAssessmentAgent(question = question, document = document, client = client, assistant_id = assistant_id)
    res = await agent.run()
    agent.close()
    return res
