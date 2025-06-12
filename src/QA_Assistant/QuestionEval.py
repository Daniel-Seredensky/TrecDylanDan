import os
import json
import enum
import time
import re
from typing import Dict, Any, List, Optional, Callable
from uuid import uuid4

from asyncinit import asyncinit
import asyncio

from InfoRetrieval.Search.Searcher import search

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Enum to track the lifecycle of the agent's work on a single question.
# ---------------------------------------------------------------------------
class QAStatus(enum.Enum):
    """Lifecycle of the agent's work on a single question."""

    NO_ANSWER = "no_answer"  # Agent has not produced any answer yet
    HAS_ANSWER = "has_answer"  # Agent produced an initial answer, may iterate
    FINISHED = "finished"  # Agent signalled it is fully confident

@asyncinit
class QuestionAssessmentAgent:
    """High‑level wrapper around an Azure OpenAI Assistant that evaluates a Q&A task."""

    MAX_ITERATIONS: int = 15
    POLL_INTERVAL_SECONDS: float = 0.5  # local back‑off when polling manually

    async def __init__(
        self,
        question: str,
        document: Dict[str, Any],
        *,
        client: Optional[AsyncOpenAI] = None,
        assistant_id: Optional[str] = None,
        model: str = os.getenv("AZURE_MODEL_KEY", "gpt-4o-mini"),
    ):
        self.question = question
        self.document = document
        self.status = QAStatus.NO_ANSWER
        self.full_answer_status: Optional[str] = None
        self.agent_id = uuid4()

        # ─── Async client setup ────────────────────────────────────
        if client is None:
            client = AsyncOpenAI(
                azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version="2024-02-15-preview",
            )
        self.client: AsyncOpenAI = client

        if assistant_id is None:
            # create_assistant is now an async method
            # so we must await it in an async constructor or after __init__.
            # Here we store a coroutine and await it in run().
            # Note: updated to use the asyncinit decorator from asyncinit package
            # now we can wait here for the results 
            self._assistant_id = await self.create_assistant(model)
        else:
            self.assistant_id = assistant_id

        # unique thread for each question
        self.thread = await self.client.beta.threads.create()

        # Registry mapping tool‑names in the Assistant schema to Python callables
        self.LOCAL_FUNCTIONS: Dict[str, Callable[..., Any]] = {
            "search": search,
            "document_selection": document_selection,
        }


    # ----------------------------------------------------------------------
    # Assistant creation helpers
    # ----------------------------------------------------------------------
    async def create_assistant(self, model: str) -> str:
        """Async creation of the Assistant with tool schema & system prompt."""
        system_prompt = self._build_system_prompt()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": (
                        "Execute a hybrid information retrieval pipeline optimized for recall and cost-efficiency on the MARCO V2.1 segmented dataset. "
                        "This pipeline retrieves up to 75 document metadata tuples by combining lexical BM25 search, semantic cross-encoder reranking, and pseudo-relevance feedback. "
                        "The tool requires two types of queries: BM25-optimized queries (to enhance recall by capturing varied phrasings and synonyms) and a master query (a concise, semantically rich paragraph to guide the reranker). "
                        "Generate 2–4 diverse and targeted BM25 queries to broadly cover relevant terminology and alternative phrasing. The master query should succinctly capture the semantic essence of the information needed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "BMQueries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of 2–4 distinct keyword-rich BM25 queries, each optimized for lexical search coverage."
                            },
                            "MasterQuery": {
                                "type": "string",
                                "description": "A concise paragraph or detailed sentence serving as the semantic seed for pseudo-relevance feedback and cross-encoder reranking."
                            }
                        },
                        "required": ["BMQueries", "MasterQuery"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "document_selection",
                    "description": (
                        "Select up to 3 documents by ID; request best fragment or full text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "documentIds": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of document IDs (max 5).",
                            },
                            "bestFragment": {
                                "type": "boolean",
                                "description": "true for best fragment; false for full text.",
                            },
                        },
                        "required": ["documentIds", "bestFragment"],
                    }
                }
            }
        ]
        
        assistant = await self.client.beta.assistants.create(
            name="Question-Assessment-Agent" ,
            model=model,
            tools=tools,
            instructions=system_prompt,
        )
        return assistant.id

    def _build_system_prompt(self) -> str:
        """Return the system prompt injected when the Assistant is created."""
        doc_text = json.dumps(self.document, ensure_ascii=False)
        prompt = (
            "You are a **Question Assessment AI Agent** specialised in information "
            "retrieval, fact‑checking, and rapid synthesis.\n\n"
            "You will be given: \n"
            "• A document (JSON) — provided inline below.\n"
            "• A question to answer.\n\n"
            f"DOCUMENT: {doc_text}\n"
            f"QUESTION: {self.question}\n\n"
            "You have access to two tools (call them exactly as specified):\n"
            "1. **Search**(KWQueries: array[str], seed: str) -> list[75] "
            "returns document metadata (url, title, headers, documentId).\n"
            "2. **document_selection**(documentIds: array[str], bestFragment: bool) "
            "-> object mapping each documentId to content (best fragment or full text).\n\n"
            "# Working style\n"
            "• First, *use tools* to collect evidence.\n"
            "• Strive to craft an initial answer quickly.\n"
            "• Always respond in the following wrappers: \n"
            "    <notepad>your running thoughts</notepad>\n"
            "    Then either: \n"
            "      <noAnswer></noAnswer> (if no answer yet) OR \n"
            "      <answer>{\"question\": ..., \"answer\": ..., \"citations\": [...], \"finished\": false}</answer>\n"
            "• After an initial answer, iteratively improve it until confident, then set "
            "  `finished` to true.\n"
            "• Cite ONLY documentIds (string IDs) in the `citations` array.\n"
            "• Stop when confident or after 15 internal steps, whichever is first.\n"
        )
        return prompt

    # ------------------------------------------------------------------
    # Public API – run the agent end‑to‑end on the current question
    # ------------------------------------------------------------------
    async def run(self) -> Dict[str, Any]:
        # Now handled in async initialization
        # Step A: ensure assistant_id is ready
        # if self._assistant_id_coro is not None:
            #self.assistant_id = await self._assistant_id_coro

        # Step B: create a new thread to drive this assistant
        # Note: moved to async initialization

        # 1️⃣ Seed user question (async)
        await self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=f"Please evaluate the question: {self.question}",
        )

        # 2️⃣ Kick-off the assistant run (async)
        run = await self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id,
        )

        # 3️⃣ Manual polling loop with tool dispatch (async)
        for _ in range(self.MAX_ITERATIONS):
            # 3a) Poll until run.status moves out of "queued"/"in_progress"
            while True:
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id, run_id=run.id
                )
                if run.status in {"queued", "in_progress"}:
                    await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
                    continue
                break

            if run.status == "completed":
                msg = await self._latest_assistant_message()
                self._update_status(msg)
                return {"status": self.status.value, "content": self.full_answer_status}

            if run.status != "requires_action":
                raise RuntimeError(f"Run ended in unexpected state: {run.status}")

            # 3b) Dispatch tools in parallel (use asyncio.to_thread for blocking tools)
            calls = run.required_action.submit_tool_outputs.tool_calls  # type: ignore[attr-defined]

            # We can fire each tool call non-blocking:
            tool_outputs = await asyncio.gather(
                *[self._dispatch_tool(call) for call in calls]
            )

            run = await self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread.id, run_id=run.id, tool_outputs=tool_outputs
            )

            # 3c) Update status after each assistant message
            msg = await self._latest_assistant_message()
            self._update_status(msg)

        # 4️⃣ Fallback if we never got a “completed” status in MAX_ITERATIONS
        await self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=(
                "Max iterations reached without a definitive answer. "
                "Compose *one* response using the required format:\n"
                "<notepad>brief reflection</notepad>\n"
                f"<answer>{{\"question\": \"{self.question}\", "
                "\"answer\": \"State which info you sought and why it was insufficient.\", "
                "\"citations\": [], \"finished\": false}}</answer>"
            ),
        )

        # Create a short follow-up run with fewer steps
        run = await self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id,
            max_rounds=3,
        )

        if run.status == "completed":
            msg = await self._latest_assistant_message()
            self._update_status(msg)
            return {"status": self.status.value, "content": self.full_answer_status}

        # If fallback still fails, return whatever you last saw
        return {"status": self.status.value, "content": self.full_answer_status}
    
    # ────────────────────────────────────────────────────────────────────────────
    # Dispatching a single tool call
    # ────────────────────────────────────────────────────────────────────────────
    async def _dispatch_tool(self, call) -> Dict[str, Any]:
        """Invoke a blocking local function on a thread pool to avoid blocking the event loop."""
        fn_name = call.function.name  # type: ignore[attr-defined]
        args = json.loads(call.function.arguments or "{}")  # type: ignore[attr-defined]
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
    def _cleanup_local_workspace(self):
        os.remove(os.getenv("BM25_RESULTS_PATH") + self.agent_id + ".json")


# ---------------------------------------------------------------------------
# Convenience procedural API
# ---------------------------------------------------------------------------

async def assess_question(question: str, document: Dict[str, Any]) -> Dict[str, Any]:
    """Simple functional entry point for callers outside the class world."""
    agent = await QuestionAssessmentAgent(question, document)
    return await agent.run()

# Example usage for when I go to parallelize them in contextBuilder
async def example():
    tasks = []
    # Suppose you have a list of (question, document) pairs:
    for question, document in $your_question_document_list:
        tasks.append(assess_question(question, document))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            print("Agent failed:", r)
        else:
            print("Agent result:", r)